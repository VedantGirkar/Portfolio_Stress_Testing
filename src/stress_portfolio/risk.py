"""Risk of the current holdings: VaR, expected shortfall, volatility, drawdown, beta, exposures.

Method: historical simulation. Today's positions (fixed dollar exposures, cash earning zero) are
repriced with each past day's total returns (`adj_close`). These are hypothetical P&Ls of today's book,
not results the portfolio earned. Losses are reported as positive numbers. VaR is a percentile of the
loss distribution, not the maximum possible loss.

Discrete tail convention: with n P&L observations and confidence a, k = ceil(n * (1 - a)).
VaR = the k-th largest loss; ES = the mean of the k largest losses.
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import Settings

VAR_CONVENTION = "VaR = k-th largest loss, ES = mean of the k largest losses, k = ceil(n x (1 - confidence))"


def daily_returns(adj: pd.DataFrame, lookback: int | None = None) -> pd.DataFrame:
    """Simple daily returns on the sessions where every column has a return (no filling)."""
    rets = (adj / adj.shift(1) - 1).dropna(how="any")
    return rets if lookback is None else rets.iloc[-lookback:]


def horizon_returns(rets: pd.DataFrame, days: int) -> pd.DataFrame:
    """Compounded returns over overlapping `days`-session windows, labelled by the window's last day."""
    if days == 1:
        return rets
    return np.expm1(np.log1p(rets).rolling(days).sum()).dropna(how="any")


def var_es(pnl: pd.Series, confidence: float) -> tuple[float, float, int]:
    """Historical VaR and ES as positive losses, using the discrete convention above."""
    losses = np.sort(-pnl.to_numpy())[::-1]
    k = max(1, math.ceil(len(losses) * (1 - confidence) - 1e-9))
    return float(losses[k - 1]), float(losses[:k].mean()), k


def normal_var_es(pnl: pd.Series, confidence: float) -> tuple[float, float]:
    """Parametric VaR/ES assuming normally distributed P&L with the sample mean and standard deviation."""
    mu, sd = float(pnl.mean()), float(pnl.std(ddof=1))
    z = NormalDist().inv_cdf(confidence)
    return -(mu - z * sd), -(mu - sd * NormalDist().pdf(z) / (1 - confidence))


def max_drawdown(path: pd.Series) -> float:
    """Positive magnitude of the deepest fall from a running peak of a value path."""
    return max(0.0, float(-(path / path.cummax() - 1).min())) if len(path) else 0.0


def kupiec_p_value(exceedances: int, n: int, confidence: float) -> float:
    """Kupiec proportion-of-failures test: p-value that the exceedance rate equals 1 - confidence."""
    p, x = 1 - confidence, exceedances
    if n == 0:
        return float("nan")

    def loglik(q):
        return (n - x) * math.log(1 - q) + x * math.log(q) if 0 < q < 1 else 0.0

    lr = -2 * (loglik(p) - loglik(x / n))
    return math.erfc(math.sqrt(max(lr, 0.0) / 2))  # chi-square(1) survival function


def var_backtest(pnl: pd.Series, s: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1-day VaR estimated each day from all earlier days only (expanding window), then compared with
    that day's P&L. Returns the daily series and a summary per confidence level."""
    rows = []
    values = pnl.to_numpy()
    for i in range(s.var_backtest_min_window, len(values)):
        prior = pd.Series(values[:i])
        row = {"date": pnl.index[i], "pnl": values[i]}
        for c in s.var_confidences:
            row[f"var_{c:.0%}"] = var_es(prior, c)[0]
        rows.append(row)
    series = pd.DataFrame(rows)
    summary = []
    for c in s.var_confidences:
        if series.empty:
            break
        hit = -series["pnl"] > series[f"var_{c:.0%}"]
        series[f"exceed_{c:.0%}"] = hit
        n, x = len(series), int(hit.sum())
        summary.append({"confidence": c, "days_tested": n, "exceedances": x, "expected": n * (1 - c),
                        "kupiec_p_value": kupiec_p_value(x, n, c)})
    return series, pd.DataFrame(summary)


def risk_report(holdings: pd.DataFrame, adj: pd.DataFrame, benchmark: pd.Series, nav: pd.DataFrame,
                s: Settings) -> dict[str, pd.DataFrame]:
    """All risk tables for one valuation of the book.

    holdings: one row per position plus a CASH row (ticker, market_value, sector). nav: realized NAV history.
    """
    stocks = holdings.loc[holdings["ticker"] != "CASH"].set_index("ticker")
    total_nav = float(holdings["market_value"].sum())
    exposure = stocks["market_value"]
    weights = exposure / total_nav

    rets = daily_returns(adj[exposure.index], s.risk_lookback_sessions)
    bench = (benchmark / benchmark.shift(1) - 1).reindex(rets.index)
    if bench.isna().any():
        raise ValueError(f"{s.benchmark_ticker} returns missing on {bench.isna().sum()} risk-window sessions")
    window = {"lookback_sessions": len(rets), "start": rets.index[0].date(), "end": rets.index[-1].date()}

    var_rows = []
    for days in s.var_horizons_days:
        pnl = horizon_returns(rets, days) @ exposure
        for c in s.var_confidences:
            hv, he, k = var_es(pnl, c)
            nv, ne = normal_var_es(pnl, c)
            for method, v, e, note in [("historical", hv, he, f"k = {k} of {len(pnl)}; " + VAR_CONVENTION),
                                       ("normal", nv, ne, "normal P&L with the sample mean and std")]:
                var_rows.append({"method": method, "horizon_days": days, "confidence": c, "observations": len(pnl),
                                 "var_usd": v, "var_pct_nav": v / total_nav, "es_usd": e, "es_pct_nav": e / total_nav,
                                 "overlapping": days > 1, **window, "convention": note})

    port = rets @ weights  # daily return of today's book, cash earning zero
    cov = rets.cov() * s.trading_days_per_year
    port_vol = float(np.sqrt(weights @ cov @ weights))
    beta = rets.apply(lambda r: r.cov(bench)) / bench.var()
    positions = pd.DataFrame({
        "company": stocks["company"], "sector": stocks["sector"], "weight_of_nav": weights,
        "volatility_annual": rets.std() * np.sqrt(s.trading_days_per_year),
        "beta": beta,
        "risk_contribution_pct": weights * (cov @ weights) / port_vol**2,  # shares of portfolio variance; sum to 1
    }).rename_axis("ticker").reset_index()

    sectors = holdings.groupby("sector", as_index=False)["market_value"].sum()
    sectors["weight_of_nav"] = sectors["market_value"] / total_nav
    sleeve = exposure / exposure.sum()
    hhi = float((sleeve**2).sum())
    path = (1 + port).cumprod()
    bench_path = (1 + bench).cumprod()
    realized = nav.set_index("date")["nav"]
    summary = pd.DataFrame([
        ("volatility", port_vol, "annual, % of NAV", "daily std x sqrt(252) of today's book"),
        ("beta", float(weights @ beta), f"vs {s.benchmark_ticker}", "sum of weight x stock beta (cash beta 0)"),
        ("correlation", float(port.corr(bench)), f"vs {s.benchmark_ticker}", "daily returns"),
        ("max_drawdown", max_drawdown(path), "% of NAV", "hypothetical path of today's book"),
        (f"{s.benchmark_ticker}_max_drawdown", max_drawdown(bench_path), "%", "same window"),
        ("herfindahl_index", hhi, "equity sleeve", "sum of squared stock weights"),
        ("effective_holdings", 1 / hhi, "stocks", "1 / Herfindahl"),
        ("largest_position", float(weights.max()), "% of NAV", str(weights.idxmax())),
        ("cash_weight", 1 - float(weights.sum()), "% of NAV", "cash earns zero in risk estimates"),
        ("realized_return", float(realized.iloc[-1] / s.initial_capital - 1), "% since funding",
         f"realized NAV over {len(realized)} session(s)"),
        ("realized_max_drawdown", max_drawdown(realized), "% of NAV", "realized NAV path"),
    ], columns=["metric", "value", "unit", "note"]).assign(**{k: str(v) for k, v in window.items()})

    corr = pd.concat([rets, bench.rename(s.benchmark_ticker)], axis=1).corr()
    series = pd.DataFrame({"portfolio_return": port, "benchmark_return": bench, "portfolio_path": path,
                           "benchmark_path": bench_path, "portfolio_drawdown": path / path.cummax() - 1,
                           "pnl_usd": rets @ exposure}).rename_axis("date").reset_index()
    backtest, backtest_summary = var_backtest(series.set_index("date")["pnl_usd"], s)
    return {
        "risk_var": pd.DataFrame(var_rows),
        "risk_summary": summary,
        "risk_positions": positions,
        "risk_sectors": sectors,
        "risk_correlation": corr.rename_axis("ticker").reset_index(),
        "risk_series": series,
        "var_backtest": backtest,
        "var_backtest_summary": backtest_summary,
    }
