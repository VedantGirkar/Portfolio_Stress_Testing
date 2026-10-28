"""Stress tests of the current holdings: historical event replay and hypothetical shocks.

Every scenario gives each stock one return over the scenario, then P&L = market value x return
(cash is unchanged). This is a replay of today's book, not an investable backtest: nobody held these
stocks through those events with today's weights.

Historical returns are compounded price ratios over the window (adj_close), never sums of daily returns.
A stock without its own prices across the window gets a flagged proxy: its sector ETF's return, else
beta x SPY's return. Market and sector shocks are defined so they do not overlap: a sector shock is an
extra move on top of the market move, and the two are compounded.
"""

import pandas as pd

from .config import Settings

COLUMNS = ["scenario", "ticker", "sector", "market_value", "shock_return", "pnl_usd", "source"]


def worst_drawdown_window(prices: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """(peak date, trough date, return) of the deepest peak-to-trough fall in a price series."""
    p = prices.dropna()
    drawdown = p / p.cummax() - 1
    trough = drawdown.idxmin()
    peak = p.loc[:trough].idxmax()
    return peak, trough, float(p[trough] / p[peak] - 1)


def window_return(prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """Compounded return start -> end (price ratio), or None without a price on both dates (e.g. not yet listed)."""
    a, b = prices.get(start), prices.get(end)
    if a is None or b is None or pd.isna(a) or pd.isna(b) or a <= 0:
        return None
    return float(b / a - 1)


def apply_shocks(scenario: str, stocks: pd.DataFrame, shocks: pd.Series, sources: pd.Series) -> pd.DataFrame:
    """Position P&L for one scenario. Returns are floored at -100% (a long stock cannot lose more)."""
    r = shocks.reindex(stocks.index).clip(lower=-1.0)
    return pd.DataFrame({"scenario": scenario, "ticker": stocks.index, "sector": stocks["sector"].values,
                         "market_value": stocks["market_value"].values, "shock_return": r.values,
                         "pnl_usd": (stocks["market_value"] * r).values,
                         "source": sources.reindex(stocks.index).values})


def replay_events(stocks: pd.DataFrame, history: pd.DataFrame, betas: pd.Series, s: Settings) -> tuple[list, list]:
    """Historical replays. history: wide adj_close of the holdings, SPY and sector ETFs."""
    etf_of = dict(s.sector_etfs)
    spy = history[s.benchmark_ticker]
    defs, positions = [], []
    for name, label, lo, hi, group in s.stress_events:
        searched = spy.loc[lo:hi].dropna()
        if len(searched) < 2:
            defs.append({"scenario": name, "label": label, "kind": "historical", "group": group,
                         "available": False, "note": f"no {s.benchmark_ticker} prices between {lo} and {hi}"})
            continue
        peak, trough, spy_ret = worst_drawdown_window(searched)
        shocks, sources = {}, {}
        for t, row in stocks.iterrows():
            own = window_return(history[t], peak, trough) if t in history else None
            etf = etf_of.get(row["sector"])
            proxy = window_return(history[etf], peak, trough) if etf in history else None
            if own is not None:
                shocks[t], sources[t] = own, "own history"
            elif proxy is not None:
                shocks[t], sources[t] = proxy, f"sector ETF proxy ({etf})"
            else:
                shocks[t], sources[t] = betas[t] * spy_ret, f"beta x {s.benchmark_ticker} proxy"
        defs.append({"scenario": name, "label": label, "kind": "historical", "group": group, "available": True,
                     "window_start": peak.date(), "window_end": trough.date(),
                     "sessions": int(len(spy.loc[peak:trough]) - 1), "benchmark_return": spy_ret,
                     "note": f"{s.benchmark_ticker} peak-to-trough within {lo}..{hi}"})
        positions.append(apply_shocks(name, stocks, pd.Series(shocks), pd.Series(sources)))
    return defs, positions


def hypothetical_shocks(stocks: pd.DataFrame, betas: pd.Series, combined_sector: str,
                        s: Settings) -> tuple[list, list]:
    """Beta-scaled market shocks, single-sector shocks (one per sector in `stocks`), and a compounded
    market-plus-sector shock in `combined_sector`."""
    defs, positions = [], []

    def add(name, label, shocks, source, note):
        defs.append({"scenario": name, "label": label, "kind": "hypothetical", "group": "train",
                     "available": True, "note": note})
        positions.append(apply_shocks(name, stocks, shocks, pd.Series(source, index=stocks.index)))

    for m in s.market_shocks:
        add(f"market{m:+.0%}", f"Market {m:+.0%}", betas * m, "beta x market", "stock return = beta x market move")
    m = s.combined_market_shock
    add(f"market{m:+.0%}_beta1", f"Market {m:+.0%}, all betas 1", pd.Series(m, index=stocks.index),
        "market, beta forced to 1", "crisis case: every stock moves with the market one-for-one")
    for sector in sorted(stocks["sector"].unique()):
        shock = (stocks["sector"] == sector) * s.sector_shock
        add(f"sector_{sector}", f"{sector} {s.sector_shock:+.0%}", shock.astype(float), "sector shock",
            "this sector only, market flat")
    sector_part = (stocks["sector"] == combined_sector) * s.combined_sector_shock
    combined = (1 + betas * m) * (1 + sector_part) - 1
    add("combined", f"Market {m:+.0%} + {combined_sector} {s.combined_sector_shock:+.0%}", combined,
        "beta x market, compounded with sector", f"riskiest sector of the current book: {combined_sector}")
    return defs, positions


def scenario_returns(stocks: pd.DataFrame, history: pd.DataFrame, betas: pd.Series, combined_sector: str,
                     s: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Scenario x stock shock returns for any set of stocks (index = ticker, column sector), using the same
    scenarios as the book's stress test, plus each scenario's train/validation group."""
    unit = stocks[["sector"]].assign(market_value=1.0)
    hist_defs, hist_pos = replay_events(unit, history, betas, s)
    hypo_defs, hypo_pos = hypothetical_shocks(unit, betas, combined_sector, s)
    matrix = pd.concat(hist_pos + hypo_pos).pivot(index="scenario", columns="ticker", values="shock_return")
    groups = pd.DataFrame(hist_defs + hypo_defs).set_index("scenario")["group"]
    return matrix, groups.reindex(matrix.index)


def run_stress(holdings: pd.DataFrame, risk_positions: pd.DataFrame, history: pd.DataFrame,
               s: Settings) -> dict[str, pd.DataFrame]:
    """Scenario totals, position attribution and sector attribution for the current book."""
    stocks = holdings.loc[holdings["ticker"] != "CASH"].set_index("ticker")[["sector", "market_value"]]
    nav = float(holdings["market_value"].sum())
    rp = risk_positions.set_index("ticker")
    betas, variance_share = rp["beta"].reindex(stocks.index), rp["risk_contribution_pct"].reindex(stocks.index)

    hist_defs, hist_pos = replay_events(stocks, history, betas, s)
    riskiest = variance_share.groupby(stocks["sector"]).sum().idxmax()
    hypo_defs, hypo_pos = hypothetical_shocks(stocks, betas, riskiest, s)
    positions = pd.concat(hist_pos + hypo_pos, ignore_index=True)
    scenarios = pd.DataFrame(hist_defs + hypo_defs)

    totals = positions.groupby("scenario").agg(
        pnl_usd=("pnl_usd", "sum"), proxied_positions=("source", lambda x: int(x.str.contains("proxy").sum())))
    scenarios = scenarios.merge(totals, left_on="scenario", right_index=True, how="left")
    scenarios["pnl_pct_nav"] = scenarios["pnl_usd"] / nav
    scenarios["nav_after"] = nav + scenarios["pnl_usd"]
    scenarios["overlaps_training_event"] = _overlaps_training(scenarios)
    sectors = positions.groupby(["scenario", "sector"], as_index=False)["pnl_usd"].sum()

    # attribution must reconcile: positions -> sectors -> scenario total
    check = sectors.groupby("scenario")["pnl_usd"].sum().reindex(totals.index)
    if (check - totals["pnl_usd"]).abs().max() > 1e-6:
        raise AssertionError("sector attribution does not sum to the scenario total")
    return {"stress_scenarios": scenarios, "stress_positions": positions[COLUMNS], "stress_sectors": sectors}


def _overlaps_training(scenarios: pd.DataFrame) -> pd.Series:
    """True for a validation event whose window overlaps a training event's window (not truly held out)."""
    out = pd.Series(False, index=scenarios.index)
    if "window_start" not in scenarios:
        return out
    hist = scenarios.loc[(scenarios["kind"] == "historical") & scenarios["available"]]
    train = hist.loc[hist["group"] == "train"]
    for i, row in hist.loc[hist["group"] == "validation"].iterrows():
        out[i] = bool(((train["window_start"] <= row["window_end"]) & (train["window_end"] >= row["window_start"])).any())
    return out
