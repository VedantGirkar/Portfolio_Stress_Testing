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
    spy = history[s.benchmark_ticker]
    defs, positions = [], []
    for name, label, lo, hi, group in s.stress_events:
        searched = spy.loc[lo:hi].dropna()
        if len(searched) < 2:
            defs.append({"scenario": name, "label": label, "kind": "historical", "group": group,
                         "available": False, "note": f"no {s.benchmark_ticker} prices between {lo} and {hi}"})
            continue
        peak, trough, spy_ret = worst_drawdown_window(searched)
        shocks, sources = _event_shocks(stocks, history, betas, peak, trough, spy_ret, s)
        defs.append({"scenario": name, "label": label, "kind": "historical", "group": group, "available": True,
                     "window_start": peak.date(), "window_end": trough.date(),
                     "sessions": int(len(spy.loc[peak:trough]) - 1), "benchmark_return": spy_ret,
                     "note": f"{s.benchmark_ticker} peak-to-trough within {lo}..{hi}"})
        positions.append(apply_shocks(name, stocks, pd.Series(shocks), pd.Series(sources)))
    return defs, positions


def _event_shocks(stocks: pd.DataFrame, history: pd.DataFrame, betas: pd.Series, start, end, spy_ret: float,
                  s: Settings) -> tuple[dict, dict]:
    """Each stock's return start -> end: own prices, else its sector ETF, else beta x SPY (flagged proxies)."""
    etf_of = dict(s.sector_etfs)
    shocks, sources = {}, {}
    for t, row in stocks.iterrows():
        own = window_return(history[t], start, end) if t in history else None
        etf = etf_of.get(row["sector"])
        proxy = window_return(history[etf], start, end) if etf in history else None
        if own is not None:
            shocks[t], sources[t] = own, "own history"
        elif proxy is not None:
            shocks[t], sources[t] = proxy, f"sector ETF proxy ({etf})"
        else:
            shocks[t], sources[t] = betas[t] * spy_ret, f"beta x {s.benchmark_ticker} proxy"
    return shocks, sources


def capped_replay(stocks: pd.DataFrame, history: pd.DataFrame, betas: pd.Series, vol_index: pd.Series,
                  max_days: int, s: Settings) -> pd.DataFrame:
    """Historical events cut at a put's remaining life: each event starts at its usual peak and ends at the
    trough or at the last session within `max_days` calendar days of the peak, whichever comes first.

    Returns per event the book's stock P&L, SPY's return and the volatility index (decimal, e.g. VIX/100) on
    the end date, which hedging uses to reprice puts. An event without a volatility reading is marked
    unavailable rather than given an assumed volatility.
    """
    if (vol_index.dropna() > 3).any():
        raise ValueError("vol_index must be a decimal (VIX / 100), not index points")
    spy = history[s.benchmark_ticker]
    rows = []
    for name, label, lo, hi, group in s.stress_events:
        searched = spy.loc[lo:hi].dropna()
        if len(searched) < 2:
            continue
        peak, trough, _ = worst_drawdown_window(searched)
        end = spy.loc[peak:min(trough, peak + pd.Timedelta(days=max_days))].dropna().index[-1]
        spy_ret = float(spy[end] / spy[peak] - 1)
        shocks, _ = _event_shocks(stocks, history, betas, peak, end, spy_ret, s)
        vol = vol_index.get(end)
        rows.append({"scenario": name, "label": label, "group": group, "window_start": peak, "window_end": trough,
                     "capped_end": end, "elapsed_years": (end - peak).days / 365.0,
                     "book_pnl_usd": float((stocks["market_value"] * pd.Series(shocks).clip(lower=-1.0)).sum()),
                     "spy_return": spy_ret, "stress_vol": vol, "available": vol is not None and pd.notna(vol)})
    return pd.DataFrame(rows)


def hypothetical_shocks(stocks: pd.DataFrame, betas: pd.Series, combined_sector: str,
                        s: Settings) -> tuple[list, list]:
    """Beta-scaled market shocks, single-sector shocks (one per sector in `stocks`), and a compounded
    market-plus-sector shock in `combined_sector`."""
    defs, positions = [], []

    def add(name, label, shocks, source, note, market):
        defs.append({"scenario": name, "label": label, "kind": "hypothetical", "group": "train",
                     "available": True, "benchmark_return": market, "note": note})
        positions.append(apply_shocks(name, stocks, shocks, pd.Series(source, index=stocks.index)))

    for m in s.market_shocks:
        add(f"market{m:+.0%}", f"Market {m:+.0%}", betas * m, "beta x market", "stock return = beta x market move",
            m)
    m = s.combined_market_shock
    add(f"market{m:+.0%}_beta1", f"Market {m:+.0%}, all betas 1", pd.Series(m, index=stocks.index),
        "market, beta forced to 1", "crisis case: every stock moves with the market one-for-one", m)
    for sector in sorted(stocks["sector"].unique()):
        shock = (stocks["sector"] == sector) * s.sector_shock
        add(f"sector_{sector}", f"{sector} {s.sector_shock:+.0%}", shock.astype(float), "sector shock",
            "this sector only, market flat", 0.0)
    sector_part = (stocks["sector"] == combined_sector) * s.combined_sector_shock
    combined = (1 + betas * m) * (1 + sector_part) - 1
    add("combined", f"Market {m:+.0%} + {combined_sector} {s.combined_sector_shock:+.0%}", combined,
        "beta x market, compounded with sector", f"riskiest sector of the current book: {combined_sector}", m)
    return defs, positions


def scenario_returns(stocks: pd.DataFrame, history: pd.DataFrame, betas: pd.Series, combined_sector: str,
                     s: Settings) -> tuple[pd.DataFrame, pd.Series]:
    """Scenario x stock shock returns for any set of stocks (index = ticker, column sector), using the same
    scenarios as the book's stress test, plus each scenario's group. A validation event inside a training
    window is grouped "validation (overlaps training)" so it never counts as an out-of-sample check."""
    unit = stocks[["sector"]].assign(market_value=1.0)
    hist_defs, hist_pos = replay_events(unit, history, betas, s)
    hypo_defs, hypo_pos = hypothetical_shocks(unit, betas, combined_sector, s)
    matrix = pd.concat(hist_pos + hypo_pos).pivot(index="scenario", columns="ticker", values="shock_return")
    defs = pd.DataFrame(hist_defs + hypo_defs)
    defs.loc[_overlaps_training(defs), "group"] = "validation (overlaps training)"
    return matrix, defs.set_index("scenario")["group"].reindex(matrix.index)


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
