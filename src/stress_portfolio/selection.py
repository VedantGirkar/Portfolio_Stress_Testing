"""Eligible universe, 100-stock shortlist, and sequential 10-stock core.

An explainable heuristic, not a global optimization.
"""

import numpy as np
import pandas as pd

from .config import Settings

FEATURE_COLUMNS = ["momentum", "downside_dev_annual", "max_drawdown", "median_dollar_volume"]


def session_calendar(adj: pd.DataFrame) -> pd.DatetimeIndex:
    """Sessions where at least half the universe has a price; drops stray single-ticker dates."""
    return adj.index[adj.notna().mean(axis=1) >= 0.5]


def window_returns(adj: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """The 252 simple returns of the feature window on the session calendar; gaps stay missing."""
    window = adj.reindex(session_calendar(adj)).iloc[-s.feature_window_prices:]
    return (window / window.shift(1) - 1).iloc[1:]


def compute_features(adj: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Per-ticker features as of the last common session.

    momentum: adj[-1-skip] / adj[-1-skip-lookback] - 1 (126 sessions ending 21 before the latest).
    downside_dev_annual: sqrt(mean(min(r, 0)^2)) over all 252 window returns, times sqrt(252).
    max_drawdown: positive magnitude of the deepest fall from a running peak over the 253-price window.
    """
    cal = session_calendar(adj)
    adj, close, volume = adj.reindex(cal), close.reindex(cal), volume.reindex(cal)
    last_valid = adj.apply(lambda c: c.last_valid_index())
    pos = pd.Series(cal.get_indexer(last_valid.fillna(cal[0])), index=adj.columns)

    window = adj.iloc[-s.feature_window_prices:]
    rets = window_returns(adj, s)
    skip, look = s.momentum_skip_sessions, s.momentum_lookback_sessions

    feats = pd.DataFrame({
        "observations": adj.notna().sum(),
        "last_price_date": last_valid,
        "sessions_stale": (len(cal) - 1 - pos).where(last_valid.notna()),
        "complete_window": window.notna().all(),
        "momentum": adj.iloc[-1 - skip] / adj.iloc[-1 - skip - look] - 1,
        "downside_dev_annual": np.sqrt((rets.clip(upper=0) ** 2).mean()) * np.sqrt(s.trading_days_per_year),
        "max_drawdown": -(window / window.cummax() - 1).min(),
        "median_dollar_volume": (close * volume).iloc[-s.liquidity_window_sessions:].median(),
    })
    feats.index.name = "ticker"
    return feats


def eligibility(membership: pd.DataFrame, feats: pd.DataFrame, failures: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """One row per membership ticker with an eligibility decision and reason."""
    df = membership.merge(feats, left_on="provider_ticker", right_index=True, how="left")
    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask, text):
        reason.loc[mask & reason.eq("")] = text

    flag(df["provider_ticker"].isin(failures["ticker"]) | df["observations"].isna(), "download failed")
    flag(df["observations"] < s.min_observations, f"fewer than {s.min_observations} price observations")
    flag(df["sessions_stale"] > s.max_staleness_sessions, "latest price is stale")
    flag(~df["complete_window"].astype("boolean").fillna(False), "gaps in the recent feature window")
    flag(df["median_dollar_volume"] < s.min_median_dollar_volume,
         f"median dollar volume below ${s.min_median_dollar_volume / 1e6:.0f}M")

    # One share class per company (SEC CIK): keep the most liquid class.
    ok = reason.eq("")
    ranked = df.loc[ok].sort_values(["median_dollar_volume", "ticker"], ascending=[False, True])
    keeper = ranked.dropna(subset=["cik"]).groupby("cik")["ticker"].first()
    dup = ok & df["cik"].notna() & ~df["ticker"].isin(keeper)
    reason.loc[dup] = "other share class: " + df.loc[dup, "cik"].map(keeper)

    df["eligible"] = reason.eq("")
    df["eligibility_reason"] = reason.replace("", "eligible")
    return df


def sector_quotas(eligible: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Base slots per sector, then remaining slots one at a time to the largest proportional deficit."""
    counts = eligible["sector"].value_counts().sort_index()
    if counts.sum() < s.shortlist_size:
        raise ValueError(f"only {counts.sum()} eligible companies; need {s.shortlist_size}")
    quota = counts.clip(upper=s.base_slots_per_sector)
    if quota.sum() > s.shortlist_size:
        raise ValueError("base sector slots exceed the shortlist size")
    target = s.shortlist_size * counts / counts.sum()
    while quota.sum() < s.shortlist_size:
        deficit = (target - quota).where(quota < counts)
        quota[deficit.idxmax()] += 1  # idxmax takes the first sector alphabetically on ties
    return pd.DataFrame({"eligible_count": counts, "proportional_target": target.round(2), "quota": quota}) \
        .rename_axis("sector").reset_index()


def build_shortlist(eligible: pd.DataFrame, quotas: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Most liquid companies per sector up to the quota, then within-shortlist percentiles and score."""
    q = quotas.set_index("sector")["quota"]
    ranked = eligible.sort_values(["sector", "median_dollar_volume", "ticker"], ascending=[True, False, True])
    return score(ranked.loc[ranked.groupby("sector").cumcount() < ranked["sector"].map(q)].reset_index(drop=True), s)


def score(short: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Within-shortlist percentiles (higher is better) and the investment score."""
    short = short.copy()
    short["momentum_pct"] = short["momentum"].rank(pct=True)
    short["low_downside_pct"] = short["downside_dev_annual"].rank(ascending=False, pct=True)
    short["low_drawdown_pct"] = short["max_drawdown"].rank(ascending=False, pct=True)
    short["investment_score"] = (s.weight_momentum * short["momentum_pct"]
                                 + s.weight_low_downside * short["low_downside_pct"]
                                 + s.weight_low_drawdown * short["low_drawdown_pct"])
    return short


def select_core(short: pd.DataFrame, returns: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Greedy picks by investment_score - penalty * normalized average correlation with picks so far."""
    corr = returns[short["provider_ticker"]].corr()
    info = short.set_index("provider_ticker")
    picks: list[dict] = []
    while len(picks) < s.core_size:
        chosen = [p["provider_ticker"] for p in picks]
        held = pd.Series([info.at[t, "sector"] for t in chosen], dtype=object).value_counts()
        slots_left = s.core_size - len(picks)
        need = max(0, s.min_sectors - len(held))
        rows = []
        for t in info.index.difference(chosen):
            in_sector = int(held.get(info.at[t, "sector"], 0))
            if in_sector >= s.max_per_sector:
                continue
            if in_sector > 0 and slots_left - 1 < need:
                continue  # an old-sector pick would make min_sectors unreachable
            if chosen:
                avg = corr.loc[t, chosen].mean()
                avg = 1.0 if pd.isna(avg) else avg  # undefined correlation: assume the worst
                norm = (avg + 1) / 2
            else:
                avg, norm = np.nan, 0.0
            score = info.at[t, "investment_score"]
            rows.append((t, score - s.correlation_penalty * norm, score, avg))
        if not rows:
            raise ValueError(f"no feasible pick for core slot {len(picks) + 1} under sector constraints")
        t, sel, score, avg = sorted(rows, key=lambda r: (-r[1], -r[2], r[0]))[0]
        picks.append({"provider_ticker": t, "selection_order": len(picks) + 1, "selection_score": sel,
                      "avg_corr_at_selection": avg, "candidates_considered": len(rows)})
    return pd.DataFrame(picks)


def run_selection(membership: pd.DataFrame, bars: pd.DataFrame, failures: pd.DataFrame,
                  s: Settings) -> dict[str, pd.DataFrame]:
    from .data import wide

    adj = wide(bars, "adj_close")
    feats = compute_features(adj, wide(bars, "close"), wide(bars, "volume"), s)
    elig = eligibility(membership, feats, failures, s)
    eligible = elig.loc[elig["eligible"]]
    quotas = sector_quotas(eligible, s)
    short = build_shortlist(eligible, quotas, s)
    picks = select_core(short, window_returns(adj, s), s)

    short = short.merge(picks, on="provider_ticker", how="left")
    short["selection_order"] = short["selection_order"].astype("Int64")
    short["role"] = np.where(short["selection_order"].notna(), "core", "reserve")
    short["target_weight"] = np.where(short["role"] == "core", 1.0 / s.core_size, 0.0)
    core_sectors = short.loc[short["role"] == "core", "sector"].value_counts()
    capped = short["sector"].map(core_sectors).fillna(0) >= s.max_per_sector
    short["reason"] = np.select(
        [short["role"] == "core", capped],
        ["core pick #" + short["selection_order"].astype(str),
         "reserve: sector already has the maximum core holdings"],
        "reserve: selection score below the core picks",
    )
    short = short.sort_values(["role", "selection_order", "investment_score"], ascending=[True, True, False])
    short = short.reset_index(drop=True)
    return {
        "eligibility": elig,
        "sector_quotas": quotas,
        "shortlist": short,
        "core": short.loc[short["role"] == "core"].reset_index(drop=True),
        "reserves": short.loc[short["role"] == "reserve"].reset_index(drop=True),
        "cutoff": session_calendar(adj)[-1],
    }
