"""Scheduled stock decisions: whether to keep the current book or move to a target.

Keeping the book is the default. A move happens when the current book breaks a limit (position cap,
sector cap, or the stress budget under scenarios known at the review) by more than the drift band, or when
the target's estimated utility beats the current book's by at least `rebalance_min_improvement` after
the full trading cost, charged once. Utility = expected return - risk_aversion / 2 x variance (annual).
"""

import pandas as pd

from .config import Settings

TOL = 1e-6
HEDGE_REVIEW = "not applicable: hedge method undecided (milestone 7)"


def quarter_end_sessions(calendar: pd.DatetimeIndex, after: pd.Timestamp) -> pd.DatetimeIndex:
    """Last session of each completed quarter after `after`: a session whose next session is in a new
    quarter. The final session in the data is never a review, since its quarter may not be over."""
    cal = calendar[calendar > after]
    q = cal.to_period("Q")
    return cal[:-1][q[:-1] != q[1:]]


def utility(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, s: Settings) -> float:
    w = w[w > 0]
    return float(mu[w.index] @ w - s.risk_aversion / 2 * (w @ cov.loc[w.index, w.index] @ w))


def breaches(w: pd.Series, sectors: pd.Series, s: Settings, worst_stress_loss: float | None = None) -> list[str]:
    """Limits the book breaks by more than the drift band: position cap, sector cap and (when given) the stress
    budget. The band stops a target built right at a cap from counting as broken after normal price drift."""
    band = s.rebalance_drift_band + TOL
    out = [f"{t} at {v:.1%} > {s.max_position_weight:.0%} cap + {s.rebalance_drift_band:.0%} band"
           for t, v in w.items() if v > s.max_position_weight + band]
    by_sector = w.groupby(sectors.reindex(w.index)).sum()
    out += [f"{sec} at {v:.1%} > {s.max_sector_weight:.0%} sector cap + {s.rebalance_drift_band:.0%} band"
            for sec, v in by_sector.items() if v > s.max_sector_weight + band]
    if worst_stress_loss is not None and worst_stress_loss > s.stress_loss_budget + band:
        out.append(f"worst stress loss {worst_stress_loss:.1%} > {s.stress_loss_budget:.0%} budget "
                   f"+ {s.rebalance_drift_band:.0%} band")
    return out


def decide(current: pd.Series, target: pd.Series, target_status: str, mu: pd.Series, cov: pd.DataFrame,
           sectors: pd.Series, cost_usd: float, nav: float, s: Settings,
           current_stress_loss: float | None = None) -> dict:
    """Keep or move. current/target: weights of NAV. Returns the decision with its reasons and numbers."""
    broken = breaches(current, sectors, s, current_stress_loss)
    row = {"breaches": "; ".join(broken), "cost_usd": cost_usd, "cost_pct_nav": cost_usd / nav,
           "utility_current": utility(current, mu, cov, s), "hedge_review": HEDGE_REVIEW}
    if target.empty:
        return row | {"action": "hold", "reason": f"no feasible target ({target_status})", "utility_target": None,
                      "improvement": None}
    row["utility_target"] = utility(target, mu, cov, s)
    row["improvement"] = row["utility_target"] - row["utility_current"] - row["cost_pct_nav"]
    if broken:
        return row | {"action": "rebalance", "reason": "current book breaks a limit: " + row["breaches"]}
    if row["improvement"] >= s.rebalance_min_improvement:
        return row | {"action": "rebalance",
                      "reason": f"after-cost gain {row['improvement']:.2%} >= {s.rebalance_min_improvement:.2%}"}
    return row | {"action": "hold",
                  "reason": f"after-cost gain {row['improvement']:.2%} < {s.rebalance_min_improvement:.2%}"}
