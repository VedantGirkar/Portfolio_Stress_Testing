"""Stock trading costs, shared by execution and reports.

Half-spread and slippage are inside the fill price; commission is the only separate cash fee.
"""

import numpy as np
import pandas as pd

from .config import Settings

SCENARIOS = {"low": 0.5, "base": 1.0, "high": 2.0}


def stock_costs(quantity: pd.Series, ref_price: pd.Series, s: Settings, scenario: str | None = None) -> pd.DataFrame:
    """Itemized costs for signed share quantities (+ buy, - sell) against a reference price.

    fill_price = ref * (1 +/- (half_spread + slippage)); spread_cost and slippage_cost are the
    implicit dollars inside that price; commission = max(per-share charge, per-order minimum).
    """
    mult = SCENARIOS[scenario or s.cost_scenario]
    qty = quantity.astype(float)
    side = np.sign(qty)
    shares = qty.abs()
    spread = ref_price * s.half_spread_bps * mult / 1e4
    slip = ref_price * s.slippage_bps * mult / 1e4
    commission = np.maximum(shares * s.commission_per_share, s.min_commission_per_order) * mult
    out = pd.DataFrame({
        "fill_price": ref_price + side * (spread + slip),
        "commission": commission.where(shares > 0, 0.0),
        "spread_cost": shares * spread,
        "slippage_cost": shares * slip,
    })
    out["total_cost"] = out[["commission", "spread_cost", "slippage_cost"]].sum(axis=1)
    return out


def scenario_summary(quantity: pd.Series, ref_price: pd.Series, s: Settings) -> pd.DataFrame:
    """Total cost of the same orders under the low, base and high friction scenarios."""
    rows = {name: stock_costs(quantity, ref_price, s, name)[["commission", "spread_cost", "slippage_cost",
                                                              "total_cost"]].sum() for name in SCENARIOS}
    return pd.DataFrame(rows).T.rename_axis("scenario").reset_index()
