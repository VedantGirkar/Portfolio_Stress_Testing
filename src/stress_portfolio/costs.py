"""Stock and option trading costs, shared by execution and reports.

Half-spread (and, for stocks, slippage) is inside the fill price; commission is the only separate cash fee.
"""

import numpy as np
import pandas as pd

from .config import Settings

SCENARIOS = {"low": 0.5, "base": 1.0, "high": 2.0}


def stock_costs(quantity: pd.Series, ref_price: pd.Series, s: Settings, scenario: str | None = None,
                split_factor=1.0) -> pd.DataFrame:
    """Itemized costs for signed share quantities (+ buy, - sell) against a reference price.

    fill_price = ref * (1 +/- (half_spread + slippage)); spread_cost and slippage_cost are the
    implicit dollars inside that price; commission = max(per-share charge, per-order minimum), charged
    on real shares: split-adjusted quantity / split_factor (the product of splits after the trade date).
    """
    mult = SCENARIOS[scenario or s.cost_scenario]
    qty = quantity.astype(float)
    side = np.sign(qty)
    shares = qty.abs()
    spread = ref_price * s.half_spread_bps * mult / 1e4
    slip = ref_price * s.slippage_bps * mult / 1e4
    commission = np.maximum(shares / split_factor * s.commission_per_share, s.min_commission_per_order) * mult
    out = pd.DataFrame({
        "fill_price": ref_price + side * (spread + slip),
        "commission": commission.where(shares > 0, 0.0),
        "spread_cost": shares * spread,
        "slippage_cost": shares * slip,
    })
    out["total_cost"] = out[["commission", "spread_cost", "slippage_cost"]].sum(axis=1)
    return out


def option_costs(contracts: pd.Series, bid: pd.Series, ask: pd.Series, multiplier: pd.Series, s: Settings,
                 scenario: str | None = None) -> pd.DataFrame:
    """Itemized costs for signed option contracts (+ buy, - sell) against a quoted bid/ask.

    Buys fill at the ask and sells at the bid. The half-spread around the mid is the implicit spread cost
    (scaled by the friction scenario, so "base" is exactly the quote); commission plus exchange/clearing fees
    per contract is the only separate cash fee. No extra slippage: the spread is already inside the fill.
    Premium is a funding cash flow, not a cost.
    """
    mult = SCENARIOS[scenario or s.cost_scenario]
    n = contracts.astype(float)
    side = np.sign(n)
    mid = (bid + ask) / 2
    half = (ask - bid) / 2 * mult
    fee = (s.option_commission_per_contract + s.option_fees_per_contract) * mult
    out = pd.DataFrame({"fill_price": (mid + side * half).clip(lower=0.0), "commission": n.abs() * fee,
                        "spread_cost": n.abs() * multiplier * half, "slippage_cost": 0.0})
    out["total_cost"] = out[["commission", "spread_cost", "slippage_cost"]].sum(axis=1)
    return out


def scenario_summary(quantity: pd.Series, ref_price: pd.Series, s: Settings) -> pd.DataFrame:
    """Total cost of the same orders under the low, base and high friction scenarios."""
    rows = {name: stock_costs(quantity, ref_price, s, name)[["commission", "spread_cost", "slippage_cost",
                                                              "total_cost"]].sum() for name in SCENARIOS}
    return pd.DataFrame(rows).T.rename_axis("scenario").reset_index()
