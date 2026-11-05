"""Orders from target weights, and simulated fills. Never changes the portfolio itself."""

import numpy as np
import pandas as pd

from .config import Settings
from .costs import option_costs, stock_costs

FILL_COLUMNS = ["ticker", "order_qty", "filled_qty", "ref_price", "fill_price", "commission", "spread_cost",
                "slippage_cost", "total_cost", "status", "reason"]
OPTION_FILL_COLUMNS = ["contract", "order_qty", "filled_qty", "bid", "ask", "multiplier", "fill_price",
                       "commission", "spread_cost", "slippage_cost", "total_cost", "status", "reason"]


def generate_orders(positions: dict[str, float], target_weights: pd.Series, decision_prices: pd.Series,
                    nav: float) -> pd.DataFrame:
    """Whole-share orders that move current holdings to `target_weights` (fractions of NAV).

    Target quantities round down so rounding never needs extra cash. No order when nothing changes.
    """
    tickers = sorted(set(positions) | set(target_weights.index))
    current = pd.Series(positions, dtype=float).reindex(tickers, fill_value=0.0)
    price = decision_prices.reindex(tickers)
    # tiny epsilon: an unchanged target like 278.9999999 shares must not round down into a 1-share sell
    target = np.floor(target_weights.reindex(tickers, fill_value=0.0) * nav / price + 1e-9)
    orders = pd.DataFrame({"ticker": tickers, "current_qty": current.values, "target_qty": target.values,
                           "decision_price": price.values})
    orders["order_qty"] = orders["target_qty"] - orders["current_qty"]
    return orders.loc[orders["order_qty"].ne(0)].reset_index(drop=True)


def execute(orders: pd.DataFrame, open_prices: pd.Series, cash: float, s: Settings,
            split_factor: pd.Series | None = None) -> pd.DataFrame:
    """Fill at the session open plus costs: sells first, then buys trimmed so cash never goes negative.
    split_factor (per ticker, default 1) converts split-adjusted shares to real shares for commissions."""
    df = orders[["ticker", "order_qty"]].copy()
    df["ref_price"] = open_prices.reindex(df["ticker"]).values
    df["split_factor"] = 1.0 if split_factor is None else split_factor.reindex(df["ticker"]).fillna(1.0).values
    bad = df["ref_price"].isna() | (df["ref_price"] <= 0) | df["order_qty"].isna()
    df["filled_qty"] = df["order_qty"].where(~bad, 0.0)
    df["status"] = np.where(bad, "rejected", "filled")
    df["reason"] = np.where(bad, "no valid open price or decision price", "")

    sells, buys = df["filled_qty"] < 0, df["filled_qty"] > 0
    available = cash - _cash_need(df.loc[sells], s)  # sells add cash (negative need)
    need = _cash_need(df.loc[buys], s)
    if need > available:
        scale = max(available, 0.0) / need
        df.loc[buys, "filled_qty"] = np.floor(df.loc[buys, "filled_qty"] * scale)
        for i in df.index[buys][::-1]:  # trim the last buys share by share if commissions still overshoot
            while df.at[i, "filled_qty"] > 0 and _cash_need(df.loc[buys], s) > available:
                df.at[i, "filled_qty"] -= 1
        trimmed = buys & (df["filled_qty"] < df["order_qty"])
        df.loc[trimmed, "status"] = "adjusted"
        df.loc[trimmed, "reason"] = "reduced to fit available cash"
        empty = buys & (df["filled_qty"] == 0)
        df.loc[empty, ["status", "reason"]] = ["rejected", "insufficient cash"]

    costs = stock_costs(df["filled_qty"], df["ref_price"].fillna(0.0), s, split_factor=df["split_factor"])
    return pd.concat([df, costs], axis=1)[FILL_COLUMNS]


def execute_options(orders: pd.DataFrame, quotes: pd.DataFrame, cash: float, s: Settings) -> pd.DataFrame:
    """Fill option orders (contract, order_qty) against the fill session's 15:45 quotes (index = contract,
    columns bid, ask, multiplier). Buys need a positive ask and a known bid; sells need a known bid, and a zero
    bid is a real fill at 0. Sells first, then buys cut to whole contracts the available cash can pay for."""
    df = orders[["contract", "order_qty"]].copy()
    q = quotes.reindex(df["contract"])
    df["bid"], df["ask"], df["multiplier"] = q["bid"].values, q["ask"].values, q["multiplier"].values
    buy = df["order_qty"] > 0
    bad = df["bid"].isna() | df["multiplier"].isna() | (buy & ~(df["ask"] > 0))
    df["filled_qty"] = df["order_qty"].where(~bad, 0.0)
    df["status"] = np.where(bad, "rejected", "filled")
    df["reason"] = np.where(bad, "no usable quote for this contract", "")

    def need(rows):
        c = option_costs(rows["filled_qty"], rows["bid"], rows["ask"], rows["multiplier"], s)
        return float((rows["filled_qty"] * rows["multiplier"] * c["fill_price"] + c["commission"]).sum())

    available = cash - need(df.loc[df["filled_qty"] < 0])
    for i in df.index[df["filled_qty"] > 0]:
        one = option_costs(pd.Series([1.0]), df.loc[[i], "bid"].reset_index(drop=True),
                           df.loc[[i], "ask"].reset_index(drop=True), df.loc[[i], "multiplier"].reset_index(drop=True), s)
        per = float(df.at[i, "multiplier"] * one.at[0, "fill_price"] + one.at[0, "commission"])
        fit = min(df.at[i, "filled_qty"], np.floor(max(available, 0.0) / per))
        if fit < df.at[i, "filled_qty"]:
            df.at[i, "status"], df.at[i, "reason"] = ("adjusted", "reduced to fit available cash") if fit > 0 \
                else ("rejected", "insufficient cash")
        df.at[i, "filled_qty"] = fit
        available -= fit * per

    costs = option_costs(df["filled_qty"], df["bid"].fillna(0.0), df["ask"].fillna(0.0), df["multiplier"].fillna(0.0), s)
    return pd.concat([df, costs], axis=1)[OPTION_FILL_COLUMNS]


def _cash_need(rows: pd.DataFrame, s: Settings) -> float:
    """Cash consumed by these fills (negative for sells): quantity x fill price + commission."""
    if rows.empty:
        return 0.0
    c = stock_costs(rows["filled_qty"], rows["ref_price"], s, split_factor=rows["split_factor"])
    return float((rows["filled_qty"] * c["fill_price"] + c["commission"]).sum())
