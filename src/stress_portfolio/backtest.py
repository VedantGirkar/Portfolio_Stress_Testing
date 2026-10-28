"""Chronological orchestration. For now: fund the core once, then hold it to the latest session."""

import pandas as pd

from . import execution
from .config import Settings
from .costs import scenario_summary
from .data import wide
from .portfolio import Portfolio
from .selection import session_calendar

EVENT_ORDER = ("1. dividends/splits for shares held at the prior close",
               "2. fills at the session open",
               "3. valuation at the session close")


def fund_and_hold(core: pd.DataFrame, bars: pd.DataFrame, decision_date: pd.Timestamp, s: Settings) -> dict:
    """Deposit capital, size orders from the decision-date close, fill at the next session's open,
    then value daily at the close. Each session follows EVENT_ORDER."""
    close = wide(bars, "close")
    cal = session_calendar(close)
    sessions = cal[cal > decision_date]
    if sessions.empty:
        raise ValueError(f"no session after the decision date {decision_date.date()}; nothing to fill yet")
    fill_date = sessions[0]
    opens, dividends, splits = wide(bars, "open"), wide(bars, "dividends"), wide(bars, "splits")
    marks = close.ffill()  # valuation marks only (e.g. a halted stock keeps its last close); never used for returns

    p = Portfolio()
    p.deposit(decision_date, s.initial_capital)
    targets = core.set_index("provider_ticker")["target_weight"] * (1 - s.cash_reserve)
    orders = execution.generate_orders(p.positions, targets, close.loc[decision_date], s.initial_capital)

    nav_rows = []
    for d in sessions:
        p.apply_corporate_actions(d, dividends.loc[d], splits.loc[d])
        if d == fill_date:
            fills = execution.execute(orders, opens.loc[d], p.cash, s)
            p.apply_fills(fills, d)
        holdings, row = p.snapshot(d, marks.loc[d])
        nav_rows.append(row)

    info = core.set_index("provider_ticker")
    holdings = holdings.assign(company=holdings["ticker"].map(info["company"]),
                               sector=holdings["ticker"].map(info["sector"]),
                               target_weight_of_nav=holdings["ticker"].map(targets))
    cash_row = pd.DataFrame([{"ticker": "CASH", "quantity": p.cash, "price": 1.0, "market_value": p.cash,
                              "weight_of_nav": p.cash / row["nav"], "company": "Cash", "sector": "Cash",
                              "target_weight_of_nav": s.cash_reserve}])
    orders = orders.merge(fills, on=["ticker", "order_qty"], how="left")
    return {
        "decision_date": decision_date,
        "fill_date": fill_date,
        "orders": orders,
        "holdings": pd.concat([holdings, cash_row], ignore_index=True),
        "ledger": p.ledger(),
        "nav": pd.DataFrame(nav_rows),
        "cost_scenarios": scenario_summary(fills["filled_qty"], fills["ref_price"].fillna(0.0), s),
    }
