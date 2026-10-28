import pandas as pd
import pytest

from stress_portfolio import costs
from stress_portfolio.portfolio import Portfolio

from .conftest import make_settings


def _fills(ticker, qty, ref, s):
    c = costs.stock_costs(pd.Series([qty]), pd.Series([ref]), s)
    return pd.DataFrame({"ticker": [ticker], "order_qty": [qty], "filled_qty": [qty], "ref_price": [ref],
                         "status": ["filled"], "reason": [""]}).join(c)


def test_buy_costs_reconcile_exactly(tmp_path):
    s = make_settings(tmp_path)
    p = Portfolio()
    p.deposit("d0", 100_000)
    fills = _fills("A", 500, 100.0, s)
    p.apply_fills(fills, "d1")
    _, row = p.snapshot("d1", pd.Series({"A": 100.0}))  # value at the reference price
    assert p.cash == pytest.approx(100_000 - 500 * fills.loc[0, "fill_price"] - fills.loc[0, "commission"])
    assert 100_000 - row["nav"] == pytest.approx(fills.loc[0, "total_cost"])  # NAV loss is exactly the cost


def test_round_trip_and_sell_limit(tmp_path):
    s = make_settings(tmp_path)
    p = Portfolio()
    p.deposit("d0", 10_000)
    p.apply_fills(_fills("A", 10, 100.0, s), "d1")
    p.apply_fills(_fills("A", -10, 110.0, s), "d2")
    assert p.positions == {}
    p.reconcile()
    with pytest.raises(ValueError, match="exceeds"):
        p.apply_fills(_fills("A", -1, 110.0, s), "d3")


def test_dividends_credit_cash_and_splits_change_nothing(tmp_path):
    s = make_settings(tmp_path)
    p = Portfolio()
    p.deposit("d0", 10_000)
    p.apply_fills(_fills("A", 10, 100.0, s), "d1")
    cash = p.cash
    p.apply_corporate_actions("d2", pd.Series({"A": 0.5, "B": 9.0}), pd.Series({"A": 2.0}))
    assert p.cash == pytest.approx(cash + 5.0)  # B is not held
    assert p.positions == {"A": 10}  # split already in split-adjusted prices
    assert p.ledger()["type"].tolist()[-2:] == ["dividend", "split"]


def test_missing_price_and_tampering_are_caught(tmp_path):
    s = make_settings(tmp_path)
    p = Portfolio()
    p.deposit("d0", 10_000)
    p.apply_fills(_fills("A", 10, 100.0, s), "d1")
    with pytest.raises(ValueError, match="no valid price"):
        p.snapshot("d1", pd.Series({"B": 1.0}))
    p.cash += 1  # change cash outside the ledger
    with pytest.raises(AssertionError, match="cash"):
        p.reconcile()
