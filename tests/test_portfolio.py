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


def _put(p, contract="SPY 2016-02-19 P 190.00", expiry="2016-02-19"):
    p.add_contract(contract, {"underlying": "SPY", "option_type": "P", "strike": 190.0, "expiration": expiry,
                              "multiplier": 100, "exercise": "American", "settlement": "physical"})
    return contract


def _option_fill(contract, qty, bid, ask, s):
    from stress_portfolio import execution
    quotes = pd.DataFrame({"bid": [bid], "ask": [ask], "multiplier": [100]}, index=[contract])
    return execution.execute_options(pd.DataFrame({"contract": [contract], "order_qty": [qty]}), quotes, 1e9, s)


def test_put_purchase_exchanges_cash_for_an_asset(tmp_path):
    s = make_settings(tmp_path)
    p = Portfolio()
    p.deposit("d0", 100_000)
    c = _put(p)
    p.apply_option_fills(_option_fill(c, 10, 2.0, 2.2, s), "d1")
    assert p.cash == pytest.approx(100_000 - 10 * 100 * 2.2 - 7.0)
    assert p.positions == {} and p.options == {c: 10}  # never mixed with stock positions
    holdings, row = p.snapshot("d1", pd.Series(dtype=float), pd.Series({c: 2.1}))  # marked at mid
    assert row["derivative_value"] == pytest.approx(2_100) and row["equity_value"] == 0
    assert 100_000 - row["nav"] == pytest.approx(10 * 100 * 0.1 + 7.0)  # half-spread + fees, no premium loss


def test_option_marks_must_exist_and_short_options_are_refused(tmp_path):
    s = make_settings(tmp_path)
    p = Portfolio()
    p.deposit("d0", 100_000)
    c = _put(p)
    p.apply_option_fills(_option_fill(c, 2, 2.0, 2.2, s), "d1")
    assert p.snapshot("d1", pd.Series(dtype=float), pd.Series({c: 0.0}))[1]["derivative_value"] == 0  # worthless is valid
    with pytest.raises(ValueError, match="no valid option mark"):
        p.snapshot("d1", pd.Series(dtype=float))
    with pytest.raises(ValueError, match="exceeds"):
        p.apply_option_fills(_option_fill(c, -3, 2.0, 2.2, s), "d2")
    with pytest.raises(KeyError):
        p.apply_option_fills(_option_fill("UNKNOWN", 1, 2.0, 2.2, s), "d2")
    with pytest.raises(ValueError, match="multiplier"):
        p.add_contract("X", {"multiplier": None})


def test_close_then_expiry_settlement_reconciles(tmp_path):
    s = make_settings(tmp_path)
    p = Portfolio()
    p.deposit("d0", 100_000)
    itm, otm = _put(p), _put(p, "SPY 2016-02-19 P 150.00")
    p.contracts[otm]["strike"] = 150.0
    p.apply_option_fills(_option_fill(itm, 3, 2.0, 2.2, s), "d1")
    p.apply_option_fills(_option_fill(otm, 2, 0.1, 0.2, s), "d1")
    p.apply_option_fills(_option_fill(itm, -1, 3.0, 3.2, s), "d2")  # partial close at the bid
    cash = p.cash
    p.expire_options("2016-02-18", underlying_price=180.0)  # not expired yet
    assert p.options == {itm: 2, otm: 2}
    p.expire_options("2016-02-19", underlying_price=180.0)
    assert p.options == {} and p.cash == pytest.approx(cash + 2 * 100 * 10.0)  # ITM: net intrinsic; OTM: 0
    types = p.ledger()["type"].tolist()
    assert types.count("exercise") == 1 and types.count("expiry") == 1
    p.reconcile()
