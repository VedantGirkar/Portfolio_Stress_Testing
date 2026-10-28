import pandas as pd

from stress_portfolio import execution

from .conftest import make_settings


def test_orders_round_down_and_skip_unchanged():
    orders = execution.generate_orders({}, pd.Series({"A": 0.5, "B": 0.5}), pd.Series({"A": 30.0, "B": 7.0}), 1000)
    assert orders.set_index("ticker")["order_qty"].to_dict() == {"A": 16, "B": 71}
    again = execution.generate_orders({"A": 16, "B": 71}, pd.Series({"A": 0.5, "B": 0.5}),
                                      pd.Series({"A": 30.0, "B": 7.0}), 1000)
    assert again.empty


def test_removed_target_becomes_a_full_sale():
    orders = execution.generate_orders({"A": 10}, pd.Series({"B": 1.0}), pd.Series({"A": 5.0, "B": 10.0}), 100)
    assert orders.set_index("ticker")["order_qty"].to_dict() == {"A": -10, "B": 10}


def test_gap_up_trims_buys_and_never_overspends(tmp_path):
    s = make_settings(tmp_path)
    orders = execution.generate_orders({}, pd.Series({"A": 0.49, "B": 0.49}), pd.Series({"A": 10.0, "B": 10.0}),
                                       10_000)
    fills = execution.execute(orders, pd.Series({"A": 11.0, "B": 11.0}), 10_000, s)  # 10% gap up > reserve
    spent = (fills["filled_qty"] * fills["fill_price"] + fills["commission"]).sum()
    assert spent <= 10_000
    assert (fills["status"] == "adjusted").all()
    assert (fills["filled_qty"] < fills["order_qty"]).all()


def test_sells_fund_buys_and_bad_prices_are_rejected(tmp_path):
    s = make_settings(tmp_path)
    orders = pd.DataFrame({"ticker": ["A", "B", "C"], "order_qty": [-100.0, 50.0, 5.0]})
    fills = execution.execute(orders, pd.Series({"A": 10.0, "B": 10.0, "C": float("nan")}), 0.0, s)
    f = fills.set_index("ticker")
    assert f.at["A", "status"] == "filled" and f.at["B", "status"] == "filled"  # sale proceeds pay for B
    assert f.at["C", "status"] == "rejected" and f.at["C", "commission"] == 0


def test_unchanged_float_weights_do_not_trade():
    nav, price = 1_002_903.3, 343.83
    weight = 279 * price / nav  # the drifted weight read back from holdings
    assert execution.generate_orders({"A": 279.0}, pd.Series({"A": weight}), pd.Series({"A": price}), nav).empty
