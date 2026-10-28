import pandas as pd
import pytest

from stress_portfolio import costs

from .conftest import make_settings


def test_known_buy_and_sell_costs(tmp_path):
    s = make_settings(tmp_path)  # 0.005/share, $1 min, 2 bp half-spread, 3 bp slippage
    c = costs.stock_costs(pd.Series([1000, -100]), pd.Series([100.0, 50.0]), s)
    buy, sell = c.iloc[0], c.iloc[1]
    assert buy["fill_price"] == pytest.approx(100.05)
    assert buy["commission"] == pytest.approx(5.0)
    assert buy["spread_cost"] == pytest.approx(20.0) and buy["slippage_cost"] == pytest.approx(30.0)
    assert buy["total_cost"] == pytest.approx(55.0)
    assert sell["fill_price"] == pytest.approx(49.975)
    assert sell["commission"] == pytest.approx(1.0)  # per-order minimum
    assert sell["total_cost"] == pytest.approx(1.0 + 100 * 0.025)


def test_zero_order_costs_nothing(tmp_path):
    c = costs.stock_costs(pd.Series([0]), pd.Series([100.0]), make_settings(tmp_path))
    assert c.loc[0, "total_cost"] == 0 and c.loc[0, "fill_price"] == 100.0


def test_scenarios_scale_every_component(tmp_path):
    summary = costs.scenario_summary(pd.Series([1000]), pd.Series([100.0]), make_settings(tmp_path))
    total = summary.set_index("scenario")["total_cost"]
    assert total["low"] == pytest.approx(total["base"] / 2)
    assert total["high"] == pytest.approx(total["base"] * 2)


def test_commission_uses_real_shares_after_splits(tmp_path):
    s = make_settings(tmp_path)
    # 40,000 split-adjusted shares = 1,000 real shares when 40:1 of splits happened later
    c = costs.stock_costs(pd.Series([40_000]), pd.Series([1.0]), s, split_factor=pd.Series([40.0]))
    assert c.loc[0, "commission"] == pytest.approx(5.0)  # not 200
    assert c.loc[0, "spread_cost"] == pytest.approx(40_000 * 2e-4)  # value-based costs unchanged
