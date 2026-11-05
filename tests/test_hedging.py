import numpy as np
import pandas as pd
import pytest

from stress_portfolio import hedging

from .conftest import make_settings


def test_textbook_put_price():
    # Hull: S=K=100, T=1, r=5%, sigma=20%, no dividends -> put 5.5735
    assert float(hedging.put_price(100, 100, 1.0, 0.05, 0.0, 0.20)) == pytest.approx(5.5735, abs=1e-4)


def test_greeks_match_finite_differences():
    args = dict(spot=480.0, strike=455.0, years=0.15, rate=0.04, div_yield=0.013, vol=0.22)
    g = {k: float(v) for k, v in hedging.put_price_greeks(**args).items()}

    def p(**kw):
        return float(hedging.put_price(**(args | kw)))

    h = 0.01
    assert g["delta"] == pytest.approx((p(spot=480 + h) - p(spot=480 - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((p(spot=480 + h) - 2 * p() + p(spot=480 - h)) / h**2, rel=1e-3)
    assert g["vega"] == pytest.approx((p(vol=0.22 + 1e-5) - p(vol=0.22 - 1e-5)) / 2e-5, rel=1e-5)
    assert g["theta"] == pytest.approx(-(p(years=0.15 + 1e-6) - p(years=0.15 - 1e-6)) / 2e-6, rel=1e-4)
    assert -1 < g["delta"] < 0 and g["gamma"] > 0 and g["vega"] > 0  # long put signs


def test_expiry_and_zero_vol_are_intrinsic_without_nans():
    out = hedging.put_price_greeks([90.0, 110.0, 90.0], 100.0, [0.0, 0.0, 0.5], 0.0, 0.0, [0.2, 0.2, 0.0])
    assert out["price"] == pytest.approx([10.0, 0.0, 10.0])
    assert out["delta"] == pytest.approx([-1.0, 0.0, -1.0])
    assert not any(np.isnan(v).any() for v in out.values())
    assert float(hedging.put_price(100, 100, -0.1, 0.0, 0.0, 0.2)) == 0.0  # past expiry, at the money


def test_contract_id_is_stable():
    assert hedging.contract_id("SPY", "2016-02-19", "p", 190) == hedging.contract_id("SPY", pd.Timestamp("2016-02-19"),
                                                                                      "P", 190.0)
    assert hedging.contract_id("SPY", "2016-02-19", "P", 190.5) == "SPY 2016-02-19 P 190.50"


def _chain():
    rows = [("SPY", "2016-01-15", 190, 1.0, 1.1),   # expires before next review + buffer
            ("SPY", "2016-02-19", 189, 2.0, 2.1),
            ("SPY", "2016-02-19", 191, 2.4, 2.5),   # 190 equidistant -> lower strike wins
            ("SPY", "2016-02-19", 200, 5.0, 5.2),
            ("SPY7", "2016-02-19", 190, 2.2, 2.3),  # adjusted root
            ("SPY", "2016-03-18", 190, 3.0, 3.2)]
    df = pd.DataFrame(rows, columns=["root", "expiration", "strike", "bid", "ask"])
    return df.assign(option_type="P", expiration=pd.to_datetime(df["expiration"]))


def test_choose_put_picks_earliest_valid_expiry_and_nearest_strike(tmp_path):
    s = make_settings(tmp_path)  # 95% moneyness, 14-day buffer
    pick = hedging.choose_put(_chain(), spot=200.0, next_review="2016-01-29", s=s)
    assert pick["expiration"] == pd.Timestamp("2016-02-19") and pick["strike"] == 189 and pick["root"] == "SPY"


def test_choose_put_skips_unquoted_and_returns_none_when_nothing_fits(tmp_path):
    s = make_settings(tmp_path)
    chain = _chain()
    chain.loc[chain["strike"].isin([189, 191]), "ask"] = 0.0  # no offer
    assert hedging.choose_put(chain, 200.0, "2016-01-29", s)["strike"] == 200
    assert hedging.choose_put(chain, 200.0, "2016-06-30", s) is None  # no expiry far enough out


def _scenarios():
    # stock book of 1M NAV: -50% with SPY -50% (instantaneous), -40% with SPY -40% over 0.1y, a mild one
    return pd.DataFrame({"book_pnl_usd": [-500_000.0, -400_000.0, -50_000.0], "spy_return": [-0.5, -0.4, -0.05],
                         "stress_vol": [0.6, 0.5, 0.25], "elapsed_years": [0.0, 0.1, 0.0]},
                        index=["crash", "slow", "mild"])


PUT = dict(entry_price=4.0, spot=400.0, strike=380.0, years=2 / 12, multiplier=100, nav=1_000_000.0, rate=0.04)


def test_hedged_losses_without_puts_is_the_stock_loss(tmp_path):
    s = make_settings(tmp_path)
    loss = hedging.hedged_losses(_scenarios(), 0, s=s, **PUT)
    assert loss.to_dict() == pytest.approx({"crash": 0.5, "slow": 0.4, "mild": 0.05})


def test_put_past_expiry_in_a_scenario_is_worth_intrinsic(tmp_path):
    s = make_settings(tmp_path)
    late = pd.DataFrame({"book_pnl_usd": [0.0], "spy_return": [-0.2], "stress_vol": [0.5], "elapsed_years": [1.0]})
    loss = hedging.hedged_losses(late, 1, s=s, **PUT)
    assert loss.iloc[0] == pytest.approx(-100 * ((380 - 320) - 4.0) / 1_000_000)


def test_size_hedge_finds_the_fewest_contracts(tmp_path):
    s = make_settings(tmp_path, hedge_premium_budget=1.0)  # budget not binding
    out = hedging.size_hedge(_scenarios(), vol=0.2, s=s, **PUT)
    assert out["status"] == "target met" and out["worst_hedged"] <= s.stress_loss_budget
    fewer = hedging.hedged_losses(_scenarios(), out["contracts"] - 1, s=s, **PUT).max()
    assert fewer > s.stress_loss_budget
    assert out["worst_unhedged"] == pytest.approx(0.5)


def test_size_hedge_reports_partial_when_budget_binds(tmp_path):
    s = make_settings(tmp_path, hedge_premium_budget=0.001)
    out = hedging.size_hedge(_scenarios(), vol=0.2, s=s, **PUT)
    assert out["status"] == "partial: premium budget binds"
    assert out["contracts"] == out["budget_cap_contracts"] and out["worst_hedged"] > s.stress_loss_budget
    assert out["annual_net_cost_usd"] <= 0.001 * PUT["nav"] + 1e-9


def test_size_hedge_buys_nothing_when_within_budget(tmp_path):
    s = make_settings(tmp_path)
    out = hedging.size_hedge(_scenarios().loc[["mild"]], vol=0.2, s=s, **PUT)
    assert out["contracts"] == 0 and out["status"] == "not needed"


def test_scenario_inputs_combines_training_rows(tmp_path):
    s = make_settings(tmp_path)
    capped = pd.DataFrame({"scenario": ["gfc", "lehman", "dotcom"], "group": ["train", "validation", "train"],
                           "book_pnl_usd": [-1.0, -2.0, -3.0], "spy_return": [-0.1, -0.2, -0.3],
                           "stress_vol": [0.4, 0.6, np.nan], "elapsed_years": [0.15, 0.1, 0.2],
                           "available": [True, True, False]})
    table = pd.DataFrame({"scenario": ["market-20%", "sector_Energy", "gfc"], "kind": ["hypothetical"] * 2 + ["historical"],
                          "group": "train", "pnl_usd": [-5.0, -6.0, -9.0], "benchmark_return": [-0.2, 0.0, -0.5]})
    out = hedging.scenario_inputs(capped, table, iv_now=0.18, s=s)
    assert list(out.index) == ["gfc", "market-20%", "sector_Energy"]  # validation and no-vol events dropped
    assert out.loc["gfc", "spy_return"] == -0.1
    assert out.loc["market-20%", "stress_vol"] == pytest.approx(0.33) and out.loc["market-20%", "elapsed_years"] == 0


def test_beta_dollar_delta_and_drift_trigger(tmp_path):
    s = make_settings(tmp_path)
    values, betas = pd.Series({"A": 600_000.0, "B": 300_000.0}), pd.Series({"A": 1.0, "B": 0.5})
    net = hedging.beta_dollar_delta(values, betas, contracts=10, put_delta=-0.25, spot=400.0, multiplier=100)
    assert net == pytest.approx(750_000 - 100_000)
    assert not hedging.needs_review(net + 90_000, net, 1_000_000, s)
    assert hedging.needs_review(net + 110_000, net, 1_000_000, s)


def test_missing_stress_volatility_is_refused(tmp_path):
    s = make_settings(tmp_path)
    bad = _scenarios().assign(stress_vol=[0.6, np.nan, 0.25])
    with pytest.raises(ValueError, match="stress volatility"):
        hedging.hedged_losses(bad, 1, s=s, **PUT)


def _plan_world():
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    spy = pd.Series(np.interp(np.arange(len(dates)), [0, 23, 53, len(dates) - 1], [90, 100, 60, 80]), index=dates)
    history = pd.DataFrame({"SPY": spy, "A": spy * 2, "^VIX": 30.0, "^IRX": 2.0})
    stocks = pd.DataFrame({"sector": ["Energy"], "market_value": [900_000.0]}, index=["A"])
    betas = pd.Series({"A": 1.0})
    table = pd.DataFrame({"scenario": ["market-20%"], "kind": ["hypothetical"], "group": ["train"],
                          "pnl_usd": [-180_000.0], "benchmark_return": [-0.2]})
    chain = pd.DataFrame({"root": "SPY", "option_type": "P", "expiration": pd.Timestamp("2020-08-21"),
                          "strike": [76.0, 80.0], "bid": [1.0, 2.0], "ask": [1.1, 2.1], "iv": [0.25, np.nan],
                          "multiplier": 100, "contract": ["P76", "P80"], "iv_source": "test quotes"})
    return stocks, betas, history, chain, table


EVENT = (("crash", "Test crash", "2020-01-01", "2020-06-30", "train"),)


def test_plan_hedge_picks_sizes_and_labels(tmp_path):
    s = make_settings(tmp_path, stress_events=EVENT, stress_loss_budget=0.2, hedge_premium_budget=1.0)
    stocks, betas, history, chain, table = _plan_world()
    plan = hedging.plan_hedge(stocks, betas, 1_000_000.0, history, chain, table, "2020-06-30", s)
    assert plan["pick"]["strike"] == 76.0 and plan["iv"] == 0.25 and plan["iv_source"] == "test quotes"
    assert plan["rate"] == pytest.approx(0.02) and plan["next_review"] == pd.Timestamp("2020-07-31")
    assert plan["sizing"]["contracts"] > 0 and plan["delta_after"] < plan["delta_before"]
    table = plan["scenarios"].set_index("scenario")
    assert table.at["crash", "hedged_loss"] < table.at["crash", "unhedged_loss"]
    assert table.at["crash", "stress_vol"] == pytest.approx(0.30)  # observed index, not assumed


def test_plan_hedge_uses_the_index_when_the_quote_has_no_iv(tmp_path):
    s = make_settings(tmp_path, stress_events=EVENT, hedge_moneyness=0.999)
    stocks, betas, history, chain, table = _plan_world()
    plan = hedging.plan_hedge(stocks, betas, 1_000_000.0, history, chain, table, "2020-06-30", s)
    assert plan["pick"]["strike"] == 80.0 and plan["iv"] == pytest.approx(0.30) and "^VIX" in plan["iv_source"]


def test_plan_hedge_reports_missing_inputs_without_crashing(tmp_path):
    s = make_settings(tmp_path, stress_events=EVENT)
    stocks, betas, history, chain, table = _plan_world()
    args = (stocks, betas, 1_000_000.0)
    assert "^VIX" in hedging.plan_hedge(*args, history.drop(columns="^VIX"), chain, table, "2020-06-30", s)["status"]
    assert "no option quotes" in hedging.plan_hedge(*args, history, chain.iloc[0:0], table, "2020-06-30", s)["status"]
    late = chain.assign(expiration=pd.Timestamp("2020-07-03"))  # before the next review + buffer
    assert "no listed put" in hedging.plan_hedge(*args, history, late, table, "2020-06-30", s)["status"]


def test_resize_plan_reports_the_contracts_actually_held(tmp_path):
    s = make_settings(tmp_path, stress_events=EVENT, stress_loss_budget=0.2, hedge_premium_budget=1.0)
    stocks, betas, history, chain, table = _plan_world()
    plan = hedging.plan_hedge(stocks, betas, 1_000_000.0, history, chain, table, "2020-06-30", s)
    less = hedging.resize_plan(plan, 1, 1_000_000.0, s)
    assert less["sizing"]["contracts"] == 1 and less["sizing"]["worst_hedged"] > plan["sizing"]["worst_hedged"]
    assert less["delta_after"] > plan["delta_after"]
