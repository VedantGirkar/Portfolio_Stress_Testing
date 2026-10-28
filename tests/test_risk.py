import math

import numpy as np
import pandas as pd
import pytest

from stress_portfolio import risk

from .conftest import make_settings


def test_discrete_var_es_on_known_losses():
    pnl = pd.Series(-np.arange(1, 101, dtype=float))  # losses 1..100
    var, es, k = risk.var_es(pnl, 0.95)
    assert (k, var, es) == (5, 96.0, 98.0)  # 5th largest loss; mean of 100..96
    var, es, k = risk.var_es(pnl, 0.99)
    assert (k, var, es) == (1, 100.0, 100.0)


def test_flat_portfolio_has_no_risk():
    pnl = pd.Series(np.zeros(50))
    assert risk.var_es(pnl, 0.99)[:2] == (0.0, 0.0)
    assert risk.normal_var_es(pnl, 0.99) == (0.0, 0.0)
    assert risk.max_drawdown(pd.Series([1.0] * 5)) == 0.0


def test_normal_var_matches_z_score():
    pnl = pd.Series(np.random.default_rng(0).normal(0, 1, 200_000))
    var, es = risk.normal_var_es(pnl, 0.95)
    assert var == pytest.approx(1.645, abs=0.02)
    assert es == pytest.approx(2.063, abs=0.02)


def test_known_drawdown_path():
    assert risk.max_drawdown(pd.Series([100, 120, 90, 110, 60, 130.0])) == pytest.approx(0.5)


def test_ten_day_returns_compound_not_sum():
    rets = pd.DataFrame({"A": [0.01] * 12})
    ten = risk.horizon_returns(rets, 10)
    assert len(ten) == 3  # overlapping windows ending on days 10, 11, 12
    assert ten["A"].iloc[0] == pytest.approx(1.01**10 - 1)


def test_daily_returns_use_only_shared_sessions():
    adj = pd.DataFrame({"A": [10.0, 11, 12, 13], "B": [np.nan, np.nan, 5, 5.5]})
    rets = risk.daily_returns(adj)
    assert len(rets) == 1 and rets["B"].iloc[0] == pytest.approx(0.1)  # no invented returns before listing


def test_var_backtest_uses_only_earlier_days(tmp_path):
    s = make_settings(tmp_path, var_backtest_min_window=100, var_confidences=(0.99,))
    pnl = pd.Series(np.r_[np.random.default_rng(1).normal(0, 1, 120), -1000.0],
                    index=pd.bdate_range("2024-01-01", periods=121))
    series, summary = risk.var_backtest(pnl, s)
    last = series.iloc[-1]
    assert last["var_99%"] == risk.var_es(pnl.iloc[:-1], 0.99)[0]  # the crash is not in its own estimate
    assert bool(last["exceed_99%"])
    assert summary.loc[0, "days_tested"] == 21


def test_kupiec_p_value():
    assert risk.kupiec_p_value(10, 1000, 0.99) == pytest.approx(1.0)
    assert risk.kupiec_p_value(40, 1000, 0.99) < 0.001


def _book():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=400)
    market = rng.normal(0.0003, 0.01, 400)
    rets = pd.DataFrame({"A": 1.5 * market + rng.normal(0, 0.005, 400),
                         "B": 0.5 * market + rng.normal(0, 0.005, 400)}, index=dates)
    adj = 100 * (1 + rets).cumprod()
    spy = pd.Series(100 * np.cumprod(1 + market), index=dates)
    holdings = pd.DataFrame({"ticker": ["A", "B", "CASH"], "market_value": [600.0, 300.0, 100.0],
                             "sector": ["X", "Y", "Cash"], "company": ["a", "b", "Cash"]})
    nav = pd.DataFrame({"date": [dates[-1]], "nav": [1000.0]})
    return holdings, adj, spy, nav


def test_risk_report_identities(tmp_path):
    s = make_settings(tmp_path, initial_capital=1000.0)
    holdings, adj, spy, nav = _book()
    out = risk.risk_report(holdings, adj, spy, nav, s)

    pos = out["risk_positions"].set_index("ticker")
    assert pos["risk_contribution_pct"].sum() == pytest.approx(1.0)
    assert pos.at["A", "beta"] == pytest.approx(1.5, abs=0.1)
    summary = out["risk_summary"].set_index("metric")["value"]
    assert summary["beta"] == pytest.approx(0.6 * pos.at["A", "beta"] + 0.3 * pos.at["B", "beta"])
    assert summary["cash_weight"] == pytest.approx(0.1)
    assert summary["effective_holdings"] == pytest.approx(1 / ((2 / 3) ** 2 + (1 / 3) ** 2))
    assert out["risk_sectors"]["weight_of_nav"].sum() == pytest.approx(1.0)

    var = out["risk_var"]
    hist = var.loc[var["method"] == "historical"].set_index(["horizon_days", "confidence"])
    assert (hist["es_usd"] >= hist["var_usd"]).all()
    assert hist.at[(1, 0.99), "var_usd"] >= hist.at[(1, 0.95), "var_usd"]
    assert hist.at[(10, 0.99), "observations"] == hist.at[(1, 0.99), "observations"] - 9
    assert math.isclose(hist.at[(1, 0.95), "var_pct_nav"], hist.at[(1, 0.95), "var_usd"] / 1000)


def test_missing_benchmark_returns_raise(tmp_path):
    holdings, adj, spy, nav = _book()
    with pytest.raises(ValueError, match="SPY returns missing"):
        risk.risk_report(holdings, adj, spy.iloc[:-5], nav, make_settings(tmp_path))
