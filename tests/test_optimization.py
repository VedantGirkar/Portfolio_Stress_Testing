import numpy as np
import pandas as pd
import pytest

from stress_portfolio import optimization as opt

from .conftest import make_settings

LOOSE = dict(max_position_weight=1.0, max_sector_weight=1.0, min_position_weight=0.0, cash_reserve=0.03)


def _two(mu=(0.08, 0.08), vol=(0.1, 0.2)):
    names = ["A", "B"]
    return (pd.Series(mu, index=names), pd.DataFrame(np.diag(np.square(vol)), index=names, columns=names),
            pd.Series(["X", "Y"], index=names))


def test_min_variance_matches_inverse_variance(tmp_path):
    s = make_settings(tmp_path, **LOOSE)
    mu, cov, sec = _two()
    w, status = opt.solve("min_variance", mu, cov, sec, pd.Series(dtype=float), s)
    assert status == "optimal"
    assert w["A"] == pytest.approx(0.8 * 0.97, abs=1e-5)  # 1/0.01 : 1/0.04 = 80% : 20% of the sleeve
    assert w.sum() == pytest.approx(0.97)


def test_infeasible_limits_are_reported(tmp_path):
    s = make_settings(tmp_path, **{**LOOSE, "max_position_weight": 0.2})
    mu, cov, sec = _two()
    w, status = opt.solve("min_variance", mu, cov, sec, pd.Series(dtype=float), s)
    assert w.empty and status == "infeasible"  # two stocks at <= 20% cannot hold 97%


def test_sector_cap_binds(tmp_path):
    s = make_settings(tmp_path, **{**LOOSE, "max_sector_weight": 0.3})
    names = ["A1", "A2", "B1", "C1", "D1"]
    mu = pd.Series([0.30, 0.30, 0.02, 0.02, 0.02], index=names)  # sector A is far more attractive
    cov = pd.DataFrame(np.eye(5) * 0.04, index=names, columns=names)
    sec = pd.Series(["A", "A", "B", "C", "D"], index=names)
    w, status = opt.solve("mean_variance", mu, cov, sec, pd.Series(dtype=float), s)
    assert status == "optimal"
    assert w[["A1", "A2"]].sum() == pytest.approx(0.3, abs=1e-5)  # held at the cap
    assert w.groupby(sec).sum().max() <= 0.3 + 1e-6


def test_stress_budget_limits_crash_exposure(tmp_path):
    s = make_settings(tmp_path, **{**LOOSE, "stress_loss_budget": 0.4, "risk_aversion": 0.5})
    mu, cov, sec = _two(mu=(0.30, 0.02), vol=(0.2, 0.2))
    train = pd.DataFrame({"A": [-0.8], "B": [0.0]}, index=["crash"])
    free, _ = opt.solve("mean_variance", mu, cov, sec, pd.Series(dtype=float), s)
    capped, status = opt.solve("stress_aware", mu, cov, sec, pd.Series(dtype=float), s, stress_train=train)
    assert free["A"] > 0.5
    assert capped["A"] == pytest.approx(0.5, abs=1e-4)  # 0.8 x w_A <= 40%
    assert status == "optimal"


def test_trading_costs_keep_the_book_still(tmp_path):
    mu, cov, sec = _two(mu=(0.10, 0.06), vol=(0.2, 0.2))
    w0 = pd.Series({"A": 0.2, "B": 0.77})
    cheap = make_settings(tmp_path, **LOOSE, half_spread_bps=0.0, slippage_bps=0.0)
    dear = make_settings(tmp_path, **LOOSE, half_spread_bps=5000.0, slippage_bps=0.0)
    moved, _ = opt.solve("mean_variance", mu, cov, sec, w0, cheap)
    stayed, _ = opt.solve("mean_variance", mu, cov, sec, w0, dear)
    assert moved["A"] > 0.3
    assert stayed["A"] == pytest.approx(0.2, abs=1e-4)


def test_small_positions_are_dropped_and_resolved(tmp_path):
    s = make_settings(tmp_path, **{**LOOSE, "min_position_weight": 0.05})
    names = ["A", "B", "C"]
    mu = pd.Series(0.08, index=names)
    cov = pd.DataFrame(np.diag([0.01, 0.01, 2.0]), index=names, columns=names)  # C is wildly risky
    w, status = opt.solve("min_variance", mu, cov, pd.Series(["X", "Y", "Z"], index=names), pd.Series(dtype=float), s)
    assert w["C"] == 0 and "dropping 1 position" in status
    assert w.sum() == pytest.approx(0.97)


def test_solve_is_repeatable(tmp_path):
    s = make_settings(tmp_path, **LOOSE)
    mu, cov, sec = _two()
    a, _ = opt.solve("mean_variance", mu, cov, sec, pd.Series(dtype=float), s)
    b, _ = opt.solve("mean_variance", mu, cov, sec, pd.Series(dtype=float), s)
    pd.testing.assert_series_equal(a, b)


def test_expected_return_blend(tmp_path):
    s = make_settings(tmp_path, expected_return_blend=0.5, equity_risk_premium=0.05)
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.0004, 0.01, 500))
    rets = pd.DataFrame({"A": 2 * bench + rng.normal(0, 0.001, 500)})
    mu, cov, beta = opt.estimates(rets, bench, s)
    assert beta["A"] == pytest.approx(2.0, abs=0.02)
    assert mu["A"] == pytest.approx(0.5 * rets["A"].mean() * 252 + 0.5 * beta["A"] * 0.05)


def test_evaluate_splits_train_and_validation(tmp_path):
    s = make_settings(tmp_path, **LOOSE)
    mu, cov, _ = _two()
    scen = pd.DataFrame({"A": [-0.5, -0.9], "B": [-0.1, 0.0]}, index=["t", "v"])
    groups = pd.Series({"t": "train", "v": "validation"})
    holdings = pd.DataFrame({"ticker": ["A", "CASH"], "quantity": [10.0, 30.0], "weight_of_nav": [0.7, 0.3],
                             "market_value": [70.0, 30.0]})
    e = opt.evaluate(pd.Series({"A": 0.7}), mu, cov, scen, groups, holdings, pd.Series({"A": 7.0, "B": 1.0}), 100, s)
    assert (e["worst_train_loss"], e["worst_validation_loss"]) == (pytest.approx(0.35), pytest.approx(0.63))
    assert e["trades"] == 0 and e["est_cost_usd"] == 0 and e["turnover"] == 0  # unchanged book costs nothing


def test_infeasible_after_dropping_names_the_cause(tmp_path):
    s = make_settings(tmp_path, **{**LOOSE, "max_position_weight": 0.15, "min_position_weight": 0.10})
    names = [f"S{i}" for i in range(7)]
    mu = pd.Series(0.08, index=names)
    cov = pd.DataFrame(np.diag([0.01] * 6 + [1.0]), index=names, columns=names)  # S6 gets only the last 7%
    w, status = opt.solve("min_variance", mu, cov, pd.Series(names, index=names), pd.Series(dtype=float), s)
    assert w.empty and status == "infeasible after dropping 1 position(s) under 10%"


def test_universe_sector_cap_scales_with_settings(tmp_path, monkeypatch):
    caps = {}

    def fake_select(short, rets, sk):
        caps[sk.core_size] = sk.max_per_sector
        return pd.DataFrame({"provider_ticker": []})

    monkeypatch.setattr(opt.selection, "select_core", fake_select)
    s = make_settings(tmp_path, core_size=10, max_per_sector=3, holding_counts=(10, 20), shortlist_size=20)
    opt.universes(pd.DataFrame(), pd.DataFrame(), s)
    assert caps == {10: 3, 20: 6}  # the core-sized universe uses the core's own cap
