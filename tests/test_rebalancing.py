import numpy as np
import pandas as pd
import pytest

from stress_portfolio import rebalancing as rb

from .conftest import make_settings

NAMES = ["A", "B", "C"]
SECTORS = pd.Series(["X", "Y", "Z"], index=NAMES)


def _est(mu=(0.10, 0.10, 0.10)):
    return pd.Series(mu, index=NAMES), pd.DataFrame(np.eye(3) * 0.04, index=NAMES, columns=NAMES)


def test_quarter_ends_only_completed_quarters():
    cal = pd.bdate_range("2023-12-20", "2024-07-10")
    ends = rb.quarter_end_sessions(cal, pd.Timestamp("2023-12-31"))
    assert list(ends) == [pd.Timestamp("2024-03-29"), pd.Timestamp("2024-06-28")]
    partial = pd.bdate_range("2024-01-02", "2024-06-28")  # data ends on a quarter's last day
    assert list(rb.quarter_end_sessions(partial, pd.Timestamp("2024-01-01"))) == [pd.Timestamp("2024-03-29")]


def test_same_book_is_kept(tmp_path):
    s = make_settings(tmp_path, max_position_weight=0.5, max_sector_weight=0.5)
    mu, cov = _est()
    w = pd.Series({"A": 0.3, "B": 0.3, "C": 0.37})
    d = rb.decide(w, w, "optimal", mu, cov, SECTORS, 0.0, 1e6, s)
    assert d["action"] == "hold" and d["improvement"] == pytest.approx(0.0)


def test_gain_must_clear_threshold_after_costs(tmp_path):
    s = make_settings(tmp_path, max_position_weight=0.5, max_sector_weight=0.5, rebalance_min_improvement=0.005)
    mu, cov = _est(mu=(0.10, 0.10, 0.20))
    current = pd.Series({"A": 0.4, "B": 0.4, "C": 0.17})
    target = pd.Series({"A": 0.3, "B": 0.3, "C": 0.37})
    gain = rb.utility(target, mu, cov, s) - rb.utility(current, mu, cov, s)
    cheap = rb.decide(current, target, "optimal", mu, cov, SECTORS, 0.0, 1e6, s)
    dear = rb.decide(current, target, "optimal", mu, cov, SECTORS, (gain - 0.001) * 1e6, 1e6, s)
    assert cheap["action"] == "rebalance" and cheap["improvement"] == pytest.approx(gain)
    assert dear["action"] == "hold" and dear["improvement"] == pytest.approx(0.001)  # cost charged once


def test_breach_forces_a_move(tmp_path):
    s = make_settings(tmp_path, max_position_weight=0.5, max_sector_weight=0.5, stress_loss_budget=0.4)
    mu, cov = _est()
    current = pd.Series({"A": 0.3, "B": 0.3, "C": 0.37})
    target = pd.Series({"A": 0.32, "B": 0.32, "C": 0.33})  # worse utility on its own
    d = rb.decide(current, target, "optimal", mu, cov, SECTORS, 100.0, 1e6, s, current_stress_loss=0.55)
    assert d["action"] == "rebalance" and "stress loss 55.0% > 40% budget" in d["reason"]
    overweight = pd.Series({"A": 0.6, "B": 0.37})
    assert "A at 60.0% > 50% cap + 2% band" in rb.breaches(overweight, SECTORS, s)


def test_drift_inside_the_band_is_not_a_breach(tmp_path):
    s = make_settings(tmp_path, max_position_weight=0.15, max_sector_weight=0.30, stress_loss_budget=0.4,
                      rebalance_drift_band=0.02)
    drifted = pd.Series({"A": 0.165, "B": 0.15, "C": 0.15})  # a 15% target that drifted to 16.5%
    assert rb.breaches(drifted, SECTORS, s, worst_stress_loss=0.415) == []
    assert rb.breaches(pd.Series({"A": 0.175}), SECTORS, s, worst_stress_loss=0.43) == [
        "A at 17.5% > 15% cap + 2% band", "worst stress loss 43.0% > 40% budget + 2% band"]


def test_no_feasible_target_holds(tmp_path):
    s = make_settings(tmp_path)
    mu, cov = _est()
    d = rb.decide(pd.Series({"A": 0.97}), pd.Series(dtype=float), "infeasible", mu, cov, SECTORS, 0.0, 1e6, s,
                  current_stress_loss=0.9)
    assert d["action"] == "hold" and d["reason"] == "no feasible target (infeasible)"
    assert d["hedge_review"].startswith("not applicable")  # hedge decisions are logged separately
