import numpy as np
import pandas as pd
import pytest

from stress_portfolio import stress

from .conftest import make_settings

EVENT = (("crash", "Test crash", "2020-01-01", "2020-12-31", "train"),
         ("inner", "Inside the crash", "2020-02-01", "2020-03-31", "validation"),
         ("old", "Before any data", "1990-01-01", "1990-12-31", "validation"))


def _book():
    holdings = pd.DataFrame({"ticker": ["A", "B", "C", "CASH"], "market_value": [500.0, 300.0, 100.0, 100.0],
                             "sector": ["Energy", "Energy", "Utilities", "Cash"]})
    risk_positions = pd.DataFrame({"ticker": ["A", "B", "C"], "beta": [1.5, 1.0, 0.5],
                                   "risk_contribution_pct": [0.6, 0.3, 0.1]})
    return holdings, risk_positions


def _history():
    """SPY rises to 100 on Feb 3, falls to 60 on Mar 16, recovers. A has full data; B lists in March
    (so it needs its sector ETF XLE); C lists late and Utilities has no ETF data (beta x SPY)."""
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    spy = pd.Series(np.interp(np.arange(len(dates)), [0, 23, 53, len(dates) - 1], [90, 100, 60, 80]), index=dates)
    return pd.DataFrame({
        "SPY": spy,
        "A": spy * 2,
        "B": spy.where(dates >= "2020-03-02"),
        "XLE": spy * 0.5 * np.where(dates >= dates[53], 0.9, 1.0),
        "C": spy.where(dates >= "2020-05-01"),
    })


def test_worst_drawdown_window_finds_peak_and_trough():
    dates = pd.bdate_range("2020-01-01", periods=6)
    peak, trough, ret = stress.worst_drawdown_window(pd.Series([100, 120, 90, 110, 60, 130.0], index=dates))
    assert (peak, trough, ret) == (dates[1], dates[4], pytest.approx(-0.5))


def test_window_return_compounds_and_needs_both_endpoints():
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.Series([100.0, 110.0, 99.0], index=dates)
    assert stress.window_return(prices, dates[0], dates[2]) == pytest.approx(-0.01)  # not +10% - 10% = 0
    assert stress.window_return(prices.where(dates > dates[0]), dates[0], dates[2]) is None


def test_zero_shock_changes_nothing():
    holdings, _ = _book()
    stocks = holdings.loc[holdings["ticker"] != "CASH"].set_index("ticker")
    out = stress.apply_shocks("flat", stocks, pd.Series(0.0, index=stocks.index), pd.Series("x", index=stocks.index))
    assert out["pnl_usd"].abs().sum() == 0


def test_single_position_shock_and_floor():
    holdings, _ = _book()
    stocks = holdings.loc[holdings["ticker"] != "CASH"].set_index("ticker")
    out = stress.apply_shocks("one", stocks, pd.Series({"A": -0.3, "B": -1.7}), pd.Series(dtype=object))
    pnl = out.set_index("ticker")["pnl_usd"]
    assert pnl["A"] == pytest.approx(-150.0)
    assert pnl["B"] == pytest.approx(-300.0)  # a long stock loses at most its value
    assert pd.isna(pnl["C"])  # a missing shock stays missing, never invented


def test_replay_uses_own_history_then_proxies(tmp_path):
    s = make_settings(tmp_path, stress_events=EVENT)
    holdings, risk_positions = _book()
    out = stress.run_stress(holdings, risk_positions, _history(), s)
    pos = out["stress_positions"].query("scenario == 'crash'").set_index("ticker")
    assert pos.at["A", "source"] == "own history"
    assert pos.at["A", "shock_return"] == pytest.approx(-0.4)  # 2 x SPY path falls 100 -> 60
    assert pos.at["B", "source"] == "sector ETF proxy (XLE)"
    assert pos.at["B", "shock_return"] == pytest.approx(0.6 * 0.9 - 1)
    assert pos.at["C", "source"] == "beta x SPY proxy"
    assert pos.at["C", "shock_return"] == pytest.approx(0.5 * -0.4)

    sc = out["stress_scenarios"].set_index("scenario")
    assert sc.at["crash", "benchmark_return"] == pytest.approx(-0.4)
    assert sc.at["crash", "proxied_positions"] == 2
    assert not sc.at["old", "available"] and pd.isna(sc.at["old", "pnl_usd"])
    assert bool(sc.at["inner", "overlaps_training_event"])  # validation window inside a training window


def test_attribution_reconciles(tmp_path):
    s = make_settings(tmp_path, stress_events=EVENT)
    holdings, risk_positions = _book()
    out = stress.run_stress(holdings, risk_positions, _history(), s)
    totals = out["stress_scenarios"].dropna(subset=["pnl_usd"]).set_index("scenario")["pnl_usd"]
    by_pos = out["stress_positions"].groupby("scenario")["pnl_usd"].sum()
    by_sector = out["stress_sectors"].groupby("scenario")["pnl_usd"].sum()
    pd.testing.assert_series_equal(by_pos.reindex(totals.index), totals, check_names=False)
    pd.testing.assert_series_equal(by_sector.reindex(totals.index), totals, check_names=False)
    nav_after = out["stress_scenarios"].set_index("scenario")["nav_after"]
    assert nav_after["crash"] == pytest.approx(1000 + totals["crash"])  # cash is unchanged


def test_hypothetical_shocks(tmp_path):
    s = make_settings(tmp_path, stress_events=())
    holdings, risk_positions = _book()
    out = stress.run_stress(holdings, risk_positions, _history(), s)
    pos = out["stress_positions"].set_index(["scenario", "ticker"])["shock_return"]
    assert pos[("market-20%", "A")] == pytest.approx(-0.30)  # beta 1.5 x -20%
    assert pos[("market-20%_beta1", "C")] == pytest.approx(-0.20)
    assert pos[("sector_Energy", "C")] == 0 and pos[("sector_Energy", "A")] == pytest.approx(-0.25)
    # Energy carries most variance, so the combined shock hits it: (1 - 1.0 x 0.2)(1 - 0.2) - 1 for B
    assert pos[("combined", "B")] == pytest.approx(0.8 * 0.8 - 1)
    assert pos[("combined", "C")] == pytest.approx(0.5 * -0.2)  # not in the shocked sector
    assert set(out["stress_scenarios"]["group"]) == {"train"}
