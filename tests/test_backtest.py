import pandas as pd
import pytest

from stress_portfolio import backtest
from stress_portfolio.data import wide

from .conftest import make_settings, make_universe

REVIEW_1, REVIEW_2 = pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30")


def _world():
    membership, bars = make_universe()
    spy = bars.groupby("date", as_index=False)[["open", "high", "low", "close", "adj_close", "volume"]].mean()
    bars = pd.concat([bars, spy.assign(ticker="SPY", dividends=0.0, splits=0.0)], ignore_index=True)
    return membership, bars


def _settings(tmp_path, **kw):
    return make_settings(tmp_path, backtest_start="2025-03-15", backtest_universe_size=10,
                         backtest_estimation_sessions=200, stress_loss_budget=0.6, rebalance_frequency="Q",
                         stress_events=(("tariff", "Test event", "2025-03-15", "2025-04-30", "validation"),), **kw)


def _run(tmp_path, bars, membership):
    return backtest.run_backtest(membership, bars, wide(bars, "adj_close"), _settings(tmp_path))


def test_strategies_share_start_and_reconcile(tmp_path):
    membership, bars = _world()
    out = _run(tmp_path, bars, membership)
    nav = out["bt_nav"]
    first = nav.loc[nav["strategy"].isin(backtest.STRATEGIES)].groupby("strategy").first()
    assert first["nav"].nunique() == 1  # same capital, same first fill, same costs
    assert (nav.dropna(subset=["cash"])["cash"] >= -1e-6).all()
    assert out["reviews"] == 2
    dec = out["bt_decisions"]
    assert set(dec["strategy"]) == {"equal_weight", "min_variance", "stress_aware"}  # buy-and-hold never reviews
    assert set(dec["date"]) == {REVIEW_1, REVIEW_2}
    assert set(out["bt_summary"]["strategy"]) == {*backtest.STRATEGIES, "SPY (reference)"}
    # review orders fill at the next session's open, never on the review day
    fills = out["bt_trades"].query("strategy == 'equal_weight'")["fill_date"].unique()
    assert REVIEW_1 not in set(fills) and pd.Timestamp("2025-04-01") in set(fills)


def test_reviews_cannot_see_later_prices(tmp_path):
    membership, bars = _world()
    clean = _run(tmp_path, bars, membership)
    crashed = bars.copy()
    hit = (crashed["date"] > REVIEW_1) & crashed["ticker"].str.startswith(("ENE", "FIN"))
    for col in ["open", "high", "low", "close", "adj_close"]:
        crashed.loc[hit, col] *= 0.3  # a 70% crash right after the first review
    later = _run(tmp_path / "b", crashed, membership)

    def at(out, date):
        return out["bt_decisions"].query("date == @date").set_index("strategy")
    pd.testing.assert_frame_equal(at(clean, REVIEW_1), at(later, REVIEW_1), check_dtype=False)  # unchanged
    # orders decided at or before the first review are identical; they fill at the (crashed) 04-01 open
    cols = ["strategy", "fill_date", "ticker", "order_qty", "decision_price"]
    early = lambda out: out["bt_trades"].query("fill_date <= '2025-04-01'")[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(early(clean), early(later))
    # control: the second review does see the crash
    assert at(clean, REVIEW_2).at["stress_aware", "utility_current"] != \
        at(later, REVIEW_2).at["stress_aware", "utility_current"]


def test_backtest_is_repeatable(tmp_path):
    membership, bars = _world()
    a, b = _run(tmp_path, bars, membership), _run(tmp_path / "b", bars, membership)
    pd.testing.assert_frame_equal(a["bt_nav"], b["bt_nav"])


def test_hedged_strategy_cannot_run_without_its_hedge(tmp_path):
    membership, bars = _world()
    with pytest.raises(NotImplementedError, match="hedge"):
        backtest.run_backtest(membership, bars, wide(bars, "adj_close"), _settings(tmp_path, hedge_method="delta"))
