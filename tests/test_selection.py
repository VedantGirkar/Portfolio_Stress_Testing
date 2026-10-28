import numpy as np
import pandas as pd
import pytest

from stress_portfolio import selection
from stress_portfolio.data import wide

from .conftest import make_settings, make_universe


def _wide(bars):
    return wide(bars, "adj_close"), wide(bars, "close"), wide(bars, "volume")


def test_known_feature_values(tmp_path):
    s = make_settings(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=300)
    p = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 300))), index=dates)
    adj = p.to_frame("A")
    vol = pd.DataFrame({"A": 1000.0}, index=dates)
    f = selection.compute_features(adj, adj, vol, s).loc["A"]

    assert f["momentum"] == pytest.approx(p.iloc[-22] / p.iloc[-148] - 1)
    r = (p / p.shift(1) - 1).iloc[-252:]
    assert f["downside_dev_annual"] == pytest.approx(np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252))
    w = p.iloc[-253:]
    assert f["max_drawdown"] == pytest.approx(-(w / w.cummax() - 1).min())
    assert f["median_dollar_volume"] == pytest.approx((p.iloc[-63:] * 1000).median())
    assert f["observations"] == 300 and f["sessions_stale"] == 0 and f["complete_window"]


def test_drawdown_of_known_path(tmp_path):
    s = make_settings(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=253)
    path = np.r_[np.full(100, 100.0), np.full(53, 60.0), np.full(100, 120.0)]
    adj = pd.DataFrame({"A": path}, index=dates)
    f = selection.compute_features(adj, adj, adj * 0 + 1, s).loc["A"]
    assert f["max_drawdown"] == pytest.approx(0.40)


def test_missing_before_listing_is_not_a_zero_return(tmp_path):
    s = make_settings(tmp_path)
    membership, bars = make_universe(n_per_sector=2, sessions=400)
    late = bars["ticker"].eq("ENE0") & (bars["date"] < bars["date"].unique()[200])
    adj, close, vol = _wide(bars.loc[~late])
    f = selection.compute_features(adj, close, vol, s)
    assert f.at["ENE0", "observations"] == 200
    # window prices are sessions 147-399; returns for 148-200 need a pre-listing price, so stay missing
    assert selection.window_returns(adj, s)["ENE0"].isna().sum() == 53
    assert not f.at["ENE0", "complete_window"]


def test_eligibility_reasons_and_share_class_dedupe(tmp_path):
    s = make_settings(tmp_path)
    membership, bars = make_universe(n_per_sector=4)
    membership.loc[membership["ticker"] == "ENE1", "cik"] = 0  # same company as ENE0
    bars = bars.loc[~(bars["ticker"].eq("FIN0") & (bars["date"] < bars["date"].unique()[200]))]
    bars = bars.loc[~(bars["ticker"].eq("HEA0") & (bars["date"] == bars["date"].max()))]
    bars.loc[bars["ticker"].eq("IND0"), "volume"] = 1.0
    feats = selection.compute_features(*_wide(bars), s)
    failures = pd.DataFrame({"ticker": ["UTI0"], "reason": ["no data"]})
    e = selection.eligibility(membership, feats, failures, s).set_index("ticker")["eligibility_reason"]

    assert e["UTI0"] == "download failed"
    assert e["FIN0"].startswith("fewer than 300")
    assert e["HEA0"] == "latest price is stale"
    assert e["IND0"].startswith("median dollar volume below")
    assert e["ENE0"] == "other share class: ENE1"  # ENE1 has more volume in the fixture
    assert e["ENE1"] == "eligible"


def test_sector_quotas(tmp_path):
    s = make_settings(tmp_path, shortlist_size=12, base_slots_per_sector=1)
    eligible = pd.DataFrame({"sector": ["A"] * 10 + ["B"] * 3 + ["C"] * 1})
    q = selection.sector_quotas(eligible, s).set_index("sector")["quota"]
    assert q.sum() == 12
    assert q["C"] == 1 and q["B"] <= 3
    assert (q <= eligible["sector"].value_counts()).all()
    with pytest.raises(ValueError, match="only 14 eligible"):
        selection.sector_quotas(eligible, make_settings(tmp_path, shortlist_size=20))


def test_run_selection_counts_weights_and_constraints(tmp_path):
    s = make_settings(tmp_path)
    membership, bars = make_universe()
    out = selection.run_selection(membership, bars, pd.DataFrame(columns=["ticker", "reason"]), s)
    core, reserves, short = out["core"], out["reserves"], out["shortlist"]

    assert len(short) == 20 and len(core) == 5 and len(reserves) == 15
    assert short["ticker"].is_unique
    assert core["target_weight"].sum() == pytest.approx(1.0)
    assert (reserves["target_weight"] == 0).all()
    assert core["sector"].value_counts().max() <= 2
    assert core["sector"].nunique() >= 3
    assert core["selection_order"].tolist() == [1, 2, 3, 4, 5]
    assert pd.isna(core.loc[0, "avg_corr_at_selection"])  # no penalty on the first pick
    assert short[["momentum_pct", "low_downside_pct", "low_drawdown_pct"]].le(1).all().all()

    again = selection.run_selection(membership, bars, pd.DataFrame(columns=["ticker", "reason"]), s)
    assert again["core"]["ticker"].tolist() == core["ticker"].tolist()  # deterministic


def _short(scores, sectors):
    return pd.DataFrame({"provider_ticker": list(scores), "investment_score": list(scores.values()),
                         "sector": sectors})


def test_correlation_penalty_prefers_diversifier(tmp_path):
    s = make_settings(tmp_path, core_size=2, min_sectors=1, correlation_penalty=0.5)
    rng = np.random.default_rng(0)
    base = rng.normal(size=252)
    rets = pd.DataFrame({"A": base, "TWIN": base + rng.normal(0, 0.01, 252), "OTHER": rng.normal(size=252)})
    short = _short({"A": 0.9, "TWIN": 0.8, "OTHER": 0.75}, ["X", "Y", "Z"])
    picks = selection.select_core(short, rets, s)
    assert picks["provider_ticker"].tolist() == ["A", "OTHER"]


def test_min_sectors_forces_new_sector(tmp_path):
    s = make_settings(tmp_path, core_size=3, min_sectors=3, max_per_sector=2, correlation_penalty=0.0)
    rets = pd.DataFrame(np.random.default_rng(0).normal(size=(252, 4)), columns=["A1", "A2", "B1", "C1"])
    short = _short({"A1": 0.9, "A2": 0.89, "B1": 0.5, "C1": 0.4}, ["A", "A", "B", "C"])
    picks = selection.select_core(short, rets, s)
    assert picks["provider_ticker"].tolist() == ["A1", "B1", "C1"]


def test_infeasible_core_raises(tmp_path):
    s = make_settings(tmp_path, core_size=3, min_sectors=1, max_per_sector=1)
    rets = pd.DataFrame(np.zeros((252, 3)), columns=["A1", "A2", "B1"])
    short = _short({"A1": 0.9, "A2": 0.8, "B1": 0.7}, ["A", "A", "B"])
    with pytest.raises(ValueError, match="no feasible pick for core slot 3"):
        selection.select_core(short, rets, s)


def test_undefined_correlation_gets_full_penalty(tmp_path):
    s = make_settings(tmp_path, core_size=2, min_sectors=1, correlation_penalty=1.0)
    rng = np.random.default_rng(0)
    rets = pd.DataFrame({"A": rng.normal(size=252), "FLAT": np.zeros(252), "B": rng.normal(size=252)})
    short = _short({"A": 0.9, "FLAT": 0.8, "B": 0.7}, ["X", "Y", "Z"])
    picks = selection.select_core(short, rets, s)
    assert picks["provider_ticker"].tolist() == ["A", "B"]  # FLAT's NaN correlation is treated as +1
