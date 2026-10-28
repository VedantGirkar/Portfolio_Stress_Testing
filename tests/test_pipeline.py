"""End-to-end select workflow and dashboard smoke test, fully offline."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from stress_portfolio import data, main, selection

from .conftest import make_settings, make_universe

NOW = datetime(2025, 7, 1, 18, 0, tzinfo=data.NY)
APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def offline_world(monkeypatch):
    membership, bars = make_universe()
    spy = bars.groupby("date", as_index=False)[["open", "high", "low", "close", "adj_close", "volume"]].mean()
    bars = pd.concat([bars, spy.assign(ticker="SPY", dividends=0.0, splits=0.0)], ignore_index=True)

    def fake_batch(tickers, start, end):
        keep = bars["ticker"].isin(tickers) & (bars["date"] >= start) & (bars["date"] < end)  # yfinance end is exclusive
        sub = bars.loc[keep].rename(columns={v: k for k, v in data._YF_FIELDS.items()})
        raw = sub.set_index(["date", "ticker"]).unstack("ticker")
        return raw.swaplevel(axis=1).sort_index(axis=1).rename_axis("Date")

    monkeypatch.setattr(data, "fetch_membership",
                        lambda url, now: (membership, {"source": "test", "retrieved_at": now.isoformat(),
                                                       "limitation": "synthetic"}))
    monkeypatch.setattr(data, "download_batch", fake_batch)
    return bars  # tests may edit these bars in place before a run


def test_select_writes_complete_run(tmp_path, offline_world):
    s = make_settings(tmp_path)
    run_dir = main.run_select(s, offline=False, now=NOW)
    for name in ["eligibility", "sector_quotas", "shortlist", "core", "reserves", "membership",
                 "download_failures", "data_quality"]:
        assert (run_dir / f"{name}.csv").exists(), name
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["counts"] == {"membership": 30, "download_failures": 0, "eligible": 30,
                              "shortlist": 20, "core": 5, "reserves": 15}
    assert json.loads((run_dir / "settings.json").read_text())["core_size"] == 5

    # offline rerun uses the saved membership and price cache, and writes a new run directory
    second = main.run_select(s, offline=True, now=NOW)
    assert second != run_dir
    assert pd.read_csv(second / "core.csv")["ticker"].tolist() == pd.read_csv(run_dir / "core.csv")["ticker"].tolist()


def test_too_many_download_failures_stop_the_run(tmp_path, offline_world, monkeypatch):
    real = data.download_batch
    monkeypatch.setattr(data, "download_batch", lambda t, start, end: real([x for x in t if x not in {"ENE0", "FIN0"}],
                                                                          start, end))
    s = make_settings(tmp_path)  # 2 of 30 missing = 6.7%, above the 2% default
    with pytest.raises(SystemExit, match="2 of 30 tickers have no prices"):
        main.run_select(s, offline=False, now=NOW)
    assert not s.runs_dir.exists()  # no run is written from a partial universe


def test_old_price_caches_are_removed(tmp_path, offline_world):
    s = make_settings(tmp_path)
    old = s.data_dir / "raw" / "prices" / "2020-01-01_2020-01-02"
    old.mkdir(parents=True)
    (old / "X.csv").write_text("stale")
    main.run_select(s, offline=False, now=NOW)
    folders = [p.name for p in (s.data_dir / "raw" / "prices").iterdir()]
    assert len(folders) == 1 and folders[0].endswith("_2025-07-02")  # only the current window survives


def test_offline_marks_uncached_tickers(tmp_path, offline_world):
    s = make_settings(tmp_path, max_download_failure_share=0.05)
    main.run_select(s, offline=False, now=NOW)
    next((s.data_dir / "raw" / "prices").rglob("ENE0.csv")).unlink()
    run_dir = main.run_select(s, offline=True, now=NOW)
    failures = pd.read_csv(run_dir / "download_failures.csv")
    assert failures.to_dict("records") == [{"ticker": "ENE0", "reason": "not in offline cache"}]
    elig = pd.read_csv(run_dir / "eligibility.csv").set_index("ticker")["eligibility_reason"]
    assert elig["ENE0"] == "download failed"
    assert len(pd.read_csv(run_dir / "core.csv")) == 5


def test_intraday_run_never_caches_todays_partial_bar(tmp_path, offline_world):
    s = make_settings(tmp_path)
    run_dir = main.run_select(s, offline=False, now=datetime(2025, 7, 1, 11, 0, tzinfo=data.NY))
    assert json.loads((run_dir / "metadata.json").read_text())["price_cutoff"] == "2025-06-30"
    cached = pd.concat(pd.read_csv(p) for p in (s.data_dir / "raw" / "prices").rglob("*.csv"))
    assert cached["date"].max() == "2025-06-30"


def test_build_funds_core_and_reconciles(tmp_path, offline_world):
    s = make_settings(tmp_path)
    run_dir = main.run_select(s, offline=False, now=NOW, fund=True)
    book = json.loads((run_dir / "metadata.json").read_text())["portfolio"]
    assert (book["decision_date"], book["fill_date"]) == ("2025-06-30", "2025-07-01")

    holdings = pd.read_csv(run_dir / "holdings.csv")
    nav = pd.read_csv(run_dir / "nav.csv").iloc[-1]
    ledger = pd.read_csv(run_dir / "ledger.csv")
    assert holdings["market_value"].sum() == pytest.approx(nav["nav"])
    assert ledger["cash_flow"].sum() == pytest.approx(nav["cash"])
    stocks = holdings.loc[holdings["ticker"] != "CASH"]
    assert set(stocks["ticker"]) == set(pd.read_csv(run_dir / "core.csv")["provider_ticker"])
    assert nav["cash"] > 0  # the reserve stays in cash
    assert holdings["weight_of_nav"].sum() == pytest.approx(1.0)

    for name in ["risk_var", "risk_summary", "risk_positions", "risk_sectors", "risk_correlation", "risk_series",
                 "var_backtest", "var_backtest_summary"]:
        assert (run_dir / f"{name}.csv").exists(), name
    risk_meta = json.loads((run_dir / "metadata.json").read_text())["risk"]
    assert risk_meta["es_1d_99_usd"] >= risk_meta["var_1d_99_usd"] > 0

    scenarios = pd.read_csv(run_dir / "stress_scenarios.csv").set_index("scenario")
    assert scenarios.at["tariff2025", "available"]  # the only event inside the synthetic data
    assert not scenarios.at["gfc", "available"]
    positions = pd.read_csv(run_dir / "stress_positions.csv")
    assert positions.groupby("scenario")["pnl_usd"].sum().round(6).to_dict() == \
        scenarios["pnl_usd"].dropna().round(6).to_dict()
    cand = pd.read_csv(run_dir / "opt_candidates.csv").set_index("candidate")
    assert cand.at["current_5", "est_cost_usd"] == 0 and cand.at["current_5", "trades"] == 0
    weights = pd.read_csv(run_dir / "opt_weights.csv")
    sums = weights.groupby("candidate")["weight"].sum()
    assert (sums.drop("current_5") - 0.97).abs().max() < 1e-6  # every new candidate keeps the 3% cash reserve
    assert (weights.groupby("candidate")["weight"].max() <= 0.25 + 1e-6).all()
    solved = cand.loc[cand.index.str.startswith("stress_aware") & cand["expected_return"].notna()]
    assert (solved["worst_train_loss"] <= 0.40 + 1e-6).all()
    assert (pd.read_csv(run_dir / "opt_frontier.csv")["holding_count"].unique() == [5, 10]).all()
    assert set(pd.read_csv(run_dir / "stress_download_failures.csv")["ticker"]) == {"XLB", "XLC", "XLE", "XLF",
                                                                                  "XLI", "XLK", "XLP", "XLRE",
                                                                                  "XLU", "XLV", "XLY"}


def test_build_cannot_see_the_fill_day(tmp_path, offline_world):
    bars = offline_world
    s = make_settings(tmp_path)
    clean = pd.read_csv(main.run_select(s, offline=False, now=NOW, fund=True) / "core.csv")["ticker"]

    fill_day = bars["date"] == pd.Timestamp("2025-07-01")
    crashed = fill_day & bars["ticker"].isin(clean)
    for col in ["open", "high", "low", "close", "adj_close"]:
        bars.loc[crashed, col] *= 0.02  # a 98% crash of every core name on the fill day

    # control: a selection that is allowed to see the fill day does react to the crash
    membership = make_universe()[0]
    seen = bars.loc[bars["date"] <= pd.Timestamp("2025-07-01")]
    no_failures = pd.DataFrame(columns=["ticker", "reason"])
    assert selection.run_selection(membership, seen, no_failures, s)["core"]["ticker"].tolist() != clean.tolist()

    run_dir = main.run_select(make_settings(tmp_path / "second"), offline=False, now=NOW, fund=True)
    assert pd.read_csv(run_dir / "core.csv")["ticker"].tolist() == clean.tolist()  # build decided without it
    orders = pd.read_csv(run_dir / "orders.csv")
    opens = bars.loc[fill_day].set_index("ticker")["open"]
    assert (orders["ref_price"].values == opens.reindex(orders["ticker"]).values).all()  # filled at the crashed open


def test_dashboard_renders_build_run(tmp_path, offline_world, monkeypatch):
    s = make_settings(tmp_path)
    main.run_select(s, offline=False, now=NOW, fund=True)
    monkeypatch.setenv("STRESS_PORTFOLIO_RUNS_DIR", str(s.runs_dir))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].label == "NAV"


def test_dashboard_renders_saved_run(tmp_path, offline_world, monkeypatch):
    s = make_settings(tmp_path)
    main.run_select(s, offline=False, now=NOW)
    monkeypatch.setenv("STRESS_PORTFOLIO_RUNS_DIR", str(s.runs_dir))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs][:2] == ["Overview", "Selection"]
    assert at.metric[0].value == "5"


def test_dashboard_without_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("STRESS_PORTFOLIO_RUNS_DIR", str(tmp_path / "none"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "No selection runs" in at.warning[0].value
