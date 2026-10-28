"""End-to-end select workflow and dashboard smoke test, fully offline."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from stress_portfolio import data, main

from .conftest import make_settings, make_universe

NOW = datetime(2025, 7, 1, 18, 0, tzinfo=data.NY)
APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def offline_world(monkeypatch):
    membership, bars = make_universe()

    def fake_batch(tickers, start, end):
        keep = bars["ticker"].isin(tickers) & (bars["date"] >= start) & (bars["date"] < end)  # yfinance end is exclusive
        sub = bars.loc[keep].rename(columns={v: k for k, v in data._YF_FIELDS.items()})
        raw = sub.set_index(["date", "ticker"]).unstack("ticker")
        return raw.swaplevel(axis=1).sort_index(axis=1).rename_axis("Date")

    monkeypatch.setattr(data, "fetch_membership",
                        lambda url, now: (membership, {"source": "test", "retrieved_at": now.isoformat(),
                                                       "limitation": "synthetic"}))
    monkeypatch.setattr(data, "download_batch", fake_batch)
    return membership


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


def test_intraday_run_never_caches_todays_partial_bar(tmp_path, offline_world):
    s = make_settings(tmp_path)
    run_dir = main.run_select(s, offline=False, now=datetime(2025, 7, 1, 11, 0, tzinfo=data.NY))
    assert json.loads((run_dir / "metadata.json").read_text())["price_cutoff"] == "2025-06-30"
    cached = pd.concat(pd.read_csv(p) for p in (s.data_dir / "raw" / "prices").rglob("*.csv"))
    assert cached["date"].max() == "2025-06-30"


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
