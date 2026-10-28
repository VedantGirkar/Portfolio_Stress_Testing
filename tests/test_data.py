from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stress_portfolio import data

HTML = """<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
<th>GICS Sub-Industry</th><th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>x</td><td>y</td><td>2010-02-16</td><td>1067983</td><td>1839</td></tr>
<tr><td>AAPL</td><td>Apple</td><td>Information Technology</td><td>x</td><td>y</td><td>1982-11-30</td><td>320193</td><td>1977</td></tr>
</table>"""


def test_parse_membership_maps_provider_symbols():
    df = data.parse_membership(HTML, "test", "2026-01-02")
    assert df["ticker"].tolist() == ["BRK.B", "AAPL"]
    assert df["provider_ticker"].tolist() == ["BRK-B", "AAPL"]
    assert df["cik"].tolist() == [1067983, 320193]


def test_parse_membership_rejects_duplicates():
    dup = HTML.replace("AAPL", "BRK.B")
    with pytest.raises(ValueError, match="duplicate"):
        data.parse_membership(dup, "test", "2026-01-02")


def test_membership_round_trip(tmp_path):
    df = data.parse_membership(HTML, "test", "2026-01-02")
    data.save_membership(df, {"source": "test"}, tmp_path / "m.csv")
    back, meta = data.load_membership(tmp_path / "m.csv")
    assert meta == {"source": "test"}
    pd.testing.assert_frame_equal(back, df, check_dtype=False)


def _fake_yf(tickers, dates):
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume",
                                                 "Dividends", "Stock Splits"]])
    raw = pd.DataFrame(10.0, index=pd.DatetimeIndex(dates, name="Date"), columns=cols)
    return raw


def test_download_reports_failures_and_uses_cache(tmp_path, monkeypatch):
    calls = []

    def fake(tickers, start, end):
        calls.append(list(tickers))
        raw = _fake_yf(tickers, ["2026-01-02", "2026-01-05"])
        raw.loc[:, ("BAD", "Close")] = np.nan  # yfinance returns all-NaN columns for a failed symbol
        return raw

    monkeypatch.setattr(data, "download_batch", fake)
    bars, failures = data.download_prices(["AAA", "BAD", "BBB"], "2026-01-01", "2026-01-06", tmp_path, chunk_size=2)
    assert calls == [["AAA", "BAD"], ["BBB"]]
    assert failures["ticker"].tolist() == ["BAD"]
    assert sorted(bars["ticker"].unique()) == ["AAA", "BBB"]

    calls.clear()
    bars2, _ = data.download_prices(["AAA", "BBB"], "2026-01-01", "2026-01-06", tmp_path)
    assert calls == []  # served from cache
    assert len(bars2) == 4


def test_corporate_actions_survive_cache(tmp_path, monkeypatch):
    def fake(tickers, start, end):
        raw = _fake_yf(tickers, ["2026-01-02", "2026-01-05"])
        raw[("AAA", "Dividends")] = [0.0, 0.25]
        raw[("AAA", "Stock Splits")] = [4.0, 0.0]
        return raw

    monkeypatch.setattr(data, "download_batch", fake)
    data.download_prices(["AAA"], "2026-01-01", "2026-01-06", tmp_path)
    bars, _ = data.download_prices(["AAA"], "2026-01-01", "2026-01-06", tmp_path)  # second call reads the cache
    assert bars["dividends"].tolist() == [0.0, 0.25]
    assert bars["splits"].tolist() == [4.0, 0.0]


def test_batch_exception_marks_each_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "download_batch", lambda *a: (_ for _ in ()).throw(RuntimeError("rate limited")))
    bars, failures = data.download_prices(["AAA", "BBB"], "2026-01-01", "2026-01-06", tmp_path)
    assert bars.empty
    assert failures["ticker"].tolist() == ["AAA", "BBB"]
    assert failures["reason"].str.contains("rate limited").all()


def test_drop_partial_session():
    bars = pd.DataFrame({"date": pd.to_datetime(["2026-10-27", "2026-10-28"]), "ticker": "A"})
    during = datetime(2026, 10, 28, 11, 0, tzinfo=data.NY)
    after = datetime(2026, 10, 28, 17, 0, tzinfo=data.NY)
    assert data.drop_partial_session(bars, during)["date"].max() == pd.Timestamp("2026-10-27")
    assert len(data.drop_partial_session(bars, after)) == 2


def test_clean_bars_policy_and_report():
    bars = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-06"]),
        "ticker": "A", "open": 1.0, "high": 1.0, "low": 1.0,
        "close": [10.0, 11.0, -1.0, 11.0], "adj_close": [10.0, 11.0, -1.0, 11.0],
        "volume": [5, 5, 0, 5], "dividends": 0.0, "splits": 0.0,
    })
    clean, report = data.clean_bars(bars)
    assert len(clean) == 3
    assert clean["close"].iloc[0] == 11.0  # duplicate kept the last row
    assert np.isnan(clean["close"].iloc[1])  # nonpositive price became missing, not filled
    row = report.iloc[0]
    assert (row["duplicate_rows"], row["nonpositive_price_rows"], row["zero_volume_days"]) == (1, 1, 1)


def test_future_split_factor():
    dates = pd.to_datetime(["2021-07-19", "2021-07-20", "2024-06-07", "2024-06-10", "2024-06-11"])
    bars = pd.DataFrame({"date": dates, "ticker": "N", "splits": [0, 4.0, 0, 10.0, 0]})
    f = data.future_split_factor(bars)["N"].tolist()
    assert f == [40.0, 10.0, 10.0, 1.0, 1.0]  # the split day itself already trades post-split
