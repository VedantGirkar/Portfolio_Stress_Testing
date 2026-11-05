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


CBOE_COLUMNS = ["underlying_symbol", "quote_date", "root", "expiration", "strike", "option_type", "open", "high", "low",
                "close", "trade_volume", "bid_size_1545", "bid_1545", "ask_size_1545", "ask_1545",
                "underlying_bid_1545", "underlying_ask_1545", "implied_underlying_price_1545",
                "active_underlying_price_1545", "implied_volatility_1545", "delta_1545", "gamma_1545", "theta_1545",
                "vega_1545", "rho_1545", "bid_size_eod", "bid_eod", "ask_size_eod", "ask_eod", "underlying_bid_eod",
                "underlying_ask_eod", "vwap", "open_interest", "delivery_code"]


def _cboe_file(path, rows, drop=()):
    """A zipped CSV in the Cboe Option EOD Summary layout (v1.1, with Calcs)."""
    base = dict.fromkeys(CBOE_COLUMNS, 0)
    df = pd.DataFrame([base | r for r in rows]).drop(columns=list(drop))
    df.to_csv(path, index=False, compression={"method": "zip", "archive_name": path.stem + ".csv"})
    return path


ROW = dict(underlying_symbol="SPY", quote_date="2016-01-04", root="SPY", expiration="2016-02-19", strike=190,
           option_type="p", bid_1545=1.5, ask_1545=1.6, underlying_bid_1545=201.0, underlying_ask_1545=201.02,
           active_underlying_price_1545=201.01, implied_volatility_1545=0.21, delta_1545=-0.3, open_interest=100)


def test_option_loader_normalizes_and_excludes_adjusted_roots(tmp_path):
    from .conftest import make_settings
    s = make_settings(tmp_path)
    path = _cboe_file(tmp_path / "UnderlyingOptionsEODCalcs_2016-01-04.zip", [
        ROW,
        ROW | {"strike": 150, "bid_1545": 0.0, "ask_1545": 0.05, "implied_volatility_1545": 0.0,
               "underlying_bid_1545": 0.0},
        ROW | {"root": "SPY7"},
        ROW | {"underlying_symbol": "QQQ", "root": "QQQ"},
    ])
    quotes, quality = data.load_option_quotes([path], s)
    assert list(quotes["strike"]) == [150, 190]  # QQQ filtered out, SPY7 excluded
    assert quality.to_dict("records") == [{"file": path.name, "root": "SPY7", "rows": 1,
                                           "action": "excluded: non-standard root, deliverable unknown"}]
    cheap, normal = quotes.iloc[0], quotes.iloc[1]
    assert cheap["bid"] == 0.0 and np.isnan(cheap["iv"]) and np.isnan(cheap["underlying_bid"])  # 0 IV = missing
    assert normal["iv"] == 0.21 and normal["option_type"] == "P" and normal["multiplier"] == 100
    assert normal["contract"] == "SPY 2016-02-19 P 190.00" and normal["quote_date"] == pd.Timestamp("2016-01-04")


def test_option_loader_requires_calcs_columns(tmp_path):
    from .conftest import make_settings
    path = _cboe_file(tmp_path / "UnderlyingOptionsEODQuotes_2016-01-04.zip", [ROW],
                      drop=["implied_volatility_1545", "delta_1545", "active_underlying_price_1545"])
    with pytest.raises(ValueError, match="with Calcs"):
        data.load_option_quotes([path], make_settings(tmp_path))


def test_yfinance_chain_keeps_standard_contracts_and_real_quotes(tmp_path):
    from .conftest import make_settings
    s = make_settings(tmp_path)
    raw = pd.DataFrame({
        "contractSymbol": ["SPY261218P00734000", "SPY1261218P00734000", "SPY261218P00700000", "SPY261218P00650000"],
        "bid": [5.0, 5.0, 0.0, 1.0], "ask": [5.1, 5.1, 0.0, 1.1], "lastPrice": [9.0, 9.0, 9.0, 9.0],
        "impliedVolatility": [0.18, 0.18, 1e-5, 0.0], "lastTradeDate": [None] * 4, "openInterest": [1] * 4,
        "contractSize": ["REGULAR"] * 4})
    chain, quality = data.normalize_yf_chain(raw, s, datetime(2026, 11, 4, 18, 0, tzinfo=data.NY))
    assert list(quality["root"]) == ["SPY1"] and quality["rows"].iloc[0] == 1
    assert list(chain["strike"]) == [650.0, 700.0, 734.0]
    row = chain.set_index("strike").loc[734.0]
    assert row["contract"] == "SPY 2026-12-18 P 734.00" and row["expiration"] == pd.Timestamp("2026-12-18")
    assert row["multiplier"] == 100 and row["quote_date"] == pd.Timestamp("2026-11-04")
    zero = chain.set_index("strike").loc[700.0]
    assert zero["bid"] == 0.0 and np.isnan(zero["ask"])  # never replaced by the 9.0 last trade
    assert np.isnan(chain.set_index("strike").loc[650.0, "iv"])
    assert data.normalize_yf_chain(pd.DataFrame(), s, datetime(2026, 11, 4))[0].empty


def test_option_chain_cache_round_trips_offline(tmp_path, monkeypatch):
    from .conftest import make_settings
    s = make_settings(tmp_path)
    cache = tmp_path / "options"
    assert data.fetch_option_chain(s, cache, True, datetime(2026, 11, 4))[0].empty  # nothing cached yet
    raw = pd.DataFrame({"contractSymbol": ["SPY261218P00734000"], "bid": [5.0], "ask": [5.1], "impliedVolatility": [0.18],
                        "lastTradeDate": [None], "openInterest": [1], "contractSize": ["REGULAR"]})
    monkeypatch.setattr(data, "download_option_chain", lambda *a: raw)
    live = data.fetch_option_chain(s, cache, False, datetime(2026, 11, 4, 18))[0]
    offline = data.fetch_option_chain(s, cache, True, datetime(2026, 11, 5))[0]
    assert offline.iloc[0]["contract"] == live.iloc[0]["contract"] and offline.iloc[0]["ask"] == 5.1
