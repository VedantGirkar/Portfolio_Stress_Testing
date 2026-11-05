"""Membership and price data: download, cache, standardize, validate."""

import hashlib
import io
import json
import shutil
import urllib.request
import warnings
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BAR_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
_YF_FIELDS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
    "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits",
}
NY = ZoneInfo("America/New_York")
MARKET_CLOSE_SETTLED = time(16, 30)  # Yahoo's daily bar is final shortly after the 16:00 close


def fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "stress-portfolio research script"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def parse_membership(html: str, source: str, snapshot_date: str) -> pd.DataFrame:
    raw = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    df = pd.DataFrame({
        "ticker": raw["Symbol"].astype(str).str.strip(),
        "company": raw["Security"],
        "sector": raw["GICS Sector"],
        "sub_industry": raw["GICS Sub-Industry"],
        "cik": raw["CIK"].astype("Int64"),
        "date_added": raw["Date added"],
    })
    df["provider_ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # Yahoo uses BRK-B for BRK.B
    df["source"] = source
    df["snapshot_date"] = snapshot_date
    if df["ticker"].duplicated().any():
        raise ValueError(f"duplicate tickers in membership: {df.loc[df['ticker'].duplicated(), 'ticker'].tolist()}")
    return df


def fetch_membership(url: str, now: datetime | None = None) -> tuple[pd.DataFrame, dict]:
    now = now or datetime.now(NY)
    html = fetch_url(url)
    df = parse_membership(html, url, now.date().isoformat())
    meta = {
        "source": url,
        "retrieved_at": now.isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(html.encode()).hexdigest(),
        "rows": len(df),
        "limitation": "Current constituents, not point-in-time membership; any backtest start before "
                      "the snapshot date carries survivorship and look-ahead bias.",
    }
    return df, meta


def save_membership(df: pd.DataFrame, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_membership(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, dtype={"ticker": str, "provider_ticker": str})
    df["cik"] = df["cik"].astype("Int64")
    return df, json.loads(path.with_suffix(".json").read_text())


def download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Raw yfinance download; kept tiny so tests can replace it."""
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return yf.download(
            tickers, start=start, end=end, auto_adjust=False, actions=True,
            group_by="ticker", threads=True, progress=False, multi_level_index=True,
        )


def _standardize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if ticker not in raw.columns.get_level_values(0):
        return pd.DataFrame(columns=BAR_COLUMNS)
    sub = raw[ticker].rename(columns=_YF_FIELDS)
    sub = sub.dropna(subset=["close"]).reset_index()
    sub = sub.rename(columns={sub.columns[0]: "date"})
    sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.normalize()
    sub["ticker"] = ticker
    for col in BAR_COLUMNS:
        if col not in sub.columns:
            sub[col] = 0.0 if col in ("dividends", "splits") else float("nan")
    return sub[BAR_COLUMNS]


def download_prices(tickers: list[str], start: str, end: str, cache_dir: Path,
                    chunk_size: int = 75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long bars table plus per-ticker failures. Cache is one CSV per ticker per (start, end)."""
    cache = cache_dir / f"{start}_{end}"
    cache.mkdir(parents=True, exist_ok=True)
    missing = [t for t in tickers if not (cache / f"{t}.csv").exists()]
    failures = []
    for i in range(0, len(missing), chunk_size):
        chunk = missing[i:i + chunk_size]
        try:
            raw = download_batch(chunk, start, end)
        except Exception as exc:  # one bad batch must not lose the others
            failures += [{"ticker": t, "reason": f"batch error: {exc}"} for t in chunk]
            continue
        for t in chunk:
            bars = _standardize(raw, t)
            if bars.empty:
                failures.append({"ticker": t, "reason": "no price data returned"})
            else:
                bars.to_csv(cache / f"{t}.csv", index=False)
    frames = [pd.read_csv(cache / f"{t}.csv", parse_dates=["date"]) for t in tickers if (cache / f"{t}.csv").exists()]
    bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BAR_COLUMNS)
    bars["ticker"] = bars["ticker"].astype(str)
    return bars, pd.DataFrame(failures, columns=["ticker", "reason"])


def prune_cache(cache_dir: Path, keep: str) -> None:
    """Delete every dated cache folder except `keep`; each day's request window gets a new folder."""
    for old in cache_dir.glob("*_*"):
        if old.is_dir() and old.name != keep:
            shutil.rmtree(old)


def drop_partial_session(bars: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Remove today's bar if the US session has not settled yet."""
    now = (now or datetime.now(NY)).astimezone(NY)
    if now.time() >= MARKET_CLOSE_SETTLED:
        return bars
    return bars.loc[bars["date"] < pd.Timestamp(now.date())].reset_index(drop=True)


def clean_bars(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the logged handling policy and return (clean bars, per-ticker quality report).

    Policy: duplicate (date, ticker) rows keep the last; nonpositive prices become missing.
    Nothing is forward-filled.
    """
    price_cols = ["open", "high", "low", "close", "adj_close"]
    dup = bars.duplicated(["date", "ticker"], keep="last")
    nonpos = (bars[price_cols] <= 0).any(axis=1)
    report = (
        bars.assign(_dup=dup, _nonpos=nonpos, _zero_vol=bars["volume"].fillna(0) <= 0)
        .groupby("ticker")
        .agg(rows=("date", "size"), first_date=("date", "min"), last_date=("date", "max"),
             duplicate_rows=("_dup", "sum"), nonpositive_price_rows=("_nonpos", "sum"),
             zero_volume_days=("_zero_vol", "sum"))
        .reset_index()
    )
    clean = bars.loc[~dup].copy()
    clean.loc[nonpos[~dup], price_cols] = float("nan")
    clean = clean.sort_values(["ticker", "date"]).reset_index(drop=True)
    stale = clean.groupby("ticker")["close"].apply(_longest_flat_run).rename("longest_flat_close_run")
    return clean, report.merge(stale.reset_index(), on="ticker", how="left")


def _longest_flat_run(close: pd.Series) -> int:
    same = close.eq(close.shift(1))
    return int(same.groupby((~same).cumsum()).sum().max()) if len(close) else 0


def future_split_factor(bars: pd.DataFrame) -> pd.DataFrame:
    """Date x ticker product of split ratios strictly after each date: split-adjusted shares / factor =
    real shares on that date (e.g. NVDA in 2016 is 40 after its 4:1 and 10:1 splits)."""
    ratio = wide(bars, "splits").fillna(0.0).replace(0.0, 1.0)
    return (ratio.iloc[::-1].cumprod().iloc[::-1].shift(-1)).fillna(1.0)


def wide(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    """Date x ticker table on the common session calendar (union of all dates)."""
    return bars.pivot(index="date", columns="ticker", values=field).sort_index()


# Cboe DataShop "Option EOD Summary" (layout v1.1), ordered with Calcs. Only these columns are used.
_CBOE_FIELDS = {
    "underlying_symbol": "underlying", "quote_date": "quote_date", "root": "root", "expiration": "expiration",
    "strike": "strike", "option_type": "option_type", "bid_1545": "bid", "ask_1545": "ask",
    "bid_size_1545": "bid_size", "ask_size_1545": "ask_size", "underlying_bid_1545": "underlying_bid",
    "underlying_ask_1545": "underlying_ask", "active_underlying_price_1545": "model_underlying_price",
    "implied_volatility_1545": "iv", "delta_1545": "delta", "gamma_1545": "gamma", "theta_1545": "theta",
    "vega_1545": "vega", "bid_eod": "bid_eod", "ask_eod": "ask_eod", "open_interest": "open_interest",
}


def load_option_quotes(paths: list[Path], s) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalized hedge-underlying quotes from Cboe daily files (zipped CSV), plus a data-quality table.

    Fill and mark prices are the 15:45 NBBO (bid/ask). Rules from the layout notes:
    - Only the standard root (s.hedge_root) is kept; roots ending in a digit are adjusted contracts whose
      deliverable the files do not give, so they are excluded and listed in the quality table.
    - The files carry no multiplier field, so `multiplier` is the OCC standard for the standard root.
    - Implied volatility 0 means Cboe could not compute one: it becomes missing, never zero volatility.
    - Underlying bid/ask 0 means not available: missing. A zero option bid is a real quote and is kept.
    - IV and Greeks are Cboe's (method not documented), labeled vendor values; our repricing uses hedging.py.
    """
    from .hedging import contract_id

    frames, quality = [], []
    for path in paths:
        raw = pd.read_csv(path)
        missing = set(_CBOE_FIELDS) - set(raw.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)} (order the file with Calcs)")
        df = raw.loc[raw["underlying_symbol"] == s.hedge_underlying, list(_CBOE_FIELDS)].rename(columns=_CBOE_FIELDS)
        other = df["root"] != s.hedge_root
        for root, n in df.loc[other, "root"].value_counts().items():
            quality.append({"file": Path(path).name, "root": root, "rows": int(n),
                            "action": "excluded: non-standard root, deliverable unknown"})
        frames.append(df.loc[~other])
    cols = list(_CBOE_FIELDS.values()) + ["multiplier", "contract", "iv_source"]
    if not frames:
        return pd.DataFrame(columns=cols), pd.DataFrame(quality)
    q = pd.concat(frames, ignore_index=True)
    q["quote_date"], q["expiration"] = pd.to_datetime(q["quote_date"]), pd.to_datetime(q["expiration"])
    q["option_type"] = q["option_type"].str.upper()
    q["iv"] = q["iv"].where(q["iv"] > 0)
    for c in ["underlying_bid", "underlying_ask", "model_underlying_price"]:
        q[c] = q[c].where(q[c] > 0)
    q["multiplier"] = s.standard_option_multiplier
    q["contract"] = [contract_id(r, e, t, k) for r, e, t, k in
                     zip(q["root"], q["expiration"], q["option_type"], q["strike"])]
    q["iv_source"] = "Cboe 15:45 (vendor-observed)"
    return q[cols].sort_values(["quote_date", "expiration", "strike", "option_type"]).reset_index(drop=True), \
        pd.DataFrame(quality)


def download_option_chain(underlying: str, max_days: int, today) -> pd.DataFrame:
    """Raw yfinance put chains for expiries within `max_days`; kept tiny so tests can replace it."""
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = yf.Ticker(underlying)
        last = pd.Timestamp(today) + pd.Timedelta(days=max_days)
        frames = [t.option_chain(e).puts.assign(expiration=e) for e in t.options if pd.Timestamp(e) <= last]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_yf_chain(raw: pd.DataFrame, s, fetched_at) -> tuple[pd.DataFrame, pd.DataFrame]:
    """yfinance puts in the same shape as load_option_quotes, plus a data-quality table.

    The root is read from the right end of the OCC symbol (the last 15 characters are date, type and strike), so
    adjusted roots such as SPY1 are recognized. Only the standard root with contractSize REGULAR is kept. A missing
    or non-positive bid/ask stays missing (never replaced by the last trade); IV outside (0, 5] becomes missing.
    Quotes are delayed and unofficial: yfinance, not an exchange feed.
    """
    from .hedging import contract_id

    cols = ["quote_date", "fetched_at", "root", "expiration", "strike", "option_type", "bid", "ask", "iv",
            "last_trade", "open_interest", "multiplier", "contract", "iv_source"]
    if raw.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame([{"root": None, "rows": 0, "action": "empty chain"}])
    sym = raw["contractSymbol"].astype(str)
    df = pd.DataFrame({"root": sym.str[:-15], "expiration": pd.to_datetime(sym.str[-15:-9], format="%y%m%d"),
                       "option_type": sym.str[-9], "strike": sym.str[-8:].astype(int) / 1000,
                       "bid": raw["bid"].where(raw["bid"] >= 0), "ask": raw["ask"].where(raw["ask"] > 0),
                       "iv": raw["impliedVolatility"].where((raw["impliedVolatility"] > 0) & (raw["impliedVolatility"] <= 5)),
                       "last_trade": raw.get("lastTradeDate"), "open_interest": raw.get("openInterest"),
                       "size": raw.get("contractSize", pd.Series("REGULAR", index=raw.index))})
    bad = (df["root"] != s.hedge_root) | (df["size"] != "REGULAR")
    quality = (df.loc[bad].groupby(["root", "size"]).size().rename("rows").reset_index()
               .assign(action="excluded: non-standard contract, deliverable unknown"))
    df = df.loc[~bad].drop(columns="size")
    ts = pd.Timestamp(fetched_at)
    df["quote_date"], df["fetched_at"] = ts.tz_localize(None).normalize() if ts.tz else ts.normalize(), ts.isoformat()
    df["multiplier"] = s.standard_option_multiplier
    df["contract"] = [contract_id(r, e, t, k) for r, e, t, k in zip(df["root"], df["expiration"], df["option_type"],
                                                                     df["strike"])]
    df["iv_source"] = "yfinance (delayed, unofficial)"
    return df[cols].sort_values(["expiration", "strike"]).reset_index(drop=True), quality


def fetch_option_chain(s, cache_dir: Path, offline: bool, now: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Today's hedge-underlying put chain from yfinance, cached as CSV; offline reuses the newest cached snapshot."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if offline:
        snaps = sorted(cache_dir.glob(f"{s.hedge_underlying}_*.csv"))
        if not snaps:
            return pd.DataFrame(), pd.DataFrame([{"root": None, "rows": 0, "action": "no cached chain (offline)"}])
        chain = pd.read_csv(snaps[-1], parse_dates=["quote_date", "expiration"])
        return chain, pd.DataFrame(columns=["root", "size", "rows", "action"])
    raw = download_option_chain(s.hedge_underlying, s.option_chain_max_days, now.date())
    chain, quality = normalize_yf_chain(raw, s, now)
    if not chain.empty:
        chain.to_csv(cache_dir / f"{s.hedge_underlying}_{now:%Y%m%d-%H%M%S}.csv", index=False)
    return chain, quality
