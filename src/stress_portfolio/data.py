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
