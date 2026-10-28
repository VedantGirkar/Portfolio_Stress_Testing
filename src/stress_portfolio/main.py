"""Command-line entry point."""

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import data, selection
from .config import ROOT, Settings


def git_revision() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        return (rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")) or None
    except OSError:
        return None


def new_run_dir(runs_dir: Path, now: datetime) -> Path:
    run_dir = runs_dir / f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a completed run
    return run_dir


def run_select(s: Settings, offline: bool, now: datetime | None = None) -> Path:
    now = now or datetime.now(data.NY)
    raw = s.data_dir / "raw"
    membership_path = raw / "membership.csv"
    price_cache = raw / "prices"

    if offline:
        membership, member_meta = data.load_membership(membership_path)
        caches = sorted(p.name for p in price_cache.glob("*_*") if p.is_dir())
        if not caches:
            raise SystemExit(f"offline mode: no cached prices in {price_cache}; run once without --offline")
        start, end = caches[-1].split("_")
    else:
        membership, member_meta = data.fetch_membership(s.membership_url, now)
        data.save_membership(membership, member_meta, membership_path)
        # yfinance end is exclusive; exclude today until its bar has settled so a partial bar is never cached
        settled = now.astimezone(data.NY).time() >= data.MARKET_CLOSE_SETTLED
        end = (now.date() + timedelta(days=1 if settled else 0)).isoformat()
        start = (now.date() - timedelta(days=s.history_calendar_days)).isoformat()

    tickers = membership["provider_ticker"].tolist()
    if offline:
        cached = {p.stem for p in (price_cache / f"{start}_{end}").glob("*.csv")}
        tickers_to_load = [t for t in tickers if t in cached]
        bars, failures = data.download_prices(tickers_to_load, start, end, price_cache, s.download_chunk_size)
        failures = pd.concat([failures, pd.DataFrame(
            {"ticker": sorted(set(tickers) - cached), "reason": "not in offline cache"})], ignore_index=True)
    else:
        bars, failures = data.download_prices(tickers, start, end, price_cache, s.download_chunk_size)
    bars = data.drop_partial_session(bars, now)
    bars, quality = data.clean_bars(bars)

    result = selection.run_selection(membership, bars, failures, s)

    run_dir = new_run_dir(s.runs_dir, now)
    for name in ["eligibility", "sector_quotas", "shortlist", "core", "reserves"]:
        result[name].to_csv(run_dir / f"{name}.csv", index=False)
    membership.to_csv(run_dir / "membership.csv", index=False)
    failures.to_csv(run_dir / "download_failures.csv", index=False)
    quality.to_csv(run_dir / "data_quality.csv", index=False)
    (run_dir / "settings.json").write_text(json.dumps(s.to_dict(), indent=2))
    meta = {
        "run_id": run_dir.name,
        "workflow": "select",
        "created_at": now.isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "data_mode": "offline cache" if offline else "live download",
        "price_source": "Yahoo Finance via yfinance",
        "price_request_window": [start, end],
        "price_cutoff": str(result["cutoff"].date()),
        "membership": member_meta,
        "counts": {
            "membership": len(membership),
            "download_failures": len(failures),
            "eligible": int(result["eligibility"]["eligible"].sum()),
            "shortlist": len(result["shortlist"]),
            "core": len(result["core"]),
            "reserves": len(result["reserves"]),
        },
        "notes": [
            "Target weights only; no orders, fills or cash ledger yet (milestone 2).",
            "Benchmark data not downloaded in this workflow.",
        ],
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="stress-portfolio",
        description="Stress-tested equity portfolio research platform.",
    )
    sub = parser.add_subparsers(dest="command")
    sel = sub.add_parser("select", help="build the 100-stock shortlist and the 10-stock core")
    sel.add_argument("--offline", action="store_true", help="use saved membership and cached prices only")
    args = parser.parse_args(argv)

    if args.command != "select":
        parser.print_help()
        return
    s = Settings()
    run_dir = run_select(s, args.offline)
    meta = json.loads((run_dir / "metadata.json").read_text())
    print(f"run saved to {run_dir}")
    print(json.dumps(meta["counts"], indent=2))
    core = pd.read_csv(run_dir / "core.csv")
    print(core[["selection_order", "ticker", "company", "sector", "investment_score", "selection_score"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
