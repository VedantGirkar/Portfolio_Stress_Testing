"""Command-line entry point."""

import argparse
import json
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import backtest, data, risk, selection
from .config import ROOT, Settings


def git_revision() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True)
        return (rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")) or None
    except OSError:
        return None


def new_run_dir(runs_dir: Path, now: datetime) -> Path:
    run_dir = runs_dir / f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a completed run
    return run_dir


def run_risk(s: Settings, book: dict, bars: pd.DataFrame, price_cache: Path, start: str, end: str,
             offline: bool, now: datetime) -> dict:
    """Load the benchmark from the same price window and measure the risk of the funded book."""
    if offline and not (price_cache / f"{start}_{end}" / f"{s.benchmark_ticker}.csv").exists():
        raise SystemExit(f"offline mode: {s.benchmark_ticker} is not cached; run `build` once without --offline")
    bench, failed = data.download_prices([s.benchmark_ticker], start, end, price_cache)
    if not failed.empty:
        raise SystemExit(f"benchmark {s.benchmark_ticker} download failed: {failed['reason'].iloc[0]}; rerun")
    bench, _ = data.clean_bars(data.drop_partial_session(bench, now))
    adj = data.wide(bars, "adj_close")
    cal = selection.session_calendar(adj)
    return risk.risk_report(book["holdings"], adj.reindex(cal),
                            data.wide(bench, "adj_close")[s.benchmark_ticker].reindex(cal), book["nav"], s)


def run_select(s: Settings, offline: bool, now: datetime | None = None, fund: bool = False) -> Path:
    """Selection run; with fund=True, select as of the prior session and buy the core at the next open."""
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
    share = len(failures) / len(tickers)
    if share > s.max_download_failure_share:
        retry = "run once without --offline" if offline else "rerun to fetch only the missing tickers"
        raise SystemExit(f"{len(failures)} of {len(tickers)} tickers have no prices ({share:.1%}), above the "
                         f"{s.max_download_failure_share:.0%} limit; {retry}. "
                         f"First failures: {failures['ticker'].head(5).tolist()}")
    if not offline:
        for old in price_cache.glob("*_*"):  # each day's window gets a new folder; keep only the current one
            if old.is_dir() and old.name != f"{start}_{end}":
                shutil.rmtree(old)
    bars = data.drop_partial_session(bars, now)
    bars, quality = data.clean_bars(bars)

    decision_date = None
    if fund:
        # decide on the prior session so the fill uses the latest session's open, never a price the decision saw
        decision_date = selection.session_calendar(data.wide(bars, "adj_close"))[-2]
    picked = bars if decision_date is None else bars.loc[bars["date"] <= decision_date]
    result = selection.run_selection(membership, picked, failures, s)
    book = backtest.fund_and_hold(result["core"], bars, decision_date, s) if fund else None
    risk_tables = run_risk(s, book, bars, price_cache, start, end, offline, now) if fund else {}

    run_dir = new_run_dir(s.runs_dir, now)
    for name in ["eligibility", "sector_quotas", "shortlist", "core", "reserves"]:
        result[name].to_csv(run_dir / f"{name}.csv", index=False)
    membership.to_csv(run_dir / "membership.csv", index=False)
    failures.to_csv(run_dir / "download_failures.csv", index=False)
    quality.to_csv(run_dir / "data_quality.csv", index=False)
    if book:
        for name in ["orders", "holdings", "ledger", "nav", "cost_scenarios"]:
            book[name].to_csv(run_dir / f"{name}.csv", index=False)
    for name, table in risk_tables.items():
        table.to_csv(run_dir / f"{name}.csv", index=False)
    (run_dir / "settings.json").write_text(json.dumps(s.to_dict(), indent=2))
    meta = {
        "run_id": run_dir.name,
        "workflow": "build" if fund else "select",
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
        "notes": [] if fund else ["Benchmark data not downloaded in this workflow."],
    }
    if book:
        last = book["nav"].iloc[-1]
        meta["portfolio"] = {
            "decision_date": str(book["decision_date"].date()),
            "fill_date": str(book["fill_date"].date()),
            "valuation_date": str(last["date"].date()),
            "initial_capital": s.initial_capital,
            "nav": round(float(last["nav"]), 2),
            "cash": round(float(last["cash"]), 2),
            "total_trading_cost": round(float(book["orders"]["total_cost"].sum()), 2),
            "event_order": list(backtest.EVENT_ORDER),
            "conventions": [
                "Selection uses data up to the decision date; orders fill at the next session's open.",
                "Fill price = open +/- half-spread and slippage; commission charged separately.",
                "Prices and dividends are split-adjusted (Yahoo); quantities are split-adjusted shares.",
                "Dividends credited on the ex-date, not the pay date.",
                "Valued at the session close; a missing close keeps the last close as its mark.",
            ],
        }
        var = risk_tables["risk_var"].set_index(["method", "horizon_days", "confidence"])
        summary = risk_tables["risk_summary"].set_index("metric")["value"]
        meta["risk"] = {
            "method": "historical simulation of today's holdings (hypothetical P&L), losses positive",
            "benchmark": s.benchmark_ticker,
            "window": risk_tables["risk_summary"].loc[0, ["start", "end"]].tolist(),
            "var_1d_99_usd": round(float(var.loc[("historical", 1, 0.99), "var_usd"]), 2),
            "es_1d_99_usd": round(float(var.loc[("historical", 1, 0.99), "es_usd"]), 2),
            "volatility_annual": round(float(summary["volatility"]), 4),
            "beta": round(float(summary["beta"]), 3),
            "convention": risk.VAR_CONVENTION,
        }
    else:
        meta["notes"].append("Target weights only; run `stress-portfolio build` for holdings, cash and NAV.")
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="stress-portfolio",
        description="Stress-tested equity portfolio research platform.",
    )
    sub = parser.add_subparsers(dest="command")
    sel = sub.add_parser("select", help="build the 100-stock shortlist and the 10-stock core")
    build = sub.add_parser("build", help="select as of the prior session, buy the core at the next open, value it")
    for p in (sel, build):
        p.add_argument("--offline", action="store_true", help="use saved membership and cached prices only")
    args = parser.parse_args(argv)

    if args.command not in {"select", "build"}:
        parser.print_help()
        return
    s = Settings()
    run_dir = run_select(s, args.offline, fund=args.command == "build")
    meta = json.loads((run_dir / "metadata.json").read_text())
    print(f"run saved to {run_dir}")
    print(json.dumps(meta["counts"], indent=2))
    core = pd.read_csv(run_dir / "core.csv")
    print(core[["selection_order", "ticker", "company", "sector", "investment_score", "selection_score"]]
          .to_string(index=False))
    if "portfolio" in meta:
        print(json.dumps({k: v for k, v in meta["portfolio"].items() if k not in ("event_order", "conventions")},
                         indent=2))
        print(json.dumps({k: v for k, v in meta["risk"].items() if k != "convention"}, indent=2))


if __name__ == "__main__":
    main()
