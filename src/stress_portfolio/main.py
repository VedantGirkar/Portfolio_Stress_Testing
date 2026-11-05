"""Command-line entry point."""

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import backtest, data, execution, hedging, optimization, risk, selection, stress
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
             offline: bool, now: datetime) -> tuple[dict, pd.Series]:
    """Load the benchmark from the same price window and measure the risk of the funded book.
    Returns the risk tables and the benchmark's adj_close on the session calendar."""
    if offline and not (price_cache / f"{start}_{end}" / f"{s.benchmark_ticker}.csv").exists():
        raise SystemExit(f"offline mode: {s.benchmark_ticker} is not cached; run `build` once without --offline")
    bench, failed = data.download_prices([s.benchmark_ticker], start, end, price_cache)
    if not failed.empty:
        raise SystemExit(f"benchmark {s.benchmark_ticker} download failed: {failed['reason'].iloc[0]}; rerun")
    bench, _ = data.clean_bars(data.drop_partial_session(bench, now))
    adj = data.wide(bars, "adj_close")
    cal = selection.session_calendar(adj)
    bench_adj = data.wide(bench, "adj_close")[s.benchmark_ticker].reindex(cal)
    return risk.risk_report(book["holdings"], adj.reindex(cal), bench_adj, book["nav"], s), bench_adj


def run_stress_tests(s: Settings, book: dict, risk_tables: dict, shortlist: pd.DataFrame, end: str, offline: bool,
                     now: datetime) -> tuple[dict, pd.DataFrame]:
    """Load long history (shortlist, benchmark, sector ETFs) in its own cache and stress the funded book.
    Returns the stress tables and the wide adj_close history (reused by the optimizer)."""
    tickers = {*book["holdings"].query("ticker != 'CASH'")["ticker"], *shortlist["provider_ticker"]}
    wide, failed = load_long_history(s, tickers, end, offline, now)
    tables = stress.run_stress(book["holdings"], risk_tables["risk_positions"], wide, s)
    tables["stress_download_failures"] = failed  # these tickers fall back to proxies
    return tables, wide


def run_hedge(s: Settings, book: dict, risk_tables: dict, stress_tables: dict, history: pd.DataFrame,
              offline: bool, now: datetime) -> dict:
    """Buy the stress-budget SPY put hedge for the funded book from today's yfinance chain, through the same ledger.

    Runs after risk, stress and optimization, which describe the stock book only. The fill is the chain snapshot
    taken now, after the valuation close: buys at the ask, puts then valued at the bid-ask mid.
    """
    chain, quality = data.fetch_option_chain(s, s.data_dir / "raw" / "options", offline, now)
    stocks = book["holdings"].query("ticker != 'CASH'").set_index("ticker")[["sector", "market_value"]]
    betas = risk_tables["risk_positions"].set_index("ticker")["beta"].reindex(stocks.index)
    nav = float(book["nav"].iloc[-1]["nav"])
    valuation = pd.Timestamp(book["nav"].iloc[-1]["date"])
    if len(chain) and pd.Timestamp(chain["quote_date"].iloc[0]) < valuation:
        plan = {"status": "not hedged: cached option quotes are older than the valuation date"}
    else:
        plan = hedging.plan_hedge(stocks, betas, nav, history, chain, stress_tables["stress_scenarios"], valuation, s)
    out = {"hedge_chain": chain, "hedge_chain_quality": quality}
    summary = {"status": plan["status"], "method": s.hedge_method, "quote_source": "yfinance (delayed, unofficial)",
               "quotes_fetched_at": chain["fetched_at"].iloc[0] if len(chain) else None,
               "valuation_date": str(valuation.date())}
    if "pick" not in plan:
        out["hedge_summary"] = pd.DataFrame(summary.items(), columns=["metric", "value"])
        return out

    pick, size, p = plan["pick"], plan["sizing"], book["portfolio"]
    fill_date = pd.Timestamp(pick["quote_date"])
    fills = pd.DataFrame(columns=execution.OPTION_FILL_COLUMNS)
    if size["contracts"] > 0:
        quotes = chain.set_index("contract")[["bid", "ask", "multiplier"]]
        orders = pd.DataFrame({"contract": [pick["contract"]], "order_qty": [size["contracts"]]})
        fills = execution.execute_options(orders, quotes, p.cash, s)
        p.add_contract(pick["contract"], {"underlying": s.hedge_underlying, "option_type": "P",
                                          "strike": float(pick["strike"]), "expiration": str(pick["expiration"].date()),
                                          "multiplier": int(pick["multiplier"]), "exercise": "American",
                                          "settlement": "physical (shares)"})
        p.apply_option_fills(fills, fill_date)
    mid = (pick["bid"] + pick["ask"]) / 2
    held = p.options.get(pick["contract"], 0.0)
    holdings, row = p.snapshot(fill_date, book["marks"], pd.Series({pick["contract"]: mid}) if held else None)
    info = book["holdings"].set_index("ticker")
    holdings = holdings.assign(company=holdings["ticker"].map(info["company"]).fillna(f"{s.hedge_underlying} put"),
                               sector=holdings["ticker"].map(info["sector"]).fillna("Hedge (SPY puts)"),
                               target_weight_of_nav=holdings["ticker"].map(info["target_weight_of_nav"]))
    cash = pd.DataFrame([{"ticker": "CASH", "quantity": p.cash, "price": 1.0, "market_value": p.cash,
                          "weight_of_nav": p.cash / row["nav"], "company": "Cash", "sector": "Cash",
                          "target_weight_of_nav": s.cash_reserve}])
    book["holdings"] = pd.concat([holdings, cash], ignore_index=True)
    book["nav"] = pd.concat([book["nav"], pd.DataFrame([row | {"holdings": len(p.positions)}])], ignore_index=True)
    book["ledger"] = p.ledger()

    filled = int(fills["filled_qty"].sum()) if len(fills) else 0
    g = plan["greeks"]
    sized = size["contracts"]
    if filled != sized:  # cash cut the order: report the hedge actually held, not the one sized
        plan = hedging.resize_plan(plan, filled, nav, s)
        size = plan["sizing"]
    summary |= {
        "contract": pick["contract"], "strike": float(pick["strike"]), "expiration": str(pick["expiration"].date()),
        "next_review": str(plan["next_review"].date()), "days_to_expiry": round(plan["years"] * 365),
        "spot": plan["spot"], "bid": float(pick["bid"]), "ask": float(pick["ask"]), "mid": float(mid),
        "iv": plan["iv"], "iv_source": plan["iv_source"], "rate": plan["rate"],
        "contracts_sized": sized, "contracts_filled": filled,
        "premium_paid_usd": float((fills["filled_qty"] * fills["multiplier"] * fills["fill_price"]).sum()) if filled else 0.0,
        "fees_usd": float(fills["commission"].sum()) if filled else 0.0,
        "spread_cost_usd": float(fills["spread_cost"].sum()) if filled else 0.0,
        "budget_cap_contracts": size["budget_cap_contracts"], "annual_net_cost_usd": size["annual_net_cost_usd"],
        "worst_train_loss_unhedged": size["worst_unhedged"], "worst_train_loss_hedged": size["worst_hedged"],
        "stress_loss_budget": s.stress_loss_budget, "premium_budget_share_nav": s.hedge_premium_budget,
        "put_delta": g["delta"], "put_gamma": g["gamma"], "put_vega": g["vega"], "put_theta": g["theta"],
        "beta_dollar_delta_before": plan["delta_before"], "beta_dollar_delta_after": plan["delta_after"],
    }
    if filled < summary["contracts_sized"]:
        summary["status"] += f"; only {filled} of {summary['contracts_sized']} filled ({fills['reason'].iloc[0]})"
    out |= {"hedge_summary": pd.DataFrame(summary.items(), columns=["metric", "value"]),
            "hedge_positions": pd.DataFrame([{
                "contract": pick["contract"], "underlying": s.hedge_underlying, "option_type": "P",
                "strike": float(pick["strike"]), "expiration": str(pick["expiration"].date()),
                "multiplier": int(pick["multiplier"]), "exercise": "American", "settlement": "physical (shares)",
                "contracts": filled, "bid": float(pick["bid"]), "ask": float(pick["ask"]), "mid": float(mid),
                "market_value": filled * int(pick["multiplier"]) * float(mid), "iv": plan["iv"],
                "iv_source": plan["iv_source"], "delta": g["delta"], "gamma": g["gamma"], "vega": g["vega"],
                "theta": g["theta"], "dollar_delta": filled * int(pick["multiplier"]) * g["delta"] * plan["spot"]}]),
            "hedge_fills": fills, "hedge_stress": plan["scenarios"], "hedge_capped_events": plan["capped"]}
    return out


def load_long_history(s: Settings, tickers, end: str, offline: bool, now: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide adj_close since stress_history_start for `tickers`, the benchmark and the sector ETFs, in its own
    cache. Missing stocks are returned as failures (stress replays fall back to proxies for them)."""
    cache = s.data_dir / "raw" / "stress"
    folder = f"{s.stress_history_start}_{end}"
    tickers = sorted({*tickers, s.benchmark_ticker, *dict(s.sector_etfs).values(), s.volatility_index_ticker,
                      s.rate_ticker})
    if offline:
        tickers = [t for t in tickers if (cache / folder / f"{t}.csv").exists()]
    history, failed = data.download_prices(tickers, s.stress_history_start, end, cache, s.download_chunk_size)
    if history.empty or s.benchmark_ticker not in set(history["ticker"]):
        how = "run once without --offline" if offline else "rerun"
        raise SystemExit(f"long history for {s.benchmark_ticker} is not available; {how}")
    if not offline:
        data.prune_cache(cache, folder)
    history, _ = data.clean_bars(data.drop_partial_session(history, now))
    return data.wide(history, "adj_close"), failed


def get_membership(s: Settings, offline: bool, now: datetime) -> tuple[pd.DataFrame, dict]:
    path = s.data_dir / "raw" / "membership.csv"
    if offline:
        return data.load_membership(path)
    membership, meta = data.fetch_membership(s.membership_url, now)
    data.save_membership(membership, meta, path)
    return membership, meta


def price_window(cache: Path, start: str, offline: bool, now: datetime) -> tuple[str, str]:
    """(start, end) of the price request. Offline reuses the newest cached window; online ends after today's
    session only once it has settled (yfinance end is exclusive), so a partial bar is never cached."""
    if offline:
        caches = sorted(p.name for p in cache.glob("*_*") if p.is_dir())
        if not caches:
            raise SystemExit(f"offline mode: no cached prices in {cache}; run once without --offline")
        return tuple(caches[-1].split("_"))
    settled = now.astimezone(data.NY).time() >= data.MARKET_CLOSE_SETTLED
    return start, (now.date() + timedelta(days=1 if settled else 0)).isoformat()


def load_prices(s: Settings, tickers: list[str], cache: Path, start: str, end: str, offline: bool,
                now: datetime) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean bars, failures and a quality report. Stops when too many tickers have no prices."""
    if offline:
        cached = {p.stem for p in (cache / f"{start}_{end}").glob("*.csv")}
        bars, failures = data.download_prices([t for t in tickers if t in cached], start, end, cache,
                                              s.download_chunk_size)
        failures = pd.concat([failures, pd.DataFrame(
            {"ticker": sorted(set(tickers) - cached), "reason": "not in offline cache"})], ignore_index=True)
    else:
        bars, failures = data.download_prices(tickers, start, end, cache, s.download_chunk_size)
    share = len(failures) / len(tickers)
    if share > s.max_download_failure_share:
        retry = "run once without --offline" if offline else "rerun to fetch only the missing tickers"
        raise SystemExit(f"{len(failures)} of {len(tickers)} tickers have no prices ({share:.1%}), above the "
                         f"{s.max_download_failure_share:.0%} limit; {retry}. "
                         f"First failures: {failures['ticker'].head(5).tolist()}")
    if not offline:
        data.prune_cache(cache, f"{start}_{end}")
    bars, quality = data.clean_bars(data.drop_partial_session(bars, now))
    return bars, failures, quality


def run_optimization(s: Settings, result: dict, picked: pd.DataFrame, bars: pd.DataFrame, book: dict,
                     risk_tables: dict, history: pd.DataFrame, bench_adj: pd.Series) -> dict:
    """Candidate allocations for 10/20/30 stocks from the shortlist, judged with one shared set of estimates."""
    short = result["shortlist"]
    adj = data.wide(bars, "adj_close")
    adj = adj.reindex(selection.session_calendar(adj))
    rets = risk.daily_returns(adj[short["provider_ticker"]], s.risk_lookback_sessions)
    bench = (bench_adj / bench_adj.shift(1) - 1).reindex(rets.index)
    est = optimization.estimates(rets, bench, s)
    positions = risk_tables["risk_positions"]
    riskiest = positions.groupby("sector")["risk_contribution_pct"].sum().idxmax()
    scen, groups = stress.scenario_returns(short.set_index("provider_ticker")[["sector"]], history, est[2],
                                           riskiest, s)
    sel_returns = selection.window_returns(data.wide(picked, "adj_close"), s)  # same information as the selection
    prices = data.wide(bars, "close").ffill().iloc[-1]  # latest valuation marks
    tables = optimization.build_candidates(short, sel_returns, est, scen, groups, book["holdings"], prices, s)
    tables["opt_estimates"]["window_start"], tables["opt_estimates"]["window_end"] = rets.index[0].date(), \
        rets.index[-1].date()
    return tables


def run_select(s: Settings, offline: bool, now: datetime | None = None, fund: bool = False) -> Path:
    """Selection run; with fund=True, select as of the prior session and buy the core at the next open."""
    now = now or datetime.now(data.NY)
    price_cache = s.data_dir / "raw" / "prices"
    membership, member_meta = get_membership(s, offline, now)
    start, end = price_window(price_cache, (now.date() - timedelta(days=s.history_calendar_days)).isoformat(),
                              offline, now)
    bars, failures, quality = load_prices(s, membership["provider_ticker"].tolist(), price_cache, start, end,
                                          offline, now)

    decision_date = None
    if fund:
        # decide on the prior session so the fill uses the latest session's open, never a price the decision saw
        decision_date = selection.session_calendar(data.wide(bars, "adj_close"))[-2]
    picked = bars if decision_date is None else bars.loc[bars["date"] <= decision_date]
    result = selection.run_selection(membership, picked, failures, s)
    book = backtest.fund_and_hold(result["core"], bars, decision_date, s) if fund else None
    risk_tables, stress_tables, opt_tables, hedge_tables = {}, {}, {}, {}
    if fund:
        risk_tables, bench_adj = run_risk(s, book, bars, price_cache, start, end, offline, now)
        stress_tables, history = run_stress_tests(s, book, risk_tables, result["shortlist"], end, offline, now)
        opt_tables = run_optimization(s, result, picked, bars, book, risk_tables, history, bench_adj)
        hedge_tables = run_hedge(s, book, risk_tables, stress_tables, history, offline, now)

    run_dir = new_run_dir(s.runs_dir, now)
    for name in ["eligibility", "sector_quotas", "shortlist", "core", "reserves"]:
        result[name].to_csv(run_dir / f"{name}.csv", index=False)
    membership.to_csv(run_dir / "membership.csv", index=False)
    failures.to_csv(run_dir / "download_failures.csv", index=False)
    quality.to_csv(run_dir / "data_quality.csv", index=False)
    if book:
        for name in ["orders", "holdings", "ledger", "nav", "cost_scenarios"]:
            book[name].to_csv(run_dir / f"{name}.csv", index=False)
    for name, table in {**risk_tables, **stress_tables, **opt_tables, **hedge_tables}.items():
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
        fills = hedge_tables.get("hedge_fills")
        hedge_cost = float(fills["total_cost"].sum()) if fills is not None and len(fills) else 0.0
        meta["portfolio"] = {
            "decision_date": str(book["decision_date"].date()),
            "fill_date": str(book["fill_date"].date()),
            "valuation_date": str(book["valuation_date"].date()),  # the stocks' close; a put's quote can be later
            "initial_capital": s.initial_capital,
            "nav": round(float(last["nav"]), 2),
            "cash": round(float(last["cash"]), 2),
            "total_trading_cost": round(float(book["orders"]["total_cost"].sum()) + hedge_cost, 2),
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
        sc = stress_tables["stress_scenarios"].dropna(subset=["pnl_usd"])
        worst = {k: sc.loc[sc.loc[sc["kind"] == k, "pnl_usd"].idxmin()] for k in ("historical", "hypothetical")}
        meta["stress"] = {
            "method": "today's holdings shocked per scenario; cash unchanged; not an investable backtest",
            "scenarios_run": len(sc),
            "scenarios_without_data": int(stress_tables["stress_scenarios"]["pnl_usd"].isna().sum()),
            **{f"worst_{k}": {"scenario": str(v["label"]), "pnl_usd": round(float(v["pnl_usd"]), 2),
                              "pnl_pct_nav": round(float(v["pnl_pct_nav"]), 4)} for k, v in worst.items()},
            "proxied_positions_total": int(sc["proxied_positions"].sum()),
        }
        cand = opt_tables["opt_candidates"]
        est_window = opt_tables["opt_estimates"].loc[0, ["window_start", "window_end"]].astype(str).tolist()
        meta["optimization"] = {
            "method": "cvxpy (CLARABEL) on heuristic 10/20/30-stock universes; convex optimum per universe only",
            "estimation_window": est_window,
            "expected_return": f"{s.expected_return_blend:.0%} historical mean + rest beta x {s.equity_risk_premium:.0%}",
            "constraints": {"long_only": True, "cash_reserve": s.cash_reserve, "max_position": s.max_position_weight,
                            "max_sector": s.max_sector_weight, "min_position": s.min_position_weight,
                            "stress_loss_budget": s.stress_loss_budget},
            "candidates": int(len(cand)),
            "infeasible": cand.loc[cand["expected_return"].isna(), "candidate"].tolist(),
        }
        meta["hedge"] = {
            **dict(hedge_tables["hedge_summary"].values),
            "label": "Observed delayed yfinance quotes for today's book; not a backtest. Risk tables cover stocks only.",
            "conventions": [
                "Sized on the valuation close; filled at the chain snapshot taken at run time (buy at the ask).",
                "Puts valued at the bid-ask mid; premium is cash exchanged for an asset, not a cost.",
                "Stress: puts fully repriced with Black-Scholes (European; SPY puts are American, so model values run low).",
                "Historical events are cut at the put's expiry, stock loss included; hypothetical shocks are instantaneous.",
            ],
        }
    else:
        meta["notes"].append("Target weights only; run `stress-portfolio build` for holdings, cash and NAV.")
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return run_dir


def run_backtest_cmd(s: Settings, offline: bool, now: datetime | None = None) -> Path:
    """Walk-forward backtest from backtest_start to the latest session; writes a `backtest` run."""
    now = now or datetime.now(data.NY)
    cache = s.data_dir / "raw" / "backtest"
    membership, member_meta = get_membership(s, offline, now)
    start, end = price_window(cache, s.backtest_history_start, offline, now)
    members = membership["provider_ticker"].tolist()
    bars, failures, quality = load_prices(s, members + [s.benchmark_ticker], cache, start, end, offline, now)
    if s.benchmark_ticker in set(failures["ticker"]):
        raise SystemExit(f"benchmark {s.benchmark_ticker} has no prices; rerun")
    adj = data.wide(bars, "adj_close")
    t0 = selection.session_calendar(adj)
    t0 = t0[t0 < pd.Timestamp(s.backtest_start)][-1]
    shortlist = selection.run_selection(membership, bars.loc[bars["date"] <= t0], failures, s)["shortlist"]
    history, hist_failed = load_long_history(s, shortlist["provider_ticker"], end, offline, now)
    out = backtest.run_backtest(membership, bars, history, s)

    run_dir = new_run_dir(s.runs_dir, now)
    for name in ["bt_nav", "bt_trades", "bt_decisions", "bt_summary", "bt_initial_core", "bt_dropped"]:
        out[name].to_csv(run_dir / f"{name}.csv", index=False)
    failures.to_csv(run_dir / "download_failures.csv", index=False)
    quality.to_csv(run_dir / "data_quality.csv", index=False)
    (run_dir / "settings.json").write_text(json.dumps(s.to_dict(), indent=2))
    period = {"M": "month", "Q": "quarter"}.get(s.rebalance_frequency, f"'{s.rebalance_frequency}' period")
    meta = {
        "run_id": run_dir.name, "workflow": "backtest", "created_at": now.isoformat(timespec="seconds"),
        "git_revision": git_revision(), "data_mode": "offline cache" if offline else "live download",
        "label": backtest.BIAS_LABEL,
        "start_decision": str(out["start"].date()), "end": str(out["bt_nav"]["date"].max().date()),
        "reviews": out["reviews"], "strategies": list(backtest.STRATEGIES),
        "policy": {"schedule": f"last session of each completed {period}; orders fill at the next open",
                   "stress_aware": f"switch if after-cost utility gain >= {s.rebalance_min_improvement:.2%}/yr or the "
                                   f"book breaks a position, sector or stress limit by more than {s.rebalance_drift_band:.0%} "
                                   "(scenarios known by the review)",
                   "equal_weight": f"reset the starting core to equal weights every {period}",
                   "min_variance": f"minimum variance over a {s.backtest_universe_size}-stock universe every {period}",
                   "buy_and_hold": "starting core, never traded again"},
        "universe": "starting 100-stock shortlist fixed for the whole backtest, re-scored at each review",
        "membership": member_meta, "event_order": list(backtest.EVENT_ORDER),
        "long_history_failures": hist_failed["ticker"].tolist(),
        "hedged_strategy": backtest.HEDGED_STRATEGY_STATUS,
        "notes": ["All strategies here are equity-only; hedge reviews start with the hedged strategy (milestone 9).",
                  "Commissions are charged on real shares (split-adjusted shares divided by later splits).",
                  "Gross CAGR adds total costs back to the final NAV and ignores their compounding."],
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
    build = sub.add_parser("build", help="select as of the prior session, buy the core at the next open, value it")
    bt = sub.add_parser("backtest", help="walk-forward test of the strategies from backtest_start to today")
    for p in (sel, build, bt):
        p.add_argument("--offline", action="store_true", help="use saved membership and cached prices only")
    args = parser.parse_args(argv)

    if args.command not in {"select", "build", "backtest"}:
        parser.print_help()
        return
    s = Settings()
    if args.command == "backtest":
        run_dir = run_backtest_cmd(s, args.offline)
        print(f"run saved to {run_dir}\n{backtest.BIAS_LABEL}")
        print(pd.read_csv(run_dir / "bt_summary.csv").round(4).to_string(index=False))
        return
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
        print(json.dumps(meta["stress"], indent=2))
        cand = pd.read_csv(run_dir / "opt_candidates.csv")
        print(cand[["candidate", "status", "expected_return", "volatility", "worst_train_loss",
                    "worst_validation_loss", "est_cost_usd", "turnover"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
