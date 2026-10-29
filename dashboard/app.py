"""Streamlit dashboard. Reads saved runs only; never downloads or recalculates results."""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit puts dashboard/ on the path, not the root

from dashboard import charts  # noqa: E402
from stress_portfolio.config import Settings  # noqa: E402

RUNS_DIR = Path(os.environ.get("STRESS_PORTFOLIO_RUNS_DIR", Settings().runs_dir))
HEDGING_PENDING = "milestones 7–8 — delta or gamma hedge (method not chosen yet)"
TABLE_COLS = ["ticker", "company", "sector", "investment_score", "momentum", "downside_dev_annual",
              "max_drawdown", "median_dollar_volume", "reason"]

st.set_page_config(page_title="Stress Portfolio", layout="wide")


def backtest_runs() -> list[Path]:
    return sorted((p for p in RUNS_DIR.glob("*") if (p / "bt_summary.csv").exists()), reverse=True)


@st.cache_data(max_entries=8)
def load_backtest(path: str) -> dict:
    p = Path(path)
    dates = {"bt_nav": ["date"], "bt_decisions": ["date"], "bt_trades": ["fill_date"]}
    tables = {n: pd.read_csv(p / f"{n}.csv", parse_dates=dates.get(n))
              for n in ["bt_nav", "bt_trades", "bt_decisions", "bt_summary", "bt_initial_core", "bt_dropped"]}
    tables["meta"] = json.loads((p / "metadata.json").read_text())
    tables["settings"] = json.loads((p / "settings.json").read_text())
    return tables


def selection_runs() -> list[Path]:
    runs = [p for p in RUNS_DIR.glob("*") if (p / "metadata.json").exists() and (p / "core.csv").exists()]
    return sorted(runs, reverse=True)


@st.cache_data
def load_run(path: str) -> dict:
    p = Path(path)
    tables = {n: pd.read_csv(p / f"{n}.csv") for n in ["core", "reserves", "shortlist", "sector_quotas", "eligibility"]}
    for n in ["holdings", "orders", "nav", "cost_scenarios", "risk_var", "risk_summary", "risk_positions",
              "risk_sectors", "risk_correlation", "risk_series", "var_backtest", "var_backtest_summary",
              "stress_scenarios", "stress_positions", "stress_sectors", "opt_candidates", "opt_weights",
              "opt_frontier", "opt_estimates"]:
        # only in `build` runs; risk tables only in builds made after milestone 3
        if (p / f"{n}.csv").exists():
            tables[n] = pd.read_csv(p / f"{n}.csv")
    tables["meta"] = json.loads((p / "metadata.json").read_text())
    return tables


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[c for c in TABLE_COLS if c in df.columns]].copy()
    out["median_dollar_volume"] = (out["median_dollar_volume"] / 1e6).round(1)
    return out.rename(columns={"median_dollar_volume": "median $ vol (M)"})


def money(x: float) -> str:
    """Short enough for a metric card: $1.003M, $34.5K, $504."""
    return f"${x / 1e6:.3f}M" if abs(x) >= 1e6 else f"${x / 1e3:.1f}K" if abs(x) >= 1e4 else f"${x:,.0f}"


st.title("Stress-tested equity portfolio")

runs = selection_runs()
if not runs:
    st.warning(f"No selection runs found in `{RUNS_DIR}`. Create one with `uv run stress-portfolio select`.")
    st.stop()

run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
r = load_run(str(run))
meta = r["meta"]
st.sidebar.markdown(
    f"**Price cutoff:** {meta['price_cutoff']}  \n"
    f"**Membership:** {meta['membership']['retrieved_at'][:10]}  \n"
    f"**Data mode:** {meta['data_mode']}  \n"
    f"**Code:** {meta.get('git_revision') or 'unknown'}"
)
st.sidebar.caption(meta["membership"]["limitation"])
bt_dirs = backtest_runs()
bt_run = st.sidebar.selectbox("Backtest run", bt_dirs, format_func=lambda p: p.name) if bt_dirs else None
bt = load_backtest(str(bt_run)) if bt_run else None

tabs = st.tabs(["Overview", "Selection", "Risk", "Stress", "Optimization", "Rebalancing", "Hedging", "Backtest"])

with tabs[0]:
    core, reserves, short = r["core"], r["reserves"], r["shortlist"]
    if "holdings" in r:
        book, h = meta["portfolio"], r["holdings"]
        st.caption(f"Simulated book: decided on {book['decision_date']} close, bought at the {book['fill_date']} "
                   f"open, valued at the {book['valuation_date']} close.")
        c = st.columns(5)
        c[0].metric("NAV", money(book['nav']), f"{book['nav'] / book['initial_capital'] - 1:+.2%}")
        c[1].metric("Cash", money(book['cash']), f"{book['cash'] / book['nav']:.1%} of NAV", delta_color="off")
        c[2].metric("Holdings", int((h["ticker"] != "CASH").sum()))
        c[3].metric("Trading costs", money(book['total_trading_cost']))
        c[4].metric("Hedge", "TBD", help="Delta or gamma method is chosen at milestone 7")
        left, right = st.columns(2)
        left.plotly_chart(charts.holdings_pie(h, "ticker", "Actual weights by holding (% of NAV)"),
                          width="stretch")
        right.plotly_chart(charts.holdings_pie(h, "sector", "Actual weights by sector (% of NAV)"),
                           width="stretch")
        st.subheader("Holdings")
        st.dataframe(
            h[["ticker", "company", "sector", "quantity", "price", "market_value", "weight_of_nav",
               "target_weight_of_nav"]],
            hide_index=True, width="stretch",
            column_config={
                "price": st.column_config.NumberColumn(format="dollar"),
                "market_value": st.column_config.NumberColumn("market value", format="dollar"),
                "weight_of_nav": st.column_config.NumberColumn("actual % of NAV", format="percent"),
                "target_weight_of_nav": st.column_config.NumberColumn("target % of NAV", format="percent"),
            },
        )
        if len(r["nav"]) > 1:
            st.plotly_chart(charts.nav_line(r["nav"]), width="stretch")
        with st.expander(f"Trades and costs ({len(r['orders'])} orders)"):
            st.dataframe(r["orders"][["ticker", "order_qty", "filled_qty", "decision_price", "ref_price",
                                      "fill_price", "commission", "spread_cost", "slippage_cost", "total_cost",
                                      "status", "reason"]].round(4).astype({"reason": "string"}).fillna({"reason": ""}),
                         hide_index=True,
                         width="stretch")
            st.caption("Cost sensitivity for the same trades (base is what was charged):")
            st.dataframe(r["cost_scenarios"].round(2), hide_index=True, width="stretch")
            st.caption(" · ".join(book["conventions"]))
    else:
        st.caption("Selection-only run: target weights of the equity sleeve, no holdings or cash. "
                   "Run `stress-portfolio build` for a funded portfolio.")
        c = st.columns(4)
        c[0].metric("Core holdings", len(core))
        c[1].metric("Reserves", len(reserves))
        c[2].metric("Core sectors", core["sector"].nunique())
        c[3].metric("Hedge", "TBD", help="Delta or gamma method is chosen at milestone 7")
        left, right = st.columns(2)
        left.plotly_chart(charts.core_weight_pie(core), width="stretch")
        right.plotly_chart(charts.core_sector_pie(core), width="stretch")
    st.subheader("Core selection")
    st.dataframe(
        fmt(core).assign(target_weight=core["target_weight"], order=core["selection_order"].astype(int))
        .set_index("order"),
        width="stretch",
    )
    st.plotly_chart(charts.shortlist_treemap(short), width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(charts.reserves_by_sector(reserves), width="stretch")
    right.plotly_chart(charts.shortlist_by_sector_bar(short), width="stretch")

with tabs[1]:
    st.plotly_chart(charts.core_scores(r["core"]), width="stretch")

    st.subheader("Reserves by sector")
    for sector, group in r["reserves"].groupby("sector"):
        n_core = int((r["core"]["sector"] == sector).sum())
        with st.expander(f"{sector} — {len(group)} reserves, {n_core} in core"):
            st.dataframe(fmt(group.sort_values("investment_score", ascending=False)), hide_index=True,
                         width="stretch")

    st.subheader("Sector quotas")
    st.dataframe(r["sector_quotas"], hide_index=True, width="stretch")

    st.subheader("Eligibility")
    st.plotly_chart(charts.eligibility_reasons(r["eligibility"]), width="stretch")
    show = st.radio("Show", ["Ineligible", "All"], horizontal=True)
    elig = r["eligibility"]
    if show == "Ineligible":
        elig = elig.loc[~elig["eligible"]]
    st.dataframe(elig[["ticker", "company", "sector", "eligibility_reason", "observations",
                       "median_dollar_volume"]], hide_index=True, width="stretch")

with tabs[2]:
    if "risk_var" not in r:
        st.info("No risk results in this run. Create a new run with `uv run stress-portfolio build`.")
    else:
        rm, var, summ = meta["risk"], r["risk_var"], r["risk_summary"].set_index("metric")["value"]
        hist = var.loc[var["method"] == "historical"].set_index(["horizon_days", "confidence"])
        nav_now = meta["portfolio"]["nav"]
        st.caption(f"Historical simulation: today's holdings repriced with each day's returns from {rm['window'][0]} "
                   f"to {rm['window'][1]}. These are hypothetical P&Ls of today's book, not past results. "
                   "Losses are positive. VaR is a loss level exceeded on about 1% (or 5%) of past days, "
                   "not the worst possible loss.")
        c = st.columns(5)
        c[0].metric("1-day VaR 99%", money(hist.at[(1, 0.99), "var_usd"]),
                    f"{hist.at[(1, 0.99), 'var_pct_nav']:.2%} of NAV", delta_color="off")
        c[1].metric("1-day ES 99%", money(hist.at[(1, 0.99), "es_usd"]),
                    f"{hist.at[(1, 0.99), 'es_pct_nav']:.2%} of NAV", delta_color="off")
        c[2].metric("10-day VaR 99%", money(hist.at[(10, 0.99), "var_usd"]),
                    f"{hist.at[(10, 0.99), 'var_pct_nav']:.2%} of NAV", delta_color="off")
        c[3].metric("Volatility", f"{summ['volatility']:.1%}", "annual", delta_color="off")
        c[4].metric(f"Beta vs {rm['benchmark']}", f"{summ['beta']:.2f}",
                    f"correlation {summ['correlation']:.2f}", delta_color="off")

        st.plotly_chart(charts.pnl_histogram(r["risk_series"], hist), width="stretch")
        st.subheader("VaR and expected shortfall")
        st.dataframe(var[["method", "horizon_days", "confidence", "observations", "var_usd", "var_pct_nav",
                          "es_usd", "es_pct_nav", "convention"]],
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="dollar") for c in ["var_usd", "es_usd"]}
                     | {c: st.column_config.NumberColumn(format="percent") for c in ["var_pct_nav", "es_pct_nav",
                                                                                      "confidence"]})
        st.caption("10-day figures use overlapping compounded 10-day returns, so neighbouring observations "
                   "share 9 days. The normal rows assume normally distributed P&L; the gap to historical shows "
                   "how fat the real tails are.")

        left, right = st.columns(2)
        left.plotly_chart(charts.growth_vs_benchmark(r["risk_series"], rm["benchmark"]), width="stretch")
        right.plotly_chart(charts.drawdown_area(r["risk_series"]), width="stretch")
        st.caption("Not a performance record: these stocks were picked partly for strong past momentum, so "
                   "replaying them over the past flatters their history (selection bias). Use these paths for "
                   "the size and shape of swings, not for expected returns.")

        st.subheader("Where the risk comes from")
        left, right = st.columns(2)
        left.plotly_chart(charts.risk_contributions(r["risk_positions"]), width="stretch")
        right.plotly_chart(charts.correlation_heatmap(r["risk_correlation"]), width="stretch")
        st.dataframe(r["risk_positions"], hide_index=True, width="stretch",
                     column_config={"weight_of_nav": st.column_config.NumberColumn("% of NAV", format="percent"),
                                    "volatility_annual": st.column_config.NumberColumn("volatility", format="percent"),
                                    "risk_contribution_pct": st.column_config.NumberColumn("share of variance",
                                                                                           format="percent")})

        st.subheader("Does the VaR model hold up?")
        st.plotly_chart(charts.var_backtest(r["var_backtest"]), width="stretch")
        st.dataframe(r["var_backtest_summary"], hide_index=True, width="stretch")
        st.caption("Each day's 1-day VaR is estimated only from earlier days, then compared with that day's "
                   "hypothetical P&L. A Kupiec p-value below 0.05 means the breach count is unlikely if the "
                   "model were right.")

        st.subheader("Concentration and other measures")
        st.dataframe(r["risk_summary"][["metric", "value", "unit", "note"]], hide_index=True, width="stretch")
        st.caption("realized_* rows come from the real NAV since funding; everything else is estimated from "
                   f"the hypothetical history. Realized NAV covers {len(r['nav'])} session(s) so far.")

with tabs[3]:
    if "stress_scenarios" not in r:
        st.info("No stress results in this run. Create a new run with `uv run stress-portfolio build`.")
    else:
        sc, pos = r["stress_scenarios"], r["stress_positions"]
        done = sc.dropna(subset=["pnl_usd"])
        st.caption("Today's holdings hit with each scenario's moves; cash is unchanged. A historical replay asks "
                   "'what would today's book have lost in that event?'. It is not a record of trades anyone made. "
                   "Historical windows are SPY's deepest peak-to-trough fall inside each event.")
        worst_h = done.loc[done.loc[done["kind"] == "historical", "pnl_usd"].idxmin()]
        worst_y = done.loc[done.loc[done["kind"] == "hypothetical", "pnl_usd"].idxmin()]
        c = st.columns(4)
        c[0].metric("Worst historical", f"{worst_h['pnl_pct_nav']:.1%}", worst_h["label"], delta_color="off")
        c[1].metric("Worst hypothetical", f"{worst_y['pnl_pct_nav']:.1%}", worst_y["label"], delta_color="off")
        c[2].metric("Scenarios run", len(done), f"{int(sc['pnl_usd'].isna().sum())} without data",
                    delta_color="off")
        c[3].metric("Proxied positions", int(done["proxied_positions"].sum()), "across all events",
                    delta_color="off")
        st.plotly_chart(charts.stress_bars(done), width="stretch")
        st.dataframe(
            sc[["label", "kind", "group", "window_start", "window_end", "sessions", "benchmark_return", "pnl_usd",
                "pnl_pct_nav", "nav_after", "proxied_positions", "overlaps_training_event", "note"]],
            hide_index=True, width="stretch",
            column_config={"benchmark_return": st.column_config.NumberColumn("SPY", format="percent"),
                           "pnl_usd": st.column_config.NumberColumn("P&L", format="dollar"),
                           "pnl_pct_nav": st.column_config.NumberColumn("% of NAV", format="percent"),
                           "nav_after": st.column_config.NumberColumn("NAV after", format="dollar")})
        st.caption("Training scenarios may be used by the optimizer later; validation scenarios are kept for "
                   "checking it. A validation event that overlaps a training window (9/11 inside the dot-com "
                   "bust, Lehman inside the GFC) is not truly held out.")

        st.subheader("What drives the loss")
        names = dict(zip(done["scenario"], done["label"]))
        pick = st.selectbox("Scenario", list(names), format_func=names.get,
                            index=list(names).index(worst_h["scenario"]))
        left, right = st.columns([3, 2])
        left.plotly_chart(charts.stress_attribution(pos.loc[pos["scenario"] == pick], names[pick]), width="stretch")
        right.plotly_chart(charts.stress_sector_attribution(r["stress_sectors"].loc[r["stress_sectors"]["scenario"]
                                                                                   == pick]), width="stretch")
        st.dataframe(pos.loc[pos["scenario"] == pick].drop(columns="scenario"), hide_index=True, width="stretch",
                     column_config={"market_value": st.column_config.NumberColumn(format="dollar"),
                                    "shock_return": st.column_config.NumberColumn("move", format="percent"),
                                    "pnl_usd": st.column_config.NumberColumn("P&L", format="dollar")})
        st.caption("Source shows where each move came from: the stock's own prices, its sector ETF (stock not yet "
                   "listed), or beta x SPY. Proxies can overstate or understate a stock's real move.")

with tabs[4]:
    if "opt_candidates" not in r:
        st.info("No optimization results in this run. Create a new run with `uv run stress-portfolio build`.")
    else:
        om, cand, wts = meta["optimization"], r["opt_candidates"], r["opt_weights"]
        con = om["constraints"]
        st.caption(
            f"Candidate portfolios built from the shortlist and judged with one shared set of estimates "
            f"({om['estimation_window'][0]} to {om['estimation_window'][1]}). Expected return = {om['expected_return']}; "
            "these are estimates, not forecasts. Stocks for each universe size come from the selection heuristic; "
            "cvxpy then sets the weights, so each result is the best for its own universe, not a global optimum. "
            "Proposals only: nothing here trades (rebalancing decisions arrive in milestone 6). The window is "
            "shorter than the Risk tab's because every shortlisted stock must have prices on every day, so the "
            "current book's volatility here differs slightly from the Risk tab.")
        c = st.columns(4)
        c[0].metric("Candidates", len(cand) - 1, f"{len(om['infeasible'])} infeasible", delta_color="off")
        c[1].metric("Stress budget", f"{con['stress_loss_budget']:.0%}", "max training-scenario loss",
                    delta_color="off")
        c[2].metric("Position / sector cap", f"{con['max_position']:.0%} / {con['max_sector']:.0%}", "of NAV",
                    delta_color="off")
        c[3].metric("Cash reserve", f"{con['cash_reserve']:.0%}", f"min position {con['min_position']:.0%}",
                    delta_color="off")
        sizes = sorted(int(k) for k in r["opt_frontier"]["holding_count"].unique())
        k = st.radio("Universe size (stocks the optimizer may use)", sizes, horizontal=True)
        current = cand.loc[cand["method"] == "current"]
        left, right = st.columns(2)
        left.plotly_chart(charts.efficient_frontier(r["opt_frontier"].query("holding_count == @k"),
                                                    cand.query("holding_count == @k"), current), width="stretch")
        right.plotly_chart(charts.stress_tradeoff(r["opt_frontier"].query("holding_count == @k"),
                                                  cand.query("holding_count == @k"), current,
                                                  con["stress_loss_budget"]), width="stretch")
        st.caption("Frontier points: lowest estimated volatility for each target return, before trading costs and "
                   "without the stress budget (approximate: 15 points). The right chart is a separate trade-off: "
                   "the same points' worst training-scenario loss against estimated return.")

        st.subheader("All candidates")
        st.dataframe(
            cand[["candidate", "status", "holdings", "expected_return", "volatility", "sharpe", "worst_train_loss",
                  "worst_train_scenario", "worst_validation_loss", "worst_validation_scenario", "trades",
                  "est_cost_usd", "turnover"]],
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="percent") for c in
                           ["expected_return", "volatility", "worst_train_loss", "worst_validation_loss", "turnover"]}
            | {"est_cost_usd": st.column_config.NumberColumn("est. cost", format="dollar"),
               "sharpe": st.column_config.NumberColumn("return / vol", format="%.2f")})
        st.caption("Training losses are what the stress-aware optimizer was allowed to see; validation losses are an "
                   "out-of-sample check it never used. Validation events inside a training window (Lehman inside "
                   "the GFC, 9/11 inside the dot-com bust) are left out of that check. Estimated cost uses the full cost model on whole-share orders "
                   "from today's holdings. 'Infeasible' means no weights satisfy every limit.")

        st.subheader("Compare weights")
        solved = [c for c in cand["candidate"] if c in set(wts["candidate"]) and not c.startswith("current")]
        pick = st.selectbox("Candidate", solved, index=solved.index("stress_aware_20") if "stress_aware_20" in solved
                            else 0)
        now_w = wts.loc[wts["candidate"].str.startswith("current")]
        st.plotly_chart(charts.weights_compare(wts.loc[wts["candidate"] == pick], now_w, pick), width="stretch")

NO_BACKTEST = "No backtest runs yet. Create one with `uv run stress-portfolio backtest`."

with tabs[5]:
    if bt is None:
        st.info(NO_BACKTEST)
    else:
        bm, dec = bt["meta"], bt["bt_decisions"]
        st.warning(f"Results are {bm['label']}.", title="Preliminary backtest", icon=":material/warning:")
        st.caption(f"Reviews from {bm['start_decision']} to {bm['end']} ({bm['reviews']} reviews). "
                   f"{bm['policy']['schedule'].capitalize()}. Stress-aware rule: {bm['policy']['stress_aware']}.")
        strategy = st.segmented_control("Strategy", ["stress_aware", "min_variance", "equal_weight"],
                                        default="stress_aware", required=True)
        log = dec.loc[dec["strategy"] == strategy]
        c = st.columns(3)
        c[0].metric("Reviews", len(log))
        c[1].metric("Rebalanced", int((log["action"] == "rebalance").sum()))
        c[2].metric("Held", int((log["action"] == "hold").sum()))
        cols = [c for c in ["date", "action", "reason", "improvement", "cost_usd", "breaches", "holdings_before",
                            "holdings_after", "hedge_review"] if c in log]
        st.dataframe(log[cols], hide_index=True, alt=f"Review-by-review decisions of the {strategy} strategy",
                     column_config={"date": st.column_config.DateColumn(),
                                    "improvement": st.column_config.NumberColumn("after-cost gain", format="percent"),
                                    "cost_usd": st.column_config.NumberColumn("est. cost", format="dollar")})
        st.caption("Hedge reviews are logged separately from stock decisions; they start once the hedge method is "
                   "chosen (milestone 7).")

        if log["improvement"].notna().any():
            st.plotly_chart(charts.rebalance_gains(log, bt["settings"]["rebalance_min_improvement"]), width="stretch",
                            alt="After-cost gain at each review against the switching hurdle")
            st.caption("The target is adopted when its gain clears the hurdle, or whenever the current book breaks a "
                       "limit (those rebalances can sit below the line).")
        trades = bt["bt_trades"].query("strategy == @strategy")
        later = trades.loc[trades["fill_date"] > trades["fill_date"].min()]  # the initial purchase dwarfs the rest
        left, right = st.columns(2)
        if later.empty:
            left.info("This strategy never traded after its initial purchase.")
        else:
            left.plotly_chart(charts.rebalance_turnover(later, bt["bt_nav"].query("strategy == @strategy")),
                              width="stretch", alt=f"Traded value per rebalance of the {strategy} strategy")
        right.plotly_chart(charts.rebalance_cumulative_costs(bt["bt_trades"].query("strategy != 'buy_and_hold'")),
                           width="stretch", alt="Cumulative trading costs of each rebalancing strategy")
        st.caption("Traded value counts buys and sells separately, so replacing the whole book shows as about 200%. "
                   "Cumulative costs include each strategy's initial purchase.")
        if not bt["bt_dropped"].empty:
            with st.expander(f"Shortlisted stocks skipped at some reviews ({len(bt['bt_dropped'])})"):
                st.dataframe(bt["bt_dropped"], hide_index=True, alt="Stocks without a full price window at a review")

with tabs[6]:
    st.info(f"Hedging is not built yet — arrives in {HEDGING_PENDING}.", icon=":material/construction:")

with tabs[7]:
    if bt is None:
        st.info(NO_BACKTEST)
    else:
        bm, summ, nav = bt["meta"], bt["bt_summary"], bt["bt_nav"]
        st.warning(f"Results are {bm['label']}. Today's winners are replayed into the past, so every strategy "
                   "here is flattered; compare strategies with each other, not with real-world returns.",
                   title="Preliminary backtest", icon=":material/warning:")
        best = summ.loc[summ["strategy"].isin(bm["strategies"])].set_index("strategy")
        c = st.columns(len(best))
        for col, (name, row) in zip(c, best.iterrows()):
            col.metric(name.replace("_", " ").capitalize(), f"{row['cagr_net']:.1%}",
                       f"max drawdown {row['max_drawdown']:.0%}", delta_color="off", help="Net CAGR after costs")
        st.plotly_chart(charts.bt_growth(nav), width="stretch", alt="NAV of each strategy and SPY over the backtest")
        left, right = st.columns(2)
        left.plotly_chart(charts.bt_drawdowns(nav), width="stretch", alt="Drawdown of each strategy over time")
        right.plotly_chart(charts.bt_holdings(nav), width="stretch", alt="Number of stocks held by each strategy")
        st.dataframe(summ, hide_index=True, alt="Backtest summary per strategy",
                     column_config={c: st.column_config.NumberColumn(format="percent")
                                    for c in ["cagr_net", "cagr_gross", "volatility", "max_drawdown"]}
                     | {c: st.column_config.NumberColumn(format="dollar") for c in ["final_nav", "total_costs"]}
                     | {"return_over_vol": st.column_config.NumberColumn("return / vol", format="%.2f")})
        st.caption("Net = after trading costs. Gross adds the total costs back to the final NAV (ignoring their "
                   "compounding). The SPY row is a reference with no costs, not a strategy. All strategies start "
                   f"from the same {bm['start_decision']} core, capital and cost model.")
        with st.expander(f"All trades ({len(bt['bt_trades'])})"):
            st.dataframe(bt["bt_trades"][["strategy", "fill_date", "ticker", "order_qty", "filled_qty", "fill_price",
                                          "commission", "total_cost", "status"]],
                         hide_index=True, alt="Every simulated fill in the backtest")
