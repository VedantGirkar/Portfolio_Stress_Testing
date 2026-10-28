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
COMING = {
    "Stress": "milestone 4 — historical and hypothetical scenarios",
    "Optimization": "milestone 5 — efficient frontier and candidate allocations",
    "Rebalancing": "milestone 6 — proposed orders and costs",
    "Hedging": "milestones 7–8 — delta or gamma hedge (method not chosen yet)",
    "Backtest": "milestones 6 and 9 — walk-forward results and benchmarks",
}
TABLE_COLS = ["ticker", "company", "sector", "investment_score", "momentum", "downside_dev_annual",
              "max_drawdown", "median_dollar_volume", "reason"]

st.set_page_config(page_title="Stress Portfolio", layout="wide")


def selection_runs() -> list[Path]:
    runs = [p for p in RUNS_DIR.glob("*") if (p / "metadata.json").exists() and (p / "core.csv").exists()]
    return sorted(runs, reverse=True)


@st.cache_data
def load_run(path: str) -> dict:
    p = Path(path)
    tables = {n: pd.read_csv(p / f"{n}.csv") for n in ["core", "reserves", "shortlist", "sector_quotas", "eligibility"]}
    for n in ["holdings", "orders", "nav", "cost_scenarios", "risk_var", "risk_summary", "risk_positions",
              "risk_sectors", "risk_correlation", "risk_series", "var_backtest", "var_backtest_summary"]:
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

tabs = st.tabs(["Overview", "Selection", "Risk", *COMING])

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

for tab, (name, when) in zip(tabs[3:], COMING.items()):
    with tab:
        st.info(f"**{name}** is not built yet — arrives in {when}.")
