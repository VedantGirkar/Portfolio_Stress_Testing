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
    "Risk": "milestone 3 — VaR, expected shortfall, drawdowns, correlations",
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
    tables["meta"] = json.loads((p / "metadata.json").read_text())
    return tables


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[c for c in TABLE_COLS if c in df.columns]].copy()
    out["median_dollar_volume"] = (out["median_dollar_volume"] / 1e6).round(1)
    return out.rename(columns={"median_dollar_volume": "median $ vol (M)"})


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

tabs = st.tabs(["Overview", "Selection", *COMING])

with tabs[0]:
    st.caption("Target weights of the equity sleeve. No orders, fills, cash or NAV yet — the ledger arrives "
               "in milestone 2.")
    core, reserves, short = r["core"], r["reserves"], r["shortlist"]
    c = st.columns(5)
    c[0].metric("Core holdings", len(core))
    c[1].metric("Reserves", len(reserves))
    c[2].metric("Core sectors", core["sector"].nunique())
    c[3].metric("Shortlist sectors", short["sector"].nunique())
    c[4].metric("Hedge", "TBD", help="Delta or gamma method is chosen at milestone 7")
    left, right = st.columns(2)
    left.plotly_chart(charts.core_weight_pie(core), width="stretch")
    right.plotly_chart(charts.core_sector_pie(core), width="stretch")
    st.subheader("Core portfolio")
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

for tab, (name, when) in zip(tabs[2:], COMING.items()):
    with tab:
        st.info(f"**{name}** is not built yet — arrives in {when}.")
