"""Reusable Plotly charts. They only draw already-calculated tables."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROLE_COLORS = {"core": "#2563eb", "reserve": "#cbd5e1"}


def _tidy(fig: go.Figure, height: int = 380) -> go.Figure:
    return fig.update_layout(height=height, margin=dict(t=40, l=10, r=10, b=10), legend_title_text="")


def core_weight_pie(core: pd.DataFrame) -> go.Figure:
    fig = px.pie(core, names="ticker", values="target_weight", hole=0.45,
                 hover_data=["company", "sector"], title="Core target weights by stock")
    return _tidy(fig.update_traces(textinfo="label+percent", sort=False))


def core_sector_pie(core: pd.DataFrame) -> go.Figure:
    by_sector = core.groupby("sector", as_index=False)["target_weight"].sum()
    fig = px.pie(by_sector, names="sector", values="target_weight", hole=0.45,
                 title="Core target weights by sector")
    return _tidy(fig.update_traces(textinfo="percent"))


def shortlist_treemap(shortlist: pd.DataFrame) -> go.Figure:
    df = shortlist.assign(count=1)
    fig = px.treemap(df, path=[px.Constant("Shortlist"), "sector", "ticker"], values="count", color="role",
                     color_discrete_map={**ROLE_COLORS, "(?)": "#64748b"}, hover_data=["company"],
                     title="100-stock shortlist by sector (blue = core, grey = reserve)")
    return _tidy(fig.update_traces(root_color="#0f172a"), height=520)


def reserves_by_sector(reserves: pd.DataFrame) -> go.Figure:
    counts = reserves["sector"].value_counts().rename_axis("sector").reset_index(name="reserves")
    fig = px.pie(counts, names="sector", values="reserves", hole=0.45, title="90 reserves by sector (count)")
    fig.update_traces(textinfo="value").update_layout(legend=dict(orientation="h", y=-0.05))
    return _tidy(fig, height=480)


def shortlist_by_sector_bar(shortlist: pd.DataFrame) -> go.Figure:
    counts = shortlist.groupby(["sector", "role"]).size().reset_index(name="stocks")
    order = shortlist["sector"].value_counts().index.tolist()
    fig = px.bar(counts, y="sector", x="stocks", color="role", orientation="h",
                 color_discrete_map=ROLE_COLORS, category_orders={"sector": order},
                 title="Shortlist composition by sector")
    return _tidy(fig, height=420)


def core_scores(core: pd.DataFrame) -> go.Figure:
    df = core.sort_values("selection_order")
    fig = go.Figure([
        go.Bar(name="Investment score", x=df["ticker"], y=df["investment_score"], marker_color="#2563eb"),
        go.Bar(name="Selection score (after correlation penalty)", x=df["ticker"], y=df["selection_score"],
               marker_color="#f59e0b"),
    ])
    fig.update_layout(barmode="group", title="Core picks in selection order", yaxis_title="score")
    return _tidy(fig)


def eligibility_reasons(eligibility: pd.DataFrame) -> go.Figure:
    counts = eligibility["eligibility_reason"].str.replace(r"other share class: .*", "other share class",
                                                          regex=True).value_counts()
    counts = counts.rename_axis("reason").reset_index(name="stocks")
    fig = px.bar(counts, y="reason", x="stocks", orientation="h", title="S&P 500 eligibility outcome")
    return _tidy(fig.update_yaxes(categoryorder="total ascending"), height=320)
