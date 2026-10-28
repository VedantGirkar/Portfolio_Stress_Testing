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


def holdings_pie(holdings: pd.DataFrame, by: str, title: str) -> go.Figure:
    by_group = holdings.groupby(by, as_index=False)["market_value"].sum()
    fig = px.pie(by_group, names=by, values="market_value", hole=0.45, title=title)
    return _tidy(fig.update_traces(textinfo="label+percent" if by == "ticker" else "percent", sort=False))


def nav_line(nav: pd.DataFrame) -> go.Figure:
    fig = px.line(nav, x="date", y=["nav", "equity_value", "cash"], title="NAV, equity and cash")
    return _tidy(fig.update_layout(yaxis_title="USD", xaxis_title=""))


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


def pnl_histogram(series: pd.DataFrame, hist: pd.DataFrame) -> go.Figure:
    fig = px.histogram(series, x="pnl_usd", nbins=80, title="Hypothetical 1-day P&L of today's book (USD)")
    for conf, color, side in [(0.95, "#f59e0b", "top right"), (0.99, "#ef4444", "top left")]:
        var = hist.at[(1, conf), "var_usd"]
        fig.add_vline(x=-var, line_color=color, line_dash="dash",
                      annotation_text=f"VaR {conf:.0%}: -${var:,.0f}", annotation_position=side)
    return _tidy(fig.update_layout(xaxis_title="daily P&L (USD)", yaxis_title="days", showlegend=False))


def growth_vs_benchmark(series: pd.DataFrame, benchmark: str) -> go.Figure:
    df = series.rename(columns={"portfolio_path": "today's book", "benchmark_path": benchmark})
    fig = px.line(df, x="date", y=["today's book", benchmark], title=f"Growth of $1: today's book vs {benchmark}")
    return _tidy(fig.update_layout(yaxis_title="value of $1", xaxis_title=""))


def drawdown_area(series: pd.DataFrame) -> go.Figure:
    fig = px.area(series, x="date", y="portfolio_drawdown", title="Drawdown of today's book (hypothetical)")
    fig.update_traces(line_color="#ef4444")
    return _tidy(fig.update_layout(yaxis_tickformat=".0%", yaxis_title="below peak", xaxis_title=""))


def risk_contributions(positions: pd.DataFrame) -> go.Figure:
    df = positions.sort_values("risk_contribution_pct", ascending=False)
    fig = go.Figure([
        go.Bar(name="Weight (% of NAV)", x=df["ticker"], y=df["weight_of_nav"], marker_color="#94a3b8"),
        go.Bar(name="Share of portfolio variance", x=df["ticker"], y=df["risk_contribution_pct"],
               marker_color="#ef4444"),
    ])
    fig.update_layout(barmode="group", title="Weight vs risk contribution", yaxis_tickformat=".0%")
    return _tidy(fig)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    m = corr.set_index("ticker")
    fig = px.imshow(m, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f",
                    title="Correlation of daily returns")
    return _tidy(fig, height=420)


def var_backtest(backtest: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=backtest["date"], y=backtest["pnl"], mode="lines", name="daily P&L",
                               line=dict(color="#94a3b8", width=1)))
    for col, color in [("var_95%", "#f59e0b"), ("var_99%", "#ef4444")]:
        if col in backtest:
            fig.add_scatter(x=backtest["date"], y=-backtest[col], name=f"-{col.replace('_', ' ').upper()}",
                            line=dict(color=color))
            hits = backtest.loc[backtest[col.replace("var", "exceed")]]
            fig.add_scatter(x=hits["date"], y=hits["pnl"], mode="markers", marker=dict(color=color, size=7),
                            name=f"breach {col[4:]}")
    fig.update_layout(title="VaR backtest: each day's VaR from earlier days only", yaxis_title="USD")
    return _tidy(fig, height=420)


def stress_bars(scenarios: pd.DataFrame) -> go.Figure:
    df = scenarios.sort_values("pnl_pct_nav")
    df = df.assign(type=df["kind"] + " / " + df["group"])
    fig = px.bar(df, y="label", x="pnl_pct_nav", color="type", orientation="h",
                 color_discrete_map={"historical / train": "#ef4444", "historical / validation": "#f59e0b",
                                     "hypothetical / train": "#8b5cf6"},
                 title="Loss of today's book in each scenario (% of NAV)")
    hist = df.loc[df["kind"] == "historical"]
    fig.add_scatter(x=hist["benchmark_return"], y=hist["label"], mode="markers", name="SPY in same window",
                    marker=dict(symbol="diamond", color="#e2e8f0", size=9, line=dict(color="#0f172a", width=1)))
    fig.update_layout(xaxis_tickformat=".0%", yaxis_title="", xaxis_title="")
    return _tidy(fig, height=max(420, 26 * len(df)))


def stress_attribution(positions: pd.DataFrame, label: str) -> go.Figure:
    df = positions.sort_values("pnl_usd")
    fig = px.bar(df, x="pnl_usd", y="ticker", color="source", orientation="h", title=f"P&L by holding: {label}")
    return _tidy(fig.update_layout(xaxis_title="USD", yaxis_title=""))


def stress_sector_attribution(sectors: pd.DataFrame) -> go.Figure:
    df = sectors.sort_values("pnl_usd")
    fig = px.bar(df, x="pnl_usd", y="sector", orientation="h", title="P&L by sector")
    return _tidy(fig.update_traces(marker_color="#ef4444").update_layout(xaxis_title="USD", yaxis_title=""))


METHOD_COLORS = {"current": "#e2e8f0", "equal_weight": "#94a3b8", "min_variance": "#22c55e",
                 "mean_variance": "#3b82f6", "stress_aware": "#ef4444"}


def _candidate_markers(fig: go.Figure, cand: pd.DataFrame, current: pd.DataFrame, x: str) -> None:
    for _, row in pd.concat([current, cand.loc[cand["method"] != "current"]]).dropna(subset=[x]).iterrows():
        fig.add_scatter(x=[row[x]], y=[row["expected_return"]], mode="markers", name=row["candidate"],
                        marker=dict(size=12, color=METHOD_COLORS[row["method"]],
                                    symbol="star" if row["method"] == "current" else "circle",
                                    line=dict(color="#0f172a", width=1)))


def efficient_frontier(frontier: pd.DataFrame, cand: pd.DataFrame, current: pd.DataFrame) -> go.Figure:
    fig = px.line(frontier.dropna(subset=["volatility"]), x="volatility", y="expected_return", markers=True,
                  title="Efficient frontier (estimated, annual, before costs)")
    fig.update_traces(name="frontier", showlegend=True, line_color="#64748b")
    _candidate_markers(fig, cand, current, "volatility")
    return _tidy(fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="volatility",
                                   yaxis_title="expected return"), height=440)


def stress_tradeoff(frontier: pd.DataFrame, cand: pd.DataFrame, current: pd.DataFrame, budget: float) -> go.Figure:
    fig = px.line(frontier.dropna(subset=["worst_train_loss"]), x="worst_train_loss", y="expected_return",
                  markers=True, title="Stress loss vs return (worst training scenario)")
    fig.update_traces(name="frontier points", showlegend=True, line_color="#64748b")
    _candidate_markers(fig, cand, current, "worst_train_loss")
    fig.add_vline(x=budget, line_dash="dash", line_color="#ef4444", annotation_text=f"budget {budget:.0%}")
    return _tidy(fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="worst loss",
                                   yaxis_title="expected return"), height=440)


def weights_compare(target: pd.DataFrame, current: pd.DataFrame, name: str) -> go.Figure:
    df = pd.concat([target.assign(portfolio=name), current.assign(portfolio="current")])
    order = df.groupby("ticker")["weight"].max().sort_values(ascending=False).index.tolist()
    fig = px.bar(df, x="ticker", y="weight", color="portfolio", barmode="group", category_orders={"ticker": order},
                 color_discrete_map={name: "#ef4444", "current": "#94a3b8"}, hover_data=["sector"],
                 title=f"Target weights: {name} vs current (% of NAV)")
    return _tidy(fig.update_layout(yaxis_tickformat=".0%", xaxis_title=""))
