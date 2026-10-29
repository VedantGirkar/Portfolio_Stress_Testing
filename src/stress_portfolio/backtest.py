"""Chronological orchestration: one daily ledger loop, the one-off funding used by `build`, and the
walk-forward backtest of the equity-only strategies.

Backtest results are preliminary: equity-only (no hedge yet) and survivorship-biased, because the
universe is today's S&P 500 constituents replayed into the past.
"""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd

from . import costs, execution, optimization, rebalancing, selection, stress
from .config import Settings
from .costs import scenario_summary
from .data import future_split_factor, wide
from .portfolio import Portfolio
from .risk import daily_returns, max_drawdown
from .selection import session_calendar

EVENT_ORDER = ("1. dividends/splits for shares held at the prior close",
               "2. fills at the session open (orders decided at an earlier close)",
               "3. valuation at the session close",
               "4. scheduled reviews at the close; their orders fill at the next session's open")
STRATEGIES = ("buy_and_hold", "equal_weight", "min_variance", "stress_aware")
BIAS_LABEL = "equity-only, preliminary, survivorship-biased (today's constituents replayed into the past)"


def simulate(bars: pd.DataFrame, decision_date: pd.Timestamp, targets: pd.Series, s: Settings,
             review: Callable | None = None, review_dates=()) -> dict:
    """Deposit capital, buy `targets` (weights of NAV) at the next open, then run every later session in
    EVENT_ORDER. review(date, portfolio, holdings, nav_row) returns new target weights or None (hold)."""
    close = wide(bars, "close")
    cal = session_calendar(close)
    sessions = cal[cal > decision_date]
    if sessions.empty:
        raise ValueError(f"no session after the decision date {decision_date.date()}; nothing to fill yet")
    opens, dividends, splits = wide(bars, "open"), wide(bars, "dividends"), wide(bars, "splits")
    factor = future_split_factor(bars)  # converts split-adjusted shares to real shares for commissions
    marks = close.ffill()  # valuation marks only (e.g. a halted stock keeps its last close); never used for returns

    p = Portfolio()
    p.deposit(decision_date, s.initial_capital)
    pending = execution.generate_orders(p.positions, targets, close.loc[decision_date], s.initial_capital)
    trades, nav_rows, review_dates = [], [], set(review_dates)
    for d in sessions:
        p.apply_corporate_actions(d, dividends.loc[d], splits.loc[d])
        if pending is not None and not pending.empty:
            fills = execution.execute(pending, opens.loc[d], p.cash, s, factor.loc[d])
            p.apply_fills(fills, d)
            trades.append(pending.merge(fills, on=["ticker", "order_qty"], how="left").assign(fill_date=d))
        pending = None
        holdings, row = p.snapshot(d, marks.loc[d])
        nav_rows.append(row | {"holdings": len(holdings)})
        if review is not None and d in review_dates:
            target = review(d, p, holdings, row)
            if target is not None:
                pending = execution.generate_orders(p.positions, target, close.loc[d], row["nav"])
    return {"portfolio": p, "holdings": holdings, "nav": pd.DataFrame(nav_rows), "fill_date": sessions[0],
            "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()}


def fund_and_hold(core: pd.DataFrame, bars: pd.DataFrame, decision_date: pd.Timestamp, s: Settings) -> dict:
    """Buy the core at the next session's open after the decision date and hold it to the latest close."""
    targets = core.set_index("provider_ticker")["target_weight"] * (1 - s.cash_reserve)
    run = simulate(bars, decision_date, targets, s)
    p, holdings, row = run["portfolio"], run["holdings"], run["nav"].iloc[-1]
    info = core.set_index("provider_ticker")
    holdings = holdings.assign(company=holdings["ticker"].map(info["company"]),
                               sector=holdings["ticker"].map(info["sector"]),
                               target_weight_of_nav=holdings["ticker"].map(targets))
    cash_row = pd.DataFrame([{"ticker": "CASH", "quantity": p.cash, "price": 1.0, "market_value": p.cash,
                              "weight_of_nav": p.cash / row["nav"], "company": "Cash", "sector": "Cash",
                              "target_weight_of_nav": s.cash_reserve}])
    orders = run["trades"].drop(columns="fill_date")
    return {
        "decision_date": decision_date,
        "fill_date": run["fill_date"],
        "orders": orders,
        "holdings": pd.concat([holdings, cash_row], ignore_index=True),
        "ledger": p.ledger(),
        "nav": run["nav"],
        "cost_scenarios": scenario_summary(orders["filled_qty"], orders["ref_price"].fillna(0.0), s),
    }


def run_backtest(membership: pd.DataFrame, bars: pd.DataFrame, history: pd.DataFrame, s: Settings) -> dict:
    """Walk-forward test of four equity-only strategies with the same capital, start, fills and costs.

    bars: daily bars of every member (and the benchmark) since backtest_history_start.
    history: wide adj_close since stress_history_start for at least the starting shortlist, the benchmark and
    the sector ETFs (stress replays). Every review uses only data up to its own close.
    """
    if s.hedge_method != "undecided":
        raise NotImplementedError("a hedged strategy needs the hedging milestone; it must not run without its hedge")
    adj = wide(bars, "adj_close")
    cal = session_calendar(adj)
    adj, close, volume = adj.reindex(cal), wide(bars, "close").reindex(cal), wide(bars, "volume").reindex(cal)
    start = cal[cal < pd.Timestamp(s.backtest_start)][-1]
    sel = selection.run_selection(membership, bars.loc[bars["date"] <= start],
                                  pd.DataFrame(columns=["ticker", "reason"]), s)
    short = sel["shortlist"]  # fixed for the whole backtest; members are re-scored at each review
    tickers = short["provider_ticker"].tolist()
    sectors = short.set_index("provider_ticker")["sector"]
    initial = sel["core"].set_index("provider_ticker")["target_weight"] * (1 - s.cash_reserve)
    reviews = rebalancing.period_end_sessions(cal, start, s.rebalance_frequency)
    bench = history[s.benchmark_ticker]
    factor = future_split_factor(bars)
    k = s.backtest_universe_size
    decisions, dropped, plans = [], [], {}

    def plan(d):
        """Everything a review at the close of d may know, shared by the optimized strategies."""
        if d not in plans:
            a = adj.loc[:d, tickers]
            feats = selection.compute_features(a, close.loc[:d, tickers], volume.loc[:d, tickers], s)
            ok = feats[["momentum", "downside_dev_annual", "max_drawdown"]].notna().all(axis=1) \
                & (feats["sessions_stale"] <= s.max_staleness_sessions)
            dropped.extend({"date": d, "ticker": t, "reason": "no complete price window at this review"}
                           for t in feats.index[~ok])
            base = short.drop(columns=[c for c in short.columns if c in feats.columns
                                       or c.endswith("_pct") or c == "investment_score"])
            scored = selection.score(base.merge(feats[ok], left_on="provider_ticker", right_index=True), s)
            live = scored["provider_ticker"].tolist()
            names = optimization.universes(scored, selection.window_returns(a[live], s),
                                           replace(s, holding_counts=(k,)))[k]
            rets = daily_returns(a[live], s.backtest_estimation_sessions)
            mu, cov, beta = optimization.estimates(rets, (bench / bench.shift(1) - 1).reindex(rets.index), s)
            plans[d] = {"names": names, "mu": mu, "cov": cov, "beta": beta, "live": live}
        return plans[d]

    def log(strategy, d, cur, target, row):
        decisions.append({"strategy": strategy, "date": d, "holdings_before": int((cur > 0).sum()),
                          "holdings_after": int((target > 0).sum()) if target is not None else None, **row})

    def weights(holdings, nav):
        return holdings.set_index("ticker")["market_value"] / nav

    def equal_weight(d, p, holdings, row):
        log("equal_weight", d, weights(holdings, row["nav"]), initial,
            {"action": "rebalance", "reason": "scheduled reset to equal weights", "hedge_review": rebalancing.HEDGE_REVIEW})
        return initial

    def min_variance(d, p, holdings, row):
        pl, cur = plan(d), weights(holdings, row["nav"])
        n = pl["names"]
        w, status = optimization.solve("min_variance", pl["mu"][n], pl["cov"].loc[n, n], sectors[n], cur, s)
        action = "rebalance" if len(w) else "hold"
        log("min_variance", d, cur, w if len(w) else None,
            {"action": action, "reason": f"scheduled minimum-variance ({status})", "hedge_review": rebalancing.HEDGE_REVIEW})
        return w if len(w) else None

    def stress_aware(d, p, holdings, row):
        pl, cur = plan(d), weights(holdings, row["nav"])
        n, mu, cov = pl["names"], pl["mu"], pl["cov"]
        cur = cur[cur.index.isin(mu.index)]  # a holding with no data at this review is left out of the estimates
        known = replace(s, stress_events=tuple(e for e in s.stress_events if pd.Timestamp(e[3]) <= d))
        share = cur * (cov.loc[cur.index, cur.index] @ cur)
        riskiest = share.groupby(sectors[cur.index]).sum().idxmax()
        scen, groups = stress.scenario_returns(sectors[pl["live"]].to_frame(), history.loc[:d], pl["beta"],
                                               riskiest, known)
        train = scen.loc[groups == "train"]
        w, status = optimization.solve("stress_aware", mu[n], cov.loc[n, n], sectors[n], cur, s, stress_train=train)
        cost = 0.0
        if len(w):
            orders = execution.generate_orders(p.positions, w, close.loc[d], row["nav"])
            px = close.loc[d].reindex(orders["ticker"]).values
            cost = float(costs.stock_costs(orders["order_qty"], px, s,
                                           split_factor=factor.loc[d].reindex(orders["ticker"]).fillna(1.0).values)
                         ["total_cost"].sum())
        cur_loss = -(train[cur.index] @ cur)  # loss per training scenario (historical and hypothetical), % of NAV
        decision = rebalancing.decide(cur, w, status, mu, cov, sectors, cost, row["nav"], s,
                                      current_stress_loss=float(cur_loss.max()))
        stress_log = {"stress_current": cur_loss.max(), "stress_current_scenario": cur_loss.idxmax()}
        if len(w):
            tgt_loss = -(train[w.index] @ w)
            stress_log |= {"stress_target": tgt_loss.max(), "stress_target_scenario": tgt_loss.idxmax()}
        log("stress_aware", d, cur, w if len(w) else None, decision | stress_log)
        return w if decision["action"] == "rebalance" else None

    policies = {"buy_and_hold": None, "equal_weight": equal_weight, "min_variance": min_variance,
                "stress_aware": stress_aware}
    navs, trades = [], []
    for name in STRATEGIES:
        run = simulate(bars, start, initial, s, policies[name], reviews)
        navs.append(run["nav"].assign(strategy=name))
        trades.append(run["trades"].assign(strategy=name))
    nav, trades = pd.concat(navs, ignore_index=True), pd.concat(trades, ignore_index=True)
    ref = bench.loc[start:cal[-1]].reindex(cal[cal >= start]).ffill()
    reference = pd.DataFrame({"date": ref.index, "nav": s.initial_capital * ref.values / ref.iloc[0],
                              "strategy": f"{s.benchmark_ticker} (reference)"}).iloc[1:]
    return {
        "bt_nav": pd.concat([nav, reference], ignore_index=True),
        "bt_trades": trades,
        "bt_decisions": pd.DataFrame(decisions),
        "bt_summary": summarize(nav, reference, trades, s),
        "bt_initial_core": sel["core"],
        "bt_dropped": pd.DataFrame(dropped, columns=["date", "ticker", "reason"]),
        "start": start,
        "reviews": len(reviews),
    }


def summarize(nav: pd.DataFrame, reference: pd.DataFrame, trades: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Net results per strategy, plus gross (costs added back, ignoring their compounding) and the reference."""
    rows = []
    for name, g in pd.concat([nav, reference]).groupby("strategy", sort=False):
        v = g.set_index("date")["nav"]
        years = (v.index[-1] - v.index[0]).days / 365.25
        daily = v.pct_change().dropna()
        paid = float(trades.loc[trades["strategy"] == name, "total_cost"].sum()) if name in STRATEGIES else 0.0
        fills = trades.loc[trades["strategy"] == name, "fill_date"].nunique() if name in STRATEGIES else 0
        rows.append({
            "strategy": name, "start": v.index[0].date(), "end": v.index[-1].date(), "final_nav": v.iloc[-1],
            "cagr_net": (v.iloc[-1] / s.initial_capital) ** (1 / years) - 1,
            "cagr_gross": ((v.iloc[-1] + paid) / s.initial_capital) ** (1 / years) - 1,
            "volatility": daily.std() * np.sqrt(s.trading_days_per_year),
            "return_over_vol": daily.mean() / daily.std() * np.sqrt(s.trading_days_per_year),
            "max_drawdown": max_drawdown(v), "total_costs": paid, "trade_days": int(fills),
            "avg_holdings": float(g["holdings"].mean()) if "holdings" in g and g["holdings"].notna().any() else None,
        })
    return pd.DataFrame(rows)
