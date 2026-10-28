"""Candidate allocations: equal weight, minimum variance, mean-variance, stress-aware, and the frontier.

Stock universes for 10, 20 and 30 holdings come from the existing sequential selection (a heuristic);
cvxpy then optimizes continuous weights inside each universe. Results are optimal for that convex
problem only, not a global optimum over all stock combinations.

Units: weights are fractions of NAV; the equity sleeve is 1 - cash_reserve. Expected return and volatility
are annual estimates from daily total returns. Trading cost in the objective is the linear spread +
slippage rate times traded weight, charged once against a one-year return; the reported cost uses the
full cost model (including commissions) on whole-share orders.
"""

import math
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd

from . import costs, execution, selection
from .config import Settings

TOL = 1e-6


def estimates(rets: pd.DataFrame, bench: pd.Series, s: Settings) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """(expected annual return, annual covariance, beta) per stock. Return = blend x historical mean +
    (1 - blend) x beta x equity premium."""
    beta = rets.apply(lambda r: r.cov(bench)) / bench.var()
    hist = rets.mean() * s.trading_days_per_year
    mu = s.expected_return_blend * hist + (1 - s.expected_return_blend) * beta * s.equity_risk_premium
    return mu, rets.cov() * s.trading_days_per_year, beta


def universes(shortlist: pd.DataFrame, sel_returns: pd.DataFrame, s: Settings) -> dict[int, list[str]]:
    """Stocks for each holding count, from the same sequential selection with the core's sector cap scaled
    to the count (2 per 10 stocks by default), so the core-sized universe is the current core."""
    out = {}
    for k in s.holding_counts:
        sk = replace(s, core_size=k, max_per_sector=math.ceil(k * s.max_per_sector / s.core_size))
        out[k] = sorted(selection.select_core(shortlist, sel_returns, sk)["provider_ticker"])
    return out


def solve(kind: str, mu: pd.Series, cov: pd.DataFrame, sectors: pd.Series, w0: pd.Series, s: Settings,
          stress_train: pd.DataFrame | None = None, target_return: float | None = None) -> tuple[pd.Series, str]:
    """Optimize weights over mu's index. kind: min_variance | mean_variance | stress_aware | max_return |
    frontier. Positions under min_position_weight are dropped and the problem re-solved."""
    names = list(mu.index)
    invest = 1 - s.cash_reserve
    upper = pd.Series(s.max_position_weight, index=names)
    dropped = 0
    for _ in range(10):
        w = cp.Variable(len(names))
        cons = [cp.sum(w) == invest, w >= 0, w <= upper.values]
        for sec in sorted(sectors.unique()):
            idx = np.flatnonzero(sectors.values == sec)
            cons.append(cp.sum(w[idx]) <= s.max_sector_weight)
        risk = cp.quad_form(w, cp.psd_wrap(cov.values))
        trade_cost = (s.half_spread_bps + s.slippage_bps) / 1e4 * cp.norm1(w - w0.reindex(names, fill_value=0).values)
        utility = mu.values @ w - s.risk_aversion / 2 * risk - trade_cost
        if kind == "min_variance" or kind == "frontier":
            objective = cp.Minimize(risk)
            if kind == "frontier":
                cons.append(mu.values @ w >= target_return)
        elif kind == "max_return":
            objective = cp.Maximize(mu.values @ w)
        else:
            objective = cp.Maximize(utility)
            if kind == "stress_aware":
                cons.append(stress_train[names].values @ w >= -s.stress_loss_budget)
        prob = cp.Problem(objective, cons)
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            cause = f" after dropping {dropped} position(s) under {s.min_position_weight:.0%}" if dropped else ""
            return pd.Series(dtype=float), f"{prob.status}{cause}"
        weights = pd.Series(np.where(w.value > TOL, w.value, 0.0), index=names)
        small = weights[(weights > 0) & (weights < s.min_position_weight - TOL)].index
        if small.empty:
            note = f" after dropping {dropped} position(s) under {s.min_position_weight:.0%}" if dropped else ""
            return weights * invest / weights.sum(), f"{prob.status}{note}"
        upper[small] = 0.0
        dropped += len(small)
    return pd.Series(dtype=float), "did not settle after dropping small positions"


def evaluate(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, scen: pd.DataFrame, groups: pd.Series,
             holdings: pd.DataFrame, prices: pd.Series, nav: float, s: Settings) -> dict:
    """Estimated return/vol, stress losses (training and validation separately), and whole-share trade cost."""
    w = weights[weights > 0]
    ret, vol = float(mu[w.index] @ w), float(np.sqrt(w @ cov.loc[w.index, w.index] @ w))
    losses = -(scen[w.index] @ w)
    train, valid = losses[groups[losses.index] == "train"], losses[groups[losses.index] == "validation"]
    qty = holdings.set_index("ticker")["quantity"].drop("CASH", errors="ignore")
    orders = execution.generate_orders(qty.to_dict(), w, prices, nav)
    cost = costs.stock_costs(orders["order_qty"], prices.reindex(orders["ticker"]).values, s)
    current = (holdings.set_index("ticker")["weight_of_nav"].drop("CASH", errors="ignore"))
    turnover = float((w.reindex(current.index.union(w.index), fill_value=0)
                      - current.reindex(current.index.union(w.index), fill_value=0)).abs().sum() / 2)
    return {"holdings": len(w), "expected_return": ret, "volatility": vol, "sharpe": ret / vol if vol else np.nan,
            "worst_train_loss": float(train.max()), "worst_train_scenario": train.idxmax(),
            "worst_validation_loss": float(valid.max()) if len(valid) else np.nan,
            "worst_validation_scenario": valid.idxmax() if len(valid) else None,
            "trades": int((orders["order_qty"] != 0).sum()), "est_cost_usd": float(cost["total_cost"].sum()),
            "turnover": turnover}


def build_candidates(shortlist: pd.DataFrame, sel_returns: pd.DataFrame, est: tuple, scen: pd.DataFrame,
                     groups: pd.Series, holdings: pd.DataFrame, prices: pd.Series, s: Settings) -> dict[str, pd.DataFrame]:
    """All candidates and frontiers. est: (mu, cov, beta) from `estimates`; scen: scenario x ticker shock
    returns; groups: scenario -> train/validation."""
    mu, cov, beta = est
    sector_of = shortlist.set_index("provider_ticker")["sector"]
    stocks = holdings.loc[holdings["ticker"] != "CASH"]
    nav = float(holdings["market_value"].sum())
    w0 = stocks.set_index("ticker")["weight_of_nav"]
    stress_train = scen.loc[groups[scen.index] == "train"]
    invest = 1 - s.cash_reserve

    rows, weights, frontier = [], [], []

    def record(k, method, w, status):
        row = {"holding_count": k, "method": method, "status": status, "candidate": f"{method}_{k}"}
        if len(w):
            row |= evaluate(w, mu, cov, scen, groups, holdings, prices, nav, s)
            weights.append(pd.DataFrame({"candidate": row["candidate"], "ticker": w.index, "weight": w.values,
                                         "sector": sector_of.reindex(w.index).values}).query("weight > 0"))
        rows.append(row)

    record(len(w0), "current", w0, "no trades")
    for k, names in universes(shortlist, sel_returns, s).items():
        m, c, sec = mu[names], cov.loc[names, names], sector_of[names]
        record(k, "equal_weight", pd.Series(invest / k, index=names), "fixed weights")
        for kind in ("min_variance", "mean_variance", "stress_aware"):
            w, status = solve(kind, m, c, sec, w0, s, stress_train=stress_train)
            record(k, kind, w, status)
        lo, _ = solve("min_variance", m, c, sec, w0, s)
        hi, _ = solve("max_return", m, c, sec, w0, s)
        if len(lo) and len(hi):
            for i, target in enumerate(np.linspace(float(m[lo.index] @ lo), float(m[hi.index] @ hi),
                                                   s.frontier_points)):
                w, status = solve("frontier", m, c, sec, w0, s, target_return=target - TOL)
                point = {"holding_count": k, "point": i, "target_return": target, "status": status}
                if len(w):
                    e = evaluate(w, mu, cov, scen, groups, holdings, prices, nav, s)
                    point |= {key: e[key] for key in ("expected_return", "volatility", "worst_train_loss")}
                frontier.append(point)

    return {"opt_candidates": pd.DataFrame(rows), "opt_weights": pd.concat(weights, ignore_index=True),
            "opt_frontier": pd.DataFrame(frontier),
            "opt_estimates": pd.DataFrame({"expected_return": mu, "volatility": np.sqrt(np.diag(cov)), "beta": beta,
                                           "sector": sector_of.reindex(mu.index)}).rename_axis("ticker").reset_index()}
