"""Mandatory derivative hedge: long SPY puts sized to the stress-loss budget (milestone 7 decision record).

Pricing is Black-Scholes-Merton for European puts with a continuous dividend yield. SPY options are American,
so the model undervalues them somewhat (most for deep in-the-money puts); it is used for stress repricing,
sizing and missing Greeks, while normal-day marks come from observed quotes.

Greek units, per share of underlying: delta per $1 move, gamma per $1 per $1, vega per 1.00 of volatility
(0.01 = one vol point), theta per year. Dollar exposure = contracts x multiplier x Greek x underlying price.
"""

import math

import numpy as np
import pandas as pd

from .config import Settings

_erf = np.vectorize(math.erf, otypes=[float])
PERIOD_YEARS = {"M": 1 / 12, "Q": 1 / 4}


def _ncdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _npdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def contract_id(root: str, expiry, option_type: str, strike: float) -> str:
    """Deterministic instrument ID: the strike is always printed with two decimals, so 190 and 190.0 match."""
    return f"{root} {pd.Timestamp(expiry):%Y-%m-%d} {option_type.upper()} {float(strike):.2f}"


def put_price_greeks(spot, strike, years, rate, div_yield, vol) -> dict[str, np.ndarray]:
    """European put price and Greeks per share. At or past expiry, or with zero volatility, the put is worth
    its (discounted forward) intrinsic value and has no vega or theta."""
    S, K, T, r, q, v = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in
                                             (spot, strike, years, rate, div_yield, vol)))
    live = (T > 0) & (v > 0)
    Tl, vl = np.where(live, T, 1.0), np.where(live, v, 1.0)  # placeholders keep the formulas finite
    sq = vl * np.sqrt(Tl)
    d1 = (np.log(S / K) + (r - q + 0.5 * vl * vl) * Tl) / sq
    d2 = d1 - sq
    disc_q, disc_r = np.exp(-q * Tl), np.exp(-r * Tl)
    price = K * disc_r * _ncdf(-d2) - S * disc_q * _ncdf(-d1)
    delta = -disc_q * _ncdf(-d1)
    gamma = disc_q * _npdf(d1) / (S * sq)
    vega = S * disc_q * _npdf(d1) * np.sqrt(Tl)
    theta = (-S * disc_q * _npdf(d1) * vl / (2 * np.sqrt(Tl)) + r * K * disc_r * _ncdf(-d2)
             - q * S * disc_q * _ncdf(-d1))

    Tz = np.maximum(T, 0.0)
    fwd_k, fwd_s = K * np.exp(-r * Tz), S * np.exp(-q * Tz)
    dead_price = np.maximum(fwd_k - fwd_s, 0.0)
    dead_delta = np.where(fwd_k > fwd_s, -np.exp(-q * Tz), 0.0)
    return {"price": np.where(live, price, dead_price), "delta": np.where(live, delta, dead_delta),
            "gamma": np.where(live, gamma, 0.0), "vega": np.where(live, vega, 0.0),
            "theta": np.where(live, theta, 0.0)}


def put_price(spot, strike, years, rate, div_yield, vol) -> np.ndarray:
    return put_price_greeks(spot, strike, years, rate, div_yield, vol)["price"]


def choose_put(chain: pd.DataFrame, spot: float, next_review, s: Settings) -> pd.Series | None:
    """The put to buy from one day's normalized quotes (see data.load_option_quotes): the earliest listed expiry
    at least `hedge_expiry_buffer_days` past the next review, then the strike nearest hedge_moneyness x spot
    (lower strike on a tie). Needs a positive ask and a known bid. None when nothing qualifies."""
    earliest = pd.Timestamp(next_review) + pd.Timedelta(days=s.hedge_expiry_buffer_days)
    ok = chain.loc[(chain["root"] == s.hedge_root) & (chain["option_type"] == "P")
                   & (pd.to_datetime(chain["expiration"]) >= earliest)
                   & (chain["ask"] > 0) & chain["bid"].notna()]
    if ok.empty:
        return None
    ok = ok.loc[pd.to_datetime(ok["expiration"]) == pd.to_datetime(ok["expiration"]).min()]
    gap = (ok["strike"] - s.hedge_moneyness * spot).abs()
    return ok.assign(_gap=gap).sort_values(["_gap", "strike"]).iloc[0].drop("_gap")


def annual_net_cost(entry_price: float, spot: float, strike: float, years: float, rate: float, vol: float,
                    s: Settings) -> float:
    """Expected net premium per share per year: entry price minus modeled resale one review later (spot and
    volatility unchanged), scaled to a year. A planning estimate; the backtest tracks the actual net cost."""
    hold = PERIOD_YEARS[s.rebalance_frequency]
    resale = float(put_price(spot, strike, max(years - hold, 0.0), rate, s.hedge_dividend_yield, vol))
    return (entry_price - resale) / hold


def hedged_losses(scenarios: pd.DataFrame, contracts: int, entry_price: float, spot: float, strike: float,
                  years: float, multiplier: int, nav: float, rate: float, s: Settings) -> pd.Series:
    """Loss as a share of NAV (positive = loss) per scenario with `contracts` puts held.

    scenarios: one row per scenario with book_pnl_usd (stocks), spy_return, stress_vol and elapsed_years (how
    far into the put's life the scenario ends). Each put is fully repriced at the scenario end, never
    approximated with Greeks; a put past expiry is worth its intrinsic value.
    """
    if not (scenarios["stress_vol"] > 0).all():
        raise ValueError("every scenario needs a positive stress volatility; drop unavailable events first")
    shocked = spot * (1.0 + scenarios["spy_return"].to_numpy())
    left = np.maximum(years - scenarios["elapsed_years"].to_numpy(), 0.0)
    value = put_price(shocked, strike, left, rate, s.hedge_dividend_yield, scenarios["stress_vol"].to_numpy())
    put_pnl = contracts * multiplier * (value - entry_price)
    return pd.Series(-(scenarios["book_pnl_usd"].to_numpy() + put_pnl) / nav, index=scenarios.index)


def size_hedge(scenarios: pd.DataFrame, entry_price: float, spot: float, strike: float, years: float,
               multiplier: int, nav: float, rate: float, vol: float, s: Settings, max_contracts: int = 10_000) -> dict:
    """Fewest whole contracts that keep every scenario's loss within stress_loss_budget, capped by the yearly
    premium budget. If the cap binds first, buy the cap and report the remaining breach (status "partial")."""
    cost = annual_net_cost(entry_price, spot, strike, years, rate, vol, s) * multiplier
    cap = max_contracts if cost <= 0 else min(max_contracts, int(s.hedge_premium_budget * nav // cost))
    unhedged = float(hedged_losses(scenarios, 0, entry_price, spot, strike, years, multiplier, nav, rate, s).max())
    n = np.arange(cap + 1)
    worst = np.array([hedged_losses(scenarios, k, entry_price, spot, strike, years, multiplier, nav, rate, s).max()
                      for k in n])
    meets = np.flatnonzero(worst <= s.stress_loss_budget)
    contracts = int(meets[0]) if len(meets) else cap
    status = ("not needed" if contracts == 0 and len(meets) else "target met" if len(meets)
              else "partial: premium budget binds")
    return {"contracts": contracts, "worst_unhedged": unhedged, "worst_hedged": float(worst[contracts]),
            "budget_cap_contracts": cap, "annual_net_cost_usd": contracts * cost, "status": status}


def scenario_inputs(capped_historical: pd.DataFrame, scenarios: pd.DataFrame, iv_now: float,
                    s: Settings) -> pd.DataFrame:
    """Training scenarios in the shape hedged_losses needs. Historical rows come from stress.capped_replay
    (observed volatility index, path cut at the put's expiry); hypothetical rows from the stress table are
    instantaneous, with today's IV plus hedge_stress_vol_bump (an assumption)."""
    usable = (capped_historical["group"] == "train") & capped_historical["available"].astype(bool)
    hist = capped_historical.loc[usable, ["scenario", "book_pnl_usd", "spy_return", "stress_vol", "elapsed_years"]]
    hypo = scenarios.loc[(scenarios["kind"] == "hypothetical") & (scenarios["group"] == "train")]
    hypo = pd.DataFrame({"scenario": hypo["scenario"], "book_pnl_usd": hypo["pnl_usd"],
                         "spy_return": hypo["benchmark_return"], "stress_vol": iv_now + s.hedge_stress_vol_bump,
                         "elapsed_years": 0.0})
    return pd.concat([hist, hypo], ignore_index=True).set_index("scenario")


def beta_dollar_delta(stock_values: pd.Series, betas: pd.Series, contracts: float, put_delta: float, spot: float,
                      multiplier: int) -> float:
    """Book exposure to the underlying in dollars: sum of value x beta for stocks plus the puts' dollar delta."""
    stocks = float((stock_values * betas.reindex(stock_values.index)).sum())
    return stocks + contracts * multiplier * put_delta * spot


def needs_review(net_delta: float, delta_at_hedge: float, nav: float, s: Settings) -> bool:
    """Extra hedge review once net beta-dollar delta drifts more than hedge_delta_drift x NAV from the last hedge."""
    return abs(net_delta - delta_at_hedge) > s.hedge_delta_drift * nav
