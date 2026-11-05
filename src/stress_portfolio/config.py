"""Project settings: one place for every research assumption."""

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # Universe and data
    membership_url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    history_calendar_days: int = 1100  # enough sessions for min_observations plus holidays
    download_chunk_size: int = 75
    max_download_failure_share: float = 0.02  # stop the run rather than select from a partial universe
    data_dir: Path = ROOT / "data"
    runs_dir: Path = ROOT / "outputs" / "runs"

    # Eligibility (sessions are trading days)
    min_observations: int = 504
    feature_window_prices: int = 253  # 252 daily returns
    max_staleness_sessions: int = 0  # latest price must be on the cutoff session
    liquidity_window_sessions: int = 63
    min_median_dollar_volume: float = 25_000_000.0

    # Shortlist
    shortlist_size: int = 100
    base_slots_per_sector: int = 5

    # Features
    momentum_lookback_sessions: int = 126
    momentum_skip_sessions: int = 21
    trading_days_per_year: int = 252

    # Investment score weights (on 0-1 percentiles; higher is better)
    weight_momentum: float = 0.50
    weight_low_downside: float = 0.30
    weight_low_drawdown: float = 0.20

    # Core selection
    core_size: int = 10
    correlation_penalty: float = 0.5
    max_per_sector: int = 2
    min_sectors: int = 5

    # Capital and execution (simulated only)
    initial_capital: float = 1_000_000.0
    cash_reserve: float = 0.03  # share of NAV kept in cash for costs and the future hedge

    # Stock trading costs; the scenario multiplier scales all of them
    commission_per_share: float = 0.005
    min_commission_per_order: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 3.0
    cost_scenario: str = "base"  # low = 0.5x, base = 1x, high = 2x

    # Risk (historical simulation of today's holdings; losses are positive numbers)
    benchmark_ticker: str = "SPY"
    var_confidences: tuple[float, ...] = (0.95, 0.99)
    var_horizons_days: tuple[int, ...] = (1, 10)  # multi-day = compounded overlapping returns, no sqrt-time
    risk_lookback_sessions: int | None = None  # None = every session all holdings share
    var_backtest_min_window: int = 252  # expanding-window VaR needs this many prior days first

    # Stress testing. Historical events: (name, label, search start, search end, group). The replay window
    # is SPY's deepest peak-to-trough fall inside the search range, found from the data.
    stress_history_start: str = "2000-01-01"
    stress_events: tuple[tuple[str, str, str, str, str], ...] = (
        ("dotcom", "Dot-com bust", "2000-01-01", "2002-12-31", "train"),
        ("sept11", "9/11 attacks", "2001-09-01", "2001-09-30", "validation"),
        ("gfc", "Global Financial Crisis", "2007-07-01", "2009-06-30", "train"),
        ("lehman", "Lehman month", "2008-09-01", "2008-10-31", "validation"),
        ("downgrade2011", "US downgrade 2011", "2011-07-01", "2011-10-31", "train"),
        ("china2015", "China devaluation 2015", "2015-08-01", "2015-09-30", "validation"),
        ("q4_2018", "Q4 2018 sell-off", "2018-09-01", "2018-12-31", "validation"),
        ("covid", "COVID crash", "2020-02-01", "2020-04-30", "train"),
        ("rates2022", "2022 rate shock", "2022-01-01", "2022-12-31", "train"),
        ("tariff2025", "April 2025 tariff shock", "2025-03-15", "2025-04-30", "validation"),
    )
    sector_etfs: tuple[tuple[str, str], ...] = (
        ("Communication Services", "XLC"), ("Consumer Discretionary", "XLY"), ("Consumer Staples", "XLP"),
        ("Energy", "XLE"), ("Financials", "XLF"), ("Health Care", "XLV"), ("Industrials", "XLI"),
        ("Information Technology", "XLK"), ("Materials", "XLB"), ("Real Estate", "XLRE"), ("Utilities", "XLU"),
    )
    # Hypothetical shocks (all training scenarios). Market moves are scaled by each stock's beta.
    market_shocks: tuple[float, ...] = (-0.10, -0.20, -0.35)
    sector_shock: float = -0.25  # one held sector, market flat
    combined_market_shock: float = -0.20  # plus combined_sector_shock in the riskiest held sector, compounded
    combined_sector_shock: float = -0.20

    # Optimization (annual estimates from daily total returns; cash earns zero)
    holding_counts: tuple[int, ...] = (10, 20, 30)  # stocks per universe; sector cap scales with max_per_sector
    expected_return_blend: float = 0.5  # weight on the historical mean; the rest on CAPM (beta x premium)
    equity_risk_premium: float = 0.05
    risk_aversion: float = 4.0  # objective: return - risk_aversion / 2 x variance - trading cost
    max_position_weight: float = 0.15  # of NAV
    max_sector_weight: float = 0.30  # of NAV
    min_position_weight: float = 0.01  # smaller positions are dropped and the problem re-solved
    stress_loss_budget: float = 0.35  # stress-aware: every training scenario loses at most this share of NAV
    frontier_points: int = 15

    # Rebalancing and backtest (equity-only until the hedge milestone; today's constituents, so survivorship-biased)
    backtest_start: str = "2016-01-01"  # first decision is the last session before this date
    backtest_history_start: str = "2013-01-01"  # enough sessions for eligibility at the start
    backtest_universe_size: int = 20  # stocks the optimized strategies may use at each review
    backtest_estimation_sessions: int = 756  # about 3 years of returns for each review's estimates
    rebalance_frequency: str = "M"  # pandas period code for scheduled reviews: "M" month-end, "Q" quarter-end
    rebalance_min_improvement: float = 0.005  # annual utility gain after costs needed to switch
    rebalance_drift_band: float = 0.02  # a limit only counts as broken once drift passes it by this much

    # Hedge (milestone 7 decision: delta, long SPY puts only, sized so the worst training stress scenario stays within
    # stress_loss_budget; see PROJECT_SPECIFICATION.md 5.11). `build` hedges with today's yfinance chain; the hedged
    # backtest (milestone 9) needs Cboe history.
    hedge_method: str = "delta"
    hedge_underlying: str = "SPY"
    hedge_root: str = "SPY"  # standard trading class only; roots ending in a digit are adjusted contracts
    standard_option_multiplier: int = 100  # OCC standard for the standard root; Cboe files carry no multiplier field
    hedge_moneyness: float = 0.95  # strike as a fraction of the underlying price
    hedge_expiry_buffer_days: int = 14  # expiry at least the next review plus this many calendar days
    hedge_premium_budget: float = 0.03  # net premium (paid minus resale) per year, share of NAV
    hedge_delta_drift: float = 0.10  # extra hedge review when net beta-dollar delta moves this share of NAV
    hedge_dividend_yield: float = 0.013  # assumed SPY yield for Black-Scholes (assumption, not observed)
    hedge_stress_vol_bump: float = 0.15  # added to today's IV under hypothetical shocks (assumption)
    volatility_index_ticker: str = "^VIX"  # observed stress volatility for historical replays (points; / 100)
    rate_ticker: str = "^IRX"  # 13-week T-bill yield in percent; the latest close is the option pricing rate
    option_chain_max_days: int = 120  # fetch listed expiries up to this many calendar days ahead
    option_commission_per_contract: float = 0.65
    option_fees_per_contract: float = 0.05  # exchange and clearing

    def __post_init__(self) -> None:
        weights = self.weight_momentum + self.weight_low_downside + self.weight_low_drawdown
        if abs(weights - 1.0) > 1e-9:
            raise ValueError(f"score weights must sum to 1, got {weights}")
        if max(self.core_size, *self.holding_counts) > self.shortlist_size:
            raise ValueError("core_size and holding_counts cannot exceed shortlist_size")
        if self.max_per_sector < 1:
            raise ValueError("max_per_sector must be at least 1")
        if self.min_sectors > self.core_size:
            raise ValueError("min_sectors cannot exceed core_size")
        if self.min_observations < self.feature_window_prices:
            raise ValueError("min_observations must cover the feature window")
        if self.momentum_lookback_sessions + self.momentum_skip_sessions >= self.feature_window_prices:
            raise ValueError("momentum window must fit inside the feature window")
        if self.initial_capital <= 0 or not 0 <= self.cash_reserve < 1:
            raise ValueError("initial_capital must be positive and cash_reserve in [0, 1)")
        if self.cost_scenario not in {"low", "base", "high"}:
            raise ValueError(f"unknown cost_scenario {self.cost_scenario!r}")
        if self.hedge_method not in {"undecided", "delta", "gamma"}:
            raise ValueError(f"unknown hedge_method {self.hedge_method!r}")
        if not 0 < self.hedge_moneyness <= 1.5 or self.hedge_premium_budget < 0:
            raise ValueError("hedge_moneyness must be in (0, 1.5] and hedge_premium_budget non-negative")

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
