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

    # Hedge (decided at milestone 7; a full-strategy run must reject "undecided")
    hedge_method: str = "undecided"

    def __post_init__(self) -> None:
        weights = self.weight_momentum + self.weight_low_downside + self.weight_low_drawdown
        if abs(weights - 1.0) > 1e-9:
            raise ValueError(f"score weights must sum to 1, got {weights}")
        if self.core_size > self.shortlist_size:
            raise ValueError("core_size cannot exceed shortlist_size")
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

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
