# Stress Portfolio

Research platform for a stylized institution's equity book: a 100-stock S&P 500 shortlist, a 10-stock core portfolio with 90 zero-weight reserves, and (in later milestones) risk, stress testing, cost-aware rebalancing, a mandatory derivative hedge and a walk-forward backtest. Simulated only; no live trading.

See `PROJECT_SPECIFICATION.md` for the full specification and `CLAUDE.md` for working rules.

## Status

| Milestone | State |
|---|---|
| 1. Data and selection | Implemented |
| 2. Ledger, costs, execution | Implemented for stocks (option positions arrive with hedging, milestone 8) |
| 3. Risk and dashboard | Implemented: historical and normal VaR/ES, volatility, drawdown, beta vs SPY, risk contributions, correlations, VaR backtest |
| 4. Stress testing | Implemented for stocks: 10 historical events replayed since 2000, market/sector/combined shocks, position and sector attribution, train/validation tags |
| 5. Optimization | Implemented: equal-weight, minimum-variance, mean-variance and stress-aware candidates for 10/20/30-stock universes (cvxpy), efficient frontier, stress-vs-return trade-off, cost estimates |
| 6. Rebalancing and preliminary backtest | Implemented: monthly decision policy (configurable `rebalance_frequency`), walk-forward backtest 2016–today of buy-and-hold, equal-weight, minimum-variance and stress-aware strategies (equity-only, survivorship-biased) |
| 7. Hedge decision | Decided: delta-based, long SPY puts, worst training stress ≤ 35% NAV, premium ≤ 3% NAV/year, Cboe DataShop SPY option EOD data (see `PROJECT_SPECIFICATION.md` §5.11 decision record) |
| 8. Hedging | `build` buys the hedge from today's yfinance chain (delayed, unofficial quotes): picks the put, sizes it to the stress budget within the premium cap, fills it through the shared ledger, and fills the Hedging tab. Rolling and resizing over time arrive with milestone 9 |
| 9–10. Hedged backtest, final validation | Not started; the backtest runs the four unhedged benchmarks only and says the hedged strategy is missing. It needs Cboe historical option files |

## Setup

```bash
uv sync
uv run pytest -q
```

## Build a selection

```bash
uv run stress-portfolio select
```

Downloads today's S&P 500 list (Wikipedia) and about three years of daily prices (Yahoo Finance via yfinance), then writes a new run to `outputs/runs/<run_id>/`: eligibility, sector quotas, shortlist, core, reserves, data-quality report, download failures, settings and metadata. Prices are cached in `data/raw/prices/`, so a rerun only downloads what is missing.

To rerun without network access, using the saved membership and cached prices:

```bash
uv run stress-portfolio select --offline
```

All selection assumptions (liquidity threshold, score weights, correlation penalty, sector limits) are in `src/stress_portfolio/config.py`.

## Build a funded portfolio

```bash
uv run stress-portfolio build
```

Selects the core as of the previous completed session, buys it with simulated capital at the next session's open (whole shares, after a cash reserve and trading costs), and values it at the latest close. Adds `orders.csv` (with itemized costs), `holdings.csv`, `ledger.csv`, `nav.csv` and `cost_scenarios.csv` to the run, plus risk tables (`risk_var.csv`, `risk_summary.csv`, `risk_positions.csv`, `risk_sectors.csv`, `risk_correlation.csv`, `risk_series.csv`, `var_backtest*.csv`). It also downloads SPY as the benchmark, plus daily history since 2000 for the holdings, SPY and the 11 sector ETFs (cached in `data/raw/stress/`) for stress tests, which add `stress_scenarios.csv`, `stress_positions.csv` and `stress_sectors.csv`. The long history covers all 100 shortlisted stocks so optimization can stress-test every candidate; it adds `opt_candidates.csv`, `opt_weights.csv`, `opt_frontier.csv` and `opt_estimates.csv`. Optimization limits (position and sector caps, stress budget, risk aversion, expected-return blend) are in `config.py`. Capital, cash reserve and cost assumptions are in `config.py`. It then fetches today's SPY put chain from yfinance (cached in `data/raw/options/`) plus ^VIX and ^IRX history, buys the hedge from the cash reserve, and adds it to `holdings.csv`, `ledger.csv` and `nav.csv`, along with `hedge_summary.csv`, `hedge_fills.csv`, `hedge_stress.csv`, `hedge_capped_events.csv`, `hedge_chain.csv` and `hedge_chain_quality.csv`. `--offline` reuses the newest cached chain; if it is older than the valuation date, the run stays unhedged and says so. Each build is a fresh funding; carrying a portfolio forward through time comes with the backtest.

## Run the backtest

```bash
uv run stress-portfolio backtest
```

Picks the starting core and 100-stock shortlist as of the last session before `backtest_start` (2016-01-01), then runs four strategies to the latest session with the same capital, fills and costs: buy-and-hold, monthly equal-weight, monthly minimum-variance, and the stress-aware policy (switch only when the book breaks a limit or the after-cost gain is at least 0.5% a year). Every month-end review (`rebalance_frequency = "M"`; `"Q"` for quarterly) uses only data up to its own close, including which historical crises it knows about. Each stress-aware review checks the book and its target against every training scenario (known historical crises plus the hypothetical market, sector and combined shocks) and logs the worst loss and its scenario in `bt_decisions.csv`; targets must stay within `stress_loss_budget` (35% of NAV), and a book that drifts past it plus the 2% band is rebalanced. Writes `bt_nav.csv`, `bt_trades.csv`, `bt_decisions.csv`, `bt_summary.csv`, `bt_initial_core.csv` and `bt_dropped.csv`. Downloads daily prices since 2013 for every current member (cached in `data/raw/backtest/`).

## Dashboard

```bash
uv run streamlit run dashboard/app.py
```

Shows the latest (or any saved) run. Build runs show NAV, cash, actual vs target weights, holdings and trades with costs, a Risk tab with VaR/ES, drawdown, beta, risk contributions, correlations and a VaR backtest, a Stress tab with every scenario's loss next to SPY's and a per-holding breakdown, an Optimization tab with the frontier, the stress-vs-return chart, all candidates and a weight comparison, and (with a backtest run) Rebalancing and Backtest tabs with the decision log, NAV and drawdown curves, holdings counts and costs. Every run shows core target weights by stock and sector, the shortlist by sector, reserves grouped by sector, scores and eligibility. A Hedging tab shows the puts held, their cost, delta and losses with and without them.

## Known limitations

- Hedge pricing uses Black-Scholes (European) for SPY's American puts, so model values are somewhat low, most for deep in-the-money puts in a crash. Fills and marks come from quotes; the model is used for stress repricing, sizing and Greeks.
- `build` hedge quotes are yfinance's: delayed and unofficial, and possibly stale outside market hours. If no usable quote exists, the run finishes unhedged and says so. Risk and optimization tables cover the stocks only; the Stress tab shows stocks only and the Hedging tab shows them with the puts. Historical events in the hedge comparison stop at the put's expiry, stock loss included, so slow bear markets count only their first weeks. SPY is not this book, so the hedge gives no guaranteed floor.

- Uses the current constituent list, not point-in-time membership. Fine for today's selection; a historical backtest needs point-in-time membership first.
- Yahoo prices and dividends are split-adjusted, so share quantities are in split-adjusted units, and dividends are credited on the ex-date rather than the pay date.
- Risk uses historical simulation: today's holdings repriced with past daily returns (hypothetical, not results the book earned). Because the stocks were chosen partly on past momentum, their replayed history is flattered; use it for the size of swings, not expected returns.
- Stress replays apply past events to today's holdings; they are not trades anyone made. Stocks not listed during an event use a flagged proxy (sector ETF, else beta x SPY), which can over- or understate the real move. Validation events that overlap a training window (9/11, Lehman) are not fully held out.
- Optimization results are optimal only within each heuristic stock universe, not over every stock combination. Expected returns are estimates (half historical, half CAPM); the historical half is flattered for momentum-picked stocks. Candidates are proposals only until the rebalancing milestone.
- The backtest is preliminary: equity-only (no hedge yet) and survivorship-biased, because today's constituents are replayed into the past. Compare strategies with each other, not with real-world returns. Commissions use real share counts (split-adjusted shares divided by later splits).
- Yahoo data is unofficial; check the provider's terms before relying on it.
