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
| 5–10. Optimization, rebalancing, hedging, backtest | Not started |

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

Selects the core as of the previous completed session, buys it with simulated capital at the next session's open (whole shares, after a cash reserve and trading costs), and values it at the latest close. Adds `orders.csv` (with itemized costs), `holdings.csv`, `ledger.csv`, `nav.csv` and `cost_scenarios.csv` to the run, plus risk tables (`risk_var.csv`, `risk_summary.csv`, `risk_positions.csv`, `risk_sectors.csv`, `risk_correlation.csv`, `risk_series.csv`, `var_backtest*.csv`). It also downloads SPY as the benchmark, plus daily history since 2000 for the holdings, SPY and the 11 sector ETFs (cached in `data/raw/stress/`) for stress tests, which add `stress_scenarios.csv`, `stress_positions.csv` and `stress_sectors.csv`. Capital, cash reserve and cost assumptions are in `config.py`. Each build is a fresh funding; carrying a portfolio forward through time comes with the backtest.

## Dashboard

```bash
uv run streamlit run dashboard/app.py
```

Shows the latest (or any saved) run. Build runs show NAV, cash, actual vs target weights, holdings and trades with costs, a Risk tab with VaR/ES, drawdown, beta, risk contributions, correlations and a VaR backtest, and a Stress tab with every scenario's loss next to SPY's and a per-holding breakdown. Every run shows core target weights by stock and sector, the shortlist by sector, reserves grouped by sector, scores and eligibility. Tabs for later milestones are present but empty.

## Known limitations

- Uses the current constituent list, not point-in-time membership. Fine for today's selection; a historical backtest needs point-in-time membership first.
- Yahoo prices and dividends are split-adjusted, so share quantities are in split-adjusted units, and dividends are credited on the ex-date rather than the pay date.
- Risk uses historical simulation: today's holdings repriced with past daily returns (hypothetical, not results the book earned). Because the stocks were chosen partly on past momentum, their replayed history is flattered; use it for the size of swings, not expected returns.
- Stress replays apply past events to today's holdings; they are not trades anyone made. Stocks not listed during an event use a flagged proxy (sector ETF, else beta x SPY), which can over- or understate the real move. Validation events that overlap a training window (9/11, Lehman) are not fully held out.
- Yahoo data is unofficial; check the provider's terms before relying on it.
