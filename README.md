# Stress Portfolio

Research platform for a stylized institution's equity book: a 100-stock S&P 500 shortlist, a 10-stock core portfolio with 90 zero-weight reserves, and (in later milestones) risk, stress testing, cost-aware rebalancing, a mandatory derivative hedge and a walk-forward backtest. Simulated only; no live trading.

See `PROJECT_SPECIFICATION.md` for the full specification and `CLAUDE.md` for working rules.

## Status

| Milestone | State |
|---|---|
| 1. Data and selection | Implemented |
| 2–10. Ledger, risk, stress, optimization, rebalancing, hedging, backtest | Not started |

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

## Dashboard

```bash
uv run streamlit run dashboard/app.py
```

Shows the latest (or any saved) run: core target weights by stock and sector, the shortlist by sector, reserves grouped by sector, scores and eligibility. Tabs for later milestones are present but empty.

## Known limitations

- Uses the current constituent list, not point-in-time membership. Fine for today's selection; a historical backtest needs point-in-time membership first.
- Target weights only: no orders, cash ledger or NAV until milestone 2.
- Yahoo data is unofficial; check the provider's terms before relying on it.
