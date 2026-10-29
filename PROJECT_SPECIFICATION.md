# Project Specification

Detailed functional specification for the stress-tested portfolio research platform. Working rules, repository structure, commands, and safeguards live in `CLAUDE.md`; this file defines what each part of the system must do. It specifies required behavior and is not a claim that any module is already implemented.

## 1. What this project does

Build an end-to-end, modular Python portfolio-management research platform for a portfolio manager overseeing a stylized financial institution's equity investment book.

The platform begins with a dated S&P 500 constituent snapshot, constructs an eligible and sector-diversified shortlist of 100 stocks, selects an initial 10-stock equity portfolio, and maintains the other 90 stocks as zero-weight reserve candidates. It values actual holdings and cash, measures normal and tail risk, applies severe historical and hypothetical stress scenarios, and proposes cost-aware portfolio changes.

Rebalancing can change existing weights, remove stocks, add reserve stocks, or change the number of equity holdings. The initial count is 10, not a permanent portfolio-size constraint. Later allocations may hold 20 or 30 stocks, subject to configured limits and the demonstrated benefit of diversification after costs.

Derivatives hedging is a mandatory component of the final project. The hedge must implement a clearly defined delta-hedging or gamma-hedging approach. The choice is deliberately deferred until the hedging module is reached. Do not silently choose one now, implement a token protective put and call the requirement satisfied, or treat hedging as optional.

The complete platform must evaluate decisions chronologically through a walk-forward backtest and expose the portfolio, risk, stress results, efficient frontier, trades, costs, and hedge behavior through an interactive dashboard.

This is a simulated investment-book study, not a live trading system and not a full bank regulatory-capital, liquidity, or solvency model. Do not claim that portfolio survival proves institution-wide survival.

## 2. Current baseline to preserve

The previously supplied starter implementation contains:

- `src/stress_portfolio/config.py`: a Settings dataclass.
- `src/stress_portfolio/data.py`: membership retrieval, saved membership loading, price downloads/cache, and structural validation.
- `src/stress_portfolio/selection.py`: feature calculation, eligibility, sector quotas, shortlist creation, and sequential core selection.
- `src/stress_portfolio/main.py`: selection-workflow command-line orchestration and exports.
- `tests/test_pipeline.py`: synthetic tests for selection and data behavior.

Inspect the actual files before editing; this description is context, not proof of current repository contents. Preserve existing CLI behavior and outputs where practical. Do not repeatedly rebuild the starter or relocate working modules for every new milestone.

The starter used a current Wikipedia constituent table and Yahoo data through yfinance. These are initial provider choices, not guarantees of availability or licensing rights. Handle failures explicitly and comply with provider terms.

## 3. Portfolio mandate and configurable assumptions

| Item | Initial design |
|---|---|
| Parent universe | S&P 500 membership snapshot |
| Research shortlist | 100 eligible stocks |
| Initial equity holdings | 10 |
| Initial equity-sleeve weights | 10% each |
| Reserve candidates | Other 90 shortlisted stocks at zero weight |
| Later equity holdings | Flexible; compare candidates up to 10, 20, and 30 |
| Equity direction | Long-only initially |
| Currency | USD initially |
| Cash | Explicit, including hedge and cost funding |
| Equity leverage | Disabled initially |
| Scheduled stock rebalancing | Monthly default (configurable; quarterly also supported) |
| Risk-triggered stock rebalancing | Added after scheduled behavior is validated |
| Hedge adjustment frequency | Separate configurable policy, decided with hedge method |
| Derivatives hedge | Required; delta or gamma approach decided later |
| Execution | Simulated only |
| Evaluation | Out-of-sample, after costs |

Numeric sector, position, turnover, hedge, and risk limits are configurable research assumptions. Do not present them as official regulatory rules or mathematically optimal choices.

Initial 10% weights refer to the equity sleeve. If cash or option assets are held, stock weights relative to total NAV differ. Reserve hedge capital before equity purchases or rescale the equity sleeve. Never invest 100% of capital in stocks and then buy options using unrecorded funds.

Keep the initial 100-stock shortlist fixed within the first backtest implementation. Re-score its members at decision dates. A future universe refresh needs an explicit policy for new members, departing members, held securities, and historical information availability; do not refresh silently.

## 4. End-to-end system workflow

1. Load validated settings and establish run ID, data cutoff, and execution convention.
2. Retrieve or reload a dated S&P 500 membership snapshot.
3. Load/download and validate stock, benchmark, corporate-action, and eventually derivative data.
4. Construct the 100-stock eligible shortlist.
5. Select the initial 10 stocks and generate equity target weights.
6. Reserve funds for required costs and the derivative hedge.
7. Generate simulated orders and execute them under an explicit fill convention.
8. Update the authoritative holdings/cash ledger.
9. Value stock and derivative positions consistently.
10. Calculate performance, risk, exposures, VaR, and expected shortfall.
11. Stress test the combined equity-and-derivative portfolio.
12. Generate alternative allocations, including different equity holding counts.
13. Compare candidates with the no-action portfolio after projected costs and hedge requirements.
14. Execute approved simulated rebalance and hedge-adjustment events.
15. Advance chronological time, record realized outcomes, and repeat.
16. Save reproducible artifacts and present them in the dashboard.

Separate the same-date planning loop from the time-progressing backtest. Improving an allocation against its optimization scenarios is not evidence that it performed better in future realized conditions.

## 5. Module specifications

### 5.1 `src/stress_portfolio/config.py` — One source of assumptions

Own the Settings dataclass or similarly simple validated configuration. Cover universe/selection parameters, date windows, capital, constraints, costs, stress settings, stock-rebalance policy, hedge method/status, hedge limits, and data mode.

Use named units: rates in decimals unless explicitly named otherwise, basis-point settings with a `_bps` suffix, per-contract/per-share charges identified clearly, and dates with defined inclusion rules.

The hedge method can remain `undecided` during development. A final full-strategy run must reject an undecided method or a disabled mandatory hedge. Equity-only development and benchmark runs remain allowed if clearly labeled.

### 5.2 `src/stress_portfolio/data.py` — Collection, normalization, validation, and caching

Inputs: membership source or saved CSV, symbols, dates, provider settings, and cache location.

Responsibilities:

- Preserve membership symbols separately from provider symbols.
- Save membership snapshot, source, timestamp, and source hash where feasible.
- Load daily equity and benchmark data, corporate actions, and later historical option data.
- Standardize column names, timestamps, symbols, and units.
- Preserve raw inputs and avoid unnecessary repeat downloads.
- Validate duplicates, nonpositive prices, missing data, volume, stale observations, and option contract fields.
- Report download failures individually.
- Provide local-data mode for offline development and tests.

Do not select holdings, silently fill all missing prices, or substitute today's option chain for a historical chain. Missing observations before listing are not zero returns. Invalid records require a logged handling policy.

Use an explicit price-adjustment convention. Preserve suitable raw/provider prices and corporate actions for accounting; use documented adjusted/total-return series for estimation. Do not double-count dividends or splits.

Current constituents are not historical constituents. Capture this limitation in retrospective run metadata. A cutoff on prices does not remove membership bias.

Outputs: standardized bars, corporate actions, membership data, derivative data when available, and quality reports.

Tests: schema validation, missing data, duplicates, symbol mapping, cache reload, date filtering, corporate-action preservation, and individual download failure handling. Most tests must be offline.

### 5.3 `src/stress_portfolio/selection.py` — Eligible universe, shortlist, and initial holdings

Preserve the current baseline unless a change is explicitly approved:

1. Require approximately 504 usable price observations.
2. Use a complete recent feature window and a fresh latest price.
3. Apply a recent median dollar-volume eligibility screen.
4. Deduplicate company share classes using a reliable company identifier where available.
5. Allocate up to five initial shortlist slots per represented sector, then distribute remaining slots by proportional target deficits.
6. Choose liquid companies within sectors until the shortlist contains 100.
7. Calculate cross-sectional feature percentiles within the shortlist.
8. Select the core sequentially with diversification constraints.

Baseline investment features:

- Momentum: 126-session adjusted-price return ending 21 sessions before the latest observation.
- Downside deviation: square root of the mean squared negative daily returns, using all 252 observations and zero contribution from nonnegative days; annualization convention documented.
- Maximum drawdown loss: positive magnitude of the deepest decline from a running peak within the trailing 253-price window.

Investment score:

`0.50 * momentum_percentile + 0.30 * low_downside_percentile + 0.20 * low_drawdown_percentile`

Higher percentile always means more desirable. Preserve the raw features and component scores.

Sequential selection score:

`investment_score - correlation_penalty * normalized_average_correlation`

For existing selected holdings, map average correlation from [-1, 1] into [0, 1]. Apply no correlation penalty to the first selected stock. Handle undefined correlations explicitly and conservatively. The initial constraints are at most two stocks per sector and at least five represented sectors.

Outputs: eligibility decisions, sector quotas, 100-stock shortlist, 10 core stocks, 90 reserves, raw features, component scores, selection order, correlations at selection, explanations, and equity-sleeve target weights.

This is an explainable heuristic, not global optimization. Do not hard-code the previously observed selected tickers or tune to reproduce them.

Tests: exact counts when feasible, no duplicates, correct weights, sector limits, deterministic ties, known feature calculations, and explicit failure on insufficient candidates.

### 5.4 `src/stress_portfolio/portfolio.py` — Authoritative accounting and valuation

Maintain one portfolio state with cash, stock quantities, derivative quantities/specifications, and transaction/event history. Use a small state object plus tables rather than a framework.

Responsibilities:

- Apply fills and fees once.
- Track dividends, splits, and derivative lifecycle cash flows.
- Value positions and calculate NAV.
- Calculate actual weights from filled holdings.
- Preserve target versus actual allocations.
- Produce dated snapshots for analytics and reporting.

Accounting identity: NAV equals cash plus market value of all positions less explicitly modeled liabilities. Never count both a cash premium expense and an unchanged option asset as separate losses. Long-option purchases exchange cash for an asset; subsequent valuation and realized cash flows determine P&L.

Tests: buy/sell reconciliation, fees, dividends, splits, rounding effects, derivative opening/closing/settlement, and NAV identities.

### 5.5 `src/stress_portfolio/costs.py` — Central cost calculations

Use shared calculations for optimization estimates, simulated execution, and reports. Initial parameters can be constant, but distinguish per-share, per-contract, per-order, percentage, and basis-point charges.

Include as applicable:

- Stock commissions and modeled spread/slippage.
- Option commissions, exchange/clearing charges, and modeled spread/slippage.
- Regulatory charges where modeled.
- Exercise/assignment event charges where applicable.
- Financing or borrow costs only if those features are later introduced.
- Optional explicitly scoped account/management costs.

Keep data-subscription/research overhead outside strategy NAV by default. Taxes and FX are out of initial scope unless explicitly added.

Do not double-count spreads embedded in bid/ask fills. Do not separately expense theta on top of market-value changes. A hedge roll consists of closing and opening transactions, not a duplicate generic fee. Premium is a funding cash flow, not an arbitrary transaction-cost percentage.

Outputs: itemized cost breakdowns and totals, with low/base/high-friction sensitivity support.

Tests: known fee examples, zero orders, sign/unit consistency, option contract multipliers, and no duplicate charging.

### 5.6 `src/stress_portfolio/execution.py` — Orders and simulated fills

Convert current holdings and target allocations into share/contract orders. Enforce cash, integer rounding policy, configured instrument limits, and the execution timestamp convention.

Responsibilities:

- Generate buys, sells, opens, closes, and rolls.
- Estimate costs through `costs.py`.
- Apply the chosen fill-price and liquidity assumptions.
- Return filled, adjusted, and rejected orders with reasons.
- Leave authoritative state mutation to `portfolio.py`.

Do not execute using information unavailable at order time. End-of-day decisions must not receive an earlier same-day fill. A documented next-session convention is acceptable.

Tests: unchanged targets produce no unnecessary trade; fees do not overspend cash; positions round consistently; derivative contract quantities and cash flows reconcile.

### 5.7 `src/stress_portfolio/risk.py` — Performance, exposures, VaR, and expected shortfall

Calculate returns, volatility, drawdowns, historical VaR, expected shortfall, correlations/covariance, sector exposures, concentration, and optionally market beta or risk contributions.

Every metric must identify horizon, lookback, confidence level, and estimation convention. Use positive values for losses in risk outputs. Define discrete-tail and percentile conventions explicitly. VaR is not maximum possible loss.

Calculate combined-portfolio risk from consistent joint stock-and-derivative valuation or scenario P&L once hedging exists. Do not model nonlinear options as ordinary fixed-weight stocks without justification.

Track VaR exceedances using estimates generated before the realized outcomes. Keep estimated and realized metrics distinct.

Outputs: metric tables, exposure tables, correlation matrices, and chart-ready series.

Tests: manually verified small examples, flat portfolio, known drawdown path, VaR/ES tail handling, consistent units and horizons.

### 5.8 `src/stress_portfolio/stress.py` — Extreme-scenario analysis and attribution

Implement historical event replay and hypothetical market/sector shocks first. Add path simulation only after deterministic scenarios are validated.

Responsibilities:

- Define named scenarios and horizons.
- Map shocks to stock exposures.
- Reprice derivative positions under price, implied-volatility, time, rate, and dividend assumptions as appropriate.
- Aggregate position P&L and reconcile attribution.
- Compare current, proposed, hedged, and unhedged portfolios.
- Keep training/optimization scenarios distinct from validation scenarios where feasible.

Do not confuse historical replay of today's portfolio with an investable historical backtest. Do not assume historical events that occur after a decision date were already known to the decision policy.

Before expiry, use suitable option repricing, not expiry payoff alone. Use full repricing for large stress moves; Greek approximations may be explanatory diagnostics, not the sole severe-shock valuation.

Validate simulated covariance matrices and save random seeds. Do not sum daily returns as if they were compounded multi-day returns. Avoid overlapping factor definitions that double-count market shocks.

Tests: zero shock, known single-position shocks, attribution identity, options at/before expiry, and reproducible simulation if introduced.

### 5.9 `src/stress_portfolio/optimization.py` — Candidate allocations and frontier

Implement progressively: equal-weight baseline, minimum variance, mean–variance frontier, then stress-aware allocation with trading-cost estimates.

Inputs: available candidates, estimates, current drifted holdings, stress scenarios, cash/hedge funding requirements, and constraints.

Generate candidates for up to 10, 20, and 30 stocks. Initial selection may remain heuristic followed by continuous weight optimization. Report solver/heuristic status and infeasibility.

Constraints can include long-only weights, budget, cash reserve, position maximum, sector limit, minimum meaningful position, turnover, stock count, and stress-loss budget.

Do not use an L1 penalty on nonnegative fully invested weights as a claimed sparsity solution; it is constant under that constraint. Do not claim global optimality for a heuristic search.

A conventional efficient frontier uses estimated return and volatility. A stress-loss/return tradeoff is a separate chart. Label approximate frontiers and maintain consistent units and horizons. Hedge-aware candidates require explicit derivative valuation and capital funding.

Outputs: targets, estimated costs, estimated risk/return, stress outcomes, constraints, and status.

Tests: feasible budgets/limits, known small problems, infeasible problems, cost inclusion, and repeatability.

### 5.10 `src/stress_portfolio/rebalancing.py` — Decision policy

Decide whether, when, and how to change positions. Begin with scheduled monthly stock decisions; add risk triggers only after validating the schedule.

Compare current holdings and proposed candidates after costs, hedge changes, and configured improvement thresholds. Keeping the current portfolio is a valid decision.

Permit weight changes, replacements, and stock-count changes. Avoid arbitrary forced turnover. A future risk-trigger policy may need cooldown/hysteresis to avoid repeated near-threshold trading.

Coordinate with hedge review, but do not assume Greek maintenance only occurs on stock-rebalance dates. Log stock-rebalance and hedge-adjustment events separately.

Outputs: action/no-action decision, target allocation, hedge-review request, reasons, projected improvement, and projected costs.

Tests: correct scheduled dates, no-action outcome, cost thresholds, constraint breaches, and stock/hedge event separation.

### 5.11 `src/stress_portfolio/hedging.py` — Mandatory derivative hedge

This module is required for final completion. Its method remains undecided until its implementation stage.

#### Decision gate before implementation

Present a concise delta-versus-gamma design comparison and obtain the user's choice. Specify:

- What exposure is being hedged: portfolio P&L, selected stocks, market exposure, or an explicitly defined option book.
- Which instruments are permitted for the hedge and why they fit that exposure.
- Delta/gamma measurement convention and units.
- Target exposure, acceptable residual exposure, and coverage goal.
- Whether the goal is full neutralization or partial downside-risk reduction.
- Allowed stock, option, short-option, and financing positions.
- Contract/price data availability and historical versus synthetic mode.
- Hedge adjustment frequency, triggers, budget, and turnover limits.
- Expiry, settlement, closing, rolling, and exercise policy.

Do not assume that a long-only stock book has a meaningful nonzero gamma to neutralize, or that the user wants a fully market-neutral book. Establish the exposure and protection objective explicitly. If the selected hedge requires permissions or financing outside the mandate, resolve that conflict before coding it.

#### Required implementation behavior

- Represent contract underlying, option type, strike, expiry, multiplier, exercise style, settlement convention, and adjustments.
- Select actual available expiries in historical-market mode.
- Price options using a model appropriate to documented assumptions and contract features.
- Calculate the Greeks needed for the chosen method.
- Size integer contracts and any permitted underlying hedge positions.
- Estimate residual exposures after rounding and constraints.
- Fund the hedge through the existing cash ledger.
- Track mark-to-market, realized P&L, fees, and lifecycle events.
- Reassess the hedge after stock allocation changes and on its own maintenance schedule.
- Report hedge effectiveness under realized and stressed conditions.

A generic put purchase alone is not sufficient unless embedded in the chosen Greek-based policy and its documented target. Do not label a hedge delta-neutral or gamma-neutral merely because it reduces a Greek.

#### Expiry coordination

Support listed expiry selection relative to scheduled stock rebalancing. A proposed policy is expiry beyond the next planned rebalance with a configurable buffer, then close/roll/resize at review. Final policy is decided with the hedge method.

Do not fabricate an expiry on an arbitrary rebalance date. Early risk-triggered stock changes must prompt hedge review. Expiration/settlement may alter underlying shares or cash; record those effects.

#### Data modes

Historical-market mode uses historical contracts and quotes. Synthetic mode uses an explicitly specified pricing/volatility/execution model. Synthetic mode can be a documented feasibility fallback if approved, not an unannounced substitute or proof of attainable historical execution.

Implied volatility is contract/expiry/strike-specific where available. Preserve its source and distinguish observed IV from assumed volatility.

Tests: known payoff and pricing cases, Greek sign/unit checks, finite-difference validation where appropriate, hedge sizing, residual exposure, integer contracts, insufficient cash, missing contracts, and complete opening-to-close/expiry reconciliation.

### 5.12 `src/stress_portfolio/backtest.py` — Chronological orchestration

Coordinate modules without reimplementing their formulas. Maintain estimation, decision, order, fill, valuation, and lifecycle timestamps.

At each event, process the defined corporate-action/settlement order, value positions, update available information, make eligible decisions, execute orders, apply fills, and record state. Document event ordering rather than relying on accidental loop order.

Include stock decisions, hedge maintenance, option expiry, dividends, and daily valuation as relevant events. Do not equate hedge adjustment frequency with the stock-rebalancing schedule.

Benchmarks:

1. Initial buy-and-hold equity portfolio.
2. Scheduled equal-weight equity rebalancing.
3. Conventional risk-optimized portfolio.
4. Stress-aware equity allocation without hedge as an ablation.
5. Complete stress-aware strategy with mandatory chosen derivative hedge.

Unhedged comparisons are valid benchmarks; they do not replace the required final hedged strategy. Use comparable capital, dates, and cost assumptions; report gross and net results.

Outputs: daily NAV, holdings, cash, trades, itemized costs, decision logs, hedge exposures/events, realized risk, benchmark results, and saved settings.

Tests: no look-ahead, deterministic short runs, correct event order, cash/NAV reconciliation, benchmark comparability, and final strategy cannot silently omit hedging.

## 6. Entry point and dashboard

### `src/stress_portfolio/main.py`

Keep command-line orchestration thin. Preserve current selection flags where possible. Add explicit workflow selection as functionality grows, without creating another top-level project or independent application entry point for every module.

A selection-only run may be used during development. A final full run must check prerequisites, including the selected and enabled hedge policy. Reject incomplete settings with clear messages.

### `dashboard/app.py`

Use a single cohesive interface with tabs initially:

- Overview: NAV, cash, actual holdings, returns, costs, hedge status.
- Selection: membership snapshot, eligibility, shortlist, scores, reasons.
- Risk: VaR, expected shortfall, drawdowns, correlations, exposures.
- Stress: historical/hypothetical scenarios, attribution, hedge comparison.
- Optimization: frontier, stress tradeoffs, candidates, constraints, turnover.
- Rebalancing: current/target weights, proposed orders, reasons, costs.
- Hedging: chosen method, instruments, Greeks, targets/residuals, expiry, rolls, P&L.
- Backtest: equity curves, drawdowns, costs, stock count, benchmark comparison.

Call backend functions; do not duplicate financial calculations. Exploratory sliders must not silently mutate official portfolio state or overwrite a saved backtest. Show timestamps, data mode, assumptions, and errors.

### `dashboard/charts.py`

Own reusable Plotly chart functions taking already calculated data. Keep calculations that affect financial results in `src/stress_portfolio/` modules. Avoid one-off chart implementations scattered across the app.

## 7. Shared tables and units

Prefer a small consistent data contract:

| Table | Key fields |
|---|---|
| Membership | snapshot date, ticker, provider ticker, company identifier, sector, source |
| Bars | date, ticker, prices, adjusted price, volume |
| Corporate actions | event date, ticker, event type, value |
| Selection | date, ticker, eligibility, features, scores, reason, target weight |
| Holdings | timestamp, instrument ID/type, quantity, price, value |
| Option contracts | contract ID, underlying, type, strike, expiry, multiplier, conventions |
| Orders/fills | IDs, timestamps, instrument, side, quantity, reference/fill price, charges, status |
| Portfolio history | timestamp, cash, equity value, derivative value, NAV |
| Stress | scenario, horizon, instrument P&L, total loss, assumptions |
| Hedge events | timestamp, method, pre/post Greeks, targets, trades, residuals, reason |
| Decisions | timestamp, action, targets, costs, benefit, explanation |

Use decimal returns internally; convert to percentages for display. Define signed quantities and long/short conventions. Option Greek units and multipliers must be explicit. Do not mix per-share Greeks with portfolio-dollar exposures without conversion.

## 8. Reproducible run artifacts

Save all results in `outputs/runs/<run_id>/`; do not create per-module output roots.

Typical artifacts:

- Resolved configuration and run metadata.
- Membership snapshot/source metadata.
- Input dataset references or preserved snapshots.
- Download failures and data-quality report.
- Eligibility, shortlist, core, reserves, and sector quotas.
- Target/actual weights and holdings history.
- Orders, fills, fees, cash flows, and daily NAV.
- Risk and stress results with assumptions.
- Frontier/candidate allocations and solver status.
- Rebalance decisions.
- Hedge positions, Greeks, events, valuation mode, and P&L.
- Benchmark comparisons and summary report.

Record data cutoff, membership date, source versions where feasible, random seeds, and code revision if available. Generate unique run IDs and do not overwrite completed runs.

## 9. Development sequence and completion gates

1. Data and selection: reproducible 100-stock shortlist and 10/90 split with explanations.
2. Portfolio accounting, costs, execution: funded trades and reconciled cash/NAV.
3. Risk and basic dashboard: validated metrics and inspectable holdings.
4. Stress: reconciled historical and hypothetical losses.
5. Optimization: feasible cost-aware candidates and correctly labeled frontier.
6. Scheduled rebalancing and preliminary backtest: chronological equity baseline.
7. Hedge decision gate: user selects delta or gamma approach after a documented exposure/data/mandate discussion.
8. Mandatory hedging: validated pricing, Greeks, sizing, execution, lifecycle, and exposure tracking.
9. Integrated hedged backtest: realistic hedge events and cost-adjusted comparisons.
10. Final dashboard, stress/hedge comparisons, documentation, and end-to-end validation.

The project is not complete at the equity-only stage. Conformal prediction, Kelly sizing, ML forecasts, tax lots, multi-currency, and live trading are not dependencies of these milestones. Do not add them without a specific request.

## 10. Final definition of done

The project is complete when a user can run a documented uv-based workflow to obtain or reload the data, construct the 100-stock universe and initial 10-stock allocation, maintain funded stock/derivative holdings, measure risk, stress test, generate and execute cost-aware rebalances with variable stock count, maintain the chosen delta- or gamma-based hedge, evaluate the complete strategy chronologically, compare credible benchmarks, and inspect/export the results in the dashboard.

All of this must use the single standardized structure, Python source files rather than notebooks, an authoritative shared ledger, tested modules, and transparent assumptions. The research conclusion must describe the protection–return–cost tradeoff honestly, including when the method fails.
