# Data dictionary

All returns are **decimal** (0.01 = 1%). "Monthly" means calendar months. File names in
`results/` are prefixed with the data mode (`synthetic_` or `real_`). Research-size synthetic
outputs are written under `results/research/`.

## Inputs

| Path | Content | Provenance |
|---|---|---|
| `data/synthetic/<size>/prices_daily.csv` | date, symbol, open, high, low, close, volume | SYNTHETIC, seeded (`provenance.json`) |
| `data/synthetic/<size>/stock_meta.csv` | ticker, industry, shares, true beta/idio vol, list/delist dates, delisting return | SYNTHETIC |
| `data/synthetic/<size>/fundamentals.csv` | ticker, fiscal_period_end, publication_date, vintage_date, values, is_restatement | SYNTHETIC |
| `data/synthetic/<size>/academic_*.csv` | factors (MKT_RF…MOM, RF), 25 portfolios, 6 funds (raw returns) | SYNTHETIC; true model in `provenance.json` |
| `data/synthetic/<size>/truth_*.csv` | true latent value/quality and true monthly style factor returns | SYNTHETIC (for diagnostics only) |
| `data/french/raw/*.zip` + `.meta.json` | French FF3, FF5 (2×3), Momentum, 25 Size/BM portfolios | URL, retrieval time (UTC), SHA-256, units = percent |
| `data/market/etfs/*.csv` | adjusted daily OHLCV of SPY, IWD, IWN, MTUM, QUAL, USMV | yfinance, `_provenance.json` |
| `data/market/stocks/*.csv` | adjusted daily OHLCV of current US large caps | yfinance, `_provenance.json` (failures listed) |
| `configs/real_universe.csv` | ticker, current sector | hand-written; survivorship-biased |

## Monthly panel columns (`panel.build_monthly_panel` + `characteristics.add_characteristics`)

| Column | Meaning | Known at t? |
|---|---|---|
| `date` (index) | formation close (last session of month t) | — |
| `month` | calendar month of `date` | yes |
| `price` | adjusted close at `date` | yes |
| `ret` | return over month t | yes |
| `ret_fwd`, `return_date` | return over month t+1 and its month-end | **no (target)** |
| `rf_fwd`, `ret_fwd_excess` | rf of month t+1; excess `ret_fwd` | **no (target)** |
| `exec_eligible` | next session after `date` | calendar knowledge |
| `industry`, `mcap` | static industry; price × shares (synthetic only) | yes |
| `f_*`, `fund_*` | as-of-joined fundamentals and their timestamps | yes (asserted) |
| `<c>_raw`, `<c>` | raw and processed (winsorised + z-scored per date) characteristic | yes |
| `q_roe`, `q_gpa`, `q_neg_leverage` | quality components | yes |

## Main output tables (`results/tables/`)

| File | Content |
|---|---|
| `*_01_factor_regressions.csv` | alpha and loadings, HAC SE/t/p/CI, R², nobs, sample dates per asset × model |
| `*_04_grs.csv` | GRS statistic, p, df, T/N/L, condition numbers, sample, assumptions |
| `*_06_factor_tearsheets.csv` | long-short mean, vol, Sharpe, HAC t, max DD, turnover per signal × weighting |
| `*_06_sort_holdings.csv`, `*_06_sort_turnover.csv` | quintile holdings and one-way turnover |
| `*_07_long_short_returns.csv` | monthly Q5−Q1 returns |
| `*_08_published_factor_comparison.csv` | correlation with French (real) or true (synthetic) factors |
| `*_09_fama_macbeth.csv` | premia, HAC SE/t/CI, months, avg N, avg R², hurdle flag |
| `*_09_fama_macbeth_linearmodels_check.csv` | own vs linearmodels estimates and SEs |
| `synthetic_10_factor_returns.csv` | monthly cross-sectional factor returns |
| `synthetic_12_risk_forecasts_ex_ante.csv` | forecasts only (made before outcomes) |
| `synthetic_12_risk_forecasts_realised.csv` | forecasts joined with realised returns, z |
| `synthetic_12_bias_statistics.csv` | bias by portfolio group and split, CIs |
| `synthetic_14_te_selection_validation.csv` | TE grid, validation IR, selected flag |
| `synthetic_14_backtest_daily_returns.csv` | backtester daily returns (net) per strategy |
| `synthetic_15_tearsheet.csv` | validation/OOS metrics, costs and traded notional |
| `synthetic_16_active_exposures.csv` | active style exposures of targets per rebalance |
| `synthetic_17/18/19_*.csv` | total, stock-level and active risk attribution |
| `synthetic_20_target_vs_realised_opt.csv` | target vs realised weights, residual cash, untradeable weight |

## Logs (`results/logs/`)

The logs folder holds:

* the Fama–MacBeth specification log and dropped months;
* the risk-model estimation log;
* the optimiser log (status, solver, attempts, every post-solve check);
* the portfolio trial log (every TE evaluated);
* the research-run stdout.

The trial logs are append-only: re-running the pipeline or notebooks appends new rows.
`results/run_metadata.json` records, per mode:

* versions, seeds and the full configuration;
* data provenance and splits;
* hold-out status, per-stage runtimes and solver outcomes.
