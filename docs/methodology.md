# Methodology

This project keeps three research layers apart. They answer different questions and use
different objects.

| Object | What it is | Where |
|---|---|---|
| Traded academic factor | A published long-short (or market-excess) portfolio *return*, e.g. French HML | `data.py`, `timeseries.py` |
| Stock characteristic | A number per stock and date, e.g. book-to-market at month end t | `characteristics.py` |
| Factor exposure | A standardised characteristic (or a dummy) used as a regressor in the risk model, column of B_t | `risk_model.py` |
| Risk-model factor return | The cross-sectional regression coefficient f_t of stock returns on B_{t-1} | `risk_model.py` |
| Expected-return forecast | **Not produced.** The composite score is a ranking, not a calibrated forecast | `characteristics.composite_score` |
| Risk attribution | Split of *predicted* volatility into factor / specific parts | `attribution.py` |
| Realised return attribution | **Not implemented** (would need returns × exposures over the holding period) | — |

## A. Time-series layer (`timeseries.py`)

For a fund or portfolio i and month t:

    r_i,t - rf_t = alpha_i + beta_i' f_t + e_i,t

* `f_t` are traded factors as published: MKT_RF is already an excess return; SMB, HML, RMW,
  CMA and MOM are zero-investment long-short returns and **never** have rf subtracted.
* Models: CAPM, FF3, **Carhart4 = FF3 + MOM**, FF5, and FF5 + MOM (labelled an extension,
  not Carhart). FF5-family models take MKT_RF/SMB/HML from the 5-factor file because SMB is
  constructed differently in the two files.
* Inference: OLS point estimates with Newey–West (Bartlett) HAC standard errors,
  `L = floor(4 (T/100)^(2/9))` by default (configurable), with the T/(T−k) small-sample
  correction; t-statistics use the normal approximation.
* Rolling betas: trailing 36-month OLS; the value at t uses observations t−35…t only.
* GRS: see [statistical_validation.md](statistical_validation.md).

## B. Cross-sectional layer

**Panel** (`panel.py`): a long `(date, ticker)` panel where `date` is the close of the last
session of each month. `ret_fwd` (month t+1) and `return_date` are targets only. Fundamentals
are attached with an availability-aware as-of join ([point_in_time_policy.md](point_in_time_policy.md)).

**Characteristics** (`characteristics.py`), all at the formation close t:

| Name | Definition | Invalid / missing policy |
|---|---|---|
| value | latest available book equity / market cap at t (E/P alternative) | book ≤ 0 or cap ≤ 0 → NaN |
| momentum | ∏(1+r) − 1 over months t−11 … t−1 (end of t−12 to end of t−1; month t skipped) | all 11 returns required |
| low_vol | −(std of daily returns over 252 sessions to t) × √252 | ≥ 126 returns required |
| quality | mean of per-date z-scores of ROE, GP/A, −debt/assets | ≥ 2 of 3 components; book ≤ 0 → ROE NaN; assets ≤ 0 → NaN |
| size | log market cap | cap ≤ 0 → NaN |

Processing per date: winsorise at the 1st/99th percentiles of that date, then z-score
(mean/std of that date only; ddof = 1). Dates with fewer than 20 valid values or zero
dispersion give NaN and are logged. Sector-relative processing (demeaning within sector) is
optional. Raw (`<name>_raw`) and processed values are both kept.

**Quintile sorts** (`sorts.py`): groups at t by `rank(method="first")` on a ticker-sorted
cross-section (deterministic ties, sizes differ by ≤ 1). EW and VW (cap at t) returns over
t+1. Top-minus-bottom = Q5 − Q1. Missing next returns are dropped with renormalisation and
logged. Turnover is one-way against drifted weights. These are frictionless academic
returns, not investable returns, and not replications of official French factors.

**Fama–MacBeth** (`fama_macbeth.py`): each month, OLS of `ret_fwd_excess` on characteristic
z-scores at t (optionally with industry dummies, one reference industry dropped). Months with
n < 30 or a rank-deficient design are dropped and logged. Inference: HAC on the mean of the
monthly slopes. Cross-checked against `linearmodels.FamaMacBeth`.

## C. Portfolio-risk layer

* Risk model Σ_t = B_t F_t B_t′ + D_t: [risk_model_conventions.md](risk_model_conventions.md).
* Attribution (`attribution.py`): Euler decomposition of volatility; active risk uses
  w − w_b; contributions are signed and add up exactly.
* Optimiser (`optimiser.py`): cvxpy, long-only, fully invested, max weight 5%, active style
  exposure bounds, ±5% industry active weights, an explicit tracking-error second-order cone,
  20% one-way turnover against drifted weights (initial deployment exempt; forced exits
  exempt), CLARABEL then SCS on the same problem, post-solve constraint checks, and a
  predeclared hold-drifted-weights fallback. Scores are composite z-scores, so the objective is
  a **score–risk trade-off**, not mean–variance utility.
* Walk-forward (`portfolio.py`): three strategies are compared: EW benchmark, simple tilt
  (top composite quintile, EW) and the optimised tilt. Targets formed at month-end t use only data to t.
* Execution: the project-02 backtester ([backtester_integration.md](backtester_integration.md)).

## Splits and the hold-out

The monthly calendar is split contiguously into train 45% / validation 20% / OOS 20% /
hold-out 15% (`configs/data.yaml`). `run_all.py` physically removes hold-out months from the
daily data before building anything, so even a forward return realised in the first hold-out
month is unavailable. Only one choice is tuned: the tracking-error budget, chosen from a
predeclared grid {2%, 4%, 6%} by validation-period information ratio of model-implied net
returns, ties going to the smaller TE. The risk model needs a burn-in, which is taken from the
training period. OOS results are reported separately. `scripts/run_holdout.py --confirm` is
the only path to the hold-out, and it has **not** been run.
