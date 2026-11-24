# Statistical validation

## HAC (Newey–West) standard errors

`timeseries.newey_west_cov` implements V = (X′X)⁻¹ S (X′X)⁻¹ with Bartlett weights
1 − l/(L+1), plus the optional T/(T−k) correction. A unit test matches statsmodels
`cov_type="HAC"` to below 1e-10, with and without `use_correction`. The default lag is
floor(4 (T/100)^(2/9)); it can be changed in `configs/timeseries.yaml`. HAC inference is
asymptotic, so small samples (for example, ETFs launched in 2013) give wide intervals.

## GRS test — verified form

With T months, N test assets and L traded factors, OLS alphas â, residuals E, factor means μ̄:

    Σ̂ = E′E / T                      (MLE residual covariance)
    Ω̂ = (F − μ̄)′(F − μ̄) / T          (MLE factor covariance)
    W = ((T − N − L)/N) · â′Σ̂⁻¹â / (1 + μ̄′Ω̂⁻¹μ̄)  ~  F(N, T − N − L)

**Correction to the guide.** The guide writes
`(T/N)((T−N−L)/(T−L−1)) â′Σ̂⁻¹â/(1+μ̄′Ω̂⁻¹μ̄)`. That expression is identical to the one above
**only if** Σ̂ uses the unbiased divisor T−L−1 **and** Ω̂ uses divisor T. With Ω̂ on divisor T−1
(the pandas/numpy default) the statistic is slightly wrong. `grs_from_guide_formula` exists so
that the tests can show both facts.

Further tests:

* With N = 1, W equals the squared classical OLS t-statistic of alpha, and the p-value
  matches. This is an algebraic identity.
* T − N − L ≤ 0 raises `GRSError`.
* A singular or ill-conditioned covariance (condition number above 1e10) raises. No
  pseudoinverse is substituted.
* A common sample is enforced across all test assets.
* Factor and test-asset column names must not overlap.

**Assumptions.** GRS is exact only under i.i.d. multivariate-normal residuals and traded
factors. It is not the HAC inference, and the two are reported separately. The project does
not assume FF5 must beat FF3. On real 1963-07 to 2026-09 data (T = 759) the statistics were
CAPM 4.17, FF3 3.63, Carhart4 3.20 and FF5 3.21. FF5 was not below Carhart4, and all four
reject at conventional levels.

## Fama–MacBeth

* Alignment: y must be a forward (t+1) column; characteristics are at t.
* Point estimates equal the mean of per-month OLS slopes (tested).
* Matched against `linearmodels.FamaMacBeth`:
  - point estimates agree to 1e-17;
  - linearmodels' `unadjusted` SE equals std(ddof=1)/√T;
  - its Bartlett `kernel` SE with bandwidth 4 equals the statsmodels HAC(4) SE with the
    T/(T−1) correction.

  These conventions were found empirically and are pinned by a test.
* Months with n < 30 or a rank-deficient design are dropped and logged
  (`results/logs/*_fama_macbeth_dropped_months.csv`). Every specification is logged
  (`*_fama_macbeth_specifications.csv`).
* **Characteristic regressions are not beta-pricing tests.** With observed characteristics
  there is no errors-in-variables problem, so the Shanken correction is not applied. That
  correction belongs to two-pass tests on *estimated* betas, which are not implemented here.
* |t| > 3 is flagged as an illustrative hurdle (Harvey–Liu–Zhu), not a multiple-testing
  correction.

## Risk-forecast validation

* Forecasts are made at month-end t from factor returns and residuals realised by t. They are
  written to `*_12_risk_forecasts_ex_ante.csv` before realised returns are joined
  (`*_12_risk_forecasts_realised.csv`). A test confirms that forecasts do not change when
  later returns are mutated.
* The portfolios are: EW, 20 seeded random long-only (Dirichlet) portfolios, the simple tilt,
  and the optimised targets. Weights are frozen over the month. This is not the same as
  evaluating a changing strategy.
* Bias statistic = √mean(z²), with z = r/σ̂, **not demeaned** (expected return treated as 0).
  The CI uses a chi-square approximation (i.i.d. normal); a block-bootstrap CI is also given.
* [0.8, 1.2] is a descriptive heuristic, not an acceptance test. Calibration is **not** a unit
  test.

## What the unit tests do not require

Positive factor returns, significant premia, outperformance, any particular factor
correlation, or a bias statistic inside any range. These are empirical outcomes, reported but
never asserted.

## Checks run

| Check | Result (this run) |
|---|---|
| `pytest` (offline, incl. a slow end-to-end run on a tiny project copy) | 81 passed, 0 failed, 0 skipped |
| `ruff check .` | no findings |
| `mypy src/factors` | no issues in 18 files |
| coverage (`pytest --cov=factors`, `cli.py` omitted) | 94% of lines |
| notebooks (executed by `scripts/build_notebooks.py`) | 4 executed, 0 cell errors |
