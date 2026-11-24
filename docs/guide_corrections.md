# Corrections and clarifications to the supplied build guide

The guide (`04_Factor_Model_Guide.pdf`) was used as a specification, but checked rather than
copied. Where this implementation differs, the reason is below.

| # | Guide | Issue | What this project does |
|---|---|---|---|
| 1 | GRS = (T/N)((T−N−L)/(T−L−1)) α̂′Σ̂⁻¹α̂/(1+μ̄′Ω̂⁻¹μ̄) | Correct only if Σ̂ uses divisor T−L−1 **and** Ω̂ uses divisor T; the usual `np.cov`/pandas Ω̂ (T−1) gives a slightly wrong statistic | MLE form ((T−N−L)/N)·…, both divisors T; equivalence and non-equivalence tested; N=1 equals squared OLS t |
| 2 | "the 5-factor model should reduce, but not eliminate, GRS" | An empirical expectation, not a rule | Nothing is forced; on real data (1963-07 to 2026-09) FF5 = 3.21 vs Carhart4 3.20 vs FF3 3.63 |
| 3 | Fama–MacBeth "to estimate factor risk premia"; Shanken mentioned | Characteristic slopes are not risk premia; Shanken applies to estimated betas | Labelled "characteristic premia (predictive slopes)"; Shanken not applied; beta-pricing two-pass not implemented |
| 4 | "assume annual figures are available about 4 to 6 months after fiscal year end" | A lag on revised data is not point-in-time; restatements leak | Vintage-aware as-of join; assumed vs observed availability recorded; real fundamentals not used because none are available |
| 5 | B = "z-scores plus industry dummies and a market column" | Market column + all industry dummies is collinear | Cap-weighted sum-to-zero industry constraint (default) or reference industry; Σ invariance tested; attribution basis dependence documented |
| 6 | "weighting by √market-cap" | Ambiguous: objective weight vs row multiplier | √cap is the objective weight; rows scaled by cap^¼; tested against statsmodels WLS |
| 7 | `factor_tilt` has "a tracking-error budget" in the text but no TE constraint in the code | The code penalises *total* risk, not active risk | Explicit SOC constraint ‖[F^½B′(w−w_b); D^½(w−w_b)]‖ ≤ TE |
| 8 | `cp.norm1(w - w_prev) <= 2*max_turnover` | Fine as one-way 20%, but `w_prev` must be the *drifted* pre-trade weights, and first deployment/forced exits need a rule | Drifted weights; initial deployment exempt; forced exits exempt; post-solve check |
| 9 | `factor_tilt` returns `w.value` regardless of status | Can silently return None or an inaccurate solution | Status checked; post-solve constraint checks; second solver; predeclared hold-drifted fallback; all logged |
| 10 | `euler_contributions` divides by σ again "# sums to 1" | It returns *shares* while the formula above it defines contributions that sum to σ | Both reported, explicitly labelled (vol contribution, variance contribution, % of σ) |
| 11 | "Bias statistic within [0.8, 1.2] on random portfolios" listed as a test pass criterion | Empirical calibration is not a software property | Reported as a diagnostic with CIs; never a unit test |
| 12 | "Custom HML vs French HML correlation > 0.5" as a pass criterion | Empirical, and impossible to check without point-in-time book data | Correlations reported as diagnostics (real momentum vs French UMD: 0.86) |
| 13 | F half-life ≈ 90 days | The guide implies daily factor returns | Monthly estimation matching the monthly horizon; half-life 18 months, configurable |
| 14 | "Bayesian shrinkage" of specific risk | No prior or likelihood specified | Called linear shrinkage toward the size-bucket mean |
| 15 | Statsmodels HAC "matches to 1e-10" | Only with matching `use_correction` | Tested both ways |
| 16 | `.github/workflows/ci.yml` in the layout | Out of scope (no Git or GitHub in this build) | Not created |
