# Risk-model conventions

Σ_t = B_t F_t B_t′ + D_t. All quantities are **monthly** (variance per month). Annualise by
multiplying variance by 12 or volatility by √12.

## B — exposures (known at the formation close t)

* `market`: a column of ones.
* `ind_<k>`: one dummy per industry. Industry labels are static: synthetic labels, or current
  sectors for real data.
* Styles `size, value, momentum, low_vol, quality`: the raw characteristic is winsorised
  (1/99) per date, then standardised to a **cap-weighted mean of 0** and an
  **equal-weighted std of 1**. Under this Barra-style convention, the cap-weighted market
  portfolio has zero style exposure. A missing style exposure is set to 0 (the cap-weighted
  mean). A month where an entire style is unavailable (for example, the first 12 months for
  momentum) is skipped and logged rather than estimated with a zero column.

## Identification

Intercept + all industry dummies is collinear. Two identifications are implemented:

| Option | How | Interpretation |
|---|---|---|
| `cap_weighted_sum_zero` (default) | keep all K industries, impose Σ c_k g_k = 0 (cap weights) by substituting g_K = −Σ_{k<K} c_k g_k / c_K | `market` ≈ the cap-weighted market return; industry factors are deviations from it |
| `reference` | drop the alphabetically first industry | intercept = reference industry at zero style exposure; other industries relative to it |

Both bases give identical fitted values, residuals and Σ. The tests show:

* residuals agree to 1e-12;
* Σ agrees to a relative 1e-6;
* the constraint holds to 1e-12.

**Individual factor attributions depend on the basis.** "How much risk is from the market
factor" changes when the basis changes. Σ, total risk and the stock-level contributions do not.

## Estimation timing and WLS

For each formation date t, `ret_fwd` (month t+1) is regressed on B_t. The result is labelled
with the realisation date t+1, so a forecast made at t uses factor returns up to t only.

Regression weight: **w_i = √(market cap_i) is the objective weight** in min Σ w_i e_i². It is
solved by multiplying the rows of y and X by √w_i = cap_i^¼. A test checks the result against
`statsmodels.WLS(weights=w)` and shows that using √cap as the *row multiplier* gives a
different (cap-weighted) estimator. The guide's phrase "weighting by √market-cap" is ambiguous
on this point.

## F — factor covariance

Single half-life EWMA (default 18 months). The decay is λ = 0.5^(1/h): an observation h
months old has half the weight of the newest. Weights are normalised to sum to 1. The mean is
**not subtracted** by default (configurable). At least 24 observations are required (no
initialisation from a prior). F is symmetrised and checked for PSD. Eigenvalues below 1e-12
are floored, and the count is recorded in the diagnostics; in the runs here, none were floored.
The guide suggests a 90-day half-life with daily data. This project estimates **monthly** factor
returns to match the monthly rebalance and forecast horizon, so the half-life is in months.
Daily estimation with horizon scaling is not implemented.

## D — specific variance

* EWMA of squared residuals per stock, using the same decay rule (half-life 18 months).
* **Linear shrinkage** toward the mean of the stock's size tercile:
  var = s·own + (1−s)·target, with s = n/(n+12).
* Stocks with fewer than 6 residuals use the bucket mean.
* Floor: 2% monthly specific volatility.

This is a transparent heuristic, **not** a Bayesian posterior. A diagonal D assumes residuals
are uncorrelated across stocks after the factors. Same-industry residual correlation not
captured by the industry factors is therefore missed.

## Forecast horizon

Forecasts are for the next calendar month. They are validated against the realised next-month
return of a portfolio with frozen weights. See [statistical_validation.md](statistical_validation.md).
