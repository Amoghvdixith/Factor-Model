# Interview notes (tied to this implementation)

These notes describe what the code does and what was observed. They are not claims of
investment performance.

**1. Time-series vs cross-sectional tests.** A time-series test (`timeseries.py`) regresses a
portfolio's excess return on *traded* factor returns. The intercept is the pricing error, and
GRS tests all N intercepts jointly. A cross-sectional test (`fama_macbeth.py`) asks whether
stock *characteristics* (or estimated betas) explain the cross-section of next-month returns.
Here the slopes are characteristic premia; there are no estimated betas. One test asks "do the
factors price these portfolios?", the other "does this characteristic predict returns?".

**2. Newey–West and the lag.** Monthly residuals can be heteroskedastic and autocorrelated,
and OLS SEs ignore that. The default is floor(4(T/100)^(2/9)): 6 for T = 759, 4 for about
100 months. The implementation matches statsmodels to 1e-10, and the tests pin which
small-sample correction is used. The lag is a bias–variance choice. With 60–160 months the
SEs are noisy whatever lag is chosen.

**3. What Fama–MacBeth solves and does not solve.** The time-series spread of the monthly
slopes absorbs cross-sectional residual correlation. Applying HAC to that slope series
handles autocorrelation. It does not solve errors-in-variables for estimated betas (Shanken),
data snooping, survivorship bias, or look-ahead in the inputs. It also does not turn a
predictive slope into a risk premium. In the real price-only run, low volatility had a
*negative* slope (t = −2.64, 166 months). That says something about this survivorship-biased
2010–2024 large-cap sample, not about a universal premium.

**4. Why profitability and investment were added; what happened to HML.** In the FF5 regressions
of the real 25 portfolios, adding RMW and CMA changed the alpha pattern (see
`real_04_grs_and_25_portfolio_alphas.png`). The GRS statistic moved from 3.63 (FF3) to 3.21
(FF5), about the same as Carhart4 (3.20), and all reject. In the 2000–2026 ETF regressions,
IWD and IWN load positively on HML, as their mandates suggest. Their FF5 alphas are negative
(about −1.7% and −1.4% a year) with confidence intervals that exclude zero in this sample.

**5. Is a new factor genuinely new?** Regress its returns on existing factors; the alpha is
what is new. Check correlations, and whether it survives in a multivariate Fama–MacBeth with
the existing characteristics. Report every specification tried (the logs here do). Use a
higher hurdle (|t| > 3 is flagged), and hold out data that is used once (`run_holdout.py`
requires `--confirm`; not yet run). The custom momentum factor correlated 0.86 with French
UMD over 2011–2024. That high correlation, despite a different universe and survivorship
bias, suggests the construction is right.

**6. Barra-style risk forecast.** B_t holds exposures known at the close of t; each month a
WLS cross-sectional regression gives factor returns f and residuals u; F is the EWMA
covariance of f (18-month half-life); D is shrunk EWMA residual variance. Then
σ²_p = x′Fx + w′Dw with x = B′w. Identification matters: market plus all industries needs a
constraint. The default is cap-weighted industries summing to zero; Σ is invariant to that
choice, individual attributions are not.

**7. Attribution that adds up.** Euler: σ is homogeneous of degree 1 in w, so
σ = Σ_i w_i ∂σ/∂w_i. Factor contributions are x_k(Fx)_k/σ and the specific contribution is
w′Dw/σ. Tests check the sum to 1e-12. Contributions can be negative: in the last OOS
rebalance (synthetic) the size factor had a negative contribution because the tilt was
short-size relative to the market factor's covariance. For tracking error, replace w by
w − w_b. This is *risk* attribution, not P&L attribution.

**8. Is the value premium dead?** This project cannot answer that. The real-data mode has no
point-in-time book equity, so no real value factor was built. The honest workflow is: get
point-in-time data, test out of sample, account for costs and multiple testing, and accept
inconclusive answers. A decade of underperformance is a small sample for a factor with roughly
10% annual volatility. Ten years gives a standard error of about 3% a year on the mean.

**Things that went wrong or surprised me (true for this build):**

* An early version forward-filled prices *across symbols* when computing realised weights. A
  −7% "residual cash" exposed it, and it was fixed.
* The engine cannot sell delisted names. That trapped value is now reported rather than hidden.
* The guide's GRS formula needs matching divisors to be right.
* In fast synthetic mode, none of the planted premia were statistically detectable in about
  90 months. Detection is a sample-size question, not a coding one.
