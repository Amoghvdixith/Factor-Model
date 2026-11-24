# Limitations

**This is educational research software.** It is not a commercial Barra or Axioma
replica, not production-ready, and not investment advice. No result here shows that any
strategy is profitable.

## Data

* **Synthetic results are about the software, not markets.** The synthetic premia, factor
  structure and noise levels were chosen by the author of the generator. Recovering them only
  shows the estimators behave as intended.
* **No real point-in-time fundamentals.** Value and quality on real stocks were **not run**.
  A schema and loader exist for user-supplied historical extracts with publication or vintage
  dates. A fixed reporting lag applied to revised data would not make it point-in-time.
* **Real stock universe is survivorship-biased:** current large caps, current sector labels,
  no delisted firms. yfinance returned nothing for MMC and BK, and only 2026 data for EA and
  EQR. Price-only results (momentum, low volatility) are biased toward firms that survived
  and grew.
* No market caps for real stocks: value weighting, size and the risk model are not run in real
  mode.
* The ETF sample is short: MTUM and QUAL start in 2013 and USMV in 2011, so alphas have wide
  intervals.
* French and ETF regressions are US-only, in USD. No currency conversion is performed.

## Statistics

* HAC inference is asymptotic. GRS assumes i.i.d. normal residuals.
* Fama–MacBeth slopes measure predictive association with characteristics. They are not
  estimates of risk premia for priced factors.
* Only 6 specifications (×2 for industry controls) were run per mode. All are logged. The
  |t| > 3 hurdle is illustrative.
* Synthetic fast mode has about 90 months of characteristic returns. Most premia are
  statistically indistinguishable from zero there, even though non-zero premia exist by
  construction. Research mode (about 190 months) recovers some of them; see the README.

## Risk model

* Monthly estimation only, single half-life EWMA, no Newey–West adjustment of F, no
  volatility-regime or eigenfactor adjustment, and diagonal specific risk.
* Static industry labels; heuristic (non-Bayesian) specific-risk shrinkage.
* Individual factor contributions depend on the identification basis.
* Bias statistics are descriptive, with wide confidence intervals for one portfolio over
  about 50–90 months.

## Optimisation and backtest

* The scores are not expected returns. γ (risk aversion) has no utility meaning, and the
  tracking-error constraint does most of the work.
* Estimation risk: the optimiser treats Σ̂ as true. The realised tracking error of the
  optimised tilt was close to the ex-ante budget in fast mode (2.0% vs 2%), but 5.1% vs a 6%
  budget in research mode.
* The TE budget was chosen on validation using model-implied (close-to-close) returns, not
  backtester returns.
* Pre-trade weights for the turnover constraint come from the model path, not the engine's
  realised holdings. Delisted holdings cannot be sold by the engine and stay frozen (about 2–4%
  of equity by the end of the samples).
* Costs are a flat 10 bps one-way. There is no market impact, borrow or financing (long-only),
  and no tax.
* Long-short factor returns in the sort analysis are frictionless academic returns.
* The protected hold-out has **not** been evaluated.
