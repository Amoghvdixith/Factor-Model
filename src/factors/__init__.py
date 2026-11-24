"""factors - an educational equity factor-model research package.

Three separate layers:

* time-series factor regressions of portfolio/fund returns on traded academic factors
  (:mod:`factors.timeseries`);
* cross-sectional characteristics, quintile sorts and Fama-MacBeth regressions
  (:mod:`factors.characteristics`, :mod:`factors.sorts`, :mod:`factors.fama_macbeth`);
* a fundamental factor risk model, risk attribution and constrained optimisation
  (:mod:`factors.risk_model`, :mod:`factors.attribution`, :mod:`factors.optimiser`).

Research software only - not a commercial risk model and not an investment system.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
