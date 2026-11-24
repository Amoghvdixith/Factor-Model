"""Ex-ante risk attribution (Euler) and risk-forecast diagnostics.

For weights w (or active weights w - w_b for tracking error)::

    x = B' w                       factor exposures
    var = x' F x + w' D w          total variance (factor part + specific part)
    sigma = sqrt(var)

Volatility contributions (sum exactly to sigma)::

    RC_k        = x_k (F x)_k / sigma      for each factor k
    RC_specific = w' D w / sigma
    MCTR_i      = (Sigma w)_i / sigma      marginal contribution of stock i
    RC_i        = w_i MCTR_i               stock component contribution (sum to sigma)

Variance contributions are the same quantities multiplied by sigma (they sum to var).
Percentage shares are RC / sigma (sum to 1).  Contributions can be negative (a hedge);
they are never clipped.  This is attribution of *predicted risk*; it says nothing about
realised profit and loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .risk_model import RiskModel, align_weights


class AttributionError(ValueError):
    """Raised for zero risk or misaligned inputs."""


@dataclass
class RiskAttribution:
    sigma: float                       # monthly volatility of w (or active w)
    variance: float
    factor_exposure: pd.Series         # x = B'w
    factor_vol_contrib: pd.Series      # RC_k
    specific_vol_contrib: float
    stock_mctr: pd.Series
    stock_vol_contrib: pd.Series
    label: str = "total"               # "total" or "active"

    def table(self) -> pd.DataFrame:
        """Long table: source, vol contribution, variance contribution, share of sigma."""
        vc = pd.concat([self.factor_vol_contrib,
                        pd.Series({"specific": self.specific_vol_contrib})])
        return pd.DataFrame({
            "exposure": pd.concat([self.factor_exposure, pd.Series({"specific": np.nan})]),
            "vol_contrib_monthly": vc,
            "var_contrib_monthly": vc * self.sigma,
            "pct_of_sigma": vc / self.sigma,
            "vol_contrib_annual": vc * np.sqrt(12),
        }).rename_axis("source")

    def grouped(self, groups: dict[str, list[str]]) -> pd.Series:
        """Sum factor contributions into groups (e.g. all industries), keep specific."""
        out = {}
        used: set[str] = set()
        for g, cols in groups.items():
            cols = [c for c in cols if c in self.factor_vol_contrib.index]
            out[g] = float(self.factor_vol_contrib[cols].sum())
            used |= set(cols)
        for c in self.factor_vol_contrib.index:
            if c not in used:
                out[c] = float(self.factor_vol_contrib[c])
        out["specific"] = self.specific_vol_contrib
        return pd.Series(out)


def euler_attribution(rm: RiskModel, w: pd.Series, benchmark: pd.Series | None = None,
                      zero_tol: float = 1e-14) -> RiskAttribution:
    """Euler decomposition of total risk (or active risk if ``benchmark`` is given)."""
    wv = align_weights(w, rm.assets)
    if benchmark is not None:
        wv = wv - align_weights(benchmark, rm.assets)
    B, F, D = rm.B.to_numpy(), rm.F.to_numpy(), rm.D.to_numpy()
    x = B.T @ wv
    Fx = F @ x
    fvar = float(x @ Fx)
    svar = float((D * wv ** 2).sum())
    var = fvar + svar
    if var <= zero_tol:
        raise AttributionError("portfolio risk is zero; contributions are undefined")
    sigma = float(np.sqrt(var))
    mctr = (B @ Fx + D * wv) / sigma
    return RiskAttribution(
        sigma=sigma, variance=var,
        factor_exposure=pd.Series(x, index=rm.F.index),
        factor_vol_contrib=pd.Series(x * Fx / sigma, index=rm.F.index),
        specific_vol_contrib=svar / sigma,
        stock_mctr=pd.Series(mctr, index=rm.assets),
        stock_vol_contrib=pd.Series(wv * mctr, index=rm.assets),
        label="active" if benchmark is not None else "total",
    )


def euler_contributions_dense(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Stock contributions w_i (Sigma w)_i / sigma from a dense covariance (reference)."""
    s = float(np.sqrt(w @ Sigma @ w))
    if s <= 0:
        raise AttributionError("zero risk")
    return w * (Sigma @ w) / s


# --------------------------------------------------------------------------- diagnostics
def bias_statistic(z: pd.Series, demean: bool = False) -> dict[str, float]:
    """Standard deviation of standardised outcomes z = r / sigma_hat.

    ``demean=False`` (default) treats the expected return as zero, the usual risk-model
    convention.  The 95% interval assumes i.i.d. normal z (chi-square for the variance)
    and is a rough guide only.  A value near 1 is consistent with calibration; ranges
    such as [0.8, 1.2] are descriptive heuristics, not formal acceptance tests.
    """
    v = z.dropna().to_numpy(float)
    n = len(v)
    if n < 2:
        return {"n": n, "bias": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    b = float(np.std(v, ddof=1)) if demean else float(np.sqrt(np.mean(v ** 2)))
    df = n - 1 if demean else n
    lo = b * np.sqrt(df / stats.chi2.ppf(0.975, df))
    hi = b * np.sqrt(df / stats.chi2.ppf(0.025, df))
    return {"n": n, "bias": b, "ci_low": float(lo), "ci_high": float(hi),
            "demeaned": demean, "mean_z": float(v.mean())}


def rolling_bias(z: pd.Series, window: int = 12, demean: bool = False) -> pd.Series:
    """Trailing-window bias statistic (uses only past and current outcomes)."""
    if demean:
        return z.rolling(window, min_periods=window).std(ddof=1)
    return np.sqrt((z ** 2).rolling(window, min_periods=window).mean())


def bootstrap_bias_ci(z: pd.Series, reps: int = 2000, seed: int = 0, block: int = 3) -> tuple[float, float]:
    """Moving-block bootstrap 95% interval for the (non-demeaned) bias statistic."""
    v = z.dropna().to_numpy(float)
    n = len(v)
    if n < 2 * block:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    stats_ = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n - block + 1, nb)
        s = np.concatenate([v[a:a + block] for a in starts])[:n]
        stats_[i] = np.sqrt(np.mean(s ** 2))
    return float(np.quantile(stats_, 0.025)), float(np.quantile(stats_, 0.975))
