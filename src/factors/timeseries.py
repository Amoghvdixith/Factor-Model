"""Time-series factor regressions, HAC inference, rolling betas and the GRS test.

Conventions
-----------
* Left-hand side: *excess* returns of the test asset, ``r_i - rf`` (same month).
* Right-hand side: traded factors as published.  ``MKT_RF`` is already an excess
  return; long-short factors (SMB, HML, RMW, CMA, MOM) are zero-investment returns and
  must NOT have the risk-free rate subtracted again.
* All returns are decimal and monthly unless stated.
* HAC (Newey-West, Bartlett kernel) standard errors use ``hac_lags`` lags; the default
  rule is ``floor(4 (T/100)^(2/9))``.  ``small_sample=True`` applies the T/(T-k)
  degrees-of-freedom correction used by statsmodels' ``use_correction=True``.
* GRS is the *classical* finite-sample test, which assumes i.i.d. multivariate-normal
  residuals; it does not use the HAC covariance.  The two are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

MODEL_NOTES = {
    "Carhart4": "Carhart (1997) four-factor model: FF3 + momentum (MOM/UMD)",
    "FF5_plus_MOM": "FF5 + momentum: an extension, NOT the standard Carhart model",
}


class GRSError(ValueError):
    """Raised when the classical GRS test is not defined for the given sample."""


def hac_lags_auto(T: int) -> int:
    """Newey-West (1994) rule of thumb ``floor(4 (T/100)^(2/9))``."""
    if T <= 0:
        raise ValueError("T must be positive")
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def resolve_lags(spec: Any, T: int) -> int:
    return hac_lags_auto(T) if spec in (None, "auto") else int(spec)


def excess_returns(raw: pd.Series | pd.DataFrame, rf: pd.Series) -> pd.Series | pd.DataFrame:
    """Subtract the same-period risk-free rate; raise if rf is missing for any period.

    Use only for raw asset returns - never for long-short factors or MKT_RF.
    """
    if raw.attrs.get("is_excess"):
        raise ValueError("returns are already excess returns; refusing to subtract rf twice")
    idx = raw.index
    missing = idx.difference(rf.dropna().index)
    if len(missing):
        raise ValueError(f"risk-free rate missing for {len(missing)} periods, e.g. {missing[0]}")
    out = raw.sub(rf.reindex(idx), axis=0) if isinstance(raw, pd.DataFrame) else raw - rf.reindex(idx)
    out.attrs["is_excess"] = True
    out.attrs["units"] = "decimal"
    return out


def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int,
                   small_sample: bool = True) -> np.ndarray:
    """HAC covariance of OLS coefficients with a Bartlett kernel.

    ``V = (X'X)^-1 S (X'X)^-1`` with ``S = sum_t u_t u_t' + sum_{l=1}^{L} w_l
    sum_t (u_t u_{t-l}' + u_{t-l} u_t')``, ``u_t = x_t e_t``, ``w_l = 1 - l/(L+1)``.
    """
    X = np.asarray(X, float)
    e = np.asarray(resid, float)
    T, k = X.shape
    u = X * e[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = u[lag:].T @ u[:-lag]
        S += w * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    if small_sample:
        V *= T / (T - k)
    return V


@dataclass
class RegressionResult:
    """Output of :func:`factor_regression` (alpha and loadings, monthly decimal)."""

    name: str
    model: str
    params: pd.Series
    se: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    ci_low: pd.Series
    ci_high: pd.Series
    r2: float
    adj_r2: float
    nobs: int
    start: str
    end: str
    hac_lags: int
    conventions: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"coef": self.params, "se_hac": self.se, "t_hac": self.tvalues,
                           "p_hac": self.pvalues, "ci95_low": self.ci_low,
                           "ci95_high": self.ci_high})
        df.index.name = "term"
        df = df.reset_index()
        df = df.assign(asset=self.name, model=self.model, r2=self.r2, adj_r2=self.adj_r2,
                       nobs=self.nobs, start=self.start, end=self.end, hac_lags=self.hac_lags)
        return df


def factor_regression(y: pd.Series, factors: pd.DataFrame, hac_lags: int | str = "auto",
                      small_sample: bool = True, name: str = "", model: str = "") -> RegressionResult:
    """OLS of excess returns on factors with HAC (Newey-West) inference.

    Rows with any missing value are dropped (common sample); the sample dates are
    reported.  Inference uses a normal approximation for the HAC t-statistics.
    """
    df = pd.concat([y.rename("y"), factors], axis=1, join="inner").dropna()
    if len(df) < factors.shape[1] + 3:
        raise ValueError(f"too few observations ({len(df)}) for {model}")
    T = len(df)
    L = resolve_lags(hac_lags, T)
    X = sm.add_constant(df[factors.columns], has_constant="add").rename(columns={"const": "alpha"})
    res = sm.OLS(df["y"], X).fit(cov_type="HAC",
                                 cov_kwds={"maxlags": L, "use_correction": small_sample})
    ci = res.conf_int(0.05)
    return RegressionResult(
        name=name or str(y.name), model=model, params=res.params, se=res.bse,
        tvalues=res.tvalues, pvalues=res.pvalues, ci_low=ci[0], ci_high=ci[1],
        r2=float(res.rsquared), adj_r2=float(res.rsquared_adj), nobs=T,
        start=str(df.index[0]), end=str(df.index[-1]), hac_lags=L,
        conventions={"lhs": "excess return", "units": "decimal monthly",
                     "se": f"Newey-West HAC, Bartlett, {L} lags, small_sample={small_sample}",
                     "inference": "normal approximation"},
    )


def rolling_betas(y: pd.Series, factors: pd.DataFrame, window: int = 36,
                  min_obs: int | None = None) -> pd.DataFrame:
    """Trailing-window OLS coefficients; the row at t uses observations t-window+1..t only."""
    min_obs = min_obs or window
    df = pd.concat([y.rename("y"), factors], axis=1, join="inner")
    cols = ["alpha", *factors.columns]
    out = pd.DataFrame(np.nan, index=df.index, columns=[*cols, "nobs"])
    Y = df["y"].to_numpy()
    F = df[factors.columns].to_numpy()
    for i in range(len(df)):
        lo = max(0, i - window + 1)
        yy, ff = Y[lo:i + 1], F[lo:i + 1]
        ok = np.isfinite(yy) & np.isfinite(ff).all(axis=1)
        if ok.sum() < min_obs:
            continue
        Xw = np.column_stack([np.ones(ok.sum()), ff[ok]])
        beta = np.linalg.lstsq(Xw, yy[ok], rcond=None)[0]
        out.iloc[i, : len(cols)] = beta
        out.iloc[i, -1] = ok.sum()
    return out


@dataclass
class GRSResult:
    statistic: float
    pvalue: float
    df1: int
    df2: int
    T: int
    N: int
    L: int
    alpha_quadratic: float
    sharpe_factor_sq: float
    cond_residual_cov: float
    cond_factor_cov: float
    sample_start: str
    sample_end: str
    mean_abs_alpha: float
    assumptions: str = (
        "Classical GRS (1989): i.i.d. multivariate-normal residuals, factors are traded "
        "excess returns, MLE covariances (divisor T) for residuals and factors, F(N, T-N-L). "
        "Not robust to heteroskedasticity/autocorrelation; not the HAC inference.")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _check_names(a: pd.DataFrame, b: pd.DataFrame) -> None:
    overlap = set(map(str, a.columns)) & set(map(str, b.columns))
    if overlap or a.columns.has_duplicates or b.columns.has_duplicates:
        raise ValueError(f"test-asset and factor column names must be unique; overlap={sorted(overlap)}")


def grs_test(test_excess: pd.DataFrame, factors: pd.DataFrame,
             cond_max: float = 1e10) -> GRSResult:
    """Gibbons-Ross-Shanken joint test that all N intercepts are zero.

    ``W = ((T - N - L) / N) * a' S^-1 a / (1 + m' O^-1 m)  ~  F(N, T - N - L)``

    with ``S = E'E / T`` (MLE residual covariance), ``m`` the factor means and
    ``O = (F - m)'(F - m) / T`` (MLE factor covariance).  Equivalently, with an unbiased
    residual covariance (divisor T - L - 1) the leading factor becomes
    ``(T/N) (T - N - L)/(T - L - 1)`` - but only if the factor covariance still uses
    divisor T.  A common sample across all assets is enforced.
    """
    _check_names(test_excess, factors)
    df = pd.concat([test_excess, factors], axis=1, join="inner").dropna()
    T, N, L = len(df), test_excess.shape[1], factors.shape[1]
    if T - N - L <= 0:
        raise GRSError(f"GRS needs T - N - L > 0 (T={T}, N={N}, L={L})")
    R = df[test_excess.columns].to_numpy()
    F = df[factors.columns].to_numpy()
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    alpha = B[0]
    E = R - X @ B
    Sigma = E.T @ E / T
    mu = F.mean(axis=0)
    Fc = F - mu
    Omega = Fc.T @ Fc / T
    c_s, c_o = float(np.linalg.cond(Sigma)), float(np.linalg.cond(Omega))
    if not np.isfinite(c_s) or c_s > cond_max:
        raise GRSError(f"residual covariance ill-conditioned (cond={c_s:.3g}); "
                       "classical GRS not computed (no pseudoinverse substitution)")
    if not np.isfinite(c_o) or c_o > cond_max:
        raise GRSError(f"factor covariance ill-conditioned (cond={c_o:.3g})")
    aq = float(alpha @ np.linalg.solve(Sigma, alpha))
    sh = float(mu @ np.linalg.solve(Omega, mu))
    W = (T - N - L) / N * aq / (1.0 + sh)
    p = float(stats.f.sf(W, N, T - N - L))
    return GRSResult(statistic=float(W), pvalue=p, df1=N, df2=T - N - L, T=T, N=N, L=L,
                     alpha_quadratic=aq, sharpe_factor_sq=sh, cond_residual_cov=c_s,
                     cond_factor_cov=c_o, sample_start=str(df.index[0]),
                     sample_end=str(df.index[-1]), mean_abs_alpha=float(np.abs(alpha).mean()))


def grs_from_guide_formula(test_excess: pd.DataFrame, factors: pd.DataFrame,
                           resid_divisor: str = "T-L-1", factor_divisor: str = "T") -> float:
    """The guide's form ``(T/N)((T-N-L)/(T-L-1)) a'S^-1 a / (1 + m'O^-1 m)``.

    Provided only to demonstrate (in tests/docs) that it equals :func:`grs_test` when
    S uses divisor T-L-1 *and* O uses divisor T, and differs otherwise.
    """
    df = pd.concat([test_excess, factors], axis=1, join="inner").dropna()
    T, N, L = len(df), test_excess.shape[1], factors.shape[1]
    R, F = df[test_excess.columns].to_numpy(), df[factors.columns].to_numpy()
    X = np.column_stack([np.ones(T), F])
    B = np.linalg.lstsq(X, R, rcond=None)[0]
    E = R - X @ B
    dv = {"T": T, "T-1": T - 1, "T-L-1": T - L - 1}
    S = E.T @ E / dv[resid_divisor]
    mu = F.mean(axis=0)
    Om = (F - mu).T @ (F - mu) / dv[factor_divisor]
    a = B[0]
    return float((T / N) * ((T - N - L) / (T - L - 1)) * (a @ np.linalg.solve(S, a))
                 / (1 + mu @ np.linalg.solve(Om, mu)))


def run_models(assets_excess: pd.DataFrame, factor_sets: dict[str, pd.DataFrame],
               hac_lags: int | str = "auto", small_sample: bool = True) -> pd.DataFrame:
    """Regress every asset on every model; long table of coefficients."""
    rows = []
    for model, F in factor_sets.items():
        for a in assets_excess.columns:
            r = factor_regression(assets_excess[a], F, hac_lags, small_sample, a, model)
            rows.append(r.to_frame())
    return pd.concat(rows, ignore_index=True)
