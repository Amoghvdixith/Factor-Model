"""Fama-MacBeth (1973) characteristic regressions.

For each formation month t, regress the next-month return of each stock on
characteristics known at t::

    r_{i,t+1} = g0_t + sum_k g_k,t z_{i,k,t} (+ industry dummies) + u_{i,t+1}

then average the monthly slopes and base inference on the time series of slopes, with
Newey-West HAC standard errors of the mean.

Important distinctions
----------------------
* These are *characteristic* regressions.  A significant average slope says the
  characteristic forecast returns in this sample; it does not prove that a priced
  risk factor exists.
* With characteristics measured without estimation error there is no generated-
  regressor problem, so the Shanken (1992) errors-in-variables correction (which
  applies to *estimated betas* in two-pass beta-pricing tests) is not applied.
* Industry dummies use a reference industry (dropped) so that they are not collinear
  with the intercept.  Each month's design matrix is rank-checked; dropped months are
  logged with reasons.
* The |t| > 3 flag is an illustrative, literature-motivated hurdle (Harvey, Liu and Zhu
  2016), not a formal multiple-testing correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .timeseries import resolve_lags


def hac_mean_test(x: pd.Series, lags: int | str | None = "auto",
                  small_sample: bool = True) -> dict[str, Any]:
    """Mean of a series with a Newey-West standard error (statsmodels OLS on a constant)."""
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    T = len(v)
    L = resolve_lags(lags, T)
    res = sm.OLS(v, np.ones(T)).fit(cov_type="HAC",
                                    cov_kwds={"maxlags": L, "use_correction": small_sample})
    m, se = float(res.params[0]), float(res.bse[0])
    t = m / se if se > 0 else np.nan
    return {"mean": m, "se": se, "t": t, "p": float(2 * stats.norm.sf(abs(t))),
            "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se, "lags": L, "T": T}


def _design(g: pd.DataFrame, xs: list[str], industry_col: str | None,
            reference: str | None) -> tuple[pd.DataFrame, str | None]:
    X = g[xs].copy()
    X.insert(0, "const", 1.0)
    ref_used = None
    if industry_col is not None:
        inds = sorted(g[industry_col].unique())
        ref_used = reference if reference in inds else inds[0]
        for ind in inds:
            if ind == ref_used:
                continue
            X[f"ind_{ind}"] = (g[industry_col] == ind).astype(float)
    return X, ref_used


@dataclass
class FMResult:
    spec: str
    xs: list[str]
    industry: bool
    gammas: pd.DataFrame
    r2: pd.Series
    nobs: pd.Series
    summary: pd.DataFrame
    month_log: list[dict] = field(default_factory=list)
    conventions: dict[str, Any] = field(default_factory=dict)


def fama_macbeth(panel: pd.DataFrame, y: str, xs: list[str], industry_col: str | None = None,
                 reference_industry: str | None = None, min_obs: int = 30,
                 hac_lags: int | str = "auto", t_hurdle: float = 3.0,
                 spec: str = "") -> FMResult:
    """Monthly cross-sectional OLS of ``y`` (a t+1 return column) on ``xs`` (known at t)."""
    if not (y.endswith("_fwd") or y.startswith("ret_fwd")):
        raise ValueError("y must be a forward (t+1) return column, e.g. 'ret_fwd_excess'")
    rows, r2s, ns, mlog = {}, {}, {}, []
    for t, g in panel.groupby(level=0, sort=True):
        cols = [y, *xs] + ([industry_col] if industry_col else [])
        g = g[cols].dropna()
        if len(g) < min_obs:
            mlog.append({"date": t, "dropped": True, "reason": f"n={len(g)} < min_obs={min_obs}"})
            continue
        X, ref = _design(g, xs, industry_col, reference_industry)
        if X.shape[0] <= X.shape[1] + 1:
            mlog.append({"date": t, "dropped": True, "reason": "n <= k+1"})
            continue
        rank = np.linalg.matrix_rank(X.to_numpy())
        if rank < X.shape[1]:
            mlog.append({"date": t, "dropped": True,
                         "reason": f"rank deficient ({rank} < {X.shape[1]})"})
            continue
        b, *_ = np.linalg.lstsq(X.to_numpy(), g[y].to_numpy(), rcond=None)
        fit = X.to_numpy() @ b
        resid = g[y].to_numpy() - fit
        sst = ((g[y] - g[y].mean()) ** 2).sum()
        r2s[t] = 1 - (resid @ resid) / sst if sst > 0 else np.nan
        rows[t] = pd.Series(b, index=X.columns)
        ns[t] = len(g)
        mlog.append({"date": t, "dropped": False, "n": len(g), "reference_industry": ref})
    if not rows:
        raise ValueError(f"no usable months for spec {spec}")
    G = pd.DataFrame(rows).T
    G.index.name = "date"
    keep = ["const", *xs]
    summ = []
    for k in keep:
        h = hac_mean_test(G[k], hac_lags)
        summ.append({"spec": spec, "term": k, "premium_monthly": h["mean"], "se_hac": h["se"],
                     "t_hac": h["t"], "p_hac": h["p"], "ci95_low": h["ci_low"],
                     "ci95_high": h["ci_high"], "hac_lags": h["lags"], "n_months": h["T"],
                     "avg_n_stocks": float(pd.Series(ns).mean()),
                     "avg_r2": float(pd.Series(r2s).mean()),
                     "industry_controls": industry_col is not None,
                     "passes_t3_hurdle": bool(abs(h["t"]) > t_hurdle) if k != "const" else None})
    return FMResult(spec=spec, xs=xs, industry=industry_col is not None, gammas=G,
                    r2=pd.Series(r2s), nobs=pd.Series(ns), summary=pd.DataFrame(summ),
                    month_log=mlog,
                    conventions={"y": y, "weighting": "equal (OLS)",
                                 "x_units": "cross-sectional z-scores",
                                 "inference": "Newey-West HAC on the mean of monthly slopes"})


def linearmodels_crosscheck(panel: pd.DataFrame, y: str, xs: list[str], min_obs: int = 30,
                            bandwidth: int | None = None) -> pd.DataFrame | None:
    """Re-estimate with ``linearmodels.FamaMacBeth`` on the matched sample (if installed).

    Months are filtered with the same rule as :func:`fama_macbeth` (complete cases,
    n >= min_obs).  Returns a frame of linearmodels estimates/std errors (unadjusted and
    kernel) or None if linearmodels is unavailable.
    """
    try:
        from linearmodels import FamaMacBeth
    except Exception:  # noqa: BLE001
        return None
    d = panel[[y, *xs]].dropna()
    cnt = d.groupby(level=0).size()
    d = d[d.index.get_level_values(0).isin(cnt[cnt >= min_obs].index)]
    d = d.swaplevel().sort_index()  # linearmodels expects (entity, time)
    exog = sm.add_constant(d[xs], has_constant="add")
    mod = FamaMacBeth(d[y], exog)
    out = {}
    un = mod.fit(cov_type="unadjusted")
    out["params"] = un.params
    out["se_unadjusted"] = un.std_errors
    if bandwidth is not None:
        kr = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bandwidth)
        out["se_kernel"] = kr.std_errors
    return pd.DataFrame(out)
