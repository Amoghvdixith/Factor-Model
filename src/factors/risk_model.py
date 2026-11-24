"""Fundamental factor risk model  Sigma_t = B_t F_t B_t' + D_t  (monthly units).

Exposures B_t
-------------
``market`` (a column of ones), one dummy per industry, and style exposures (size, value,
momentum, low_vol, quality).  Style exposures are re-standardised at each date to a
cap-weighted mean of zero and an equal-weighted standard deviation of one, after
cross-sectional winsorisation.  A missing style exposure is set to 0 (the cap-weighted
mean) and counted.

Identification
--------------
An intercept plus a full set of industry dummies is perfectly collinear.  Two options:

* ``reference``: drop one industry (the alphabetically first); the intercept is then the
  return of the reference industry at zero style exposure and the other industry factor
  returns are relative to it.
* ``cap_weighted_sum_zero`` (default): keep all industries and impose
  ``sum_k c_k g_k = 0`` with c_k the industry cap weights, implemented by substitution
  (``g_K = -sum_{k<K} c_k g_k / c_K``).  The market factor is then close to the
  cap-weighted market return.

Both give identical fitted values, residuals and Sigma (tested); individual factor
attributions differ because they depend on the factor basis.

Estimation timing
-----------------
Factor returns for month t+1 are estimated by regressing ``ret_fwd`` (month t+1) on B_t
(known at the close of t) and labelled with the month-end at which they are realised.
A risk forecast made at t uses only factor returns and residuals realised by t.

WLS weights
-----------
``w_i = sqrt(mcap_i)`` is the *objective weight* in ``min sum_i w_i e_i^2``.  The
regression is solved by multiplying the rows of y and X by ``sqrt(w_i) = mcap_i^(1/4)``.
Using ``sqrt(mcap)`` as the row multiplier would instead imply objective weights
proportional to mcap.

Factor covariance F
-------------------
Single half-life EWMA: decay ``lam = 0.5 ** (1 / half_life)``; weight of the observation
``a`` periods old is ``lam ** a``, weights normalised to sum to one; not demeaned by
default (monthly factor means are small relative to volatility).  At least
``min_periods`` observations are required.  F is symmetrised; eigenvalues below
``eigen_floor`` are raised to the floor (logged).

Specific risk D
---------------
Diagonal: EWMA variance of each stock's residuals (same decay rule), linearly shrunk
toward the mean specific variance of its size bucket at t with weight ``n / (n + k)`` on
the stock's own estimate (n = number of residual observations).  Stocks with fewer
than ``min_obs`` residuals use the bucket mean.  A floor on specific volatility applies.
This is a transparent heuristic shrinkage, not a Bayesian posterior.  Assuming a diagonal
D means residuals are uncorrelated across stocks after the factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .characteristics import winsorise_cs

log = logging.getLogger(__name__)


class RiskModelError(ValueError):
    """Raised for invalid risk-model inputs."""


def ewma_decay(half_life: float) -> float:
    """Decay factor lam with lam ** half_life == 0.5."""
    if half_life <= 0:
        raise RiskModelError("half_life must be positive")
    return float(0.5 ** (1.0 / half_life))


def ewma_weights(n: int, half_life: float) -> np.ndarray:
    """Normalised weights, oldest first; newest has the largest weight."""
    lam = ewma_decay(half_life)
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    return w / w.sum()


def ewma_cov(f: pd.DataFrame, half_life: float, min_periods: int = 24, demean: bool = False,
             eigen_floor: float = 1e-12) -> tuple[pd.DataFrame, dict[str, Any]]:
    """EWMA covariance of the rows of ``f`` (time order ascending, complete rows only)."""
    x = f.dropna()
    n = len(x)
    if n < min_periods:
        raise RiskModelError(f"EWMA needs {min_periods} observations, have {n}")
    w = ewma_weights(n, half_life)
    X = x.to_numpy(float)
    if demean:
        X = X - w @ X
    C = (X * w[:, None]).T @ X
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    n_floor = int((vals < eigen_floor).sum())
    if n_floor:
        vals = np.maximum(vals, eigen_floor)
        C = (vecs * vals) @ vecs.T
        C = 0.5 * (C + C.T)
    diag = {"n_obs": n, "half_life": half_life, "decay": ewma_decay(half_life),
            "demeaned": demean, "min_eigen_raw": float(np.linalg.eigvalsh(0.5 * (C + C.T)).min()),
            "n_eigen_floored": n_floor, "eigen_floor": eigen_floor}
    return pd.DataFrame(C, index=f.columns, columns=f.columns), diag


def standardise_style(x: pd.Series, cap: pd.Series) -> pd.Series:
    """Cap-weighted mean zero, equal-weighted std one (single date)."""
    v = x.dropna()
    if len(v) < 3:
        return x * np.nan
    w = np.nan_to_num(cap.reindex(v.index).to_numpy(float), nan=0.0)
    w = np.maximum(w, 0.0)
    vv = v.to_numpy(float)
    mu = float((vv * w).sum() / w.sum()) if w.sum() > 0 else float(vv.mean())
    sd = float(vv.std(ddof=1))
    if sd <= 1e-12:
        return x * np.nan
    return (x - mu) / sd


def exposures_at(g: pd.DataFrame, styles: list[str], industries: list[str]) -> pd.DataFrame:
    """Full exposure matrix B for one date: market, all industry dummies, styles."""
    B = pd.DataFrame(index=g.index)
    B["market"] = 1.0
    for ind in industries:
        B[f"ind_{ind}"] = (g["industry"] == ind).astype(float)
    for s in styles:
        z = standardise_style(g[f"{s}_rm"], g["mcap"])
        B[s] = z.fillna(0.0)
    return B


def _estimation_design(B: pd.DataFrame, cap: pd.Series, industries: list[str],
                       identification: str) -> tuple[pd.DataFrame, Any]:
    """Return the full-rank design and a function mapping theta -> full factor returns."""
    ind_cols = [f"ind_{i}" for i in industries]
    other = [c for c in B.columns if c not in ind_cols]
    if identification == "reference":
        ref = ind_cols[0]
        X = B.drop(columns=[ref])

        def back(theta: pd.Series) -> pd.Series:
            out = theta.copy()
            out[ref] = 0.0  # reference industry is absorbed in the intercept
            return out[[*other[:1], *ind_cols, *other[1:]]]
        return X, back
    if identification == "cap_weighted_sum_zero":
        c = pd.Series({col: float(cap[B[col] > 0].sum()) for col in ind_cols})
        if (c <= 0).any():
            raise RiskModelError(f"empty industry at this date: {list(c.index[c <= 0])}")
        last = ind_cols[-1]
        X = B[other].copy()
        for col in ind_cols[:-1]:
            X[col] = B[col] - (c[col] / c[last]) * B[last]

        def back(theta: pd.Series) -> pd.Series:
            out = theta.copy()
            out[last] = -float(sum(c[k] * theta[k] for k in ind_cols[:-1])) / c[last]
            return out[[*other[:1], *ind_cols, *other[1:]]]
        return X, back
    raise RiskModelError(f"unknown identification {identification!r}")


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Solve min sum_i w_i (y_i - x_i'b)^2 by scaling rows with sqrt(w_i)."""
    s = np.sqrt(w)
    return np.linalg.lstsq(X * s[:, None], y * s, rcond=None)[0]


@dataclass
class FactorReturns:
    """Cross-sectionally estimated factor returns and residuals."""

    factor_returns: pd.DataFrame          # index return_date, columns factor names
    residuals: pd.DataFrame               # index return_date, columns tickers
    exposures: dict[pd.Timestamp, pd.DataFrame]   # formation date -> B (full)
    weights: dict[pd.Timestamp, pd.Series]
    r2: pd.Series
    log: list[dict] = field(default_factory=list)
    identification: str = "cap_weighted_sum_zero"


def prepare_style_inputs(panel: pd.DataFrame, styles: list[str], lower: float = 0.01,
                         upper: float = 0.99) -> pd.DataFrame:
    """Add ``<style>_rm`` columns: winsorised raw characteristics for the risk model.

    Uses the *raw* characteristic (log cap for size; the quality composite for quality)
    winsorised per date; standardisation happens per date in :func:`exposures_at`.
    """
    p = panel.copy()
    for s in styles:
        src = f"{s}_raw" if f"{s}_raw" in p.columns else s
        p[f"{s}_rm"] = winsorise_cs(p[src], lower, upper) if p[src].notna().any() else np.nan
    return p


def estimate_factor_returns(panel: pd.DataFrame, styles: list[str],
                            identification: str = "cap_weighted_sum_zero",
                            ret_col: str = "ret_fwd", min_stocks: int = 30) -> FactorReturns:
    """Monthly WLS cross-sectional regressions of next-month returns on B_t."""
    p = prepare_style_inputs(panel, styles)
    styles = [s for s in styles if p[f"{s}_rm"].notna().any()]
    industries = sorted(p["industry"].dropna().unique())
    fr: dict[Any, pd.Series] = {}
    res: dict[Any, pd.Series] = {}
    expo: dict[pd.Timestamp, pd.DataFrame] = {}
    wts: dict[pd.Timestamp, pd.Series] = {}
    r2: dict[Any, float] = {}
    lg: list[dict[str, Any]] = []
    for t_key, g in p.groupby(level=0, sort=True):
        t = pd.Timestamp(t_key)  # type: ignore[arg-type]
        g = g.droplevel(0)
        g = g[g["mcap"] > 0]
        B = exposures_at(g, styles, industries)
        expo[t] = B
        wts[t] = np.sqrt(g["mcap"])
        ok = g[ret_col].notna()
        rd = g["return_date"].dropna()
        if ok.sum() < min_stocks or rd.empty:
            lg.append({"date": t, "estimated": False, "reason": f"n={int(ok.sum())}"})
            continue
        present = [i for i in industries if (g.loc[ok, "industry"] == i).any()]
        if len(present) < len(industries):
            lg.append({"date": t, "estimated": False, "reason": "industry with no returns"})
            continue
        Bt = B[ok]
        X, back = _estimation_design(Bt, g.loc[ok, "mcap"], industries, identification)
        w = np.sqrt(g.loc[ok, "mcap"].to_numpy(float))
        y = g.loc[ok, ret_col].to_numpy(float)
        if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
            lg.append({"date": t, "estimated": False, "reason": "rank deficient"})
            continue
        theta = pd.Series(wls(X.to_numpy(float), y, w), index=X.columns)
        f_full = back(theta)
        fit = Bt[f_full.index].to_numpy() @ f_full.to_numpy()
        e = y - fit
        ybar = np.average(y, weights=w)
        sst = float((w * (y - ybar) ** 2).sum())
        r2[rd.iloc[0]] = 1 - float((w * e ** 2).sum()) / sst if sst > 0 else np.nan
        fr[rd.iloc[0]] = f_full
        res[rd.iloc[0]] = pd.Series(e, index=Bt.index)
        lg.append({"date": t, "estimated": True, "n": int(ok.sum()),
                   "n_missing_style_filled": int(g[[f"{s}_rm" for s in styles]].isna().any(axis=1).sum())})
    F = pd.DataFrame(fr).T.sort_index()
    F.index.name = "return_date"
    R = pd.DataFrame(res).T.sort_index()
    R.index.name = "return_date"
    return FactorReturns(F, R, expo, wts, pd.Series(r2).sort_index(), lg, identification)


def specific_variance(resid: pd.DataFrame, tickers: pd.Index, size: pd.Series,
                      half_life: float, min_obs: int = 6, shrink_k: float = 12,
                      n_buckets: int = 3, floor_vol: float = 0.02) -> tuple[pd.Series, pd.DataFrame]:
    """Shrunk EWMA specific variance for ``tickers`` (see module docstring)."""
    R = resid.reindex(columns=tickers)
    lam = ewma_decay(half_life)
    n_obs = R.notna().sum()
    raw = pd.Series(np.nan, index=tickers)
    for tk in tickers:
        x = R[tk].dropna()
        if len(x) == 0:
            continue
        pos = R.index.get_indexer(x.index)
        ages = (len(R) - 1 - pos).astype(float)
        w = lam ** ages
        raw[tk] = float((w * x.to_numpy() ** 2).sum() / w.sum())
    buckets = pd.qcut(size.reindex(tickers).rank(method="first"), n_buckets,
                      labels=False) if size.notna().sum() >= n_buckets else pd.Series(0, index=tickers)
    eligible = raw.where(n_obs >= min_obs)
    target = eligible.groupby(buckets).transform("mean")
    target = target.fillna(eligible.mean())
    shrink = n_obs / (n_obs + shrink_k)
    var = (shrink * raw + (1 - shrink) * target).where(n_obs >= min_obs, target)
    var = var.clip(lower=floor_vol ** 2)
    diag = pd.DataFrame({"raw_var": raw, "n_obs": n_obs, "bucket": buckets,
                         "target_var": target, "shrink_weight": shrink.where(n_obs >= min_obs, 0.0),
                         "spec_var": var, "floored": var <= floor_vol ** 2 + 1e-15})
    return var, diag


@dataclass
class RiskModel:
    """Forecast for the month after ``date`` in MONTHLY variance units."""

    date: pd.Timestamp
    B: pd.DataFrame
    F: pd.DataFrame
    D: pd.Series
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if list(self.B.columns) != list(self.F.index):
            raise RiskModelError("B columns and F index must match in order")
        if not self.B.index.equals(self.D.index):
            raise RiskModelError("B and D must share the same asset index")

    @property
    def assets(self) -> pd.Index:
        return self.B.index

    def sigma(self) -> pd.DataFrame:
        B = self.B.to_numpy()
        S = B @ self.F.to_numpy() @ B.T + np.diag(self.D.to_numpy())
        return pd.DataFrame(S, index=self.assets, columns=self.assets)

    def portfolio_variance(self, w: pd.Series) -> float:
        wv = align_weights(w, self.assets)
        x = self.B.to_numpy().T @ wv
        return float(x @ self.F.to_numpy() @ x + (self.D.to_numpy() * wv ** 2).sum())

    def portfolio_vol(self, w: pd.Series, annualise: bool = False) -> float:
        v = self.portfolio_variance(w)
        return float(np.sqrt(v * (12 if annualise else 1)))


def align_weights(w: pd.Series, assets: pd.Index) -> np.ndarray:
    """Weights aligned to ``assets``; assets missing from w get 0; unknown names raise."""
    unknown = w.index.difference(assets)
    if len(unknown) and (w.reindex(unknown).abs() > 0).any():
        raise RiskModelError(f"weights on assets not in the risk model: {list(unknown[:5])}")
    return w.reindex(assets).fillna(0.0).to_numpy(float)


def build_risk_model(fr: FactorReturns, panel: pd.DataFrame, t: pd.Timestamp,
                     cfg: dict[str, Any]) -> RiskModel:
    """Risk model at formation date ``t`` using information realised by t only."""
    if t not in fr.exposures:
        raise RiskModelError(f"no exposures at {t}")
    fc, sc = cfg["factor_cov"], cfg["specific_risk"]
    hist = fr.factor_returns[fr.factor_returns.index <= t]
    F, fdiag = ewma_cov(hist, fc["half_life_months"], fc["min_periods"], fc["demean"],
                        fc["eigen_floor"])
    B = fr.exposures[t]
    B = B[F.index.tolist()]
    resid = fr.residuals[fr.residuals.index <= t]
    g = panel.xs(t, level=0)
    size = np.log(g["mcap"].reindex(B.index))
    D, ddiag = specific_variance(resid, B.index, size, sc["half_life_months"], sc["min_obs"],
                                 sc["shrink_k"], sc["n_size_buckets"], sc["floor_vol_monthly"])
    diag = {"factor_cov": fdiag, "n_assets": len(B), "n_factors": B.shape[1],
            "specific_floored": int(ddiag["floored"].sum()),
            "specific_short_history": int((ddiag["n_obs"] < sc["min_obs"]).sum()),
            "last_factor_return_date": str(hist.index[-1].date()), "units": "monthly variance"}
    return RiskModel(date=t, B=B, F=F, D=D, diagnostics=diag)


def is_psd(M: np.ndarray, tol: float = 1e-10) -> bool:
    M = np.asarray(M)
    return bool(np.allclose(M, M.T, atol=1e-12) and np.linalg.eigvalsh(M).min() >= -tol)
