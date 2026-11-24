"""Long-only, risk-constrained factor-tilt optimisation with cvxpy.

Problem (monthly units)::

    maximise    s' w  -  gamma * a' Sigma a  -  c * sum|w - w_drift|
    subject to  sum(w) = 1, 0 <= w <= max_w                      (full investment, long only)
                lo_k <= (B' a)_k <= hi_k   for listed style factors (active exposure bounds)
                |(Ind' a)_j| <= ind_max     for every industry j  (industry active weights)
                sqrt(a' Sigma a) <= TE_monthly                    (explicit TE cone)
                0.5 * sum|w - w_drift| <= tau + forced_exit_weight (one-way turnover)

with ``a = w - w_b`` and ``a' Sigma a = ||F^(1/2) B' a||^2 + ||D^(1/2) a||^2``.

* ``s`` are composite characteristic z-scores, NOT calibrated expected returns, so the
  objective is a *score-risk trade-off*; ``gamma`` has no utility interpretation.
* ``w_drift`` are the previous holdings drifted by realised returns (pre-trade weights).
  Names that left the universe (``forced_exits``) must be sold; that forced selling and
  its reinvestment are exempt from the turnover budget (documented policy).
* The first deployment from cash is exempt from the turnover limit if configured.
* Fallback (declared in advance): if the primary solver does not return ``optimal`` or a
  post-solve constraint check fails, the *same* problem is re-solved with the next
  solver.  If all fail, the portfolio keeps its drifted weights (continuing names,
  renormalised) - no trade - and the failure is recorded.  Constraints are never relaxed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from .risk_model import RiskModel


class InfeasibleSpecError(ValueError):
    """Raised by the pre-check when constraints are infeasible by construction."""


@dataclass
class OptResult:
    weights: pd.Series
    status: str
    solver: str | None
    fallback_used: bool
    message: str
    checks: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    objective: float | None = None
    runtime_s: float = 0.0


def feasibility_precheck(n_assets: int, max_w: float) -> None:
    """Full investment with a cap needs at least ceil(1 / max_w) names."""
    if n_assets * max_w < 1 - 1e-12:
        raise InfeasibleSpecError(
            f"{n_assets} assets x max weight {max_w:.2%} < 100%: full investment infeasible "
            f"(need >= {int(np.ceil(1 / max_w))} names)")


def _sqrt_psd(F: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (F + F.T))
    return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T


def check_constraints(w: np.ndarray, wb: np.ndarray, w_drift: np.ndarray, B: np.ndarray,
                      F: np.ndarray, D: np.ndarray, ind: np.ndarray, style_idx: dict[str, int],
                      cfg: dict[str, Any], te_monthly: float, turnover_budget: float | None,
                      tol: float) -> dict[str, Any]:
    """Evaluate every constraint on a candidate solution (post-solve verification)."""
    a = w - wb
    x = B.T @ a
    te = float(np.sqrt(max(x @ F @ x + (D * a ** 2).sum(), 0.0)))
    out: dict[str, Any] = {
        "sum_w": float(w.sum()), "min_w": float(w.min()), "max_w": float(w.max()),
        "te_monthly": te, "te_limit_monthly": te_monthly,
        "turnover_one_way": float(0.5 * np.abs(w - w_drift).sum()),
        "turnover_budget": turnover_budget,
        "max_abs_industry_active": float(np.abs(ind.T @ a).max()) if ind.size else 0.0,
    }
    viol = []
    if abs(out["sum_w"] - 1) > tol:
        viol.append("sum_w")
    if out["min_w"] < -tol:
        viol.append("long_only")
    if out["max_w"] > cfg["max_weight"] + tol:
        viol.append("max_weight")
    if te > te_monthly * (1 + 1e-4) + tol:
        viol.append("tracking_error")
    if turnover_budget is not None and out["turnover_one_way"] > turnover_budget + tol:
        viol.append("turnover")
    if out["max_abs_industry_active"] > cfg["industry_active_max"] + tol:
        viol.append("industry_active")
    for k, (lo, hi) in cfg.get("active_exposure_bounds", {}).items():
        if k in style_idx:
            v = float(x[style_idx[k]])
            out[f"active_{k}"] = v
            if v < lo - tol or v > hi + tol:
                viol.append(f"exposure_{k}")
    out["violations"] = viol
    out["ok"] = not viol
    return out


def optimise_tilt(scores: pd.Series, rm: RiskModel, w_drift: pd.Series, benchmark: pd.Series,
                  cfg: dict[str, Any], te_annual: float, initial: bool = False,
                  forced_exit_weight: float = 0.0) -> OptResult:
    """Solve the factor-tilt problem on the risk model's asset universe."""
    t0 = time.perf_counter()
    assets = rm.assets
    s = scores.reindex(assets).fillna(0.0).to_numpy(float)
    wd = w_drift.reindex(assets).fillna(0.0).to_numpy(float)
    wb = benchmark.reindex(assets).fillna(0.0).to_numpy(float)
    B, F, D = rm.B.to_numpy(float), rm.F.to_numpy(float), rm.D.to_numpy(float)
    ind_cols = [i for i, c in enumerate(rm.B.columns) if c.startswith("ind_")]
    ind = B[:, ind_cols]
    style_idx = {c: i for i, c in enumerate(rm.B.columns)
                 if c in cfg.get("active_exposure_bounds", {})}
    te_m = te_annual / np.sqrt(12)
    max_w = cfg["max_weight"]
    tau = cfg["turnover_one_way_max"]
    exempt = initial and cfg.get("initial_deployment_exempt", True)
    budget = None if exempt else tau + forced_exit_weight
    tol = cfg.get("constraint_check_tol", 1e-5)
    hold = wd / wd.sum() if wd.sum() > 0 else wb.copy()

    try:
        feasibility_precheck(len(assets), max_w)
    except InfeasibleSpecError as exc:
        chk = check_constraints(hold, wb, wd, B, F, D, ind, style_idx, cfg, te_m, budget, tol)
        return OptResult(pd.Series(hold, index=assets), "precheck_infeasible", None, True,
                         str(exc), chk, [], None, time.perf_counter() - t0)

    Fh = _sqrt_psd(F)
    w = cp.Variable(len(assets))
    a = w - wb
    active_var = cp.sum_squares(Fh @ (B.T @ a)) + cp.sum_squares(cp.multiply(np.sqrt(D), a))
    te_vec = cp.hstack([Fh @ (B.T @ a), cp.multiply(np.sqrt(D), a)])
    cons = [cp.sum(w) == 1, w >= 0, w <= max_w, cp.norm(te_vec, 2) <= te_m]
    if ind.size:
        cons += [cp.abs(ind.T @ a) <= cfg["industry_active_max"]]
    for k, (lo, hi) in cfg.get("active_exposure_bounds", {}).items():
        if k in style_idx:
            cons += [B[:, style_idx[k]] @ a >= lo, B[:, style_idx[k]] @ a <= hi]
    if budget is not None:
        cons += [0.5 * cp.norm1(w - wd) <= budget]
    tc = cfg.get("tcost_bps_penalty", 0.0) / 1e4
    obj = s @ w - cfg["risk_aversion"] * active_var - (tc * cp.norm1(w - wd) if tc else 0)
    prob = cp.Problem(cp.Maximize(obj), cons)
    attempts = []
    for solver in cfg.get("solvers", ["CLARABEL", "SCS"]):
        rec: dict[str, Any] = {"solver": solver}
        try:
            prob.solve(solver=solver)
            rec["status"] = prob.status
        except Exception as exc:  # noqa: BLE001 - recorded, then fallback
            rec["status"] = f"error: {exc}"
            attempts.append(rec)
            continue
        if prob.status == cp.OPTIMAL and w.value is not None:
            sol = np.where(np.abs(w.value) < 1e-10, 0.0, w.value)
            chk = check_constraints(sol, wb, wd, B, F, D, ind, style_idx, cfg, te_m, budget, tol)
            rec["checks_ok"] = chk["ok"]
            attempts.append(rec)
            if chk["ok"]:
                return OptResult(pd.Series(sol, index=assets), "optimal", solver, False,
                                 "solved", chk, attempts, float(prob.value),
                                 time.perf_counter() - t0)
        else:
            attempts.append(rec)
    chk = check_constraints(hold, wb, wd, B, F, D, ind, style_idx, cfg, te_m, budget, tol)
    return OptResult(pd.Series(hold, index=assets), "failed_hold_drifted", None, True,
                     f"all solvers failed or violated checks: {attempts}", chk, attempts, None,
                     time.perf_counter() - t0)
