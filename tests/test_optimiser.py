import numpy as np
import pandas as pd
import pytest

from factors.optimiser import InfeasibleSpecError, feasibility_precheck, optimise_tilt
from tests.conftest import PORT_CFG, make_rm


def _setup(n=40, seed=0):
    rm = make_rm(n=n, k=5, seed=seed)
    rng = np.random.default_rng(seed)
    scores = pd.Series(rng.standard_normal(n), index=rm.assets)
    bench = pd.Series(1 / n, index=rm.assets)
    return rm, scores, bench


def test_precheck_cap_needs_enough_names():
    feasibility_precheck(20, 0.05)
    with pytest.raises(InfeasibleSpecError):
        feasibility_precheck(19, 0.05)


def test_initial_deployment_satisfies_constraints():
    rm, s, b = _setup()
    res = optimise_tilt(s, rm, pd.Series(dtype=float), b, PORT_CFG, 0.04, initial=True)
    w = res.weights
    assert res.status == "optimal" and not res.fallback_used and res.checks["ok"]
    assert w.sum() == pytest.approx(1, abs=1e-6) and w.min() >= -1e-8 and w.max() <= 0.05 + 1e-6
    a = (w - b).to_numpy()
    te = np.sqrt(a @ rm.sigma().to_numpy() @ a) * np.sqrt(12)
    assert te <= 0.04 * (1 + 1e-4) + 1e-6
    assert res.checks["turnover_budget"] is None       # initial deployment exempt


def test_turnover_constraint_uses_drifted_weights():
    rm, s, b = _setup()
    rng = np.random.default_rng(9)
    wd = pd.Series(rng.dirichlet(np.ones(40) * 5), index=rm.assets).clip(upper=0.05)
    wd /= wd.sum()
    res = optimise_tilt(s, rm, wd, b, {**PORT_CFG, "turnover_one_way_max": 0.05}, 0.06)
    assert res.status == "optimal"
    assert 0.5 * (res.weights - wd).abs().sum() <= 0.05 + 1e-5
    res2 = optimise_tilt(s, rm, wd, b, {**PORT_CFG, "turnover_one_way_max": 0.05}, 0.06,
                         forced_exit_weight=0.03)
    assert res2.checks["turnover_budget"] == pytest.approx(0.08)


def test_infeasible_problem_falls_back_to_drifted_weights_without_relaxing():
    rm, s, b = _setup()
    wd = pd.Series(0.0, index=rm.assets)
    wd.iloc[:20] = 0.05                       # far from the benchmark
    cfg = {**PORT_CFG, "turnover_one_way_max": 0.01}
    res = optimise_tilt(s, rm, wd, b, cfg, te_annual=0.001)   # TE 0.1% unreachable within 1% turnover
    assert res.fallback_used and res.status == "failed_hold_drifted"
    pd.testing.assert_series_equal(res.weights, wd / wd.sum(), check_names=False)
    assert "tracking_error" in res.checks["violations"]   # the fallback is reported, not hidden


def test_solver_fallback_order_is_recorded():
    rm, s, b = _setup()
    cfg = {**PORT_CFG, "solvers": ["NOT_A_SOLVER", "CLARABEL"]}
    res = optimise_tilt(s, rm, pd.Series(dtype=float), b, cfg, 0.04, initial=True)
    assert res.solver == "CLARABEL" and res.attempts[0]["status"].startswith("error")


def test_precheck_failure_returns_benchmark_on_initial():
    rm, s, b = _setup(n=15)
    res = optimise_tilt(s, rm, pd.Series(dtype=float), b, PORT_CFG, 0.04, initial=True)
    assert res.status == "precheck_infeasible" and res.fallback_used
