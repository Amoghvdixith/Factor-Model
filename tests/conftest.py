"""Shared offline fixtures (tiny seeded synthetic data; no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.data import validate_fundamentals
from factors.panel import asof_join_fundamentals, build_monthly_panel
from factors.synthetic import SyntheticSpec, generate_fundamentals, generate_market

CHAR_CFG = {
    "momentum": {"lookback_months": 12, "skip_months": 1},
    "low_vol": {"lookback_days": 252, "min_obs": 126},
    "value": {"measure": "book_to_market"},
    "quality": {"min_components": 2},
    "processing": {"winsor_lower": 0.01, "winsor_upper": 0.99, "min_cross_section": 10},
}

RISK_CFG = {
    "factor_cov": {"half_life_months": 12, "min_periods": 12, "demean": False, "eigen_floor": 1e-12},
    "specific_risk": {"half_life_months": 12, "min_obs": 6, "shrink_k": 12, "n_size_buckets": 3,
                      "floor_vol_monthly": 0.02},
}

PORT_CFG = {
    "max_weight": 0.05, "turnover_one_way_max": 0.20, "initial_deployment_exempt": True,
    "risk_aversion": 1000.0, "industry_active_max": 0.05, "tcost_bps_penalty": 0.0,
    "active_exposure_bounds": {"size": [-0.3, 0.3], "value": [-1, 1], "momentum": [-1, 1],
                               "low_vol": [-1, 1], "quality": [-1, 1]},
    "solvers": ["CLARABEL", "SCS"], "constraint_check_tol": 1e-5,
    "backtest": {"initial_cash": 1_000_000, "cost_bps_one_way": 10, "cash_buffer": 0.002,
                 "fractional": False, "execution_delay": 1},
}


@pytest.fixture(scope="session")
def tiny_market():
    spec = SyntheticSpec(n_stocks=60, n_years=4, seed=11, n_industries=4)
    mk = generate_market(spec)
    fund = generate_fundamentals(spec, mk)
    return spec, mk, fund


@pytest.fixture(scope="session")
def tiny_panel(tiny_market):
    from factors.characteristics import add_characteristics

    spec, mk, fund = tiny_market
    close = mk["ohlcv"]["close"]
    meta = mk["meta"].set_index("ticker")
    p = build_monthly_panel(close, meta["industry"], meta["shares"], mk["rf"])
    fj = asof_join_fundamentals(p, validate_fundamentals(fund))
    return add_characteristics(p, close, fj, CHAR_CFG), close


def make_rm(n: int = 30, k: int = 4, seed: int = 0):
    """A small random risk model (B, F PSD, D) for attribution/optimiser tests."""
    from factors.risk_model import RiskModel

    rng = np.random.default_rng(seed)
    assets = pd.Index([f"A{i:02d}" for i in range(n)])
    ind = np.eye(2)[rng.integers(0, 2, n)]
    styles = rng.standard_normal((n, k - 3))
    B = pd.DataFrame(np.column_stack([np.ones(n), ind, styles]), index=assets,
                     columns=["market", "ind_X", "ind_Y"] + [s for s in ["size", "value", "momentum", "low_vol", "quality"][: k - 3]])
    A = rng.standard_normal((B.shape[1], B.shape[1])) * 0.02
    F = pd.DataFrame(A @ A.T + np.eye(B.shape[1]) * 1e-4, index=B.columns, columns=B.columns)
    D = pd.Series(rng.uniform(0.002, 0.01, n), index=assets)
    return RiskModel(date=pd.Timestamp("2020-01-31"), B=B, F=F, D=D)
