import numpy as np
import pandas as pd
import pytest

from factors.attribution import (
    AttributionError,
    bias_statistic,
    euler_attribution,
    euler_contributions_dense,
    rolling_bias,
)
from factors.risk_model import RiskModelError
from tests.conftest import make_rm


def test_euler_contributions_add_up():
    rm = make_rm()
    w = pd.Series(1 / 30, index=rm.assets)
    a = euler_attribution(rm, w)
    assert a.factor_vol_contrib.sum() + a.specific_vol_contrib == pytest.approx(a.sigma, abs=1e-12)
    assert a.stock_vol_contrib.sum() == pytest.approx(a.sigma, abs=1e-12)
    t = a.table()
    assert t["pct_of_sigma"].sum() == pytest.approx(1.0, abs=1e-12)
    assert t["var_contrib_monthly"].sum() == pytest.approx(a.variance, abs=1e-14)
    S = rm.sigma().to_numpy()
    assert np.allclose(a.stock_vol_contrib.to_numpy(), euler_contributions_dense(w.to_numpy(), S))
    assert np.allclose(a.stock_mctr.to_numpy(), S @ w.to_numpy() / a.sigma)


def test_negative_contributions_are_kept():
    rm = make_rm(seed=5)
    rng = np.random.default_rng(5)
    w = pd.Series(rng.normal(0, 0.1, 30), index=rm.assets)   # long-short -> hedges
    a = euler_attribution(rm, w)
    assert (a.stock_vol_contrib < 0).any()
    assert a.stock_vol_contrib.sum() == pytest.approx(a.sigma, abs=1e-12)


def test_active_risk_attribution_equals_attribution_of_active_weights():
    rm = make_rm()
    rng = np.random.default_rng(1)
    w = pd.Series(rng.dirichlet(np.ones(30)), index=rm.assets)
    b = pd.Series(1 / 30, index=rm.assets)
    act = euler_attribution(rm, w, b)
    direct = euler_attribution(rm, w - b)
    assert act.sigma == pytest.approx(direct.sigma) and act.label == "active"
    assert np.allclose(act.factor_vol_contrib, direct.factor_vol_contrib)
    te = np.sqrt((w - b).to_numpy() @ rm.sigma().to_numpy() @ (w - b).to_numpy())
    assert act.sigma == pytest.approx(te)


def test_asset_ordering_and_missing_names():
    rm = make_rm()
    w = pd.Series(np.linspace(1, 2, 30), index=rm.assets)
    w = w / w.sum()
    a = euler_attribution(rm, w)
    b = euler_attribution(rm, w.sample(frac=1, random_state=0))
    assert a.sigma == pytest.approx(b.sigma)
    pd.testing.assert_series_equal(a.stock_vol_contrib, b.stock_vol_contrib)
    w2 = w.iloc[:10]                      # missing names are zero weight
    assert euler_attribution(rm, w2).stock_vol_contrib.iloc[10:].abs().max() == 0
    with pytest.raises(RiskModelError):   # unknown names with weight raise
        euler_attribution(rm, pd.concat([w, pd.Series({"ZZZ": 0.1})]))


def test_zero_risk_raises():
    rm = make_rm()
    with pytest.raises(AttributionError):
        euler_attribution(rm, pd.Series(0.0, index=rm.assets))
    w = pd.Series(1 / 30, index=rm.assets)
    with pytest.raises(AttributionError):
        euler_attribution(rm, w, benchmark=w)   # zero active risk


def test_bias_statistic_definitions():
    z = pd.Series([1.0, -1.0, 1.0, -1.0])
    b = bias_statistic(z)
    assert b["bias"] == pytest.approx(1.0) and b["ci_low"] < 1 < b["ci_high"]
    bd = bias_statistic(z + 1, demean=True)
    assert bd["bias"] == pytest.approx(np.std([1, -1, 1, -1], ddof=1))
    rb = rolling_bias(pd.Series([2.0] * 15), 12)
    assert rb.iloc[:11].isna().all() and rb.iloc[11:].eq(2.0).all()
