import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from factors.risk_model import (
    RiskModelError,
    build_risk_model,
    estimate_factor_returns,
    ewma_cov,
    ewma_decay,
    ewma_weights,
    is_psd,
    specific_variance,
    wls,
)
from tests.conftest import RISK_CFG

STYLES = ["size", "value", "momentum", "low_vol", "quality"]


def test_ewma_decay_and_weights():
    lam = ewma_decay(12)
    assert lam ** 12 == pytest.approx(0.5)
    w = ewma_weights(25, 12)
    assert w.sum() == pytest.approx(1) and w[-1] / w[-13] == pytest.approx(2.0)
    with pytest.raises(RiskModelError):
        ewma_decay(0)


def test_ewma_cov_matches_manual_and_is_psd():
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(0, 0.03, (40, 3)), columns=list("abc"))
    C, d = ewma_cov(f, 10, min_periods=20)
    w = ewma_weights(40, 10)
    X = f.to_numpy()
    assert np.allclose(C.to_numpy(), (X * w[:, None]).T @ X)
    assert is_psd(C.to_numpy()) and d["n_eigen_floored"] == 0
    Cd, _ = ewma_cov(f, 10, min_periods=20, demean=True)
    m = w @ X
    assert np.allclose(Cd.to_numpy(), ((X - m) * w[:, None]).T @ (X - m))
    with pytest.raises(RiskModelError):
        ewma_cov(f.iloc[:10], 10, min_periods=20)


def test_ewma_cov_regularises_singular_input():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.03, 30)
    f = pd.DataFrame({"a": a, "b": a})     # perfectly collinear -> singular
    C, d = ewma_cov(f, 10, min_periods=10, eigen_floor=1e-8)
    assert d["n_eigen_floored"] == 1 and np.linalg.eigvalsh(C.to_numpy()).min() >= 1e-8 - 1e-15


def test_wls_weight_is_objective_weight_not_row_multiplier():
    rng = np.random.default_rng(2)
    X = sm.add_constant(rng.standard_normal((50, 2)))
    y = X @ [0.1, 1, -1] + rng.normal(0, 1, 50) * np.linspace(0.5, 2, 50)
    w = rng.uniform(1, 10, 50)
    b = wls(X, y, w)
    assert np.allclose(b, sm.WLS(y, X, weights=w).fit().params)
    assert not np.allclose(b, sm.WLS(y, X, weights=np.sqrt(w)).fit().params)


def test_identification_invariance_of_sigma(tiny_panel):
    p, _ = tiny_panel
    a = estimate_factor_returns(p, STYLES, "cap_weighted_sum_zero", min_stocks=20)
    b = estimate_factor_returns(p, STYLES, "reference", min_stocks=20)
    assert np.nanmax(np.abs(a.residuals.to_numpy() - b.residuals.to_numpy())) < 1e-12
    t = sorted(a.exposures)[-2]
    ra, rb_ = build_risk_model(a, p, t, RISK_CFG), build_risk_model(b, p, t, RISK_CFG)
    rel = np.abs(ra.sigma().to_numpy() - rb_.sigma().to_numpy()).max() / np.abs(ra.sigma().to_numpy()).max()
    assert rel < 1e-6
    # individual factor returns are basis dependent
    assert not np.allclose(a.factor_returns["market"], b.factor_returns["market"])


def test_cap_weighted_industry_constraint_holds(tiny_panel):
    p, _ = tiny_panel
    fr = estimate_factor_returns(p, STYLES, "cap_weighted_sum_zero", min_stocks=20)
    n_checked = 0
    for t in sorted(fr.exposures):
        g = p.xs(t, level=0)
        rd = g["return_date"].dropna()
        if rd.empty or rd.iloc[0] not in fr.factor_returns.index:
            continue
        ok = g["ret_fwd"].notna() & (g["mcap"] > 0)
        B = fr.exposures[t].loc[g.index[ok]]
        ind = [c for c in B.columns if c.startswith("ind_")]
        cap = g.loc[ok, "mcap"]
        c = np.array([cap[B[k] > 0].sum() for k in ind])
        f = fr.factor_returns.loc[rd.iloc[0], ind].to_numpy()
        assert abs(c @ f) / c.sum() < 1e-12
        n_checked += 1
    assert n_checked > 10


def test_factor_returns_use_exposures_known_before_returns(tiny_panel):
    p, _ = tiny_panel
    fr = estimate_factor_returns(p, STYLES, min_stocks=20)
    dates = sorted(fr.exposures)
    # each factor return is labelled with the month after the formation date
    est = [m["date"] for m in fr.log if m["estimated"]]
    assert len(est) == len(fr.factor_returns)
    for t in est[:5]:
        rd = p.xs(t, level=0)["return_date"].dropna().iloc[0]
        assert rd > t and rd in fr.factor_returns.index
    skipped = [m for m in fr.log if not m["estimated"]]
    assert all("reason" in m for m in skipped)   # early months lack momentum/low-vol: logged
    # mutating returns after t leaves the risk model at t unchanged
    t = dates[30]
    p2 = p.copy()
    late = p2["return_date"] > t
    p2.loc[late, "ret_fwd"] = p2.loc[late, "ret_fwd"] * -3 + 0.5
    r1 = build_risk_model(fr, p, t, RISK_CFG)
    r2 = build_risk_model(estimate_factor_returns(p2, STYLES, min_stocks=20), p2, t, RISK_CFG)
    assert np.allclose(r1.sigma().to_numpy(), r2.sigma().to_numpy(), rtol=1e-10, atol=1e-14)


def test_specific_variance_floor_shrinkage_and_short_history():
    idx = pd.date_range("2018-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(3)
    R = pd.DataFrame(rng.normal(0, 0.05, (24, 6)), index=idx, columns=list("abcdef"))
    R["e"] = 0.0001                     # tiny residuals -> floored
    R.loc[idx[:-3], "f"] = np.nan        # short history (3 obs)
    size = pd.Series([1, 2, 3, 4, 5, 6.0], index=R.columns)
    var, d = specific_variance(R, R.columns, size, 12, min_obs=6, shrink_k=12, n_buckets=2,
                               floor_vol=0.02)
    assert var["e"] > d.loc["e", "raw_var"]             # shrunk toward the bucket mean
    tiny = pd.DataFrame(1e-4, index=idx, columns=list("xyz"))
    vt, dt = specific_variance(tiny, tiny.columns, pd.Series([1, 2, 3.0], index=list("xyz")), 12,
                               min_obs=6, shrink_k=12, n_buckets=1, floor_vol=0.02)
    assert np.allclose(vt, 0.02 ** 2) and dt["floored"].all()
    assert d.loc["f", "shrink_weight"] == 0 and var["f"] == pytest.approx(max(d.loc["f", "target_var"], 4e-4))
    a = d.loc["a"]
    assert a["shrink_weight"] == pytest.approx(24 / 36)
    assert var["a"] == pytest.approx(max(a["shrink_weight"] * a["raw_var"] + (1 - a["shrink_weight"]) * a["target_var"], 4e-4))


def test_sigma_reconstruction_and_portfolio_variance(tiny_panel):
    p, _ = tiny_panel
    fr = estimate_factor_returns(p, STYLES, min_stocks=20)
    t = sorted(fr.exposures)[-2]
    rm = build_risk_model(fr, p, t, RISK_CFG)
    S = rm.sigma().to_numpy()
    B, F, D = rm.B.to_numpy(), rm.F.to_numpy(), rm.D.to_numpy()
    assert np.allclose(S, B @ F @ B.T + np.diag(D)) and is_psd(S)
    w = pd.Series(np.random.default_rng(4).dirichlet(np.ones(len(rm.assets))), index=rm.assets)
    assert rm.portfolio_variance(w) == pytest.approx(w.to_numpy() @ S @ w.to_numpy(), rel=1e-12)
    assert rm.portfolio_vol(w, annualise=True) == pytest.approx(np.sqrt(12 * rm.portfolio_variance(w)))
