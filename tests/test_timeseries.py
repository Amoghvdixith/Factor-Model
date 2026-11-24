import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from factors.timeseries import (
    excess_returns,
    factor_regression,
    hac_lags_auto,
    newey_west_cov,
    rolling_betas,
)


def _sim(T=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1970-01", periods=T, freq="M")
    F = pd.DataFrame(rng.normal(0.005, 0.04, (T, 3)), index=idx, columns=["MKT_RF", "SMB", "HML"])
    beta = np.array([1.1, 0.4, -0.3])
    y = pd.Series(0.002 + F.to_numpy() @ beta + rng.normal(0, 0.01, T), index=idx, name="y")
    return y, F, beta


def test_hac_lag_rule():
    assert hac_lags_auto(100) == 4
    assert hac_lags_auto(759) == int(np.floor(4 * (7.59) ** (2 / 9)))


def test_regression_recovers_known_loadings():
    y, F, beta = _sim()
    r = factor_regression(y, F, hac_lags=4)
    for k, b in zip(F.columns, beta, strict=True):
        assert abs(r.params[k] - b) < 4 * r.se[k]
    assert abs(r.params["alpha"] - 0.002) < 4 * r.se["alpha"]
    assert r.nobs == 600 and r.hac_lags == 4


@pytest.mark.parametrize("corr", [True, False])
def test_newey_west_matches_statsmodels(corr):
    rng = np.random.default_rng(3)
    T = 250
    X = sm.add_constant(rng.standard_normal((T, 2)))
    e = np.convolve(rng.standard_normal(T + 2), [1, 0.6, 0.3])[:T] * (1 + np.abs(X[:, 1]))
    y = X @ [0.1, 1.0, -0.5] + e
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": corr})
    V = newey_west_cov(X, res.resid, 5, small_sample=corr)
    assert np.max(np.abs(V - res.cov_params())) < 1e-10


def test_excess_return_alignment_and_no_double_subtraction():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    raw = pd.Series([0.02, 0.01, -0.01], index=idx)
    rf = pd.Series([0.001, 0.002, 0.003], index=idx)
    ex = excess_returns(raw, rf)
    assert ex.tolist() == pytest.approx([0.019, 0.008, -0.013])
    with pytest.raises(ValueError):
        excess_returns(ex, rf)
    with pytest.raises(ValueError):
        excess_returns(raw, rf.iloc[:2])


def test_rolling_betas_are_causal():
    y, F, _ = _sim(T=120)
    rb = rolling_betas(y, F, window=36)
    assert rb.iloc[:35].isna().all().all() and rb.iloc[35:]["nobs"].eq(36).all()
    y2 = y.copy()
    y2.iloc[80:] = y2.iloc[80:] * 5 + 1.0   # mutate the future
    rb2 = rolling_betas(y2, F, window=36)
    pd.testing.assert_frame_equal(rb.iloc[:80], rb2.iloc[:80])
    t = 60  # the row at t equals OLS on observations t-35..t
    X = sm.add_constant(F.iloc[t - 35:t + 1])
    b = sm.OLS(y.iloc[t - 35:t + 1], X).fit().params.to_numpy()
    assert np.allclose(rb.iloc[t, :4].to_numpy(float), b)
