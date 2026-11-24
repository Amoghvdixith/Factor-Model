import numpy as np
import pandas as pd
import pytest

from factors.fama_macbeth import fama_macbeth, hac_mean_test, linearmodels_crosscheck


def _panel(T=60, n=80, slope=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in pd.date_range("2010-01-31", periods=T, freq="ME"):
        x = rng.standard_normal(n)
        z = rng.standard_normal(n)
        ind = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
        y = 0.005 + slope * x + 0.0 * z + rng.normal(0, 0.05, n)
        for i in range(n):
            rows.append({"date": t, "ticker": f"S{i}", "x": x[i], "z": z[i], "industry": ind[i],
                         "ret_fwd_excess": y[i]})
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_fm_recovers_slope_and_requires_forward_returns():
    p = _panel()
    r = fama_macbeth(p, "ret_fwd_excess", ["x", "z"], min_obs=30, hac_lags=3)
    s = r.summary.set_index("term")
    assert abs(s.loc["x", "premium_monthly"] - 0.01) < 4 * s.loc["x", "se_hac"]
    assert s.loc["x", "n_months"] == 60
    with pytest.raises(ValueError):
        fama_macbeth(p.rename(columns={"ret_fwd_excess": "ret"}), "ret", ["x"])


def test_fm_slope_mean_equals_average_of_monthly_ols():
    p = _panel(T=12)
    r = fama_macbeth(p, "ret_fwd_excess", ["x"], min_obs=30, hac_lags=0)
    manual = []
    for _, g in p.groupby(level=0):
        X = np.column_stack([np.ones(len(g)), g["x"]])
        manual.append(np.linalg.lstsq(X, g["ret_fwd_excess"], rcond=None)[0][1])
    assert r.summary.set_index("term").loc["x", "premium_monthly"] == pytest.approx(np.mean(manual))


def test_industry_dummies_use_reference_and_rank_deficient_months_dropped():
    p = _panel(T=6)
    r = fama_macbeth(p, "ret_fwd_excess", ["x"], industry_col="industry", min_obs=30, hac_lags=0)
    assert "ind_A" not in r.gammas.columns and {"ind_B", "ind_C"} <= set(r.gammas.columns)
    p2 = p.copy()
    first = p2.index.get_level_values(0)[0]
    p2.loc[first, "z"] = 1.0  # constant regressor -> collinear with the intercept
    r2 = fama_macbeth(p2, "ret_fwd_excess", ["x", "z"], min_obs=30, hac_lags=0)
    dropped = [m for m in r2.month_log if m["dropped"]]
    assert len(dropped) == 1 and "rank" in dropped[0]["reason"]


def test_min_obs_drop_logged():
    p = _panel(T=4, n=20)
    with pytest.raises(ValueError):
        fama_macbeth(p, "ret_fwd_excess", ["x"], min_obs=30)


def test_hac_mean_equals_statsmodels_convention():
    x = pd.Series(np.random.default_rng(2).standard_normal(100))
    h0 = hac_mean_test(x, 0, small_sample=False)
    assert h0["se"] == pytest.approx(x.std(ddof=0) / np.sqrt(100))


def test_linearmodels_crosscheck_matches():
    pytest.importorskip("linearmodels")
    p = _panel(T=36)
    own = fama_macbeth(p, "ret_fwd_excess", ["x", "z"], min_obs=30, hac_lags=4)
    lm = linearmodels_crosscheck(p, "ret_fwd_excess", ["x", "z"], 30, bandwidth=4)
    s = own.summary.set_index("term")
    assert np.allclose(lm["params"].to_numpy(), s["premium_monthly"].to_numpy(), atol=1e-12)
    # linearmodels' Bartlett kernel (bandwidth 4) equals statsmodels HAC(4) with T/(T-1) correction
    assert np.allclose(lm["se_kernel"].to_numpy(), s["se_hac"].to_numpy(), rtol=1e-8)
    T = len(own.gammas)
    assert np.allclose(lm["se_unadjusted"].to_numpy(),
                       (own.gammas[["const", "x", "z"]].std(ddof=1) / np.sqrt(T)).to_numpy(), rtol=1e-8)
