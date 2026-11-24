import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from factors.timeseries import GRSError, grs_from_guide_formula, grs_test


def _data(T=240, N=6, L=2, seed=0, alpha=0.0):
    rng = np.random.default_rng(seed)
    F = pd.DataFrame(rng.normal(0.005, 0.04, (T, L)), columns=[f"f{i}" for i in range(L)])
    B = rng.normal(1, 0.3, (L, N))
    R = pd.DataFrame(alpha + F.to_numpy() @ B + rng.normal(0, 0.02, (T, N)),
                     columns=[f"p{i}" for i in range(N)])
    return R, F


def test_grs_n1_equals_squared_ols_t():
    R, F = _data(N=1)
    g = grs_test(R, F)
    res = sm.OLS(R["p0"], sm.add_constant(F)).fit()
    assert g.statistic == pytest.approx(res.tvalues.iloc[0] ** 2, rel=1e-10)
    assert g.pvalue == pytest.approx(res.pvalues.iloc[0], rel=1e-8)
    assert (g.df1, g.df2) == (1, 240 - 1 - 2)


def test_guide_formula_equivalence_requires_matching_divisors():
    R, F = _data()
    g = grs_test(R, F).statistic
    assert grs_from_guide_formula(R, F, "T-L-1", "T") == pytest.approx(g, rel=1e-12)
    assert grs_from_guide_formula(R, F, "T-L-1", "T-1") != pytest.approx(g, rel=1e-6)
    assert grs_from_guide_formula(R, F, "T", "T") != pytest.approx(g, rel=1e-6)


def test_grs_pvalue_and_degrees_of_freedom():
    R, F = _data(alpha=0.01)
    g = grs_test(R, F)
    assert g.pvalue == pytest.approx(stats.f.sf(g.statistic, 6, 240 - 6 - 2))
    assert g.pvalue < 1e-6  # large true alphas are detected


def test_grs_edge_cases():
    R, F = _data(T=8, N=6, L=2)
    with pytest.raises(GRSError):
        grs_test(R, F)  # T - N - L = 0
    R, F = _data()
    R["dup"] = R["p0"]  # singular residual covariance
    with pytest.raises(GRSError):
        grs_test(R, F)
    with pytest.raises(ValueError):
        grs_test(R.rename(columns={"p1": "f0"}), F)


def test_grs_common_sample():
    R, F = _data()
    R.iloc[5, 2] = np.nan
    g = grs_test(R, F)
    assert g.T == 239
    g2 = grs_test(R.drop(index=5), F.drop(index=5))
    assert g.statistic == pytest.approx(g2.statistic)
