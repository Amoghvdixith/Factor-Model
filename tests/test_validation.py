import numpy as np
import pandas as pd
import pytest

from factors.risk_model import build_risk_model, estimate_factor_returns
from factors.validation import (
    HoldoutGuard,
    HoldoutViolation,
    TrialLog,
    compute_splits,
    forecast_table,
    random_portfolios,
    realise,
)
from tests.conftest import RISK_CFG

STYLES = ["size", "value", "momentum", "low_vol", "quality"]


def test_splits_are_contiguous_and_complete():
    m = pd.period_range("2000-01", periods=100, freq="M")
    sp = compute_splits(m, {"train": 0.45, "validation": 0.2, "oos": 0.2, "holdout": 0.15})
    assert sp["train"][0] == m[0] and sp["holdout"][1] == m[-1]
    order = ["train", "validation", "oos", "holdout"]
    for a, b in zip(order, order[1:], strict=False):
        assert sp[b][0] == sp[a][1] + 1
    with pytest.raises(ValueError):
        compute_splits(m, {"train": 0.5, "validation": 0.2, "oos": 0.2, "holdout": 0.2})


def test_holdout_guard_and_truncation():
    g = HoldoutGuard(pd.Period("2020-04", "M"))
    df = pd.DataFrame({"date": pd.bdate_range("2020-03-25", "2020-04-10"), "x": 1})
    out = g.truncate_daily(df)
    assert out["date"].max() < pd.Timestamp("2020-04-01")
    with pytest.raises(HoldoutViolation):
        g.check(df["date"])


def test_run_holdout_refuses_without_confirm():
    from factors.pipeline import HoldoutLocked, run_holdout

    with pytest.raises(HoldoutLocked):
        run_holdout(confirm=False)


def test_trial_log_appends(tmp_path):
    log = TrialLog(tmp_path / "t.csv")
    log.append({"spec": "a", "value": 1})
    log.append({"spec": "b", "value": 2})
    df = pd.read_csv(tmp_path / "t.csv")
    assert df["spec"].tolist() == ["a", "b"]


def test_random_portfolios_reproducible():
    a = random_portfolios(pd.Index(list("abcde")), 3, 7, 20200131)
    b = random_portfolios(pd.Index(list("abcde")), 3, 7, 20200131)
    c = random_portfolios(pd.Index(list("abcde")), 3, 7, 20200228)
    assert all(a[k].equals(b[k]) for k in a)
    assert not a["random_00"].equals(c["random_00"])
    assert all(abs(v.sum() - 1) < 1e-12 and (v >= 0).all() for v in a.values())


def test_forecasts_are_made_before_outcomes(tiny_panel):
    p, _ = tiny_panel
    fr = estimate_factor_returns(p, STYLES, min_stocks=20)
    dates = [t for t in sorted(fr.exposures) if (fr.factor_returns.index <= t).sum() >= 12][:6]
    rms = {t: build_risk_model(fr, p, t, RISK_CFG) for t in dates}
    fc = forecast_table(rms, n_random=2, seed=1)
    assert "realised_return" not in fc.columns
    # forecasts do not change when outcomes after each forecast date are altered
    p2 = p.copy()
    late = p2["return_date"] > dates[-1]
    p2.loc[late, "ret_fwd"] = 9.0
    fr2 = estimate_factor_returns(p2, STYLES, min_stocks=20)
    fc2 = forecast_table({t: build_risk_model(fr2, p2, t, RISK_CFG) for t in dates}, n_random=2, seed=1)
    assert np.allclose(fc["sigma_hat_monthly"], fc2["sigma_hat_monthly"])
    rz = realise(fc, p)
    row = rz.iloc[0]
    g = p.xs(row["date"], level=0)
    w = pd.Series(1 / len(rms[row["date"]].assets), index=rms[row["date"]].assets)
    assert row["realised_return"] == pytest.approx(float((w * g["ret_fwd"].reindex(w.index).fillna(0)).sum()))
    assert row["z"] == pytest.approx(row["realised_return"] / row["sigma_hat_monthly"])
