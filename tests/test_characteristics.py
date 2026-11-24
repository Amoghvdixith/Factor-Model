import numpy as np
import pandas as pd
import pytest

from factors.characteristics import (
    add_characteristics,
    book_to_market,
    composite_quality,
    low_vol,
    momentum_12_1,
    quality_components,
    winsorise_cs,
    zscore_cs,
)


def test_momentum_endpoints_skip_most_recent_month():
    idx = pd.period_range("2020-01", periods=14, freq="M")
    r = pd.DataFrame({"A": 0.0}, index=idx)
    t = 13                                   # formation month index
    r.iloc[t, 0] = 0.50                      # month t: skipped
    r.iloc[t - 12, 0] = 0.40                 # month t-12: outside the window
    r.iloc[t - 1, 0] = 0.10                  # month t-1: included
    r.iloc[t - 11, 0] = 0.20                 # month t-11: included
    m = momentum_12_1(r)
    assert m.iloc[t, 0] == pytest.approx(1.10 * 1.20 - 1)
    assert np.isnan(m.iloc[10, 0])           # fewer than 11 past returns


def test_momentum_requires_all_returns():
    idx = pd.period_range("2020-01", periods=14, freq="M")
    r = pd.DataFrame({"A": 0.01}, index=idx)
    r.iloc[8, 0] = np.nan
    assert np.isnan(momentum_12_1(r).iloc[13, 0])


def test_low_vol_min_obs_and_sign():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"A": rng.normal(0, 0.01, 300), "B": rng.normal(0, 0.02, 300)}, index=idx)
    d.iloc[:200, 1] = np.nan
    lv = low_vol(d, pd.DatetimeIndex([idx[-1]]), 252, 126)
    assert lv["A"].iloc[0] < 0 and np.isnan(lv["B"].iloc[0])   # B has only 100 obs
    assert lv["A"].iloc[0] == pytest.approx(-d["A"].iloc[-252:].std() * np.sqrt(252))


def test_value_and_quality_invalid_denominators():
    be = pd.Series([10.0, -5.0, 0.0, 10.0])
    me = pd.Series([100.0, 100.0, 100.0, 0.0])
    assert book_to_market(be, me).isna().tolist() == [False, True, True, True]
    f = pd.DataFrame({"book_equity": [10.0, -1.0], "net_income": [1.0, 1.0],
                      "gross_profit": [3.0, 3.0], "total_assets": [20.0, 0.0], "total_debt": [5.0, 5.0]})
    q = quality_components(f)
    assert q.iloc[0].tolist() == pytest.approx([0.1, 0.15, -0.25])
    assert q.iloc[1].isna().all()


def _cs(values_by_date):
    parts = []
    for d, v in values_by_date.items():
        parts.append(pd.Series(v, index=pd.MultiIndex.from_product(
            [[pd.Timestamp(d)], [f"T{i:02d}" for i in range(len(v))]])))
    return pd.concat(parts)


def test_winsorise_and_zscore_are_per_date():
    rng = np.random.default_rng(1)
    x = _cs({"2020-01-31": rng.normal(0, 1, 50), "2020-02-29": rng.normal(5, 3, 50)})
    w = winsorise_cs(x, 0.05, 0.95)
    a = x.xs(pd.Timestamp("2020-01-31"))
    assert w.xs(pd.Timestamp("2020-01-31")).max() == pytest.approx(a.quantile(0.95))
    z = zscore_cs(w, min_n=10)
    for _, g in z.groupby(level=0):
        assert g.mean() == pytest.approx(0, abs=1e-12) and g.std() == pytest.approx(1)
    x2 = x.copy()
    x2.loc[pd.Timestamp("2020-02-29")] = 1e6    # mutate another date
    z2 = zscore_cs(winsorise_cs(x2, 0.05, 0.95), min_n=10)
    pd.testing.assert_series_equal(z.xs(pd.Timestamp("2020-01-31")), z2.xs(pd.Timestamp("2020-01-31")))


def test_zscore_edge_cases():
    x = _cs({"2020-01-31": np.ones(30), "2020-02-29": np.arange(5.0)})
    z = zscore_cs(x, min_n=10)
    assert z.isna().all()       # zero dispersion and too-small cross-section
    x = _cs({"2020-01-31": np.arange(30.0)})
    w = pd.Series(np.r_[np.zeros(29), 1.0], index=x.index)
    zw = zscore_cs(x, weights=w, min_n=10)
    assert zw.iloc[-1] == pytest.approx(0.0)   # all weight on the last name -> its z is 0


def test_sector_relative_demeaning():
    x = _cs({"2020-01-31": np.r_[np.full(15, 10.0), np.full(15, 0.0)] + np.tile([0.0, 1.0, 2.0], 10)})
    by = pd.Series(["S1"] * 15 + ["S2"] * 15, index=x.index)
    z = zscore_cs(x, min_n=10, by=by)
    assert z.groupby(by).mean().abs().max() < 1e-12


def test_quality_min_components():
    comp = pd.DataFrame({"roe": _cs({"2020-01-31": np.arange(30.0)}),
                         "gpa": _cs({"2020-01-31": np.arange(30.0)[::-1]}),
                         "neg_leverage": _cs({"2020-01-31": np.r_[np.full(29, np.nan), 1.0]})})
    comp.iloc[0, 1] = np.nan
    q = composite_quality(comp, min_components=2, min_n=10)
    assert np.isnan(q.iloc[0]) and q.iloc[1:].notna().all()


def test_add_characteristics_keeps_raw_and_processed(tiny_panel):
    p, close = tiny_panel
    for c in ("value", "momentum", "low_vol", "quality", "size"):
        assert f"{c}_raw" in p.columns and c in p.columns


def test_characteristics_future_mutation_invariance(tiny_market):
    from factors.data import validate_fundamentals
    from factors.panel import asof_join_fundamentals, build_monthly_panel
    from tests.conftest import CHAR_CFG

    spec, mk, fund = tiny_market
    close = mk["ohlcv"]["close"].copy()
    meta = mk["meta"].set_index("ticker")
    def run(c, f):
        p = build_monthly_panel(c, meta["industry"], meta["shares"], mk["rf"])
        return add_characteristics(p, c, asof_join_fundamentals(p, validate_fundamentals(f)), CHAR_CFG)
    base = run(close, fund)
    cut = base.index.get_level_values(0).unique()[30]
    c2 = close.copy()
    c2.loc[c2.index > cut] *= 3.0
    f2 = fund.copy()
    f2.loc[f2["publication_date"] > cut, "book_equity"] *= -7
    mut = run(c2, f2)
    cols = ["value", "momentum", "low_vol", "quality", "size", "value_raw", "momentum_raw"]
    a = base.loc[base.index.get_level_values(0) <= cut, cols]
    b = mut.loc[mut.index.get_level_values(0) <= cut, cols]
    pd.testing.assert_frame_equal(a, b)
