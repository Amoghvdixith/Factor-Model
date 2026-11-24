import numpy as np
import pandas as pd
import pytest

from factors.sorts import assign_groups, quintile_portfolios, turnover_from_holdings


def _panel(n=30, dates=("2020-01-31", "2020-02-29"), seed=0, sig=None):
    rng = np.random.default_rng(seed)
    rows = []
    for k, d in enumerate(dates):
        for i in range(n):
            rows.append({"date": pd.Timestamp(d), "ticker": f"T{i:02d}",
                         "s": float(i) if sig is None else sig[i],
                         "ret_fwd": rng.normal(0, 0.05), "mcap": float(i + 1),
                         "return_date": pd.Timestamp(dates[k]) + pd.offsets.MonthEnd(1)})
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_groups_with_ties_are_balanced_and_deterministic():
    x = pd.Series([1.0] * 12, index=[f"T{i:02d}" for i in range(12)])
    g = assign_groups(x, 5)
    counts = g.value_counts().sort_index()
    assert counts.max() - counts.min() <= 1
    assert assign_groups(x.sample(frac=1, random_state=3), 5).sort_index().equals(g.sort_index())


def test_top_minus_bottom_arithmetic_ew_and_vw():
    p = _panel()
    for wt in ("ew", "vw"):
        r = quintile_portfolios(p, "s", wt, min_stocks=10)
        g = p.xs(pd.Timestamp("2020-01-31"), level=0)
        q1, q5 = g.iloc[:6], g.iloc[24:]
        if wt == "ew":
            e1, e5 = q1["ret_fwd"].mean(), q5["ret_fwd"].mean()
        else:
            e1 = np.average(q1["ret_fwd"], weights=q1["mcap"])
            e5 = np.average(q5["ret_fwd"], weights=q5["mcap"])
        row = r.returns.loc[pd.Timestamp("2020-02-29")]
        assert row["Q1"] == pytest.approx(e1) and row["Q5"] == pytest.approx(e5)
        assert row["LS"] == pytest.approx(e5 - e1)
        assert (r.counts == 6).all().all()


def test_missing_returns_renormalised_and_logged():
    p = _panel()
    p.loc[(pd.Timestamp("2020-01-31"), "T00"), "ret_fwd"] = np.nan
    r = quintile_portfolios(p, "s", "ew", min_stocks=10)
    g = p.xs(pd.Timestamp("2020-01-31"), level=0).iloc[1:6]
    assert r.returns.loc[pd.Timestamp("2020-02-29"), "Q1"] == pytest.approx(g["ret_fwd"].mean())
    assert any(e["event"] == "missing_next_return" for e in r.log)


def test_small_universe_skipped():
    p = pd.concat([_panel(n=8, dates=("2020-01-31",)), _panel(n=30, dates=("2020-02-29",))])
    r = quintile_portfolios(p, "s", "ew", min_stocks=10)
    assert list(r.counts.index) == [pd.Timestamp("2020-02-29")]
    assert r.log[0]["event"] == "skipped_small_universe" and r.log[0]["n"] == 8
    with pytest.raises(ValueError):
        quintile_portfolios(_panel(n=8), "s", "ew", min_stocks=10)


def test_turnover_uses_drifted_weights():
    H = pd.DataFrame({
        "formation_date": [pd.Timestamp("2020-01-31")] * 2 + [pd.Timestamp("2020-02-28")] * 2,
        "ticker": ["A", "B", "A", "C"], "group": 1, "weight": [0.5, 0.5, 0.5, 0.5],
        "ret_fwd": [0.10, -0.10, 0.0, 0.0]})
    t = turnover_from_holdings(H, 1)["Q1"]
    # drifted: A 0.55, B 0.45 -> target A 0.5, C 0.5: 0.5*(0.05 + 0.45 + 0.5) = 0.5
    assert t.iloc[1] == pytest.approx(0.5)
    assert np.isnan(t.iloc[0])
