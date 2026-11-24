import numpy as np
import pandas as pd
import pytest

from factors.data import validate_fundamentals
from factors.panel import (
    PanelError,
    asof_join_fundamentals,
    assert_point_in_time,
    build_monthly_panel,
    truncate_panel,
)


def _keys(dates, tickers=("X",)):
    idx = pd.MultiIndex.from_product([pd.to_datetime(dates), list(tickers)], names=["date", "ticker"])
    return pd.DataFrame(index=idx)


def _fund():
    return validate_fundamentals(pd.DataFrame({
        "ticker": ["X", "X", "X"],
        "fiscal_period_end": ["2019-12-31", "2019-12-31", "2020-12-31"],
        "currency": "USD",
        "publication_date": ["2020-03-15", "2020-03-15", "2021-03-20"],
        "vintage_date": ["2020-03-15", "2020-09-10", "2021-03-20"],   # restatement in Sept
        "book_equity": [100.0, 90.0, 120.0]}))


def test_asof_join_respects_availability_and_vintages():
    keys = _keys(["2020-02-28", "2020-03-31", "2020-08-31", "2020-09-30", "2021-03-31"])
    j = asof_join_fundamentals(keys, _fund())
    be = j["book_equity"].droplevel(1)
    assert np.isnan(be.loc["2020-02-28"])            # not yet published
    assert be.loc["2020-03-31"] == 100.0             # original vintage
    assert be.loc["2020-08-31"] == 100.0             # restatement not yet available
    assert be.loc["2020-09-30"] == 90.0              # restatement now available
    assert be.loc["2021-03-31"] == 120.0             # newer fiscal period
    av = j["fund_available_at"].dropna()
    assert (av.to_numpy() <= av.index.get_level_values(0).to_numpy()).all()


def test_no_backfill_of_current_values():
    j = asof_join_fundamentals(_keys(["2019-06-28"]), _fund())
    assert j["book_equity"].isna().all()


def test_stale_fundamentals_dropped():
    j = asof_join_fundamentals(_keys(["2022-12-30"]), _fund(), max_age_days=550)
    assert j["book_equity"].isna().all()   # FY2020 is > 550 days old


def test_assert_point_in_time_raises():
    bad = _keys(["2020-01-31"])
    bad["fund_available_at"] = pd.Timestamp("2020-02-15")
    with pytest.raises(PanelError):
        assert_point_in_time(bad)


def test_monthly_panel_alignment_and_delisting():
    idx = pd.bdate_range("2020-01-01", "2020-05-29")
    close = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    close.loc["2020-02-03":, "A"] = 110.0
    close.loc["2020-03-16", "B"] = 25.0     # delisting with -50% return
    close.loc["2020-03-17":, "B"] = np.nan
    rf = pd.Series(0.001, index=pd.period_range("2020-01", "2020-05", freq="M"))
    p = build_monthly_panel(close, pd.Series({"A": "I1", "B": "I2"}), pd.Series({"A": 1.0, "B": 2.0}), rf)
    a = p.xs("A", level=1)
    assert a.loc["2020-01-31", "ret_fwd"] == pytest.approx(0.10)
    assert a.loc["2020-02-28", "ret"] == pytest.approx(0.10)
    assert a.loc["2020-01-31", "ret_fwd_excess"] == pytest.approx(0.099)
    b = p.xs("B", level=1)
    assert b.loc["2020-02-28", "ret_fwd"] == pytest.approx(-0.5)   # delisting return included
    assert "2020-03-31" not in b.index.astype(str)                 # not alive after delisting
    assert (p["return_date"].dropna() > p["return_date"].dropna().index.get_level_values(0)).all()
    ee = p["exec_eligible"].dropna()
    assert (ee.to_numpy() > ee.index.get_level_values(0).to_numpy()).all()
    assert p["exec_eligible"].isna().sum() == 1          # final date: no later session
    assert p.loc[(pd.Timestamp("2020-01-31"), "B"), "mcap"] == 100.0


def test_truncate_panel_removes_future_targets(tiny_panel):
    p, _ = tiny_panel
    t = p.index.get_level_values(0).unique()[20]
    tp = truncate_panel(p, t)
    assert tp.index.get_level_values(0).max() == t
    assert tp.xs(t, level=0)["ret_fwd"].isna().all()
    prev = p.index.get_level_values(0).unique()[19]
    assert tp.xs(prev, level=0)["ret_fwd"].notna().any()
