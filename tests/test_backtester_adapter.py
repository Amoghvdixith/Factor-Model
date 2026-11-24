import sys

import numpy as np
import pandas as pd
import pytest

from tests.conftest import PORT_CFG

bt = pytest.importorskip("backtester")

from factors.backtester_adapter import (  # noqa: E402
    BacktesterUnavailable,
    make_strategy_classes,
    market_data_from_long,
    run_schedule,
)


def _market(n=25, days=90, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=days)
    rows = []
    for j in range(n):
        c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, days)))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.002, days))
        for i, d in enumerate(idx):
            rows.append({"date": d, "symbol": f"S{j:02d}", "open": o[i], "close": c[i],
                         "high": max(o[i], c[i]) * 1.001, "low": min(o[i], c[i]) * 0.999,
                         "volume": 1e6})
    return market_data_from_long(pd.DataFrame(rows)), idx


def _month_ends(idx):
    s = pd.Series(idx, index=idx)
    return list(s.groupby(idx.to_period("M")).max())


def test_schedule_executes_at_next_open_with_costs():
    md, idx = _market()
    me = _month_ends(idx)[:2]
    sched = {t: pd.Series(1 / 25, index=[f"S{j:02d}" for j in range(25)]) for t in me}
    ex = run_schedule(md, sched, "ew", PORT_CFG, "SYNTHETIC", 0.05)
    fills = ex.result.fills
    first_fill = pd.Timestamp(fills["timestamp"].min())
    assert first_fill > me[0]                                   # never at the signal close
    assert first_fill == idx[idx.searchsorted(me[0], side="right")]   # the next session open
    assert (fills["fill_price"] > 0).all()
    assert ex.costs["commissions"] > 0
    tvr = ex.target_vs_realised
    assert len(tvr) == 2 and (tvr["residual_cash"].abs() < 0.02).all()
    assert (tvr["realised_max_weight"] <= 0.05 + 0.005).all()


def test_strategy_only_emits_on_schedule_dates():
    md, idx = _market(days=60)
    me = _month_ends(idx)[0]
    Strat, Sizer = make_strategy_classes()
    s = Strat({me: pd.Series({"S00": 0.5, "S01": 0.5})})
    engine = bt.Backtest(md, s, sizer=Sizer(), config=bt.BacktestConfig(fractional=True))
    engine.run()
    assert s.emitted == [me]
    assert Sizer().weights([bt.SignalEvent(me, "S00", 0.3)], None) == {"S00": 0.3}


def test_backtester_unavailable_raises(monkeypatch):
    import factors.backtester_adapter as ad

    monkeypatch.setitem(sys.modules, "backtester", None)
    with pytest.raises(BacktesterUnavailable):
        ad._bt()
    assert ad.backtester_version() is None
