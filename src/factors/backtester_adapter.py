"""Thin adapter to the local ``backtester`` package (project 02), public API only.

The backtester's own conventions are respected, not re-implemented:

* a strategy is called at each session CLOSE; orders created then execute no earlier
  than the next session's OPEN (``execution_delay`` sessions later);
* a returned list of ``SignalEvent`` is a complete target view (unmentioned holdings go
  to zero);
* costs (here a proportional one-way cost), whole-share rounding, residual cash and the
  ``cash_buffer`` are handled by the engine;
* prices are adjusted (the engine does not model dividend cash events).

This module adds only:

* :class:`TargetScheduleStrategy` - emits precomputed monthly targets at month-end
  closes.  Targets are keyed by formation date and may only be emitted at or after that
  date (asserted), so the schedule cannot leak future information into the engine.
* :class:`PassThroughSizer` - uses ``SignalEvent.direction`` as the target weight.
* helpers to run a schedule and compare realised with target weights.

If the backtester is not installed, :class:`BacktesterUnavailable` is raised.  There is no
substitute execution engine in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class BacktesterUnavailable(ImportError):
    """Raised when the local backtester package cannot be imported."""


def _bt() -> Any:
    try:
        import backtester
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BacktesterUnavailable(
            "the local 'backtester' package (project 02) is not installed; install it with "
            "`uv pip install <path-to-Backtester>`") from exc
    return backtester


def backtester_version() -> str | None:
    try:
        return str(_bt().__version__)
    except BacktesterUnavailable:
        return None


def make_strategy_classes() -> tuple[type, type]:
    """Create adapter classes bound to the installed backtester's base classes."""
    bt = _bt()

    class TargetScheduleStrategy(bt.Strategy):  # type: ignore[name-defined,misc]
        """Emit target weights from a {formation_date: Series} schedule at month-end."""

        name = "target_schedule"

        def __init__(self, schedule: dict[pd.Timestamp, pd.Series], label: str = "") -> None:
            super().__init__(label=label, n_rebalances=len(schedule))
            self.schedule = {pd.Timestamp(k): v for k, v in schedule.items()}
            self.emitted: list[pd.Timestamp] = []

        def on_bar(self, event: Any) -> list | None:
            now = self.view.now
            if now not in self.schedule:
                return None
            # look-ahead guard: a target formed at t may only be emitted at the close of t
            assert all(d <= now for d in self.emitted + [now])
            w = self.schedule[now]
            members = set(self.view.universe())
            sigs = [self.signal(event, s, float(v)) for s, v in w.items()
                    if v > 0 and s in members]
            self.emitted.append(now)
            return sigs

    class PassThroughSizer(bt.Sizer):  # type: ignore[name-defined,misc]
        """Target weight = signal direction (weights already sum to <= 1)."""

        def weights(self, signals: list, view: Any) -> dict[str, float]:
            return {s.symbol: float(s.direction) for s in signals}

    return TargetScheduleStrategy, PassThroughSizer


@dataclass
class ExecutedSchedule:
    label: str
    result: Any                      # backtester.BacktestResult
    returns: pd.Series               # daily strategy returns (from trade start)
    realised_weights: pd.DataFrame   # rows: execution date, cols: tickers (end-of-day)
    target_vs_realised: pd.DataFrame
    costs: dict[str, float]


def market_data_from_long(prices: pd.DataFrame, provenance: dict | None = None) -> Any:
    return _bt().MarketData.from_long(prices, provenance)


def run_schedule(market: Any, schedule: dict[pd.Timestamp, pd.Series], label: str,
                 cfg: dict[str, Any], data_mode: str, max_weight: float) -> ExecutedSchedule:
    """Run a monthly target schedule through the backtester."""
    bt = _bt()
    Strat, Sizer = make_strategy_classes()
    bcfg = cfg["backtest"]
    first = min(schedule)
    config = bt.BacktestConfig(
        initial_cash=float(bcfg["initial_cash"]), fractional=bool(bcfg["fractional"]),
        cash_buffer=float(bcfg["cash_buffer"]), trade_start=market.index[
            market.index.searchsorted(first, side="right")],
        execution_delay=int(bcfg["execution_delay"]), universe_exit="liquidate")
    engine = bt.Backtest(
        market, Strat(schedule, label=label), sizer=Sizer(),
        constraints=bt.PortfolioConstraints(max_gross=1.0, max_net=1.0,
                                            max_position=max_weight + 1e-9, allow_short=False),
        costs=bt.CostModel.proportional(float(bcfg["cost_bps_one_way"])),
        config=config, label=label, data_mode=data_mode)
    res = engine.run()
    rets = res.returns()
    rw = realised_weights(res, market, schedule)
    tvr = compare_targets(schedule, rw, market)
    return ExecutedSchedule(label, res, rets, rw, tvr, res.cost_summary())


def realised_weights(res: Any, market: Any, schedule: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    """End-of-day weights on each execution day (first session after each formation date)."""
    fills = res.fills
    close = market.frame("close").ffill()   # last observed close per symbol (stale marks)
    eq = res.equity["equity"]
    if fills.empty:
        return pd.DataFrame()
    q = fills.pivot_table(index="timestamp", columns="symbol", values="quantity", aggfunc="sum")
    pos = q.reindex(close.index).fillna(0.0).cumsum()
    out = {}
    for t in sorted(schedule):
        k = close.index.searchsorted(t, side="right")
        if k >= len(close.index):
            continue
        d = close.index[k]
        cols = pos.columns
        val = pos.loc[d, cols] * close.loc[d, cols]
        out[d] = val / float(eq.loc[d])
    return pd.DataFrame(out).T.fillna(0.0)


def compare_targets(schedule: dict[pd.Timestamp, pd.Series], rw: pd.DataFrame,
                    market: Any) -> pd.DataFrame:
    """Per rebalance: |target - realised| summaries and residual cash."""
    rows = []
    idx = market.index
    for t in sorted(schedule):
        k = idx.searchsorted(t, side="right")
        if k >= len(idx) or idx[k] not in rw.index:
            continue
        d = idx[k]
        tw = schedule[t]
        r = rw.loc[d]
        allt = tw.index.union(r.index)
        diff = (tw.reindex(allt, fill_value=0) - r.reindex(allt, fill_value=0)).abs()
        tradable = market.frame("close").loc[d].reindex(r.index).notna()
        stuck = float(r[(~tradable) & (r.abs() > 0)].sum())
        rows.append({"formation_date": t, "execution_date": d,
                     "sum_abs_diff": float(diff.sum()), "max_abs_diff": float(diff.max()),
                     "target_invested": float(tw.sum()), "realised_invested": float(r.sum()),
                     "residual_cash": 1.0 - float(r.sum()),
                     "untradeable_weight": stuck,
                     "sum_abs_diff_excl_untradeable": float(diff.sum()) - abs(stuck),
                     "realised_max_weight": float(r.max()) if len(r) else np.nan})
    return pd.DataFrame(rows)
