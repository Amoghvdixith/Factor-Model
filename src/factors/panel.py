"""Point-in-time long panel indexed by (date, ticker).

Timestamps kept on every row
----------------------------
* ``date`` - the *formation* timestamp: the close of the last session of a month.
  Everything in the row except the ``*_fwd`` columns is known at that close.
* ``month`` - the calendar month (Period) that ends at ``date``.
* ``ret`` - return over the month ending at ``date`` (close-to-close, adjusted prices).
* ``ret_fwd`` / ``return_date`` - return over the NEXT month and the month-end at
  which it is realised.  These are *targets*, never inputs to signals.
* ``exec_eligible`` - the first session after ``date``: the earliest bar at which a
  trade based on this row may execute (next open, in the backtester convention).
* fundamental columns carry ``fund_fiscal_period_end``, ``fund_publication_date``,
  ``fund_vintage_date``/``fund_available_at`` and ``fund_availability_source``.

A final calendar month that is incomplete in the data (the last session is before the
month's last business day) is dropped.

Delistings: a stock that stops trading during month m+1 gets ``ret_fwd`` equal to its
last observed close over the month-m close, minus one.  In synthetic data that last close
already embeds the delisting return; real yfinance data contain no delisted firms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PanelError(ValueError):
    """Raised when a panel violates a structural or timing invariant."""


def validate_long(df: pd.DataFrame, keys: tuple[str, str] = ("date", "ticker"),
                  required: tuple[str, ...] = ()) -> None:
    """Check uniqueness, sorting and required columns of a long table."""
    cols = list(df.columns) + list(df.index.names or [])
    miss = [c for c in (*keys, *required) if c not in cols]
    if miss:
        raise PanelError(f"missing columns {miss}")
    d = df.reset_index() if any(k in (df.index.names or []) for k in keys) else df
    if d.duplicated(list(keys)).any():
        raise PanelError("duplicate (date, ticker) rows")
    if not d[list(keys)].apply(tuple, axis=1).is_monotonic_increasing:
        raise PanelError("panel is not sorted by (date, ticker)")


def month_end_sessions(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last session of each calendar month in a session calendar."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


def build_monthly_panel(close: pd.DataFrame, industry: pd.Series | None = None,
                        shares: pd.Series | None = None,
                        rf: pd.Series | None = None) -> pd.DataFrame:
    """Build the monthly point-in-time panel from wide daily adjusted closes.

    Parameters
    ----------
    close: wide daily closes (index sessions, columns tickers); NaN = no bar.
    industry: ticker -> industry label (static; see limitations).
    shares: ticker -> shares outstanding (static, synthetic only).  If None, ``mcap``
        is NaN and value-weighting / size are unavailable.
    rf: monthly risk-free rate (PeriodIndex, decimal) used for excess ``ret_fwd``.
    """
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise PanelError("daily close index must be sorted and unique")
    last = pd.Timestamp(close.index[-1])
    if last < last + pd.offsets.BMonthEnd(0):
        # the data end before the month's last business day: drop the partial month so
        # that no month-end row (and no monthly return) is built from an incomplete month
        close = close[close.index < last.to_period("M").start_time]
    me = month_end_sessions(pd.DatetimeIndex(close.index))
    month = pd.DatetimeIndex(close.index).to_period("M")
    last_obs = close.groupby(month).last()          # last observed close in month
    at_me = close.loc[me]                           # close exactly at month-end session
    at_me.index = me.to_period("M")
    ret = last_obs / at_me.shift(1) - 1.0           # needs alive at previous month-end
    ret_fwd = ret.shift(-1)
    months = at_me.index
    nxt_date = pd.Series(list(me[1:]) + [pd.NaT], index=months)
    sess = pd.DatetimeIndex(close.index)
    pos = sess.searchsorted(me, side="right")
    exec_elig = pd.Series([sess[int(p)] if p < len(sess) else pd.NaT for p in pos],
                          index=months, dtype="datetime64[ns]")

    parts = {"price": at_me, "ret": ret.reindex(index=months),
             "ret_fwd": ret_fwd.reindex(index=months)}
    long = pd.concat({k: v.stack() for k, v in parts.items()}, axis=1)
    long.index.names = ["month", "ticker"]
    long = long.reset_index()
    long = long[long["price"].notna()].copy()       # alive at the formation close
    long["date"] = long["month"].map(pd.Series(me, index=months))
    long["return_date"] = long["month"].map(nxt_date)
    long["exec_eligible"] = long["month"].map(exec_elig)
    if industry is not None:
        long["industry"] = long["ticker"].map(industry)
    else:
        long["industry"] = "Unknown"
    if shares is not None:
        long["mcap"] = long["price"] * long["ticker"].map(shares)
    else:
        long["mcap"] = np.nan
    if rf is not None:
        rf_next = rf.copy()
        rf_next.index = rf_next.index - 1            # rf of month m+1 aligned to row m
        long["rf_fwd"] = long["month"].map(rf_next)
        long["ret_fwd_excess"] = long["ret_fwd"] - long["rf_fwd"]
    long = long.set_index(["date", "ticker"]).sort_index()
    validate_long(long)
    if (long["return_date"].dropna() <= long["return_date"].dropna().index.get_level_values(0)).any():
        raise PanelError("return_date must be after the formation date")
    return long


def asof_join_fundamentals(keys: pd.DataFrame, fundamentals: pd.DataFrame,
                           max_age_days: int = 550) -> pd.DataFrame:
    """Availability-aware as-of join of fundamentals onto (date, ticker) keys.

    For each formation date t and ticker, among records with ``available_at <= t``, take
    the most recent *fiscal period*, and for that period its latest vintage available by
    t.  A restatement published after t is never used at t, and current values are never
    backfilled to earlier dates.  Records whose fiscal period ended more than
    ``max_age_days`` before t are treated as stale (NaN).

    ``keys`` must have a (date, ticker) MultiIndex; the result shares that index.
    """
    need = {"ticker", "fiscal_period_end", "available_at"}
    if not need.issubset(fundamentals.columns):
        raise PanelError(f"fundamentals need columns {sorted(need)} (use data.validate_fundamentals)")
    f = fundamentals.sort_values(["available_at"]).reset_index(drop=True)
    value_cols = [c for c in f.columns if c not in
                  ("ticker", "fiscal_period_end", "publication_date", "vintage_date",
                   "available_at", "availability_source", "currency", "is_restatement",
                   "assumed_lag_days")]
    out_rows = []
    dates = keys.index.get_level_values(0).unique().sort_values()
    for t in dates:
        avail = f[f["available_at"] <= t]
        if avail.empty:
            continue
        latest = (avail.sort_values(["ticker", "fiscal_period_end", "vintage_date"])
                  .groupby("ticker", sort=False).tail(1))
        latest = latest[(t - latest["fiscal_period_end"]).dt.days <= max_age_days]
        tick = keys.xs(t, level=0).index
        latest = latest[latest["ticker"].isin(tick)].copy()
        latest["date"] = t
        out_rows.append(latest)
    if not out_rows:
        res = pd.DataFrame(index=keys.index)
        for c in value_cols:
            res[c] = np.nan
        return res
    j = pd.concat(out_rows, ignore_index=True)
    ren = {"fiscal_period_end": "fund_fiscal_period_end",
           "publication_date": "fund_publication_date", "vintage_date": "fund_vintage_date",
           "available_at": "fund_available_at", "availability_source": "fund_availability_source",
           "is_restatement": "fund_is_restatement"}
    j = j.rename(columns=ren).set_index(["date", "ticker"])
    keep = value_cols + [v for k, v in ren.items() if v in j.columns]
    res = j[keep].reindex(keys.index)
    assert_point_in_time(res)
    return res


def assert_point_in_time(panel: pd.DataFrame) -> None:
    """Raise if any fundamental was used before it was available."""
    if "fund_available_at" not in panel.columns:
        return
    d = panel.index.get_level_values(0)
    av = pd.to_datetime(panel["fund_available_at"])
    bad = av.notna() & (av > d)
    if bad.any():
        raise PanelError(f"{int(bad.sum())} rows use fundamentals before their availability")


def truncate_panel(panel: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Information set at the close of ``t``: rows up to t, future targets removed."""
    p = panel[panel.index.get_level_values(0) <= t].copy()
    fut = p["return_date"] > t
    for c in [c for c in p.columns if c.endswith("_fwd") or c.startswith("ret_fwd")]:
        p.loc[fut, c] = np.nan
    return p


def wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pivot a panel column to a (date x ticker) frame."""
    return panel[col].unstack("ticker")
