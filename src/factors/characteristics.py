"""Stock characteristics and cross-sectional processing.

Definitions (all computed at the formation close ``t`` from information available at t)
--------------------------------------------------------------------------------------
* **value** - book-to-market: latest *available* book equity / market cap at t.  Book
  equity <= 0 -> NaN (B/M is not meaningful for negative book; documented exclusion).
  Market cap <= 0 or missing -> NaN.  Alternative: earnings yield = net income / market
  cap (negative earnings kept: a negative yield is still informative and ordered).
* **momentum (12-1)** - compounded return from the close at the end of month t-12 to
  the close at the end of month t-1: the 11 monthly returns of months t-11 ... t-1.
  Month t itself is skipped (short-term reversal).  All 11 returns must be present.
* **low_vol** - minus the annualised standard deviation of daily returns over the last
  ``lookback_days`` sessions up to and including t, with at least ``min_obs`` returns.
  Higher = less volatile.
* **quality** - equal-weighted mean of cross-sectional z-scores of ROE (net income /
  book equity, book > 0), gross profits / total assets (assets > 0) and negative
  leverage (-(total debt / total assets)); at least ``min_components`` must be present.
* **size** - log market cap (risk-model exposure; not a "premium" signal here).

Processing: per-date winsorisation at configurable quantiles, then a per-date z-score.
Z-scores never use pooled or future observations.  Raw values are retained alongside
processed values (``<name>_raw`` and ``<name>``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- raw signals
def momentum_12_1(monthly_ret: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """12-1 momentum from a (month x ticker) frame of monthly returns.

    Value at row t = prod(1 + r_{t-lookback+1+skip-1} ... r_{t-skip}) - 1, i.e. for the
    default the 11 returns of months t-11..t-1.  Requires all of them.
    """
    n = lookback - skip
    lr = monthly_ret.apply(np.log1p)
    s = lr.rolling(n, min_periods=n).sum().shift(skip)
    return s.apply(np.expm1)


def low_vol(daily_ret: pd.DataFrame, at: pd.DatetimeIndex, lookback: int = 252,
            min_obs: int = 126, periods_per_year: int = 252) -> pd.DataFrame:
    """Negative trailing annualised daily-return volatility sampled at dates ``at``."""
    sd = daily_ret.rolling(lookback, min_periods=min_obs).std(ddof=1)
    return -(sd.reindex(at) * np.sqrt(periods_per_year))


def book_to_market(book_equity: pd.Series, mcap: pd.Series) -> pd.Series:
    """B/M with non-positive book or market cap set to NaN."""
    be = book_equity.where(book_equity > 0)
    me = mcap.where(mcap > 0)
    return be / me


def earnings_yield(net_income: pd.Series, mcap: pd.Series) -> pd.Series:
    """E/P (negative earnings kept); non-positive market cap -> NaN."""
    return net_income / mcap.where(mcap > 0)


def quality_components(f: pd.DataFrame) -> pd.DataFrame:
    """ROE, gross profitability and negative leverage with invalid denominators -> NaN."""
    be = f["book_equity"].where(f["book_equity"] > 0)
    ta = f["total_assets"].where(f["total_assets"] > 0)
    return pd.DataFrame({
        "roe": f["net_income"] / be,
        "gpa": f["gross_profit"] / ta,
        "neg_leverage": -(f["total_debt"] / ta),
    }, index=f.index)


# --------------------------------------------------------------------------- processing
def winsorise_cs(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip each date's cross-section at its own quantiles (index level 0 = date).

    Quantiles use linear interpolation over that date's non-missing values only.
    """
    g = x.groupby(level=0)
    lo = g.transform("quantile", lower)
    hi = g.transform("quantile", upper)
    v = x.to_numpy(float)
    out = np.where(np.isnan(v), np.nan, np.minimum(np.maximum(v, lo.to_numpy(float)),
                                                    hi.to_numpy(float)))
    return pd.Series(out, index=x.index, name=x.name)


def zscore_cs(x: pd.Series, weights: pd.Series | None = None, min_n: int = 20,
              by: pd.Series | None = None) -> pd.Series:
    """Per-date z-score: (x - mean) / std.

    ``weights`` (e.g. market cap) gives a weighted mean; the dispersion is always the
    equal-weighted standard deviation (ddof=1).  Cross-sections with fewer than
    ``min_n`` valid values, or zero dispersion, return NaN for that date (logged).
    ``by`` (e.g. sector) demeans within groups first (sector-relative processing).
    """
    xs = x.astype(float).copy()
    if by is not None:
        keys = [xs.index.get_level_values(0), by.reindex(xs.index).to_numpy()]
        xs = xs - xs.groupby(keys).transform("mean")
    lvl = xs.index.get_level_values(0)
    g = xs.groupby(level=0)
    n = g.transform("count")
    sd = g.transform("std")  # ddof=1
    if weights is not None:
        w = weights.reindex(xs.index).astype(float).clip(lower=0).where(xs.notna(), 0.0).fillna(0.0)
        num = (xs.fillna(0.0) * w).groupby(lvl).transform("sum")
        den = w.groupby(lvl).transform("sum")
        mu = (num / den).where(den > 0, g.transform("mean"))
    else:
        mu = g.transform("mean")
    bad = (n < min_n) | ~np.isfinite(sd) | (sd <= 1e-12)
    if bad.any():
        for d in pd.Index(lvl[bad.to_numpy()]).unique():
            log.info("zscore: date %s skipped (n < %d or zero dispersion) -> NaN", d, min_n)
    return ((xs - mu) / sd).where(~bad)


def process(x: pd.Series, lower: float = 0.01, upper: float = 0.99, min_n: int = 20,
            weights: pd.Series | None = None, by: pd.Series | None = None) -> pd.Series:
    """Winsorise then z-score cross-sectionally."""
    return zscore_cs(winsorise_cs(x, lower, upper), weights=weights, min_n=min_n, by=by)


def composite_quality(comp: pd.DataFrame, min_components: int = 2, lower: float = 0.01,
                      upper: float = 0.99, min_n: int = 20) -> pd.Series:
    """Mean of processed component z-scores; NaN if fewer than ``min_components``."""
    z = pd.DataFrame({c: process(comp[c], lower, upper, min_n) for c in comp.columns})
    cnt = z.notna().sum(axis=1)
    q = z.mean(axis=1, skipna=True)
    return q.where(cnt >= min_components)


# --------------------------------------------------------------------------- panel API
def add_characteristics(panel: pd.DataFrame, daily_close: pd.DataFrame,
                        fundamentals_joined: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """Add raw and processed characteristics to the monthly panel.

    ``fundamentals_joined`` must come from :func:`factors.panel.asof_join_fundamentals`
    (same index as ``panel``) or be None (price-only mode: value/quality unavailable).
    """
    p = panel.copy()
    pc = cfg.get("processing", {})
    lo, hi, mn = pc.get("winsor_lower", 0.01), pc.get("winsor_upper", 0.99), pc.get("min_cross_section", 20)
    by = p["industry"] if pc.get("sector_relative", False) else None

    # momentum from monthly returns (month x ticker)
    mret = p["ret"].unstack("ticker")
    mcfg = cfg.get("momentum", {})
    mom = momentum_12_1(mret, mcfg.get("lookback_months", 12), mcfg.get("skip_months", 1))
    p["momentum_raw"] = mom.stack().reindex(p.index)

    # low volatility from daily returns
    dret = daily_close / daily_close.shift(1) - 1.0
    lcfg = cfg.get("low_vol", {})
    dates = pd.DatetimeIndex(p.index.get_level_values(0).unique())
    lv = low_vol(dret, dates, lcfg.get("lookback_days", 252), lcfg.get("min_obs", 126))
    p["low_vol_raw"] = lv.stack().reindex(p.index)

    p["size_raw"] = np.log(p["mcap"].where(p["mcap"] > 0))

    if fundamentals_joined is not None:
        f = fundamentals_joined.reindex(p.index)
        for c in f.columns:
            p[f"f_{c}"] = f[c]
        measure = cfg.get("value", {}).get("measure", "book_to_market")
        if measure == "book_to_market":
            p["value_raw"] = book_to_market(f["book_equity"], p["mcap"])
        else:
            p["value_raw"] = earnings_yield(f["net_income"], p["mcap"])
        comp = quality_components(f)
        for c in comp.columns:
            p[f"q_{c}"] = comp[c]
        qcfg = cfg.get("quality", {})
        p["quality_raw"] = composite_quality(comp, qcfg.get("min_components", 2), lo, hi, mn)
    else:
        p["value_raw"] = np.nan
        p["quality_raw"] = np.nan

    for name in ("value", "momentum", "low_vol", "quality", "size"):
        raw = p[f"{name}_raw"]
        if raw.notna().any():
            p[name] = process(raw, lo, hi, mn, by=by if name != "size" else None)
        else:
            p[name] = np.nan
    return p


def composite_score(panel: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted mean of available processed characteristics, re-standardised per date.

    Missing characteristics are skipped (weights renormalised) - documented policy.
    """
    cols = [c for c in weights if c in panel.columns and panel[c].notna().any()]
    if not cols:
        raise ValueError("no characteristics available for the composite")
    z = panel[cols]
    w = pd.Series({c: weights[c] for c in cols})
    num = (z * w).sum(axis=1, min_count=1)
    den = z.notna().mul(w).sum(axis=1)
    raw = (num / den.where(den > 0)).rename("composite")
    return zscore_cs(raw, min_n=5)
