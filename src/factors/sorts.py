"""Quintile (n-group) characteristic sorts and long-short factor portfolios.

Timing: a characteristic known at the close of month t assigns stocks to groups; the
group earns the stock returns of month t+1 (``ret_fwd``).  Returns are labelled by the
month in which they are earned (``return_date``).

Policies
--------
* Ties: ranks use ``method="first"`` on a ticker-sorted cross-section, so tied values
  are split deterministically; group sizes differ by at most one.
* Small universes: a date with fewer than ``min_stocks`` valid signals is skipped and
  logged.
* Value weights use market cap at the formation close t (known before the return).
* Missing next-month returns for a stock alive at t: the stock is dropped from that
  month's group return and the remaining weights are renormalised; every occurrence is
  counted in the log.  Delisting returns are included when the data provide them.
* Turnover: one-way, ``0.5 * sum |w_t - w_drift|`` where ``w_drift`` are the previous
  weights drifted by their realised returns; stocks leaving count as sells.

These are frictionless academic long-short returns, not investable strategy returns, and
are not replications of the official French factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def assign_groups(x: pd.Series, n_groups: int = 5) -> pd.Series:
    """Group labels 1..n by ascending value (n = highest); deterministic tie-breaking."""
    v = x.dropna().sort_index()
    if v.empty:
        return pd.Series(dtype=float)
    r = v.rank(method="first")
    g = ((r - 1) * n_groups // len(v)).astype(int) + 1
    return pd.Series(g, index=v.index).reindex(x.index)


@dataclass
class SortResult:
    returns: pd.DataFrame            # index return_date, columns Q1..Qn, plus 'LS'
    holdings: pd.DataFrame           # formation date, ticker, group, weight
    turnover: pd.DataFrame           # index formation date, columns Q1..Qn
    counts: pd.DataFrame             # stocks per group per date
    log: list[dict] = field(default_factory=list)
    weighting: str = "ew"
    signal: str = ""


def quintile_portfolios(panel: pd.DataFrame, signal: str, weighting: str = "ew",
                        n_groups: int = 5, min_stocks: int = 25, ret_col: str = "ret_fwd",
                        cap_col: str = "mcap", by: str | None = None) -> SortResult:
    """Form n-group portfolios each month and compute next-month returns.

    ``by`` (e.g. "industry") gives sector-relative sorts: groups are assigned within each
    sector and pooled.
    """
    if weighting not in ("ew", "vw"):
        raise ValueError("weighting must be 'ew' or 'vw'")
    log: list[dict] = []
    rets, hold, cnts = [], [], []
    for t, g in panel.groupby(level=0, sort=True):
        g = g.droplevel(0)
        valid = g[g[signal].notna()]
        if weighting == "vw":
            valid = valid[valid[cap_col] > 0]
        if len(valid) < min_stocks:
            log.append({"date": t, "event": "skipped_small_universe", "n": len(valid)})
            continue
        if by is None:
            grp = assign_groups(valid[signal], n_groups)
        else:
            grp = valid.groupby(by, group_keys=False)[signal].apply(
                lambda s: assign_groups(s, n_groups))
        rd = valid["return_date"].dropna()
        ret_date = rd.iloc[0] if len(rd) else pd.NaT
        row: dict[str, Any] = {"return_date": ret_date, "formation_date": t}
        cnt: dict[str, Any] = {"formation_date": t}
        for q in range(1, n_groups + 1):
            members = grp.index[grp == q]
            w = (pd.Series(1.0, index=members) if weighting == "ew"
                 else valid.loc[members, cap_col].astype(float))
            w = w / w.sum()
            r = valid.loc[members, ret_col]
            miss = r.isna()
            if miss.any():
                log.append({"date": t, "event": "missing_next_return", "group": q,
                            "n": int(miss.sum())})
            wr = w[~miss] / w[~miss].sum() if (~miss).any() else w[~miss]
            row[f"Q{q}"] = float((wr * r[~miss]).sum()) if (~miss).any() else np.nan
            cnt[f"Q{q}"] = len(members)
            hold.append(pd.DataFrame({"formation_date": t, "ticker": members, "group": q,
                                      "weight": w.to_numpy(), "ret_fwd": r.to_numpy()}))
        rets.append(row)
        cnts.append(cnt)
    if not rets:
        raise ValueError(f"no valid sort dates for {signal}")
    R = pd.DataFrame(rets)
    R = R[R["return_date"].notna()].set_index("return_date")
    R["LS"] = R[f"Q{n_groups}"] - R["Q1"]
    H = pd.concat(hold, ignore_index=True)
    TO = turnover_from_holdings(H, n_groups)
    return SortResult(returns=R, holdings=H, turnover=TO,
                      counts=pd.DataFrame(cnts).set_index("formation_date"), log=log,
                      weighting=weighting, signal=signal)


def drift_weights(w: pd.Series, r: pd.Series) -> pd.Series:
    """Weights after one period of returns (missing return -> held at zero return)."""
    g = w * (1.0 + r.reindex(w.index).fillna(0.0))
    s = g.sum()
    return g / s if s != 0 else g


def turnover_from_holdings(H: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    """One-way turnover per group: 0.5 * sum|w_t - drift(w_{t-1})| (first date: NaN)."""
    out = {}
    dates = sorted(H["formation_date"].unique())
    for q in range(1, n_groups + 1):
        hq = H[H["group"] == q]
        series = {}
        prev = None
        for t in dates:
            cur = hq[hq["formation_date"] == t]
            w = pd.Series(cur["weight"].to_numpy(), index=cur["ticker"])
            if prev is not None:
                wd = drift_weights(prev[0], prev[1])
                allt = w.index.union(wd.index)
                series[t] = 0.5 * float((w.reindex(allt, fill_value=0)
                                         - wd.reindex(allt, fill_value=0)).abs().sum())
            else:
                series[t] = np.nan
            prev = (w, pd.Series(cur["ret_fwd"].to_numpy(), index=cur["ticker"]))
        out[f"Q{q}"] = pd.Series(series)
    return pd.DataFrame(out)


def max_drawdown(r: pd.Series) -> float:
    """Maximum drawdown of the compounded wealth index (1 + r).cumprod()."""
    w = (1 + r.fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min())


def tearsheet(r: pd.Series, periods_per_year: int = 12, hac_lags: int | None = None) -> dict:
    """Summary statistics for a (long-short) monthly return series.

    The Sharpe ratio of a zero-investment long-short return uses no risk-free
    subtraction.  ``t_mean_hac`` uses Newey-West on the mean.
    """
    from .fama_macbeth import hac_mean_test

    x = r.dropna()
    if len(x) < 3:
        return {"n_months": len(x)}
    m, s = x.mean(), x.std(ddof=1)
    hm = hac_mean_test(x, hac_lags)
    return {
        "n_months": int(len(x)), "start": str(x.index[0]), "end": str(x.index[-1]),
        "mean_monthly": float(m), "ann_mean": float(m * periods_per_year),
        "ann_vol": float(s * np.sqrt(periods_per_year)),
        "sharpe_ann": float(m / s * np.sqrt(periods_per_year)) if s > 0 else np.nan,
        "t_mean_hac": hm["t"], "hac_lags": hm["lags"],
        "max_drawdown": max_drawdown(x),
        "hit_rate": float((x > 0).mean()),
    }
