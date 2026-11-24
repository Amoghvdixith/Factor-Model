"""Sample splits, hold-out protection, trial logging and out-of-sample risk validation.

Splits are contiguous blocks of the monthly calendar: train, validation, out-of-sample
(OOS) evaluation and a protected final hold-out.  ``run_all.py`` physically removes the
hold-out months from the daily data before anything is computed (so even a forward
return realised in the first hold-out month is unavailable), and
:class:`HoldoutGuard` raises if hold-out dates appear in research data.

Risk-forecast validation is genuinely out of sample: the forecast for month t+1 is made
from a risk model built with factor returns and residuals realised by t, recorded in a
forecast table, and only then joined with the realised month t+1 return.  Weights are
frozen over the month (no intra-month rebalancing); this is not the same as evaluating
a strategy whose holdings change.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .risk_model import RiskModel


class HoldoutViolation(RuntimeError):
    """Raised when protected hold-out data reach research code."""


def compute_splits(months: pd.PeriodIndex, fractions: dict[str, float]) -> dict[str, tuple[pd.Period, pd.Period]]:
    """Contiguous blocks of ``months`` in the order train, validation, oos, holdout."""
    order = ["train", "validation", "oos", "holdout"]
    tot = sum(fractions[k] for k in order)
    if abs(tot - 1) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    n = len(months)
    bounds, start = {}, 0
    cum = 0.0
    for k in order:
        cum += fractions[k]
        end = n if k == "holdout" else int(round(cum * n))
        if end <= start:
            raise ValueError(f"split {k} is empty")
        bounds[k] = (months[start], months[end - 1])
        start = end
    return bounds


@dataclass
class HoldoutGuard:
    holdout_start: pd.Period

    def check(self, dates: pd.Index | pd.Series | list) -> None:
        d = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
        if len(d) and (d.to_period("M") >= self.holdout_start).any():
            raise HoldoutViolation(f"data on/after hold-out start {self.holdout_start} present")

    def truncate_daily(self, prices_long: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        cut = self.holdout_start.start_time
        out = prices_long[pd.to_datetime(prices_long[date_col]) < cut].copy()
        self.check(out[date_col])
        return out


class TrialLog:
    """Append-only CSV log of every specification evaluated."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        row = {"logged_at_utc": datetime.now(UTC).isoformat(timespec="seconds"), **row}
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow({k: (json.dumps(v) if isinstance(v, dict | list) else v)
                        for k, v in row.items()})


def random_portfolios(assets: pd.Index, n: int, seed: int, date_key: int) -> dict[str, pd.Series]:
    """Reproducible long-only random portfolios (flat Dirichlet) for one date."""
    out = {}
    for i in range(n):
        rng = np.random.default_rng([seed, i, date_key])
        w = rng.dirichlet(np.ones(len(assets)))
        out[f"random_{i:02d}"] = pd.Series(w, index=assets)
    return out


def forecast_table(risk_models: dict[pd.Timestamp, RiskModel],
                   extra: dict[str, dict[pd.Timestamp, pd.Series]] | None = None,
                   n_random: int = 20, seed: int = 7) -> pd.DataFrame:
    """Ex-ante monthly volatility forecasts (made at t, for month t+1). No outcomes used."""
    rows = []
    for t, rm in sorted(risk_models.items()):
        ports = {"equal_weight": pd.Series(1.0 / len(rm.assets), index=rm.assets)}
        ports.update(random_portfolios(rm.assets, n_random, seed, int(t.strftime("%Y%m%d"))))
        for name, sched in (extra or {}).items():
            if t in sched:
                ports[name] = sched[t]
        for name, w in ports.items():
            w = w[w.index.isin(rm.assets)]
            rows.append({"date": t, "portfolio": name,
                         "sigma_hat_monthly": rm.portfolio_vol(w),
                         "weights_json": json.dumps({k2: round(float(v), 10) for k2, v in w.items() if v != 0})})
    return pd.DataFrame(rows)


def realise(forecasts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts with realised next-month returns of the frozen portfolios."""
    out = []
    for _, row in forecasts.iterrows():
        t = row["date"]
        g = panel.xs(t, level=0)
        w = pd.Series(json.loads(row["weights_json"]))
        r = g["ret_fwd"].reindex(w.index)
        rd = g["return_date"].dropna()
        if r.isna().all() or rd.empty:
            continue
        out.append({**row.drop(labels="weights_json").to_dict(), "return_date": rd.iloc[0],
                    "realised_return": float((w * r.fillna(0.0)).sum()),
                    "n_missing_returns": int(r.isna().sum())})
    df = pd.DataFrame(out)
    df["z"] = df["realised_return"] / df["sigma_hat_monthly"]
    return df
