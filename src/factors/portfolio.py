"""Walk-forward monthly target construction for the three compared strategies.

At each formation close t, using only information available at t:

* ``ew``   - equal weight across the investable universe (the benchmark);
* ``tilt`` - equal weight across the top composite-score quintile (simple factor tilt);
* ``opt``  - the risk-constrained optimised tilt (:func:`factors.optimiser.optimise_tilt`).

Pre-trade weights are the previous targets drifted by the realised close-to-close
returns of month t (``drift``).  Names that left the universe at t are forced exits.

The *model-implied* return of month t+1 is ``sum_i w_i,t r_i,t+1`` (close-to-close,
frictionless) and a net version subtracts ``cost_bps * sum|w_t - w_drift|``.  These
model returns are used only for the validation-period selection of the tracking-error
budget; reported strategy performance comes from the backtester (next-open execution,
costs, whole-share rounding, residual cash).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .optimiser import optimise_tilt
from .risk_model import RiskModel
from .sorts import assign_groups


@dataclass
class WalkForward:
    strategy: str
    targets: dict[pd.Timestamp, pd.Series]
    gross: pd.Series                 # model-implied return, indexed by return_date
    net: pd.Series
    turnover: pd.Series              # one-way, indexed by formation date
    opt_log: list[dict[str, Any]] = field(default_factory=list)
    te_annual: float | None = None


def drift(w: pd.Series, r: pd.Series) -> pd.Series:
    """Drift weights by returns (unnormalised value relatives normalised to sum 1)."""
    g = w * (1.0 + r.reindex(w.index).fillna(0.0))
    tot = g.sum()
    return g / tot if tot > 0 else g


def run_walk_forward(panel: pd.DataFrame, risk_models: dict[pd.Timestamp, RiskModel],
                     dates: list[pd.Timestamp], strategy: str, scores: pd.Series,
                     port_cfg: dict[str, Any], te_annual: float | None = None,
                     cost_bps: float = 10.0) -> WalkForward:
    """Build monthly targets over ``dates`` (formation closes, ascending)."""
    targets: dict[pd.Timestamp, pd.Series] = {}
    gross, net, tov, olog = {}, {}, {}, []
    prev: pd.Series | None = None
    prev_ret: pd.Series | None = None
    for t in dates:
        rm = risk_models[t]
        U = rm.assets
        g = panel.xs(t, level=0)
        if prev is None:
            wd_full = pd.Series(dtype=float)
        else:
            wd_full = drift(prev, prev_ret if prev_ret is not None else pd.Series(dtype=float))
        exits = wd_full.index.difference(U)
        exit_w = float(wd_full.reindex(exits).sum()) if len(exits) else 0.0
        wd = wd_full.reindex(U).fillna(0.0)
        bench = pd.Series(1.0 / len(U), index=U)
        if strategy == "ew":
            w = bench.copy()
        elif strategy == "tilt":
            sc = scores.xs(t, level=0).reindex(U)
            grp = assign_groups(sc, 5)
            top = grp.index[grp == 5]
            w = pd.Series(1.0 / len(top), index=top).reindex(U).fillna(0.0)
        elif strategy == "opt":
            if te_annual is None:
                raise ValueError("opt strategy needs te_annual")
            sc = scores.xs(t, level=0).reindex(U)
            res = optimise_tilt(sc, rm, wd, bench, port_cfg, te_annual,
                                initial=prev is None, forced_exit_weight=exit_w)
            w = res.weights
            olog.append({"date": t, "status": res.status, "solver": res.solver,
                         "fallback_used": res.fallback_used, "runtime_s": res.runtime_s,
                         "n_assets": len(U), "forced_exit_weight": exit_w,
                         "initial": prev is None, "te_annual_limit": te_annual,
                         "attempts": str(res.attempts),
                         **{f"chk_{k}": v for k, v in res.checks.items() if k != "violations"},
                         "violations": ";".join(res.checks.get("violations", []))})
        else:
            raise ValueError(f"unknown strategy {strategy}")
        allt = w.index.union(wd_full.index)
        trade = (w.reindex(allt, fill_value=0.0) - wd_full.reindex(allt, fill_value=0.0)).abs()
        tov[t] = 0.5 * float(trade.sum()) if prev is not None else 0.5 * float(w.abs().sum())
        targets[t] = w[w > 0]
        r = g["ret_fwd"].reindex(w.index)
        rd = g["return_date"].dropna()
        if len(rd) and r.notna().any():
            gr = float((w * r.fillna(0.0)).sum())
            gross[rd.iloc[0]] = gr
            net[rd.iloc[0]] = gr - cost_bps / 1e4 * 2 * tov[t]
        prev, prev_ret = w[w > 0], g["ret_fwd"].reindex(w[w > 0].index)
    return WalkForward(strategy, targets, pd.Series(gross).sort_index(),
                       pd.Series(net).sort_index(), pd.Series(tov).sort_index(), olog, te_annual)


def information_ratio(active: pd.Series, periods: int = 12) -> float:
    a = active.dropna()
    if len(a) < 3 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * np.sqrt(periods))
