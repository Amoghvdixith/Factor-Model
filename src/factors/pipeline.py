"""Experiment pipeline shared by the scripts, the CLI and the notebooks.

Stages (each writes labelled PNG figures to ``results/figures`` and the numbers behind
them to ``results/tables``, prefixed by the data mode):

A. time series  - CAPM/FF3/Carhart/FF5 regressions, rolling betas, alphas, GRS
B. custom factors - characteristics, quintile sorts, long-short factors, comparisons
C. Fama-MacBeth - characteristic premia with HAC inference, linearmodels cross-check
D/E. risk model and portfolio - B/F/D model, OOS risk validation, optimisation,
   backtester execution, attribution

Data modes: ``synthetic`` (offline, labelled SYNTHETIC everywhere) and ``real``
(French factors + yfinance prices; stock-level analysis is price-only).  The protected
hold-out is removed from all research data before any computation.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import plotting as P
from .attribution import bias_statistic, bootstrap_bias_ci, euler_attribution, rolling_bias
from .characteristics import add_characteristics, composite_score
from .config import load_all, project_root, rel
from .data import (
    DataUnavailableError,
    load_cached_prices,
    load_french,
    model_factor_frame,
    monthly_returns_from_daily,
    validate_fundamentals,
)
from .fama_macbeth import fama_macbeth, linearmodels_crosscheck
from .panel import asof_join_fundamentals, build_monthly_panel
from .portfolio import information_ratio, run_walk_forward
from .risk_model import build_risk_model, estimate_factor_returns
from .sorts import quintile_portfolios, tearsheet
from .synthetic import SyntheticSpec, load_synthetic, write_synthetic
from .timeseries import GRSError, excess_returns, grs_test, rolling_betas, run_models
from .validation import HoldoutGuard, TrialLog, compute_splits, forecast_table, realise

log = logging.getLogger("factors.pipeline")
plt = P.plt


# =========================================================================== context
@dataclass
class Context:
    mode: str = "synthetic"
    size: str = "fast"
    root: Path = field(default_factory=project_root)
    cfg: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    include_holdout: bool = False     # only run_holdout() sets this, after --confirm

    def __post_init__(self) -> None:
        if self.mode not in ("synthetic", "real"):
            raise ValueError("mode must be 'synthetic' or 'real'")
        self.cfg = self.cfg or load_all(self.root)
        base = self.root / "results"
        if self.mode == "synthetic" and self.size != "fast":
            base = base / self.size          # research-size outputs never overwrite the demo
        self.figdir = base / "figures"
        self.tabdir = base / "tables"
        self.logdir = base / "logs"
        for d in (self.figdir, self.tabdir, self.logdir):
            d.mkdir(parents=True, exist_ok=True)
        self.label = "SYNTHETIC" if self.mode == "synthetic" else "REAL"

    def fig(self, nn: str, name: str) -> Path:
        return self.figdir / f"{self.mode}_{nn}_{name}.png"

    def tab(self, name: str) -> Path:
        return self.tabdir / f"{self.mode}_{name}.csv"

    def stage_meta(self, stage: str, info: dict[str, Any]) -> None:
        self.meta.setdefault("stages", {})[stage] = info


def _fmt_period(a: Any, b: Any) -> str:
    return f"{a} to {b}"


# =========================================================================== data access
def synthetic_dir(ctx: Context) -> Path:
    return ctx.root / ctx.cfg["data"]["synthetic"]["directory"] / ctx.size


def ensure_synthetic(ctx: Context, regenerate: bool = False) -> dict[str, Any]:
    d = synthetic_dir(ctx)
    if regenerate or not (d / "provenance.json").exists():
        write_synthetic(SyntheticSpec.from_config(ctx.cfg["data"], ctx.size), d)
    if "syn" not in ctx.cache:
        ctx.cache["syn"] = load_synthetic(d)
    return ctx.cache["syn"]


def splits_and_guard(ctx: Context, daily_dates: pd.Series) -> tuple[dict, HoldoutGuard]:
    months = pd.PeriodIndex(pd.to_datetime(daily_dates).dt.to_period("M").unique(), freq="M").sort_values()
    sp = compute_splits(months, ctx.cfg["data"]["splits"])
    guard = HoldoutGuard(sp["holdout"][0])
    ctx.meta["splits"] = {k: [str(a), str(b)] for k, (a, b) in sp.items()}
    ctx.meta["holdout"] = {"start": str(sp["holdout"][0]), "end": str(sp["holdout"][1]),
                           "consumed_by_run_all": False}
    return sp, guard


def research_stock_data(ctx: Context) -> dict[str, Any]:
    """Monthly panel with characteristics (hold-out removed) for the current mode."""
    if "stock" in ctx.cache:
        return ctx.cache["stock"]
    ccfg = ctx.cfg["characteristics"]
    if ctx.mode == "synthetic":
        syn = ensure_synthetic(ctx)
        sp, guard = splits_and_guard(ctx, syn["prices"]["date"])
        px = syn["prices"] if ctx.include_holdout else guard.truncate_daily(syn["prices"])
        close = px.pivot(index="date", columns="symbol", values="close")
        meta = syn["meta"].set_index("ticker")
        panel = build_monthly_panel(close, meta["industry"], meta["shares"], syn["rf"])
        fund = validate_fundamentals(syn["fundamentals"])
        fj = asof_join_fundamentals(panel, fund)
        panel = add_characteristics(panel, close, fj, ccfg)
        universe = f"{close.shape[1]} synthetic stocks"
        out = {"panel": panel, "close": close, "prices_long": px, "splits": sp, "guard": guard,
               "universe": universe, "truth_style": syn.get("truth_style"),
               "provenance": syn["provenance"]}
    else:
        mcfg = ctx.cfg["data"]["market"]
        sdir = ctx.root / mcfg["directory"] / "stocks"
        uni = pd.read_csv(ctx.root / mcfg["stock_universe_file"], comment="#")
        px = load_cached_prices(sdir, uni["ticker"])
        sp, guard = splits_and_guard(ctx, px["date"])
        px = px if ctx.include_holdout else guard.truncate_daily(px)
        close = px.pivot(index="date", columns="symbol", values="close")
        fr = load_french(ctx.cfg["data"], ctx.root)
        rf = fr["factors"]["RF_ff3"].dropna()
        panel = build_monthly_panel(close, uni.set_index("ticker")["sector"], None, rf)
        panel = add_characteristics(panel, close, None, ccfg)
        prov = json.loads((sdir / "_provenance.json").read_text(encoding="utf-8"))
        out = {"panel": panel, "close": close, "prices_long": px, "splits": sp, "guard": guard,
               "universe": f"{close.shape[1]} current US large caps (price-only, survivorship-biased)",
               "truth_style": None, "provenance": prov, "french": fr}
    p = out["panel"]
    out["dates_str"] = _fmt_period(p.index.get_level_values(0).min().date(),
                                   p.index.get_level_values(0).max().date())
    ctx.cache["stock"] = out
    return out


# =========================================================================== stage A
def _ts_inputs(ctx: Context) -> dict[str, Any]:
    tcfg = ctx.cfg["timeseries"]
    models = tcfg["models"]
    if ctx.mode == "synthetic":
        syn = ensure_synthetic(ctx)
        fac = syn["factors"]
        rf = fac["RF"]
        sets = {m: fac[cols] for m, cols in models.items()}
        assets = excess_returns(syn["funds"], rf)
        p25 = excess_returns(syn["p25"], rf)
        source = "synthetic academic factors (true model for the 25 portfolios: FF5, zero alpha)"
        universe = "6 synthetic style funds; 25 synthetic size/BM portfolios"
    else:
        fr = load_french(ctx.cfg["data"], ctx.root)
        fac = fr["factors"]
        rf = fac["RF_ff3"]
        sets = {m: model_factor_frame(fac, cols) for m, cols in models.items()}
        mcfg = ctx.cfg["data"]["market"]
        etf = load_cached_prices(ctx.root / mcfg["directory"] / "etfs", list(mcfg["etfs"]))
        close = etf.pivot(index="date", columns="symbol", values="close")
        mret = monthly_returns_from_daily(close)
        mret = mret[mret.index.isin(rf.dropna().index)]
        assets = excess_returns(mret, rf)
        p25 = excess_returns(fr["p25"].loc[fr["p25"].index.isin(rf.dropna().index)], rf)
        source = "Kenneth French Data Library, US, monthly (decimal)"
        universe = f"US style ETFs {list(assets.columns)}; French 25 size/BM VW portfolios"
        ctx.meta["french_provenance"] = fr["provenance"]
        ctx.meta["french_coverage"] = fr["coverage"]
    return {"sets": sets, "assets": assets, "p25": p25, "rf": rf, "source": source,
            "universe": universe}


def stage_timeseries(ctx: Context) -> dict[str, Any]:
    t0 = time.perf_counter()
    tcfg = ctx.cfg["timeseries"]
    inp = _ts_inputs(ctx)
    sets, assets, p25 = inp["sets"], inp["assets"], inp["p25"]
    lags = tcfg["hac_lags"]
    coef = run_models(assets, sets, lags, tcfg["hac_small_sample_correction"])
    coef.to_csv(ctx.tab("01_factor_regressions"), index=False)
    fdates = _fmt_period(assets.index.min(), assets.index.max())
    base = {"data_mode": ctx.label, "universe": inp["universe"], "frequency": "monthly",
            "sample": "full sample (descriptive, in-sample)", "source": inp["source"]}

    # 01 loadings (Carhart4) with HAC 95% CIs
    model = "Carhart4"
    c = coef[(coef["model"] == model) & (coef["term"] != "alpha")]
    terms = list(sets[model].columns)
    fig, axes = plt.subplots(1, len(terms), figsize=(13, 4.2), sharey=True)
    names = list(assets.columns)
    for ax, term in zip(axes, terms, strict=True):
        d = c[c["term"] == term].set_index("asset").reindex(names)
        y = np.arange(len(names))
        ax.barh(y, d["coef"], color=P.SERIES[0], height=0.55)
        ax.errorbar(d["coef"], y, xerr=[d["coef"] - d["ci95_low"], d["ci95_high"] - d["coef"]],
                    fmt="none", ecolor=P.INK2, elinewidth=1, capsize=2)
        P.zero_line(ax, "x")
        ax.set_yticks(y, names)
        ax.set_title(term)
    fig.suptitle("Carhart four-factor loadings with Newey-West 95% CIs", fontweight="bold")
    P.save(fig, ctx.fig("01", "factor_loadings_hac"),
           {**base, "dates": fdates, "model": f"{model}, OLS, HAC lags={lags}"},
           c.set_index(["asset", "term"]))

    # 02 rolling betas
    win = tcfg["rolling_window"]
    rb = {a: rolling_betas(assets[a], sets[model], win, tcfg["rolling_min_obs"]) for a in names}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, a in enumerate(names):
        col = P.SERIES[i % len(P.SERIES)]
        xt = pd.PeriodIndex(rb[a].index).to_timestamp()
        axes[0].plot(xt, rb[a]["MKT_RF"], color=col, label=a, lw=1.6)
        axes[1].plot(xt, rb[a]["HML"], color=col, label=a, lw=1.6)
    axes[0].set_title(f"Rolling {win}-month market beta")
    axes[1].set_title(f"Rolling {win}-month HML loading")
    P.zero_line(axes[1])
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    rbt = pd.concat({a: rb[a] for a in names}, axis=0, names=["asset", "month"])
    P.save(fig, ctx.fig("02", "rolling_betas"),
           {**base, "dates": fdates, "model": f"{model}, trailing {win}-month OLS (causal window)"},
           rbt)

    # 03 alpha comparison across models (annualised, HAC CI)
    al = coef[coef["term"] == "alpha"].copy()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    mlist = [m for m in ["CAPM", "FF3", "Carhart4", "FF5"] if m in sets]
    width = 0.8 / len(mlist)
    x = np.arange(len(names))
    for j, m in enumerate(mlist):
        d = al[al["model"] == m].set_index("asset").reindex(names)
        ax.bar(x + j * width, 12 * d["coef"], width * 0.92, color=P.SERIES[j], label=m)
        ax.errorbar(x + j * width, 12 * d["coef"],
                    yerr=[12 * (d["coef"] - d["ci95_low"]), 12 * (d["ci95_high"] - d["coef"])],
                    fmt="none", ecolor=P.INK2, elinewidth=0.8, capsize=1.5)
    P.zero_line(ax)
    ax.set_xticks(x + width * (len(mlist) - 1) / 2, names)
    ax.set_ylabel("alpha, % per year (x12 monthly)")
    ax.yaxis.set_major_formatter(matplotlib_pct())
    ax.legend(ncol=len(mlist))
    ax.set_title("Alpha by model with Newey-West 95% CIs")
    P.save(fig, ctx.fig("03", "alpha_comparison"),
           {**base, "dates": fdates, "model": "CAPM/FF3/Carhart4/FF5, HAC"},
           al.set_index(["asset", "model"]))

    # 04 GRS on the 25 portfolios, common sample across models
    grs_rows = []
    common = p25.dropna().index
    for m in tcfg["grs_models"]:
        common = common.intersection(sets[m].dropna().index)
    alphas25 = {}
    for m in tcfg["grs_models"]:
        try:
            g = grs_test(p25.loc[common], sets[m].loc[common], tcfg["grs_condition_number_max"])
            grs_rows.append({"model": m, **g.to_dict()})
        except GRSError as exc:
            grs_rows.append({"model": m, "error": str(exc)})
        a = run_models(p25.loc[common], {m: sets[m].loc[common]}, lags)
        alphas25[m] = a[a["term"] == "alpha"].set_index("asset")["coef"]
    grs = pd.DataFrame(grs_rows)
    grs.to_csv(ctx.tab("04_grs"), index=False)
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    ok = grs[grs["statistic"].notna()] if "statistic" in grs.columns else grs.iloc[0:0]
    ax.bar(ok["model"], ok["statistic"], color=P.SERIES[0], width=0.55)
    for i, r in enumerate(ok.itertuples()):
        ax.text(i, float(r.statistic), f"p={float(r.pvalue):.3g}", ha="center", va="bottom", fontsize=8,  # type: ignore[arg-type]
                color=P.INK2)
    ax.set_title("GRS statistic (25 size/BM portfolios)")
    vmax = max(float(np.abs(pd.concat(alphas25.values())).max()) * 1200, 1e-6)
    hm = [m for m in ["FF3", "FF5"] if m in alphas25]
    for k, m in enumerate(hm):
        axh = fig.add_subplot(gs[k + 1])
        A = alphas25[m].reindex([f"S{s}B{b}" for s in range(1, 6) for b in range(1, 6)])
        M = (1200 * A.to_numpy()).reshape(5, 5)
        im = axh.imshow(M, cmap=P.DIVERGING, vmin=-vmax, vmax=vmax)
        for s in range(5):
            for b in range(5):
                axh.text(b, s, f"{M[s, b]:.1f}", ha="center", va="center", fontsize=7.5,
                         color=P.INK)
        axh.set_xticks(range(5), ["Lo", "2", "3", "4", "Hi"])
        axh.set_yticks(range(5), ["Small", "2", "3", "4", "Big"])
        axh.set_xlabel("book-to-market")
        axh.grid(False)
        axh.set_title(f"{m} alpha, % per year")
    fig.colorbar(im, ax=fig.axes[1:], shrink=0.8, label="alpha, % per year (shared scale)")
    P.save(fig, ctx.fig("04", "grs_and_25_portfolio_alphas"),
           {**base, "universe": "25 size/BM portfolios", "dates": _fmt_period(common.min(), common.max()),
            "model": "classical GRS, MLE covariances, F(N, T-N-L); common sample across models"},
           pd.DataFrame(alphas25))
    info = {"runtime_s": round(time.perf_counter() - t0, 2), "n_regressions": int(coef[["asset", "model"]].drop_duplicates().shape[0]),
            "grs": grs.to_dict(orient="records"), "assets_sample": fdates,
            "grs_sample": _fmt_period(common.min(), common.max()), "source": inp["source"]}
    ctx.stage_meta("timeseries", info)
    return {"coef": coef, "grs": grs, "rolling": rb}


def matplotlib_pct(decimals: int = 0) -> Any:
    from matplotlib.ticker import PercentFormatter
    return PercentFormatter(1.0, decimals=decimals)


# =========================================================================== stage B
CHARS = ["value", "momentum", "low_vol", "quality"]


def stage_custom_factors(ctx: Context) -> dict[str, Any]:
    t0 = time.perf_counter()
    S = research_stock_data(ctx)
    panel = S["panel"]
    ccfg = ctx.cfg["characteristics"]
    chars = [c for c in CHARS if panel[c].notna().any()]
    unavailable = [c for c in CHARS if c not in chars]
    base = {"data_mode": ctx.label, "universe": S["universe"], "dates": S["dates_str"],
            "frequency": "monthly", "sample": "full research sample excl. hold-out (descriptive)"}

    # 05a distributions, 05b correlations
    fig, axes = plt.subplots(1, len(chars), figsize=(3.2 * len(chars), 3.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, chars, strict=True):
        ax.hist(panel[c].dropna(), bins=50, color=P.SERIES[0], edgecolor=P.SURFACE, linewidth=0.5)
        ax.set_title(c)
        ax.set_xlabel("z-score (per-date)")
    desc = panel[[f"{c}_raw" for c in chars] + chars].describe().T
    P.save(fig, ctx.fig("05a", "characteristic_distributions"),
           {**base, "model": "winsorised 1/99 per date, z-scored per date; pooled for display"},
           desc)
    cs_corr = panel[chars].groupby(level=0).corr().groupby(level=1).mean().reindex(index=chars, columns=chars)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cs_corr.to_numpy(), cmap=P.DIVERGING, vmin=-1, vmax=1)
    for i in range(len(chars)):
        for j in range(len(chars)):
            ax.text(j, i, f"{cs_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(chars)), chars, rotation=30)
    ax.set_yticks(range(len(chars)), chars)
    ax.grid(False)
    ax.set_title("Average cross-sectional correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)
    P.save(fig, ctx.fig("05b", "characteristic_correlations"),
           {**base, "model": "Pearson correlation per date, averaged over dates"}, cs_corr)

    # sorts
    weightings = ["ew", "vw"] if panel["mcap"].notna().any() else ["ew"]
    res, tears, ls = {}, [], {}
    nq = ccfg["sorts"]["n_groups"]
    for c in chars:
        for wt in weightings:
            r = quintile_portfolios(panel, c, wt, nq, ccfg["sorts"]["min_stocks"])
            res[(c, wt)] = r
            ls[f"{c}_{wt}"] = r.returns["LS"]
            tears.append({"signal": c, "weighting": wt, **tearsheet(r.returns["LS"]),
                          "avg_turnover_Q1": float(r.turnover["Q1"].mean()),
                          f"avg_turnover_Q{nq}": float(r.turnover[f"Q{nq}"].mean()),
                          "n_log_events": len(r.log)})
    tear = pd.DataFrame(tears)
    tear.to_csv(ctx.tab("06_factor_tearsheets"), index=False)
    pd.concat({f"{c}_{w}": r.holdings for (c, w), r in res.items()}, names=["sort"]).to_csv(
        ctx.tab("06_sort_holdings"))
    pd.concat({f"{c}_{w}": r.turnover for (c, w), r in res.items()}, names=["sort"]).to_csv(
        ctx.tab("06_sort_turnover"))
    LS = pd.DataFrame(ls)
    LS.to_csv(ctx.tab("07_long_short_returns"))

    # 06 quintile mean returns
    fig, axes = plt.subplots(1, len(weightings), figsize=(6 * len(weightings), 4), squeeze=False)
    qmeans = {}
    for k, wt in enumerate(weightings):
        ax = axes[0, k]
        width = 0.8 / len(chars)
        for j, c in enumerate(chars):
            q = res[(c, wt)].returns[[f"Q{i}" for i in range(1, nq + 1)]].mean() * 12
            qmeans[(c, wt)] = q
            ax.bar(np.arange(nq) + j * width, q, width * 0.9, color=P.SERIES[j], label=c)
        ax.set_xticks(np.arange(nq) + width * (len(chars) - 1) / 2, [f"Q{i}" for i in range(1, nq + 1)])
        ax.yaxis.set_major_formatter(matplotlib_pct())
        ax.set_title(f"Mean quintile return, {wt.upper()} (annualised)")
        P.zero_line(ax)
        ax.legend(ncol=2)
    P.save(fig, ctx.fig("06", "quintile_returns"),
           {**base, "model": "quintiles at t, returns over t+1, frictionless", "costs": "none (academic)"},
           pd.DataFrame(qmeans))

    # 07 cumulative long-short
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for j, c in enumerate(chars):
        for wt in weightings:
            s = (1 + LS[f"{c}_{wt}"].fillna(0)).cumprod()
            ax.plot(s.index, s, color=P.SERIES[j], lw=1.8 if wt == "ew" else 1.2,
                    ls="-" if wt == "ew" else "--", label=f"{c} {wt.upper()}")
    ax.axhline(1, color=P.AXIS, lw=1)
    ax.set_ylabel("growth of 1 (compounded long-short)")
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Custom Q5-minus-Q1 factor returns (frictionless, zero-investment)")
    P.save(fig, ctx.fig("07", "custom_factor_cumulative"),
           {**base, "model": "top-minus-bottom quintile; compounding of LS returns is a display convention",
            "costs": "none; not investable returns"}, LS)

    # 08 comparison with published / true factors
    comp_rows = []
    if ctx.mode == "synthetic" and S.get("truth_style") is not None:
        ref = S["truth_style"].copy()
        ref.index = ref.index.to_timestamp(how="end").normalize()
        ref_label = "TRUE synthetic style factor returns"
        pairs = [(f"{c}_ew", c) for c in ("value", "momentum", "quality") if f"{c}_ew" in LS]
    else:
        fr = S.get("french") or load_french(ctx.cfg["data"], ctx.root)
        ref = fr["factors"][["MOM", "HML_ff3", "MKT_RF_ff3"]].copy()
        ref.index = ref.index.to_timestamp(how="end").normalize()
        ref_label = "Kenneth French US factors"
        pairs = [(f"{c}_ew", "MOM") for c in ("momentum",) if f"{c}_ew" in LS]
        pairs += [("low_vol_ew", "MKT_RF_ff3")] if "low_vol_ew" in LS else []
    LSm = LS.copy()
    LSm.index = pd.DatetimeIndex(LSm.index).to_period("M").to_timestamp(how="end").normalize()
    fig, axes = plt.subplots(1, max(len(pairs), 1), figsize=(4.3 * max(len(pairs), 1), 4), squeeze=False)
    for k, (mine, theirs) in enumerate(pairs):
        j = pd.concat([LSm[mine], ref[theirs]], axis=1, join="inner").dropna()
        rho = float(j.corr().iloc[0, 1]) if len(j) > 2 else np.nan
        comp_rows.append({"custom": mine, "reference": theirs, "reference_source": ref_label,
                          "n_months": len(j), "correlation": rho,
                          "start": str(j.index.min().date()) if len(j) else None,
                          "end": str(j.index.max().date()) if len(j) else None})
        ax = axes[0, k]
        ax.scatter(j[theirs], j[mine], s=14, color=P.SERIES[k], alpha=0.7, edgecolors=P.SURFACE,
                   linewidths=0.5)
        P.zero_line(ax)
        P.zero_line(ax, "x")
        ax.set_xlabel(theirs)
        ax.set_ylabel(mine)
        ax.set_title(f"corr = {rho:.2f} (n={len(j)})")
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(ctx.tab("08_published_factor_comparison"), index=False)
    fig.suptitle(f"Custom factors vs {ref_label} (diagnostic, not a replication test)",
                 fontweight="bold")
    P.save(fig, ctx.fig("08", "published_factor_comparison"),
           {**base, "model": "monthly correlation on common months"}, comp)
    info = {"runtime_s": round(time.perf_counter() - t0, 2), "characteristics": chars,
            "unavailable": {c: "no point-in-time fundamentals in this mode" for c in unavailable},
            "weightings": weightings, "n_panel_rows": int(len(panel)),
            "tearsheet": tear.to_dict(orient="records"), "comparison": comp_rows}
    ctx.stage_meta("custom_factors", info)
    return {"tear": tear, "ls": LS, "comparison": comp, "sorts": res}


# =========================================================================== stage C
def stage_fama_macbeth(ctx: Context) -> dict[str, Any]:
    t0 = time.perf_counter()
    S = research_stock_data(ctx)
    panel = S["panel"]
    fcfg = ctx.cfg["characteristics"]["fama_macbeth"]
    y = "ret_fwd_excess" if fcfg.get("return_convention") == "excess" else "ret_fwd"
    available = {c for c in [*CHARS, "size"] if panel[c].notna().any()}
    trial = TrialLog(ctx.logdir / f"{ctx.mode}_fama_macbeth_specifications.csv")
    rows, mlogs = [], []
    for name, xs in fcfg["specs"].items():
        if not set(xs) <= available:
            trial.append({"spec": name, "xs": xs, "status": "not_run",
                          "reason": f"unavailable: {sorted(set(xs) - available)}"})
            continue
        for ind in fcfg["industry_controls"]:
            spec = f"{name}{'_ind' if ind else ''}"
            r = fama_macbeth(panel, y, xs, "industry" if ind else None, None,
                             fcfg["min_obs_per_month"], fcfg["hac_lags"], fcfg["t_hurdle"], spec)
            rows.append(r.summary)
            dropped = [m for m in r.month_log if m["dropped"]]
            mlogs += [{"spec": spec, **m} for m in dropped]
            trial.append({"spec": spec, "xs": xs, "industry_controls": ind, "status": "run",
                          "n_months": int(r.summary["n_months"].iloc[0]),
                          "n_dropped_months": len(dropped)})
    summ = pd.concat(rows, ignore_index=True)
    summ.to_csv(ctx.tab("09_fama_macbeth"), index=False)
    pd.DataFrame(mlogs).to_csv(ctx.logdir / f"{ctx.mode}_fama_macbeth_dropped_months.csv", index=False)
    # reference cross-check (multivariate, matched sample)
    multi = [x for x in fcfg["specs"].get("multi", []) if x in available] or sorted(available)
    own = fama_macbeth(panel, y, multi, None, None, fcfg["min_obs_per_month"], 4, spec="check")
    lm = linearmodels_crosscheck(panel, y, multi, fcfg["min_obs_per_month"], bandwidth=4)
    if lm is not None:
        own_s = own.summary.set_index("term")
        T = len(own.gammas)
        lm["own_params"] = own_s["premium_monthly"]
        lm["own_se_hac_L4"] = own_s["se_hac"]
        lm["own_se_unadjusted"] = own.gammas.std(ddof=1) / np.sqrt(T)
        lm["max_abs_param_diff"] = (lm["params"] - lm["own_params"]).abs().max()
        lm.to_csv(ctx.tab("09_fama_macbeth_linearmodels_check"))
    # figure: premia with CI
    slopes = summ[summ["term"] != "const"].copy()
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(slopes) + 1.5))
    yy = np.arange(len(slopes))[::-1]
    terms = sorted(slopes["term"].unique())
    col = {t: P.SERIES[i] for i, t in enumerate(terms)}
    for yv, row in zip(yy, slopes.to_dict("records"), strict=True):
        pm = float(row["premium_monthly"])
        ax.errorbar(100 * pm, yv, xerr=[[100 * (pm - row["ci95_low"])],
                    [100 * (row["ci95_high"] - pm)]], fmt="o", color=col[row["term"]],
                    ms=5, elinewidth=1.4, capsize=2)
        flag = " |t|>3" if row["passes_t3_hurdle"] else ""
        ax.text(100 * float(row["ci95_high"]), yv, f"  t={float(row['t_hac']):.2f}{flag}",
                va="center", fontsize=7.5, color=P.INK2)
    ax.set_yticks(yy, [f"{s_}: {t_}" for s_, t_ in zip(slopes["spec"], slopes["term"], strict=True)],
                  fontsize=8)
    P.zero_line(ax, "x")
    ax.set_xlabel("average monthly slope, % per month per 1 s.d. of characteristic")
    ax.set_title("Fama-MacBeth characteristic premia with Newey-West 95% CIs")
    P.save(fig, ctx.fig("09", "fama_macbeth_premia"),
           {"data_mode": ctx.label, "universe": S["universe"], "dates": S["dates_str"],
            "frequency": "monthly", "model": f"monthly OLS of {y} on z-scores; HAC on slope means",
            "sample": "full research sample excl. hold-out; |t|>3 is an illustrative hurdle"},
           slopes)
    info = {"runtime_s": round(time.perf_counter() - t0, 2), "y": y, "n_specs_run": len(rows),
            "linearmodels_check": None if lm is None else {
                "max_abs_param_diff": float(lm["max_abs_param_diff"].iloc[0]),
                "max_abs_se_kernel_vs_own_hac_diff": float((lm["se_kernel"] - lm["own_se_hac_L4"]).abs().max()),
                "max_abs_se_unadj_diff": float((lm["se_unadjusted"] - lm["own_se_unadjusted"]).abs().max())}}
    ctx.stage_meta("fama_macbeth", info)
    return {"summary": summ, "lm": lm}


# =========================================================================== stage D/E
def _metrics(r: pd.Series, bench: pd.Series | None = None, ppy: int = 252) -> dict[str, Any]:
    r = r.dropna()
    if len(r) < 5:
        return {"n_days": len(r)}
    w = (1 + r).cumprod()
    out: dict[str, Any] = {"n_days": len(r), "start": str(r.index[0].date()),
                           "end": str(r.index[-1].date()),
           "ann_return": float(w.iloc[-1] ** (ppy / len(r)) - 1),
           "ann_vol": float(r.std(ddof=1) * np.sqrt(ppy)),
           "max_drawdown": float((w / w.cummax() - 1).min())}
    out["return_over_vol"] = out["ann_return"] / out["ann_vol"] if out["ann_vol"] > 0 else np.nan
    if bench is not None:
        a = (r - bench.reindex(r.index)).dropna()
        out["tracking_error"] = float(a.std(ddof=1) * np.sqrt(ppy))
        out["info_ratio"] = float(a.mean() / a.std(ddof=1) * np.sqrt(ppy)) if a.std() > 0 else np.nan
    return out


def stage_risk_and_portfolio(ctx: Context) -> dict[str, Any]:
    t0 = time.perf_counter()
    if ctx.mode != "synthetic":
        info = {"status": "not_run",
                "reason": "real stock-level mode has no point-in-time market caps or fundamentals; "
                          "the fundamental risk model, optimiser and backtest are demonstrated on "
                          "synthetic data only"}
        ctx.stage_meta("risk_and_portfolio", info)
        return info
    S = research_stock_data(ctx)
    panel = S["panel"]
    rcfg, pcfg = ctx.cfg["risk_model"], ctx.cfg["portfolio"]
    styles = rcfg["style_factors"]
    fr = estimate_factor_returns(panel, styles, rcfg["identification"])
    fr.factor_returns.to_csv(ctx.tab("10_factor_returns"))
    pd.DataFrame(fr.log).to_csv(ctx.logdir / "synthetic_risk_model_estimation_log.csv", index=False)
    dates = sorted(panel.index.get_level_values(0).unique())
    mp = rcfg["factor_cov"]["min_periods"]
    rm_dates = [t for t in dates if t in fr.exposures and (fr.factor_returns.index <= t).sum() >= mp]
    risk_models = {t: build_risk_model(fr, panel, t, rcfg) for t in rm_dates}
    sp = S["splits"]
    per = {t: next((k for k, (a, b) in sp.items() if a <= pd.Period(t, "M") <= b), "?")
           for t in rm_dates}
    base = {"data_mode": ctx.label, "universe": S["universe"], "frequency": "monthly"}
    if not rm_dates:
        raise RuntimeError("no dates with enough factor-return history for a risk model")
    last = rm_dates[-1]

    # 10 factor covariance / correlation (at the last research date)
    rm_last = risk_models[last]
    Fm = rm_last.F
    vol = np.sqrt(np.diag(Fm) * 12)
    sd = np.sqrt(np.diag(Fm.to_numpy()))
    corr = pd.DataFrame(Fm.to_numpy() / np.outer(sd, sd), index=Fm.index, columns=Fm.columns)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    im = axes[0].imshow(corr.to_numpy(), cmap=P.DIVERGING, vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(corr)), corr.index, rotation=90, fontsize=8)
    axes[0].set_yticks(range(len(corr)), corr.index, fontsize=8)
    axes[0].grid(False)
    axes[0].set_title("Factor correlation (EWMA)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    axes[1].barh(np.arange(len(vol))[::-1], vol, color=P.SERIES[0], height=0.6)
    axes[1].set_yticks(np.arange(len(vol))[::-1], corr.index, fontsize=8)
    axes[1].xaxis.set_major_formatter(matplotlib_pct())
    axes[1].set_title("Forecast factor volatility (annualised)")
    hl = rcfg["factor_cov"]["half_life_months"]
    P.save(fig, ctx.fig("10", "factor_covariance"),
           {**base, "dates": f"forecast made at {last.date()}",
            "model": f"EWMA half-life {hl}m, not demeaned, {rcfg['identification']}"},
           pd.concat([Fm, pd.Series(vol, index=Fm.index, name="ann_vol")], axis=1))

    # 11 specific risk distribution
    D = rm_last.D
    sz = np.log(panel.xs(last, level=0)["mcap"].reindex(D.index))
    bucket = pd.qcut(sz.rank(method="first"), 3, labels=["small", "mid", "large"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, b in enumerate(["small", "mid", "large"]):
        v = np.sqrt(D[bucket == b] * 12)
        ax.hist(v, bins=20, alpha=0.6, color=P.SERIES[j], label=f"{b} (median {v.median():.0%})")
    ax.axvline(rcfg["specific_risk"]["floor_vol_monthly"] * np.sqrt(12), color=P.INK2, lw=1, ls=":")
    ax.xaxis.set_major_formatter(matplotlib_pct())
    ax.set_xlabel("forecast specific volatility (annualised)")
    ax.legend()
    ax.set_title("Specific-risk forecasts by size bucket")
    P.save(fig, ctx.fig("11", "specific_risk"),
           {**base, "dates": f"forecast made at {last.date()}",
            "model": f"EWMA residual variance, linear shrinkage to size-bucket mean, floor "
                     f"{rcfg['specific_risk']['floor_vol_monthly']:.0%}/month"},
           pd.DataFrame({"spec_var_monthly": D, "size_bucket": bucket}))

    # ---------- portfolio: TE selection on validation, then val+OOS walk-forward
    scores = composite_score(panel, ctx.cfg["characteristics"]["composite_weights"])
    cost = float(pcfg["backtest"]["cost_bps_one_way"])
    val_dates = [t for t in rm_dates if per[t] == "validation"]
    oos_dates = [t for t in rm_dates if per[t] == "oos"]
    trial = TrialLog(ctx.logdir / "synthetic_portfolio_trials.csv")
    ew_val = run_walk_forward(panel, risk_models, val_dates, "ew", scores, pcfg, cost_bps=cost)
    sel = []
    for te in pcfg["te_grid_annual"]:
        wf = run_walk_forward(panel, risk_models, val_dates, "opt", scores, pcfg, te, cost)
        active_ret = (wf.net - ew_val.net).dropna()
        ir = information_ratio(active_ret)
        n_fb = sum(o["fallback_used"] for o in wf.opt_log)
        sel.append({"te_annual": te, "validation_ir_net_model": ir, "n_months": len(active_ret),
                    "n_fallbacks": n_fb})
        trial.append({"stage": "te_selection", "split": "validation", "te_annual": te,
                      "metric": "IR of model-implied net active return vs EW", "value": ir,
                      "n_months": len(active_ret), "n_fallbacks": n_fb})
    seldf = pd.DataFrame(sel)
    best = seldf.sort_values(["validation_ir_net_model", "te_annual"],
                             ascending=[False, True]).iloc[0]
    te_star = float(best["te_annual"]) if np.isfinite(best["validation_ir_net_model"]) else pcfg["te_budget_annual"]
    seldf["selected"] = seldf["te_annual"] == te_star
    seldf.to_csv(ctx.tab("14_te_selection_validation"), index=False)
    trial.append({"stage": "selection", "selected_te_annual": te_star,
                  "rule": "max validation IR; ties -> smaller TE; predeclared grid"})

    run_dates = val_dates + oos_dates
    wfs = {"ew": run_walk_forward(panel, risk_models, run_dates, "ew", scores, pcfg, cost_bps=cost),
           "tilt": run_walk_forward(panel, risk_models, run_dates, "tilt", scores, pcfg, cost_bps=cost),
           "opt": run_walk_forward(panel, risk_models, run_dates, "opt", scores, pcfg, te_star, cost)}
    olog = pd.DataFrame(wfs["opt"].opt_log)
    olog.to_csv(ctx.logdir / "synthetic_optimiser_log.csv", index=False)

    # ---------- OOS risk-forecast validation (forecasts first, outcomes joined after)
    vcfg = rcfg["validation"]
    fc = forecast_table(risk_models, {"optimised": wfs["opt"].targets, "simple_tilt": wfs["tilt"].targets},
                        vcfg["n_random_portfolios"], vcfg["random_seed"])
    fc.drop(columns="weights_json").to_csv(ctx.tab("12_risk_forecasts_ex_ante"), index=False)
    rz = realise(fc, panel)
    rz["period"] = rz["date"].map(per)
    rz["group"] = np.where(rz["portfolio"].str.startswith("random_"), "random (20)", rz["portfolio"])
    rz.to_csv(ctx.tab("12_risk_forecasts_realised"), index=False)
    brows = []
    for (g, prd), d in list(rz.groupby(["group", "period"])) + [((g, "all"), d) for g, d in rz.groupby("group")]:
        bst = bias_statistic(d["z"])
        lo, hi = bootstrap_bias_ci(d["z"], vcfg["bootstrap_reps"], 0) if g != "random (20)" else (np.nan, np.nan)
        brows.append({"group": g, "period": prd, **bst, "boot_ci_low": lo, "boot_ci_high": hi})
    bias = pd.DataFrame(brows)
    bias.to_csv(ctx.tab("12_bias_statistics"), index=False)

    # 12 predicted vs realised volatility
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    W = vcfg["rolling_window_months"]
    for j, g in enumerate(["equal_weight", "optimised"]):
        d = rz[rz["group"] == g].set_index("return_date").sort_index()
        if d.empty:
            continue
        pred = d["sigma_hat_monthly"] * np.sqrt(12)
        realised = d["realised_return"].rolling(W, min_periods=W).apply(
            lambda x: np.sqrt(np.mean(x ** 2)), raw=True) * np.sqrt(12)
        axes[0].plot(pred.index, pred, color=P.SERIES[j], lw=1.8, label=f"{g}: predicted")
        axes[0].plot(realised.index, realised, color=P.SERIES[j], lw=1.2, ls="--",
                     label=f"{g}: realised (trailing {W}m RMS)")
    axes[0].yaxis.set_major_formatter(matplotlib_pct())
    axes[0].set_title("Predicted vs realised volatility (annualised)")
    axes[0].legend(fontsize=7.5)
    bb = bias[bias["period"] == "all"].set_index("group")
    order = [g for g in ["equal_weight", "simple_tilt", "optimised", "random (20)"] if g in bb.index]
    axes[1].bar(order, bb.loc[order, "bias"], color=[P.SERIES[i] for i in range(len(order))], width=0.55)
    axes[1].errorbar(range(len(order)), bb.loc[order, "bias"],
                     yerr=[bb.loc[order, "bias"] - bb.loc[order, "ci_low"],
                           bb.loc[order, "ci_high"] - bb.loc[order, "bias"]],
                     fmt="none", ecolor=P.INK2, capsize=3)
    axes[1].axhline(1, color=P.INK2, lw=1)
    for lv in (0.8, 1.2):
        axes[1].axhline(lv, color=P.AXIS, lw=1, ls=":")
    axes[1].set_title("Bias statistic (RMS of r / sigma_hat), all OOS months")
    for i, g in enumerate(order):
        axes[1].text(i, 0.02, f"n={int(bb.loc[g, 'n'])}", ha="center", fontsize=8, color=P.INK2)  # type: ignore[call-overload,arg-type]
    P.save(fig, ctx.fig("12", "predicted_vs_realised_vol"),
           {**base, "dates": _fmt_period(rz["return_date"].min().date(), rz["return_date"].max().date()),
            "model": "ex-ante forecasts made at t for month t+1; frozen weights; not demeaned",
            "sample": "out-of-sample by construction (train/validation/OOS months); dotted lines 0.8/1.2 are a heuristic"},
           bias)

    # 13 rolling calibration
    fig, ax = plt.subplots(figsize=(11, 4))
    rnd = rz[rz["group"] == "random (20)"].pivot(index="return_date", columns="portfolio", values="z")
    rb = rnd.apply(lambda s: rolling_bias(s, W))
    ax.fill_between(rb.index, rb.quantile(0.1, axis=1), rb.quantile(0.9, axis=1),
                    color=P.SERIES[2], alpha=0.25, label="random portfolios, 10-90% band")
    ax.plot(rb.index, rb.median(axis=1), color=P.SERIES[2], lw=1.4, label="random, median")
    roll_tab = {"random_median": rb.median(axis=1)}
    for j, g in [(0, "equal_weight"), (1, "optimised")]:
        zser = rz[rz["group"] == g].set_index("return_date")["z"].sort_index()
        if len(zser):
            rr = rolling_bias(zser, W)
            roll_tab[g] = rr
            ax.plot(rr.index, rr, color=P.SERIES[j], lw=1.8, label=g)
    ax.axhline(1, color=P.INK2, lw=1)
    ax.set_title(f"Rolling {W}-month bias statistic")
    ax.legend(fontsize=8, ncol=2)
    P.save(fig, ctx.fig("13", "rolling_bias"),
           {**base, "dates": "OOS forecast months", "model": f"trailing {W}-month RMS of standardised outcomes",
            "sample": "out-of-sample forecasts; descriptive diagnostic, not a pass/fail test"},
           pd.DataFrame(roll_tab))

    # ---------- backtester execution
    from .backtester_adapter import backtester_version, market_data_from_long, run_schedule

    px = S["prices_long"]
    start_px = run_dates[0] - pd.Timedelta(days=10)
    md = market_data_from_long(px[px["date"] >= start_px][["date", "symbol", "open", "high", "low", "close", "volume"]],
                               {"data_mode": ctx.label})
    execd = {}
    for k, lab in [("ew", "equal_weight_benchmark"), ("tilt", "simple_tilt"), ("opt", "optimised_tilt")]:
        execd[k] = run_schedule(md, wfs[k].targets, lab, pcfg, ctx.label, pcfg["max_weight"])
    daily = pd.DataFrame({k: e.returns for k, e in execd.items()})
    daily.to_csv(ctx.tab("14_backtest_daily_returns"))
    oos_start = pd.Period(sp["oos"][0], "M").start_time
    first_trade = daily.index[0]
    rows = []
    for strat in daily:
        k = str(strat)
        for prd, (a, b) in {"validation": (first_trade, oos_start - pd.Timedelta(days=1)),
                            "oos": (oos_start, daily.index[-1])}.items():
            r = daily.loc[a:b, k]
            m = _metrics(r, daily.loc[a:b, "ew"] if k != "ew" else None)
            costs = {f"cost_{c}": v for c, v in execd[k].costs.items()} if prd == "oos" else {}
            rows.append({"strategy": k, "period": prd, **m, **costs})
    tear = pd.DataFrame(rows)
    tear.to_csv(ctx.tab("15_tearsheet"), index=False)
    cost_note = f"{cost:.0f} bps one-way proportional, next-open execution, whole shares"
    names = {"ew": "equal-weight benchmark", "tilt": "simple tilt (top quintile)", "opt": f"optimised tilt (TE {te_star:.0%})"}

    # 14 equity curves
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for j, k in enumerate(["ew", "tilt", "opt"]):
        w = (1 + daily[k].fillna(0)).cumprod()
        ax.plot(w.index, w, color=P.SERIES[j], lw=1.8, label=names[k])
    ax.axvspan(oos_start, daily.index[-1], color=P.GRID, alpha=0.5, lw=0)  # type: ignore[arg-type]
    ax.text(oos_start, ax.get_ylim()[1], " OOS", va="top", fontsize=9, color=P.INK2)  # type: ignore[arg-type]
    ax.text(first_trade, ax.get_ylim()[1], " validation", va="top", fontsize=9,  # type: ignore[arg-type]
            color=P.INK2)
    ax.set_ylabel("growth of 1 (net of costs)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
    ax.set_title("Backtested equity curves (through the project-02 backtester)")
    P.save(fig, ctx.fig("14", "equity_curves"),
           {**base, "dates": _fmt_period(first_trade.date(), daily.index[-1].date()),
            "frequency": "daily marks, monthly rebalance", "model": f"risk model + optimiser; TE chosen on validation = {te_star:.0%}",
            "sample": "validation then OOS (shaded); hold-out not used", "costs": cost_note},
           (1 + daily.fillna(0)).cumprod())

    # 15 drawdowns and tearsheet
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for j, k in enumerate(["ew", "tilt", "opt"]):
        w = (1 + daily[k].fillna(0)).cumprod()
        axes[0].plot(w.index, w / w.cummax() - 1, color=P.SERIES[j], lw=1.4, label=names[k])
    axes[0].yaxis.set_major_formatter(matplotlib_pct())
    axes[0].set_title("Drawdown")
    axes[0].legend(fontsize=8)
    axes[1].axis("off")
    tt = tear[tear["period"] == "oos"].set_index("strategy")
    def _f(k: str, c: str) -> float:
        return float(tt.loc[k, c])  # type: ignore[arg-type]

    cells = [[names[k], f"{_f(k, 'ann_return'):.1%}", f"{_f(k, 'ann_vol'):.1%}",
              f"{_f(k, 'max_drawdown'):.1%}",
              "-" if k == "ew" else f"{_f(k, 'tracking_error'):.1%}",
              "-" if k == "ew" else f"{_f(k, 'info_ratio'):.2f}"] for k in ["ew", "tilt", "opt"]]
    tbl = axes[1].table(cellText=cells, colLabels=["OOS", "ann. ret", "vol", "max DD", "TE", "IR"],
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    axes[1].set_title("OOS tearsheet (net of costs)")
    P.save(fig, ctx.fig("15", "drawdowns_tearsheet"),
           {**base, "dates": _fmt_period(first_trade.date(), daily.index[-1].date()),
            "sample": "drawdowns: validation+OOS; table: OOS only", "costs": cost_note}, tear)

    # 16 exposures over time (active style exposures, targets)
    ex_rows = []
    for t in run_dates:
        rm = risk_models[t]
        bench = pd.Series(1 / len(rm.assets), index=rm.assets)
        for k in ["tilt", "opt"]:
            w = wfs[k].targets[t].reindex(rm.assets).fillna(0)
            x = rm.B[styles].T @ (w - bench)
            ex_rows.append({"date": t, "strategy": k, **x.to_dict()})
    ex = pd.DataFrame(ex_rows)
    ex.to_csv(ctx.tab("16_active_exposures"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for a, k in zip(axes, ["tilt", "opt"], strict=True):
        d = ex[ex["strategy"] == k].set_index("date")
        for j, s in enumerate(styles):
            a.plot(d.index, d[s], color=P.SERIES[j], lw=1.6, label=s)
        P.zero_line(a)
        a.set_title(f"Active style exposure: {names[k]}")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    P.save(fig, ctx.fig("16", "factor_exposures_over_time"),
           {**base, "dates": _fmt_period(run_dates[0].date(), run_dates[-1].date()),
            "model": "target weights x B_t minus EW benchmark (risk-model z-scores)",
            "sample": "validation + OOS rebalances"}, ex)

    # 17-19 attribution at the last OOS rebalance
    ta = (oos_dates or run_dates)[-1]
    rm = risk_models[ta]
    w_opt = wfs["opt"].targets[ta]
    bench = pd.Series(1 / len(rm.assets), index=rm.assets)
    att = euler_attribution(rm, w_opt)
    tab_total = att.table()
    tab_total.to_csv(ctx.tab("17_risk_attribution_total"))
    groups = {"industries": [c for c in rm.F.index if c.startswith("ind_")]}
    gtot = att.grouped(groups)
    meta_attr = {**base, "dates": f"weights and forecast at {ta.date()} (last OOS rebalance)",
                 "model": "Euler: RC_k = x_k (F x)_k / sigma; RC_spec = w'Dw / sigma",
                 "sample": "ex-ante risk; not realised P&L attribution"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    full = att.factor_vol_contrib.copy()
    full["specific"] = att.specific_vol_contrib
    o = full.sort_values()
    axes[0].barh(o.index, o * np.sqrt(12), color=[P.SERIES[0] if v >= 0 else P.SERIES[7] for v in o],
                 height=0.6)
    P.zero_line(axes[0], "x")
    axes[0].xaxis.set_major_formatter(matplotlib_pct(1))
    axes[0].set_title(f"Signed volatility contributions (sum = {att.sigma * np.sqrt(12):.1%} ann.)")
    axes[0].tick_params(axis="y", labelsize=8)
    gsort = gtot.sort_values()
    axes[1].barh(gsort.index, gsort / att.sigma,
                 color=[P.SERIES[0] if v >= 0 else P.SERIES[7] for v in gsort], height=0.6)
    P.zero_line(axes[1], "x")
    axes[1].xaxis.set_major_formatter(matplotlib_pct())
    axes[1].set_title("Share of total predicted volatility (industries grouped)")
    P.save(fig, ctx.fig("17", "risk_contributions_signed"), meta_attr,
           pd.concat([tab_total, gtot.rename("grouped_vol_contrib_monthly")], axis=1))

    st = pd.DataFrame({"weight": w_opt.reindex(rm.assets).fillna(0), "mctr_monthly": att.stock_mctr,
                       "component_vol_monthly": att.stock_vol_contrib})
    st["pct_of_sigma"] = st["component_vol_monthly"] / att.sigma
    st.to_csv(ctx.tab("18_stock_risk_contributions"))
    top = st[st["weight"] > 0].sort_values("component_vol_monthly", ascending=False).head(15)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    yy = np.arange(len(top))[::-1]
    axes[0].barh(yy, top["pct_of_sigma"], color=P.SERIES[0], height=0.6)
    axes[0].set_yticks(yy, top.index, fontsize=8)
    axes[0].xaxis.set_major_formatter(matplotlib_pct(1))
    axes[0].set_title("Component contribution (% of sigma), top 15")
    axes[1].barh(yy, top["mctr_monthly"] * np.sqrt(12), color=P.SERIES[1], height=0.6)
    axes[1].xaxis.set_major_formatter(matplotlib_pct())
    axes[1].set_title("Marginal contribution d sigma / d w (annualised)")
    P.save(fig, ctx.fig("18", "stock_marginal_component_risk"), meta_attr, top)

    act = euler_attribution(rm, w_opt, bench)
    act.table().to_csv(ctx.tab("19_active_risk_attribution"))
    te_rows = [{"date": o["date"], "ex_ante_te_annual": o["chk_te_monthly"] * np.sqrt(12),
                "te_limit_annual": o["te_annual_limit"]} for o in wfs["opt"].opt_log]
    ted = pd.DataFrame(te_rows).set_index("date")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.2]})
    ga = act.grouped(groups).sort_values()
    axes[0].barh(ga.index, ga * np.sqrt(12), color=[P.SERIES[0] if v >= 0 else P.SERIES[7] for v in ga], height=0.6)
    P.zero_line(axes[0], "x")
    axes[0].xaxis.set_major_formatter(matplotlib_pct(1))
    axes[0].set_title(f"Active-risk contributions (TE = {act.sigma * np.sqrt(12):.2%} ann.)")
    axes[1].plot(ted.index, ted["ex_ante_te_annual"], color=P.SERIES[0], lw=1.8, label="ex-ante TE of target")
    axes[1].plot(ted.index, ted["te_limit_annual"], color=P.INK2, lw=1, ls="--", label="TE limit")
    axes[1].yaxis.set_major_formatter(matplotlib_pct(1))
    axes[1].legend()
    axes[1].set_title("Ex-ante tracking error at each rebalance")
    P.save(fig, ctx.fig("19", "tracking_error_attribution"),
           {**meta_attr, "model": "active weights w - w_EW; Euler on active risk"},
           pd.concat([act.table(), ted.reset_index().set_index(ted.index.astype(str))], axis=0))

    # 20 target vs executed constraints
    tvr = execd["opt"].target_vs_realised
    tvr.to_csv(ctx.tab("20_target_vs_realised_opt"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(olog["date"], olog["chk_max_w"], color=P.SERIES[0], lw=1.6, label="target max weight")
    if len(tvr):
        axes[0].plot(tvr["execution_date"], tvr["realised_max_weight"], color=P.SERIES[1], lw=1.2,
                     ls="--", label="realised max weight (after fills)")
    axes[0].axhline(pcfg["max_weight"], color=P.INK2, lw=1, ls=":", label="cap")
    axes[0].yaxis.set_major_formatter(matplotlib_pct(1))
    axes[0].legend(fontsize=7.5)
    axes[0].set_title("Position cap")
    axes[1].plot(olog["date"], olog["chk_turnover_one_way"], color=P.SERIES[0], lw=1.6, label="target one-way turnover")
    axes[1].plot(olog["date"], olog["chk_turnover_budget"].astype(float), color=P.INK2, lw=1, ls=":", label="budget (incl. forced exits)")
    axes[1].yaxis.set_major_formatter(matplotlib_pct())
    axes[1].legend(fontsize=7.5)
    axes[1].set_title("Turnover vs limit")
    if len(tvr):
        axes[2].plot(tvr["execution_date"], tvr["sum_abs_diff"], color=P.SERIES[2], lw=1.6, label="sum |target - realised|")
        axes[2].plot(tvr["execution_date"], tvr["untradeable_weight"], color=P.SERIES[6], lw=1.6,
                     label="weight stuck in delisted names (no bar)")
        axes[2].plot(tvr["execution_date"], tvr["residual_cash"], color=P.SERIES[3], lw=1.6, label="residual cash")
    axes[2].yaxis.set_major_formatter(matplotlib_pct(1))
    axes[2].legend(fontsize=7.5)
    axes[2].set_title("Target vs executed")
    P.save(fig, ctx.fig("20", "target_vs_executed_constraints"),
           {**base, "dates": _fmt_period(run_dates[0].date(), run_dates[-1].date()),
            "model": "post-solve checks (targets) and backtester fills (realised)", "costs": cost_note},
           olog.drop(columns=["attempts"]))

    # 21 costs and turnover
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for j, k in enumerate(["ew", "tilt", "opt"]):
        axes[0].plot(wfs[k].turnover.index[1:], wfs[k].turnover.iloc[1:], color=P.SERIES[j], lw=1.5, label=names[k])
        eqf = execd[k].result.equity
        cc = (eqf["commissions"] / pcfg["backtest"]["initial_cash"])
        axes[1].plot(cc.index, cc, color=P.SERIES[j], lw=1.6, label=names[k])
    axes[0].axhline(pcfg["turnover_one_way_max"], color=P.INK2, lw=1, ls=":")
    axes[0].yaxis.set_major_formatter(matplotlib_pct())
    axes[0].set_title("Target one-way turnover per rebalance (excl. initial)")
    axes[0].legend(fontsize=8)
    axes[1].yaxis.set_major_formatter(matplotlib_pct(2))
    axes[1].set_title("Cumulative costs paid (% of initial capital)")
    P.save(fig, ctx.fig("21", "costs_and_turnover"),
           {**base, "dates": _fmt_period(first_trade.date(), daily.index[-1].date()), "costs": cost_note},
           pd.DataFrame({k: wfs[k].turnover for k in wfs}))

    solver_summary = olog["status"].value_counts().to_dict() if len(olog) else {}
    result_info: dict[str, Any] = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "identification": rcfg["identification"], "n_risk_model_dates": len(rm_dates),
        "factor_return_r2_mean": float(fr.r2.mean()),
        "te_selected_annual": te_star, "te_selection": sel,
        "solver_status_counts": solver_summary,
        "solvers_used": olog["solver"].value_counts(dropna=False).astype(int).to_dict() if len(olog) else {},
        "fallbacks": int(olog["fallback_used"].sum()) if len(olog) else 0,
        "post_solve_violations": int((olog["violations"].fillna("") != "").sum()) if len(olog) else 0,
        "bias_all": bias[bias["period"] == "all"].to_dict(orient="records"),
        "tearsheet_oos": tear[tear["period"] == "oos"].to_dict(orient="records"),
        "backtester_version": backtester_version(),
        "attribution_date": str(ta.date()),
        "total_sigma_annual": att.sigma * np.sqrt(12),
        "factor_share_of_sigma": float(att.factor_vol_contrib.sum() / att.sigma),
        "active_te_annual": act.sigma * np.sqrt(12),
        "target_vs_realised_mean_sum_abs_diff": float(tvr["sum_abs_diff"].mean()) if len(tvr) else None,
        "mean_residual_cash": float(tvr["residual_cash"].mean()) if len(tvr) else None,
        "final_untradeable_weight": float(tvr["untradeable_weight"].iloc[-1]) if len(tvr) else None,
    }
    ctx.stage_meta("risk_and_portfolio", result_info)
    return {"fr": fr, "risk_models": risk_models, "wfs": wfs, "execd": execd, "bias": bias,
            "tear": tear, "attribution": att, "active": act, "olog": olog}


# =========================================================================== metadata
def environment() -> dict[str, Any]:
    import importlib

    v: dict[str, str | None] = {"python": platform.python_version(), "platform": platform.system()}
    for m in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "cvxpy", "clarabel",
              "scs", "linearmodels", "yfinance", "backtester", "factors"):
        try:
            v[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001
            v[m] = None
    return v


def write_metadata(ctx: Context, started: float, stages: list[str]) -> Path:
    path = ctx.root / "results" / "run_metadata.json"
    allm = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    syn_prov = None
    if ctx.mode == "synthetic":
        syn_prov = ctx.cache.get("syn", {}).get("provenance")
    entry = {
        "data_mode": ctx.label, "size": ctx.size,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_s": round(time.perf_counter() - started, 2), "stages_run": stages,
        "environment": environment(), "configs": ctx.cfg,
        "seeds": {"synthetic": ctx.cfg["data"]["synthetic"]["seed"],
                  "random_portfolios": ctx.cfg["risk_model"]["validation"]["random_seed"]},
        "synthetic_provenance": syn_prov,
        **ctx.meta,
    }
    if ctx.mode == "real":
        mdir = ctx.root / ctx.cfg["data"]["market"]["directory"]
        for k in ("etfs", "stocks"):
            f = mdir / k / "_provenance.json"
            if f.exists():
                entry[f"market_provenance_{k}"] = json.loads(f.read_text(encoding="utf-8"))
    key = f"{ctx.mode}_{ctx.size}" if ctx.mode == "synthetic" else "real"
    # a partial run (some stages only) updates those stages and keeps the others' records
    old = allm.get(key, {})
    entry["stages"] = {**old.get("stages", {}), **ctx.meta.get("stages", {})}
    allm[key] = {**old, **entry}
    path.write_text(json.dumps(allm, indent=2, default=str), encoding="utf-8")
    return path


STAGES = {"timeseries": stage_timeseries, "custom_factors": stage_custom_factors,
          "fama_macbeth": stage_fama_macbeth, "risk_and_portfolio": stage_risk_and_portfolio}


def run(mode: str = "synthetic", size: str = "fast", stages: list[str] | None = None,
        regenerate: bool = False) -> Context:
    """Run the selected stages and write ``results/run_metadata.json``."""
    started = time.perf_counter()
    ctx = Context(mode=mode, size=size)
    if mode == "synthetic":
        ensure_synthetic(ctx, regenerate)
    stages = stages or list(STAGES)
    for s in stages:
        log.info("stage %s (%s)", s, mode)
        try:
            STAGES[s](ctx)
        except DataUnavailableError as exc:
            ctx.stage_meta(s, {"status": "not_run", "reason": str(exc)})
            log.warning("stage %s not run: %s", s, exc)
    write_metadata(ctx, started, stages)
    return ctx


class HoldoutLocked(RuntimeError):
    """Raised when the protected hold-out is requested without explicit confirmation."""


def run_holdout(confirm: bool = False, size: str = "fast", rerun: bool = False) -> dict[str, Any]:
    """Evaluate the pre-selected strategies ONCE on the protected hold-out (synthetic).

    Requires ``confirm=True``.  Uses the TE budget already selected on the validation
    split (read from results/tables); nothing is re-tuned.  Writes a lock file so that a
    second evaluation needs ``rerun=True`` and is logged as such.
    """
    if not confirm:
        raise HoldoutLocked("hold-out evaluation needs --confirm; it is never run by run_all")
    ctx = Context(mode="synthetic", size=size, include_holdout=True)
    hdir = ctx.root / "results" / "holdout"
    lock = hdir / f"synthetic_{size}.lock"
    if lock.exists() and not rerun:
        raise HoldoutLocked(f"hold-out already evaluated ({rel(lock)}); pass --rerun to repeat (logged)")
    sel_path = ctx.tab("14_te_selection_validation")
    if not sel_path.exists():
        raise HoldoutLocked("no validation selection found; run run_all.py first")
    sel = pd.read_csv(sel_path)
    te = float(sel.loc[sel["selected"], "te_annual"].iloc[0])
    ensure_synthetic(ctx)
    S = research_stock_data(ctx)
    panel = S["panel"]
    rcfg, pcfg = ctx.cfg["risk_model"], ctx.cfg["portfolio"]
    fr = estimate_factor_returns(panel, rcfg["style_factors"], rcfg["identification"])
    h0 = pd.Period(S["splits"]["holdout"][0], "M")
    hdates = [t for t in sorted(panel.index.get_level_values(0).unique())
              if pd.Period(t, "M") >= h0 and t in fr.exposures]
    rms = {t: build_risk_model(fr, panel, t, rcfg) for t in hdates}
    scores = composite_score(panel, ctx.cfg["characteristics"]["composite_weights"])
    cost = float(pcfg["backtest"]["cost_bps_one_way"])
    wfs = {k: run_walk_forward(panel, rms, hdates, k, scores, pcfg, te if k == "opt" else None, cost)
           for k in ("ew", "tilt", "opt")}
    from .backtester_adapter import market_data_from_long, run_schedule

    px = S["prices_long"]
    md = market_data_from_long(px[px["date"] >= hdates[0] - pd.Timedelta(days=10)]
                               [["date", "symbol", "open", "high", "low", "close", "volume"]])
    daily = pd.DataFrame({k: run_schedule(md, wfs[k].targets, k, pcfg, ctx.label,
                                          pcfg["max_weight"]).returns for k in wfs})
    rows = [{"strategy": k, **_metrics(daily[k], daily["ew"] if k != "ew" else None)} for k in daily]
    hdir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(hdir / f"synthetic_{size}_holdout_tearsheet.csv", index=False)
    TrialLog(ctx.logdir / "holdout_evaluations.csv").append(
        {"size": size, "te_annual": te, "rerun": rerun or lock.exists(),
         "holdout": f"{S['splits']['holdout'][0]}..{S['splits']['holdout'][1]}"})
    lock.write_text(datetime.now(UTC).isoformat(timespec="seconds"), encoding="utf-8")
    return {"tearsheet": out, "te_annual": te}


__all__ = ["Context", "run", "run_holdout", "STAGES", "rel"]
