"""(Re)create and execute the four notebooks.  They only call package functions.

    python scripts/build_notebooks.py            # build + execute (offline; real-data cells
                                                 # are skipped with a message if no cache)
    python scripts/build_notebooks.py --no-exec  # build only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """import logging

import pandas as pd
from IPython.display import Image, display

from factors import pipeline as pl
from factors.data import DataUnavailableError  # noqa: F401 - used by the real-data cells

logging.basicConfig(level=logging.WARNING)
pd.set_option("display.width", 160, "display.max_columns", 20)

def show(ctx, nn, name):
    display(Image(filename=str(ctx.fig(nn, name))))"""


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s)


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s)


def nb01() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(cells=[
        md("# 01 - Time-series factor regressions (CAPM, FF3, Carhart, FF5) and GRS\n\n"
           "Every number below is produced by `factors.pipeline.stage_timeseries`. The first part uses "
           "**SYNTHETIC** data (known truth: the 25 portfolios follow FF5 with zero alpha; `ACTIVE_FUND` "
           "has a true alpha of 0.15%/month). The second part uses **REAL** Kenneth French data and US "
           "style ETFs if they have been downloaded (`scripts/download_french_data.py`, "
           "`scripts/download_market_data.py`)."),
        code(SETUP),
        md("## SYNTHETIC"),
        code("syn = pl.Context('synthetic', 'fast')\npl.ensure_synthetic(syn)\nres = pl.stage_timeseries(syn)\n"
             "c = res['coef']\nc[(c.term == 'alpha')].pivot(index='asset', columns='model', values='t_hac').round(2)"),
        code("res['grs'][['model', 'statistic', 'pvalue', 'df1', 'df2', 'T']]"),
        code("show(syn, '01', 'factor_loadings_hac')\nshow(syn, '04', 'grs_and_25_portfolio_alphas')"),
        md("## REAL (French factors, US style ETFs)"),
        code("try:\n    real = pl.Context('real')\n    rr = pl.stage_timeseries(real)\n"
             "    display(rr['grs'][['model', 'statistic', 'pvalue', 'df1', 'df2', 'T', 'sample_start', 'sample_end']])\n"
             "    c = rr['coef']\n"
             "    display(c[c.model == 'Carhart4'].pivot(index='asset', columns='term', values='coef').round(3))\n"
             "    show(real, '02', 'rolling_betas')\n    show(real, '03', 'alpha_comparison')\n"
             "except DataUnavailableError as e:\n    print('REAL data not cached - skipped:', e)"),
        md("**Reading the output.** Alpha is the average return not explained by the factors *in this "
           "sample*; its HAC confidence interval is wide for short ETF histories. GRS is the classical "
           "finite-sample test (i.i.d. normal residuals) and is reported separately from HAC inference."),
    ])


def nb02() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(cells=[
        md("# 02 - Custom characteristics and quintile factor portfolios\n\n"
           "Characteristics are formed at month-end t from information available at t; quintiles earn "
           "month t+1 returns. These are frictionless academic long-short returns, not investable "
           "strategies and not replications of the official French factors."),
        code(SETUP),
        md("## SYNTHETIC (all four characteristics; fundamentals are point-in-time by construction)"),
        code("syn = pl.Context('synthetic', 'fast')\nr = pl.stage_custom_factors(syn)\n"
             "r['tear'][['signal', 'weighting', 'n_months', 'ann_mean', 'ann_vol', 'sharpe_ann', 't_mean_hac', 'max_drawdown']].round(3)"),
        code("r['comparison']"),
        code("show(syn, '05b', 'characteristic_correlations')\nshow(syn, '07', 'custom_factor_cumulative')"),
        md("## REAL (price-only: momentum and low volatility; current large caps -> survivorship bias)"),
        code("try:\n    real = pl.Context('real')\n    rr = pl.stage_custom_factors(real)\n"
             "    display(rr['tear'][['signal', 'weighting', 'n_months', 'ann_mean', 'sharpe_ann', 't_mean_hac']].round(3))\n"
             "    display(rr['comparison'])\n    show(real, '08', 'published_factor_comparison')\n"
             "except DataUnavailableError as e:\n    print('REAL data not cached - skipped:', e)"),
    ])


def nb03() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(cells=[
        md("# 03 - Fama-MacBeth characteristic regressions\n\n"
           "Monthly cross-sectional OLS of next-month excess returns on characteristic z-scores; HAC "
           "inference on the slope means. A significant slope shows predictive association in this "
           "sample, not proof of a priced risk factor. |t| > 3 is an illustrative hurdle only."),
        code(SETUP),
        code("syn = pl.Context('synthetic', 'fast')\nr = pl.stage_fama_macbeth(syn)\n"
             "r['summary'][['spec', 'term', 'premium_monthly', 't_hac', 'n_months', 'avg_r2', 'passes_t3_hurdle']].round(4)"),
        md("Cross-check against `linearmodels.FamaMacBeth` on the matched sample (identical point "
           "estimates; the Bartlett-kernel SE with bandwidth 4 equals the HAC(4) SE with the T/(T-1) "
           "correction)."),
        code("r['lm']"),
        code("show(syn, '09', 'fama_macbeth_premia')"),
        code("try:\n    real = pl.Context('real')\n    rr = pl.stage_fama_macbeth(real)\n"
             "    display(rr['summary'][['spec', 'term', 'premium_monthly', 't_hac', 'n_months']].round(4))\n"
             "except DataUnavailableError as e:\n    print('REAL data not cached - skipped:', e)"),
    ])


def nb04() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(cells=[
        md("# 04 - Fundamental risk model, OOS validation, optimisation and attribution (SYNTHETIC)\n\n"
           "Sigma = B F B' + D, estimated from cross-sectional WLS factor returns; forecasts are made at "
           "t for month t+1 using only information available at t. The optimiser is a score-risk "
           "trade-off (scores are not calibrated expected returns). Strategies are executed through the "
           "project-02 backtester with next-open fills and 10 bps one-way costs. The hold-out is not "
           "used."),
        code(SETUP),
        code("syn = pl.Context('synthetic', 'fast')\nr = pl.stage_risk_and_portfolio(syn)\n"
             "r['bias'][r['bias'].period == 'all'][['group', 'n', 'bias', 'ci_low', 'ci_high']].round(3)"),
        code("r['tear'][['strategy', 'period', 'ann_return', 'ann_vol', 'max_drawdown', 'tracking_error', 'info_ratio']].round(3)"),
        code("att = r['attribution']\nt = att.table()\nprint('sum of contributions =', t.vol_contrib_monthly.sum(), ' sigma =', att.sigma)\nt.round(5)"),
        code("r['olog'][['date', 'status', 'solver', 'fallback_used', 'chk_te_monthly', 'chk_turnover_one_way', 'violations']].tail()"),
        code("for nn, name in [('12', 'predicted_vs_realised_vol'), ('14', 'equity_curves'), ('17', 'risk_contributions_signed')]:\n    show(syn, nn, name)"),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    a = ap.parse_args()
    NB.mkdir(exist_ok=True)
    books = {"01_ff_time_series.ipynb": nb01(), "02_custom_factors.ipynb": nb02(),
             "03_fama_macbeth.ipynb": nb03(), "04_risk_model_and_portfolio.ipynb": nb04()}
    for name, nb in books.items():
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB / name
        if not a.no_exec:
            from nbclient import NotebookClient

            NotebookClient(nb, timeout=900, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.write(nb, path)
        print(f"{'executed' if not a.no_exec else 'built'}: notebooks/{name}")


if __name__ == "__main__":
    main()
