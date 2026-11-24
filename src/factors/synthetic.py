"""Seeded synthetic data with a KNOWN factor structure (clearly labelled SYNTHETIC).

The synthetic market exists so the whole pipeline runs offline and so estimators can be
checked against known truth.  Nothing generated here is evidence about real markets.

Generated objects
-----------------
* Daily OHLCV for ``n_stocks`` stocks (adjusted-price convention), industries, shares
  outstanding, late listings and delistings (the delisting return is embedded in the
  final close).
* Annual fundamentals with fiscal year ends (December or June), observed publication
  dates (random lag) and a fraction of later *restatements* (new vintages).
* A monthly risk-free series.
* Monthly "academic" factors (MKT_RF, SMB, HML, RMW, CMA, MOM, RF), 25 size/BM test
  portfolios whose TRUE model is FF5 with zero alpha, and six style "funds" with known
  loadings (one with drifting HML loading and one with a non-zero true alpha).

Daily stock return model (all premia enter only through factor means)::

    r_it = beta_i * m_t + g_{ind(i),t} + sum_k x_ik,(month) * s_kt + e_it

where m is the market, g industry shocks, s_k style factor returns for size, value,
quality and momentum, and x the *true* exposures fixed within each month using
information available at the previous month end (momentum is computed causally from
simulated prices).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYNTHETIC_LABEL = "SYNTHETIC"


@dataclass
class SyntheticSpec:
    n_stocks: int = 120
    n_years: int = 10
    start: str = "2012-01-02"
    n_industries: int = 8
    seed: int = 20240607
    delisting_prob_per_year: float = 0.01
    delisting_return_range: tuple[float, float] = (-0.5, -0.1)
    late_listing_fraction: float = 0.10
    revision_prob: float = 0.15
    publication_lag_days: tuple[int, int] = (60, 120)
    academic_months: int = 360
    rf_annual: float = 0.02
    # monthly style premia (means of daily style factor returns x 21)
    style_premia_monthly: dict[str, float] = field(default_factory=lambda: {
        "size": -0.0010, "value": 0.0030, "quality": 0.0020, "momentum": 0.0025})
    style_vol_annual: dict[str, float] = field(default_factory=lambda: {
        "size": 0.05, "value": 0.06, "quality": 0.04, "momentum": 0.07})

    @classmethod
    def from_config(cls, data_cfg: dict[str, Any], size: str = "fast") -> SyntheticSpec:
        s = data_cfg["synthetic"]
        sz = s["sizes"][size]
        return cls(
            n_stocks=int(sz["n_stocks"]), n_years=int(sz["n_years"]), start=str(sz["start"]),
            n_industries=int(s["n_industries"]), seed=int(s["seed"]),
            delisting_prob_per_year=float(s["delisting_prob_per_year"]),
            delisting_return_range=tuple(s["delisting_return_range"]),  # type: ignore[arg-type]
            late_listing_fraction=float(s["late_listing_fraction"]),
            revision_prob=float(s["revision_prob"]),
            publication_lag_days=tuple(s["publication_lag_days"]),  # type: ignore[arg-type]
        )


INDUSTRY_NAMES = ["Energy", "Materials", "Industrials", "ConsDisc", "Staples",
                  "Health", "Financials", "Tech", "Telecom", "Utilities", "RealEstate", "Other"]


def _ar1(rng: np.random.Generator, n_steps: int, n: int, phi: float) -> np.ndarray:
    x = np.empty((n_steps, n))
    x[0] = rng.standard_normal(n)
    s = np.sqrt(1 - phi ** 2)
    for t in range(1, n_steps):
        x[t] = phi * x[t - 1] + s * rng.standard_normal(n)
    return x


def _zs(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    out = (x - m) / s if s > 0 else np.zeros_like(x)
    return np.where(np.isfinite(out), out, 0.0)


def generate_market(spec: SyntheticSpec) -> dict[str, Any]:
    """Simulate the synthetic stock market. Returns frames and the known truth."""
    rng = np.random.default_rng(spec.seed)
    n, k_ind = spec.n_stocks, spec.n_industries
    dates = pd.bdate_range(spec.start, periods=252 * spec.n_years)
    T = len(dates)
    tickers = [f"S{i:04d}" for i in range(n)]
    months = dates.to_period("M")
    month_codes, month_index = pd.factorize(months)
    n_months = len(month_index)

    industry = np.array([INDUSTRY_NAMES[i] for i in rng.integers(0, k_ind, n)])
    # guarantee every industry has stocks
    for j in range(k_ind):
        industry[j] = INDUSTRY_NAMES[j]
    beta = np.clip(rng.normal(1.0, 0.25, n), 0.4, 1.8)
    idio_ann = np.clip(np.exp(rng.normal(np.log(0.20), 0.35, n)), 0.08, 0.60)
    log_cap0 = rng.normal(22.5, 1.2, n)
    price0 = rng.uniform(20, 200, n)
    shares = np.exp(log_cap0) / price0

    # latent slowly-moving characteristics (monthly)
    value_true = _ar1(rng, n_months, n, 0.97)
    quality_true = _ar1(rng, n_months, n, 0.98)

    # listing / delisting
    list_idx = np.zeros(n, dtype=int)
    late = rng.random(n) < spec.late_listing_fraction
    late[:k_ind] = False
    list_idx[late] = rng.integers(21, int(0.6 * T), late.sum())
    daily_hazard = spec.delisting_prob_per_year / 252
    delist_idx = np.full(n, T, dtype=int)  # T = never
    draw = rng.geometric(daily_hazard, n) if daily_hazard > 0 else np.full(n, T + 1)
    cand = list_idx + draw
    has_delist = cand < T - 1
    has_delist[:k_ind] = False
    delist_idx[has_delist] = cand[has_delist]
    delist_ret = rng.uniform(*spec.delisting_return_range, n)

    # market regime and factor returns (daily)
    regime = np.exp(_ar1(rng, n_months, 1, 0.85)[:, 0] * 0.25)  # monthly vol multiplier
    v = regime[month_codes]
    mkt = 0.07 / 252 + 0.16 / np.sqrt(252) * v * rng.standard_normal(T)
    ind_ret = 0.07 / np.sqrt(252) * v[:, None] * rng.standard_normal((T, k_ind))
    styles = ["size", "value", "quality", "momentum"]
    s_ret = np.column_stack([
        spec.style_premia_monthly[s] / 21 + spec.style_vol_annual[s] / np.sqrt(252) * v
        * rng.standard_normal(T) for s in styles])
    eps = (idio_ann / np.sqrt(252))[None, :] * v[:, None] * rng.standard_normal((T, n))
    ind_code = np.array([INDUSTRY_NAMES.index(s) for s in industry])

    logp = np.full((T, n), np.nan)
    cur = np.log(price0).copy()
    month_start = np.r_[0, np.flatnonzero(np.diff(month_codes)) + 1]
    month_end = np.r_[month_start[1:], T]
    expo_hist = np.zeros((n_months, n, len(styles)))
    for m, (a, b) in enumerate(zip(month_start, month_end, strict=True)):
        # exposures fixed for month m from information at the end of month m-1
        cap_prev = np.exp(cur) * shares
        size_x = _zs(np.log(cap_prev))
        if m >= 13:
            p_end = logp[month_start[m - 1] - 1] if month_start[m - 1] > 0 else cur
            p_beg = logp[month_start[m - 12] - 1] if month_start[m - 12] > 0 else np.log(price0)
            mom_raw = p_end - p_beg  # log return from end m-13 to end m-2 (skip m-1)
            mom_x = _zs(mom_raw)
        else:
            mom_x = np.zeros(n)
        x = np.column_stack([size_x, value_true[m - 1] if m else value_true[0],
                             quality_true[m - 1] if m else quality_true[0], mom_x])
        expo_hist[m] = x
        r = (beta[None, :] * mkt[a:b, None] + ind_ret[a:b][:, ind_code]
             + s_ret[a:b] @ x.T + eps[a:b])
        lr = np.log1p(np.clip(r, -0.95, None))
        path = cur[None, :] + np.cumsum(lr, axis=0)
        logp[a:b] = path
        cur = path[-1]

    close = np.exp(logp)
    # delisting: embed delisting return in the final close, NaN afterwards
    for i in range(n):
        if delist_idx[i] < T:
            d = delist_idx[i]
            close[d, i] = close[d - 1, i] * (1 + delist_ret[i])
            close[d + 1:, i] = np.nan
        if list_idx[i] > 0:
            close[: list_idx[i], i] = np.nan
    # OHLC: open from an overnight share of the log return
    prev = np.vstack([np.full((1, n), np.nan), close[:-1]])
    lr = np.log(close / prev)
    lr0 = np.where(np.isfinite(lr), lr, 0.0)
    open_ = np.where(np.isfinite(prev), prev * np.exp(0.3 * lr0), close)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, (T, n))))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, (T, n))))
    vol = shares[None, :] * 0.006 * np.exp(rng.normal(0, 0.3, (T, n)))
    vol = np.round(vol)
    mask = np.isfinite(close)
    for a in (open_, hi, lo, vol):
        a[~mask] = np.nan

    def wide(a: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(a, index=dates, columns=tickers)

    rf_m = pd.Series((1 + spec.rf_annual) ** (1 / 12) - 1
                     + 0.0005 * _ar1(rng, n_months, 1, 0.9)[:, 0],
                     index=month_index, name="RF")
    rf_m = rf_m.clip(lower=0.0)
    ohlcv = {"open": wide(open_), "high": wide(hi), "low": wide(lo), "close": wide(close),
             "volume": wide(vol)}
    meta = pd.DataFrame({
        "ticker": tickers, "industry": industry, "shares": shares, "beta_true": beta,
        "idio_vol_true_annual": idio_ann,
        "list_date": [dates[i] for i in list_idx],
        "delist_date": [dates[i] if i < T else pd.NaT for i in delist_idx],
        "delist_return": np.where(delist_idx < T, delist_ret, np.nan),
    })
    style_m = pd.DataFrame(s_ret, index=dates, columns=styles).groupby(months).apply(
        lambda x: (1 + x).prod() - 1)
    style_m.index.name = "month"
    truth = {
        "style_returns_monthly": style_m,
        "style_premia_monthly": spec.style_premia_monthly,
        "style_vol_annual": spec.style_vol_annual,
        "value_true": pd.DataFrame(value_true, index=month_index, columns=tickers),
        "quality_true": pd.DataFrame(quality_true, index=month_index, columns=tickers),
    }
    return {"ohlcv": ohlcv, "meta": meta, "rf": rf_m, "truth": truth,
            "month_index": month_index}


def generate_fundamentals(spec: SyntheticSpec, market: dict[str, Any]) -> pd.DataFrame:
    """Annual fundamentals with publication lags and restatements (vintages)."""
    rng = np.random.default_rng(spec.seed + 1)
    close = market["ohlcv"]["close"]
    meta = market["meta"].set_index("ticker")
    vt, qt = market["truth"]["value_true"], market["truth"]["quality_true"]
    first = close.index[0]
    rows = []
    for t in close.columns:
        fy_month = 12 if rng.random() < 0.75 else 6
        sh = meta.loc[t, "shares"]
        for year in range(first.year - 2, close.index[-1].year + 1):
            fpe = pd.Timestamp(year=year, month=fy_month, day=1) + pd.offsets.MonthEnd(0)
            if fpe > close.index[-1]:
                continue
            per = fpe.to_period("M")
            m = per if per in vt.index else (vt.index[0] if per < vt.index[0] else vt.index[-1])
            v, q = float(vt.loc[m, t]), float(qt.loc[m, t])
            px = close[t].loc[:fpe].dropna()
            p = float(px.iloc[-1]) if len(px) else float(close[t].dropna().iloc[0])
            mcap = p * sh
            book = mcap * np.exp(-1.0 + 0.5 * v + rng.normal(0, 0.2))
            if rng.random() < 0.03:
                book = -abs(book) * 0.2  # negative book equity case
            assets = abs(book) * np.exp(0.7 - 0.3 * q + rng.normal(0, 0.15))
            debt = max(assets - abs(book), 0.0)
            ni = abs(book) * (0.10 + 0.05 * q + rng.normal(0, 0.03))
            gp = assets * (0.30 + 0.08 * q + rng.normal(0, 0.03))
            lag = int(rng.integers(spec.publication_lag_days[0], spec.publication_lag_days[1] + 1))
            pub = fpe + pd.Timedelta(days=lag)
            base = {"ticker": t, "fiscal_period_end": fpe, "currency": "USD",
                    "publication_date": pub, "vintage_date": pub,
                    "book_equity": book, "net_income": ni, "gross_profit": gp,
                    "total_assets": assets, "total_debt": debt, "is_restatement": False}
            rows.append(base)
            if rng.random() < spec.revision_prob:
                rev = dict(base)
                rev["vintage_date"] = pub + pd.Timedelta(days=int(rng.integers(150, 400)))
                rev["net_income"] = ni * (1 + rng.normal(0, 0.15))
                rev["book_equity"] = book * (1 + rng.normal(0, 0.05))
                rev["is_restatement"] = True
                rows.append(rev)
    return pd.DataFrame(rows)


def generate_academic(spec: SyntheticSpec) -> dict[str, Any]:
    """Synthetic monthly factors, 25 test portfolios (true model FF5, alpha=0) and funds."""
    rng = np.random.default_rng(spec.seed + 2)
    T = spec.academic_months
    idx = pd.period_range("1996-01", periods=T, freq="M", name="month")
    names = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    mu = np.array([0.006, 0.002, 0.003, 0.003, 0.003, 0.006])
    sd = np.array([0.045, 0.030, 0.030, 0.020, 0.020, 0.045])
    corr = np.eye(6)
    for (a, b, c) in [(0, 1, 0.25), (2, 4, 0.6), (2, 5, -0.3), (0, 2, -0.1), (3, 4, 0.1),
                      (0, 3, -0.2)]:
        corr[a, b] = corr[b, a] = c
    cov = np.outer(sd, sd) * corr
    f = rng.multivariate_normal(mu, cov, T)
    factors = pd.DataFrame(f, index=idx, columns=names)
    factors["RF"] = np.clip(0.0015 + 0.0008 * _ar1(rng, T, 1, 0.95)[:, 0], 0, None)
    loadings = {}
    port = {}
    for s in range(1, 6):
        for b in range(1, 6):
            L = np.array([1.1 - 0.05 * s, 1.2 - 0.3 * s, -0.5 + 0.3 * b,
                          -0.5 + 0.15 * b + 0.05 * s, -0.4 + 0.2 * b])
            resid_sd = 0.02 - 0.002 * s
            name = f"S{s}B{b}"
            loadings[name] = dict(zip(names[:5], L, strict=True))
            port[name] = factors["RF"].to_numpy() + f[:, :5] @ L + rng.normal(0, resid_sd, T)
    p25 = pd.DataFrame(port, index=idx)
    # funds
    drift = np.linspace(0.7, 0.1, T)
    fund_spec: dict[str, tuple[dict[str, Any], float]] = {
        "VALUE_FUND": ({"MKT_RF": 1.0, "HML": drift}, 0.0),
        "SMALL_FUND": ({"MKT_RF": 1.05, "SMB": 0.8}, 0.0),
        "MOMENTUM_FUND": ({"MKT_RF": 1.0, "MOM": 0.5}, 0.0),
        "QUALITY_FUND": ({"MKT_RF": 0.9, "RMW": 0.4, "CMA": 0.2}, 0.0),
        "MARKET_FUND": ({"MKT_RF": 1.0}, 0.0),
        "ACTIVE_FUND": ({"MKT_RF": 0.95, "SMB": 0.2, "HML": 0.2}, 0.0015),
    }
    funds = {}
    fund_truth = {}
    for name, (ld, alpha) in fund_spec.items():
        r = factors["RF"].to_numpy() + alpha + rng.normal(0, 0.01, T)
        for k, val in ld.items():
            r = r + np.asarray(val) * factors[k].to_numpy()
        funds[name] = r
        fund_truth[name] = {"alpha_monthly": alpha,
                            **{k: (float(np.mean(v)) if np.ndim(v) else v) for k, v in ld.items()}}
    for d in (factors, p25):
        d.attrs["units"] = "decimal"
    return {"factors": factors, "p25": p25, "funds": pd.DataFrame(funds, index=idx),
            "p25_true_loadings": loadings, "fund_truth": fund_truth,
            "factor_mean_true": dict(zip(names, mu.tolist(), strict=True))}


def write_synthetic(spec: SyntheticSpec, directory: Path) -> dict[str, Any]:
    """Generate and write all synthetic datasets with a SYNTHETIC provenance file."""
    directory.mkdir(parents=True, exist_ok=True)
    mk = generate_market(spec)
    fund = generate_fundamentals(spec, mk)
    ac = generate_academic(spec)
    long = []
    for f, df in mk["ohlcv"].items():
        long.append(df.stack().dropna().rename(f))
    px = pd.concat(long, axis=1).reset_index()
    px.columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
    px = px.sort_values(["symbol", "date"])
    px.to_csv(directory / "prices_daily.csv", index=False, float_format="%.6f")
    mk["meta"].to_csv(directory / "stock_meta.csv", index=False)
    mk["rf"].rename_axis("month").to_csv(directory / "rf_monthly.csv")
    fund.to_csv(directory / "fundamentals.csv", index=False)
    ac["factors"].to_csv(directory / "academic_factors.csv")
    ac["p25"].to_csv(directory / "academic_p25.csv")
    ac["funds"].to_csv(directory / "academic_funds.csv")
    mk["truth"]["value_true"].to_csv(directory / "truth_value.csv")
    mk["truth"]["quality_true"].to_csv(directory / "truth_quality.csv")
    mk["truth"]["style_returns_monthly"].to_csv(directory / "truth_style_returns.csv")
    prov = {
        "data_mode": SYNTHETIC_LABEL,
        "warning": "Synthetic data generated by factors.synthetic. Not real market data.",
        "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec).items()},
        "n_price_rows": int(len(px)), "n_fundamental_rows": int(len(fund)),
        "n_restatements": int(fund["is_restatement"].sum()),
        "n_delisted": int(mk["meta"]["delist_date"].notna().sum()),
        "n_late_listed": int((mk["meta"]["list_date"] > px["date"].min()).sum()),
        "units": "decimal returns; prices in synthetic USD",
        "academic_true_model": "25 portfolios: FF5 with zero alpha; funds: see fund_truth",
        "fund_truth": ac["fund_truth"],
        "factor_mean_true_monthly": ac["factor_mean_true"],
        "stock_style_premia_monthly_true": spec.style_premia_monthly,
    }
    (directory / "provenance.json").write_text(json.dumps(prov, indent=2, default=str),
                                               encoding="utf-8")
    return prov


def load_synthetic(directory: Path) -> dict[str, Any]:
    """Load previously written synthetic data."""
    if not (directory / "provenance.json").exists():
        raise FileNotFoundError(
            f"no synthetic data in {directory.name}/; run scripts/generate_synthetic_data.py")
    prov = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    px = pd.read_csv(directory / "prices_daily.csv", parse_dates=["date"])
    meta = pd.read_csv(directory / "stock_meta.csv", parse_dates=["list_date", "delist_date"])
    rf_df = pd.read_csv(directory / "rf_monthly.csv")
    rf = pd.Series(rf_df["RF"].to_numpy(),
                   index=pd.PeriodIndex(rf_df["month"], freq="M", name="month"), name="RF")
    fund = pd.read_csv(directory / "fundamentals.csv",
                       parse_dates=["fiscal_period_end", "publication_date", "vintage_date"])

    def _acad(name: str) -> pd.DataFrame:
        d = pd.read_csv(directory / name)
        d.index = pd.PeriodIndex(d.pop("month"), freq="M", name="month")
        d.attrs["units"] = "decimal"
        return d

    style = _acad("truth_style_returns.csv") if (directory / "truth_style_returns.csv").exists() else None
    return {"prices": px, "meta": meta, "rf": rf, "fundamentals": fund, "truth_style": style,
            "factors": _acad("academic_factors.csv"), "p25": _acad("academic_p25.csv"),
            "funds": _acad("academic_funds.csv"), "provenance": prov}
