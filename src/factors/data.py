"""Data loading: Kenneth French library, cached market prices and user fundamentals.

Conventions
-----------
* All returns leave this module as **decimal** returns (0.01 = 1%).  French files are
  published in percent; :func:`parse_french_csv` divides by 100 exactly once and the
  returned frame carries ``attrs["units"] == "decimal"`` so a second conversion can be
  detected (:func:`assert_decimal`).
* French missing-value codes (-99.99 and -999) become NaN *before* scaling.
* Monthly academic series are indexed by a monthly ``PeriodIndex`` (the return month).
* A failed download raises :class:`DataUnavailableError`.  Nothing in this package ever
  replaces failed real data with synthetic observations.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MISSING_CODES = (-99.99, -999.0)

FRENCH_RENAME = {
    "Mkt-RF": "MKT_RF", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA",
    "RF": "RF", "Mom": "MOM", "UMD": "MOM",
}


class DataUnavailableError(RuntimeError):
    """Raised when requested real data cannot be obtained (never silently replaced)."""


class UnitError(ValueError):
    """Raised when return units are ambiguous or converted twice."""


class SchemaError(ValueError):
    """Raised when an input table does not match the documented schema."""


# --------------------------------------------------------------------------- units
def percent_to_decimal(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a percent-return frame to decimal returns exactly once."""
    if df.attrs.get("units") == "decimal":
        raise UnitError("frame is already in decimal units; refusing to divide by 100 again")
    out = df.replace(list(MISSING_CODES), np.nan).astype(float) / 100.0
    out.attrs.update(df.attrs)
    out.attrs["units"] = "decimal"
    return out


def assert_decimal(df: pd.DataFrame | pd.Series, max_abs: float = 2.0) -> None:
    """Heuristic guard: monthly decimal returns should not routinely exceed ``max_abs``."""
    if df.attrs.get("units") not in (None, "decimal"):
        raise UnitError(f"expected decimal units, got {df.attrs.get('units')!r}")
    vals = np.asarray(df, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size and np.nanmedian(np.abs(vals)) > 0.5:
        raise UnitError("median |return| > 50%: data look like percent, not decimal")
    if vals.size and np.abs(vals).max() > max_abs * 100:
        raise UnitError("returns far outside any plausible decimal range")


# --------------------------------------------------------------------------- French
def parse_french_csv(text: str, section: int = 0) -> pd.DataFrame:
    """Parse the ``section``-th block of monthly (YYYYMM) rows in a French CSV.

    The first block of a French factor file is the monthly table; later blocks are
    annual tables (YYYY rows, ignored because they do not match YYYYMM) or, for the
    portfolio files, other weighting schemes.  Returns decimal returns indexed by a
    monthly PeriodIndex named ``month``.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(lines):
        if re.match(r"^\s*\d{6}\s*,", lines[i]):
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            header = lines[j] if j >= 0 else ""
            rows = []
            while i < len(lines) and re.match(r"^\s*\d{6}\s*,", lines[i]):
                rows.append(lines[i])
                i += 1
            blocks.append((header, rows))
        else:
            i += 1
    if len(blocks) <= section:
        raise SchemaError(f"French file has {len(blocks)} monthly blocks; wanted index {section}")
    header, rows = blocks[section]
    cols = [c.strip() for c in header.split(",")][1:]
    data = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    if data.shape[1] - 1 != len(cols):
        raise SchemaError("French header/row width mismatch")
    data.columns = ["yyyymm", *cols]
    idx = pd.PeriodIndex([pd.Period(f"{int(v) // 100}-{int(v) % 100:02d}", "M")
                          for v in data["yyyymm"]], name="month")
    raw = data.drop(columns="yyyymm").set_index(idx).astype(float)
    raw.attrs["units"] = "percent"
    out = percent_to_decimal(raw)
    out = out.rename(columns={c: FRENCH_RENAME.get(c, c) for c in out.columns})
    if not out.index.is_monotonic_increasing or out.index.has_duplicates:
        raise SchemaError("French months are not sorted/unique")
    return out


def rename_25_portfolios(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the 25 size x book-to-market columns positionally to S{size}B{bm}.

    French orders the 25 columns size-major (small to big) and, within size, book-to-
    market low to high, e.g. ``SMALL LoBM, ME1 BM2, ..., BIG HiBM``.
    """
    if df.shape[1] != 25:
        raise SchemaError(f"expected 25 portfolio columns, got {df.shape[1]}")
    out = df.copy()
    out.columns = [f"S{s}B{b}" for s in range(1, 6) for b in range(1, 6)]
    out.attrs.update(df.attrs)
    return out


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download_french(cfg: dict[str, Any], root: Path, keys: Iterable[str] | None = None,
                    timeout: float = 60.0) -> dict[str, dict[str, Any]]:
    """Download French zip files into ``<french.directory>/raw`` with provenance JSON.

    Raises :class:`DataUnavailableError` on any failure (no fallback).
    """
    import requests  # optional dependency, imported lazily

    fr = cfg["french"]
    raw_dir = root / fr["directory"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, Any]] = {}
    for key in keys or fr["files"].keys():
        fname = fr["files"][key]
        url = fr["base_url"] + fname
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "factor-model-research/0.1"})
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise DataUnavailableError(f"French download failed for {fname}: {exc}") from exc
        content = resp.content
        try:
            zipfile.ZipFile(io.BytesIO(content)).namelist()
        except zipfile.BadZipFile as exc:
            raise DataUnavailableError(f"{fname} is not a valid zip archive") from exc
        (raw_dir / fname).write_bytes(content)
        meta = {
            "source": "Kenneth R. French Data Library", "url": url, "file": fname,
            "region": fr.get("region", "US"), "frequency": fr.get("frequency", "monthly"),
            "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "sha256": _sha256(content), "bytes": len(content),
            "units_published": "percent", "missing_codes": list(MISSING_CODES),
        }
        (raw_dir / f"{fname}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        out[key] = meta
    return out


def _read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not names:
            raise SchemaError(f"no CSV inside {path.name}")
        return zf.read(names[0]).decode("latin-1")


def load_french(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Load cached French data (monthly, decimal).

    Returns a dict with ``factors`` (MKT_RF, SMB, HML, RMW, CMA, MOM, RF on the union of
    months; columns may be NaN where a file does not cover a month), ``p25`` (25
    value-weighted size/BM portfolio *raw* returns) and ``provenance``.
    """
    fr = cfg["french"]
    raw_dir = root / fr["directory"] / "raw"
    paths = {k: raw_dir / v for k, v in fr["files"].items()}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise DataUnavailableError(
            f"French files not cached: {missing}. Run scripts/download_french_data.py")
    ff3 = parse_french_csv(_read_zip_text(paths["ff3"]))
    ff5 = parse_french_csv(_read_zip_text(paths["ff5"]))
    mom = parse_french_csv(_read_zip_text(paths["mom"]))
    p25 = rename_25_portfolios(parse_french_csv(_read_zip_text(paths["p25"]), section=0))
    # FF3 and FF5 versions of MKT_RF/SMB/HML/RF differ slightly (different breakpoints
    # for SMB); keep the FF5 file's columns for FF5 models and FF3's for FF3 models.
    factors = ff3.add_suffix("_ff3").join(ff5.add_suffix("_ff5"), how="outer")
    factors = factors.join(mom, how="outer")
    factors.attrs["units"] = "decimal"
    prov = {}
    for k, p in paths.items():
        m = p.with_name(p.name + ".meta.json")
        prov[k] = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {"file": p.name}
    coverage = {
        "ff3": [str(ff3.index[0]), str(ff3.index[-1])],
        "ff5": [str(ff5.index[0]), str(ff5.index[-1])],
        "mom": [str(mom.index[0]), str(mom.index[-1])],
        "p25": [str(p25.index[0]), str(p25.index[-1])],
    }
    return {"factors": factors, "p25": p25, "provenance": prov, "coverage": coverage}


def model_factor_frame(french_factors: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Select a model's factor columns from :func:`load_french` output.

    FF5-family models take MKT_RF/SMB/HML/RF from the 5-factor file; CAPM/FF3/Carhart
    from the 3-factor file.  MOM is the separate momentum file.
    """
    five = any(c in ("RMW", "CMA") for c in model_cols)
    suffix = "_ff5" if five else "_ff3"
    out = {}
    for c in model_cols:
        if c == "MOM":
            out[c] = french_factors["MOM"]
        else:
            out[c] = french_factors[f"{c}{suffix}"]
    df = pd.DataFrame(out)
    df.attrs["units"] = "decimal"
    df.attrs["source_file"] = "FF5" if five else "FF3"
    return df


# --------------------------------------------------------------------------- prices
def download_prices(tickers: Iterable[str], start: str, directory: Path,
                    end: str | None = None, pause: float = 0.0) -> dict[str, Any]:
    """Download adjusted daily OHLCV with yfinance into ``directory/<ticker>.csv``.

    Each successful ticker is cached with OHLC adjusted for splits and dividends
    (``auto_adjust=True``).  Failures are recorded and **not** replaced.
    """
    import yfinance as yf  # optional dependency

    directory.mkdir(parents=True, exist_ok=True)
    ok, failed = [], {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False,
                             threads=False, multi_level_index=False)
        except Exception as exc:  # noqa: BLE001
            failed[t] = repr(exc)
            continue
        if df is None or df.empty:
            failed[t] = "empty response"
            continue
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df.to_csv(directory / f"{t}.csv")
        ok.append(t)
        if pause:
            time.sleep(pause)
    meta = {
        "source": "Yahoo Finance via yfinance", "adjustment": "auto_adjust=True (split+dividend adjusted OHLC)",
        "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "start": start, "end": end, "succeeded": ok, "failed": failed,
        "currency": "USD (assumed for US listings; not verified per ticker)",
        "survivorship": "tickers chosen from CURRENT listings; delisted firms absent",
    }
    (directory / "_provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_cached_prices(directory: Path, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Load cached daily OHLCV CSVs into a long frame (date, symbol, open..volume)."""
    files = sorted(directory.glob("*.csv"))
    want = set(tickers) if tickers is not None else None
    parts = []
    for f in files:
        sym = f.stem
        if want is not None and sym not in want:
            continue
        d = pd.read_csv(f, parse_dates=["date"])
        d["symbol"] = sym
        parts.append(d)
    if not parts:
        raise DataUnavailableError(f"no cached price files in {directory.name}/")
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def monthly_returns_from_daily(close: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month returns from wide daily adjusted closes (PeriodIndex ``month``).

    A month's return needs an observed close at the last session of the previous month
    *and* at least one observation in the month (the last observed close is used).
    """
    month = pd.DatetimeIndex(close.index).to_period("M")
    last = close.groupby(month).last()
    # close observed at the final session of each month
    last_session = close.groupby(month).tail(1)
    last_session.index = pd.DatetimeIndex(last_session.index).to_period("M")
    prev = last_session.shift(1)
    out = last / prev - 1.0
    out.index.name = "month"
    out.attrs["units"] = "decimal"
    return out.iloc[1:]


# --------------------------------------------------------------------------- fundamentals
FUNDAMENTAL_REQUIRED = ["ticker", "fiscal_period_end", "currency"]
FUNDAMENTAL_VALUES = ["book_equity", "net_income", "gross_profit", "total_assets", "total_debt"]


def validate_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a historical fundamentals extract and compute ``available_at``.

    Required columns: ``ticker, fiscal_period_end, currency`` plus either an observed
    ``publication_date`` or an ``assumed_lag_days``.  An optional ``vintage_date`` marks
    when a (possibly restated) version became available; it defaults to the
    publication date.  ``availability_source`` records whether availability was
    *observed* or *assumed* from a lag - an assumed lag is not proof that a revised
    database value was point-in-time.
    """
    miss = [c for c in FUNDAMENTAL_REQUIRED if c not in df.columns]
    if miss:
        raise SchemaError(f"fundamentals missing columns {miss}")
    if "publication_date" not in df.columns and "assumed_lag_days" not in df.columns:
        raise SchemaError("fundamentals need publication_date or assumed_lag_days")
    d = df.copy()
    d["fiscal_period_end"] = pd.to_datetime(d["fiscal_period_end"])
    pub = pd.to_datetime(d["publication_date"]) if "publication_date" in d else pd.Series(pd.NaT, index=d.index)
    lag = d["assumed_lag_days"] if "assumed_lag_days" in d else pd.Series(np.nan, index=d.index)
    assumed = d["fiscal_period_end"] + pd.to_timedelta(lag, unit="D")
    d["availability_source"] = np.where(pub.notna(), "observed", "assumed_lag")
    d["publication_date"] = pub.where(pub.notna(), assumed)
    if d["publication_date"].isna().any():
        raise SchemaError("rows with neither publication_date nor assumed_lag_days")
    if "vintage_date" in d.columns:
        d["vintage_date"] = pd.to_datetime(d["vintage_date"]).fillna(d["publication_date"])
    else:
        d["vintage_date"] = d["publication_date"]
    if (d["vintage_date"] < d["publication_date"]).any():
        raise SchemaError("vintage_date earlier than publication_date")
    if (d["publication_date"] < d["fiscal_period_end"]).any():
        raise SchemaError("publication_date earlier than fiscal_period_end")
    d["available_at"] = d["vintage_date"]
    dup = d.duplicated(["ticker", "fiscal_period_end", "vintage_date"])
    if dup.any():
        raise SchemaError("duplicate (ticker, fiscal_period_end, vintage_date) rows")
    for c in FUNDAMENTAL_VALUES:
        if c not in d.columns:
            d[c] = np.nan
    return d.sort_values(["ticker", "fiscal_period_end", "vintage_date"]).reset_index(drop=True)


def normalise_currency(df: pd.DataFrame, value_cols: list[str], currency_col: str = "currency",
                       target: str | None = None) -> pd.DataFrame:
    """Normalise GBp (pence) to GBP and refuse silent currency mixing.

    Values quoted in ``GBp``/``GBX`` are divided by 100 and relabelled ``GBP``.  If more
    than one currency remains (or it differs from ``target``), a :class:`UnitError` is
    raised: converting currencies needs FX data, which this package does not invent.
    """
    d = df.copy()
    pence = d[currency_col].isin(["GBp", "GBX", "GBx"])
    d.loc[pence, value_cols] = d.loc[pence, value_cols].astype(float) / 100.0
    d.loc[pence, currency_col] = "GBP"
    cur = set(d[currency_col].dropna().unique())
    if len(cur) > 1:
        raise UnitError(f"mixed currencies {sorted(cur)}; supply FX-converted data")
    if target is not None and cur and cur != {target}:
        raise UnitError(f"currency {cur} differs from target {target}")
    return d
