"""Offline tests of the download/cache code paths with fake providers (no network)."""

import io
import json
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest

from factors.data import (
    DataUnavailableError,
    download_french,
    download_prices,
    load_cached_prices,
    load_french,
    model_factor_frame,
)

CFG = {"french": {"base_url": "https://example.invalid/", "region": "US", "frequency": "monthly",
                  "directory": "data/french",
                  "files": {"ff3": "ff3.zip", "ff5": "ff5.zip", "mom": "mom.zip", "p25": "p25.zip"}}}


def _csv(cols, rows):
    head = "header line\n\n," + ",".join(cols) + "\n"
    return head + "\n".join(f"{m}," + ",".join(f"{v:.2f}" for v in r) for m, r in rows) + "\n"


def _zip(text):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("data.csv", text)
    return b.getvalue()


def _payloads():
    months = [202001, 202002, 202003]
    rng = np.random.default_rng(0)
    return {
        "ff3.zip": _zip(_csv(["Mkt-RF", "SMB", "HML", "RF"], [(m, rng.normal(0, 3, 4)) for m in months])),
        "ff5.zip": _zip(_csv(["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], [(m, rng.normal(0, 3, 6)) for m in months])),
        "mom.zip": _zip(_csv(["Mom"], [(m, rng.normal(0, 3, 1)) for m in months])),
        "p25.zip": _zip(_csv([f"c{i}" for i in range(25)], [(m, rng.normal(0, 3, 25)) for m in months])),
    }


class _Resp:
    def __init__(self, content, status=200):
        self.content, self.status = content, status

    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError(f"HTTP {self.status}")


def test_download_and_load_french_offline(tmp_path, monkeypatch):
    pay = _payloads()
    fake = types.SimpleNamespace(get=lambda url, **kw: _Resp(pay[url.rsplit("/", 1)[1]]))
    monkeypatch.setitem(sys.modules, "requests", fake)
    meta = download_french(CFG, tmp_path)
    assert set(meta) == {"ff3", "ff5", "mom", "p25"}
    assert json.loads((tmp_path / "data/french/raw/ff3.zip.meta.json").read_text())["units_published"] == "percent"
    fr = load_french(CFG, tmp_path)
    assert fr["factors"].attrs["units"] == "decimal" and fr["p25"].shape == (3, 25)
    f5 = model_factor_frame(fr["factors"], ["MKT_RF", "SMB", "HML", "RMW", "CMA"])
    assert f5.attrs["source_file"] == "FF5" and list(f5.columns)[-1] == "CMA"
    c4 = model_factor_frame(fr["factors"], ["MKT_RF", "SMB", "HML", "MOM"])
    assert c4.attrs["source_file"] == "FF3"


def test_failed_french_download_raises_without_fallback(tmp_path, monkeypatch):
    fake = types.SimpleNamespace(get=lambda url, **kw: _Resp(b"", 503))
    monkeypatch.setitem(sys.modules, "requests", fake)
    with pytest.raises(DataUnavailableError):
        download_french(CFG, tmp_path)
    fake2 = types.SimpleNamespace(get=lambda url, **kw: _Resp(b"not a zip"))
    monkeypatch.setitem(sys.modules, "requests", fake2)
    with pytest.raises(DataUnavailableError):
        download_french(CFG, tmp_path)
    with pytest.raises(DataUnavailableError):
        load_french(CFG, tmp_path / "empty")


def test_download_prices_records_failures(tmp_path, monkeypatch):
    def fake_download(t, **kw):
        if t == "BAD":
            return pd.DataFrame()
        if t == "ERR":
            raise RuntimeError("boom")
        idx = pd.bdate_range("2020-01-01", periods=3, tz="UTC")
        return pd.DataFrame({"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 10},
                            index=idx)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    meta = download_prices(["OK", "BAD", "ERR"], "2020-01-01", tmp_path)
    assert meta["succeeded"] == ["OK"] and set(meta["failed"]) == {"BAD", "ERR"}
    px = load_cached_prices(tmp_path)
    assert px["symbol"].unique().tolist() == ["OK"] and px["date"].dt.tz is None
    with pytest.raises(DataUnavailableError):
        load_cached_prices(tmp_path / "nothing")
