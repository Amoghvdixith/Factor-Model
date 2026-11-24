"""Reproducibility and an end-to-end smoke run of the pipeline on a tiny project copy."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from factors.synthetic import SyntheticSpec, generate_academic, generate_market

ROOT = Path(__file__).resolve().parents[1]


def _hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()


def test_synthetic_generation_is_deterministic():
    s = SyntheticSpec(n_stocks=20, n_years=2, seed=5, n_industries=4)
    a, b = generate_market(s), generate_market(s)
    assert _hash(a["ohlcv"]["close"]) == _hash(b["ohlcv"]["close"])
    c = generate_market(SyntheticSpec(n_stocks=20, n_years=2, seed=6, n_industries=4))
    assert _hash(a["ohlcv"]["close"]) != _hash(c["ohlcv"]["close"])
    assert _hash(generate_academic(s)["p25"]) == _hash(generate_academic(s)["p25"])


@pytest.mark.slow
def test_end_to_end_tiny_project(tmp_path, monkeypatch):
    pytest.importorskip("backtester")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    cfg = yaml.safe_load((tmp_path / "configs" / "data.yaml").read_text())
    cfg["synthetic"]["sizes"]["fast"] = {"n_stocks": 70, "n_years": 7, "start": "2012-01-02"}
    cfg["synthetic"]["n_industries"] = 4
    (tmp_path / "configs" / "data.yaml").write_text(yaml.safe_dump(cfg))
    rcfg = yaml.safe_load((tmp_path / "configs" / "risk_model.yaml").read_text())
    rcfg["factor_cov"]["min_periods"] = 12
    rcfg["validation"]["n_random_portfolios"] = 3
    rcfg["validation"]["bootstrap_reps"] = 50
    (tmp_path / "configs" / "risk_model.yaml").write_text(yaml.safe_dump(rcfg))
    monkeypatch.setenv("FACTORS_ROOT", str(tmp_path))
    from factors.pipeline import run

    ctx = run("synthetic", "fast")
    figs = sorted(p.name for p in (tmp_path / "results" / "figures").glob("*.png"))
    assert len(figs) >= 21 and all(f.startswith("synthetic_") for f in figs)
    meta = json.loads((tmp_path / "results" / "run_metadata.json").read_text())["synthetic_fast"]
    assert meta["data_mode"] == "SYNTHETIC" and meta["holdout"]["consumed_by_run_all"] is False
    st = meta["stages"]["risk_and_portfolio"]
    assert st["solver_status_counts"] and st["post_solve_violations"] == 0
    # the research panel never reaches the hold-out
    h0 = pd.Period(meta["holdout"]["start"], "M")
    panel = ctx.cache["stock"]["panel"]
    assert panel.index.get_level_values(0).max().to_period("M") < h0
    assert pd.to_datetime(ctx.cache["stock"]["prices_long"]["date"]).max().to_period("M") < h0
