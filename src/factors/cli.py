"""Command-line interface: ``factors <command> [options]`` (also used by scripts/)."""

from __future__ import annotations

import argparse
import logging
import sys

from . import pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="factors", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate synthetic data")
    g.add_argument("--size", choices=["fast", "research"], default="fast")
    d = sub.add_parser("download", help="download French factors and/or market prices")
    d.add_argument("what", choices=["french", "market", "all"])
    r = sub.add_parser("run", help="run pipeline stages")
    r.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    r.add_argument("--size", choices=["fast", "research"], default="fast")
    r.add_argument("--stages", nargs="*", choices=list(pipeline.STAGES), default=None)
    r.add_argument("--regenerate", action="store_true", help="regenerate synthetic data first")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if a.cmd == "generate":
        ctx = pipeline.Context(mode="synthetic", size=a.size)
        pipeline.ensure_synthetic(ctx, regenerate=True)
        print(f"synthetic data written to {pipeline.rel(pipeline.synthetic_dir(ctx))}")
        return 0
    if a.cmd == "download":
        from .config import load_config, project_root
        from .data import download_french, download_prices

        root = project_root()
        cfg = load_config("data", root)
        if a.what in ("french", "all"):
            meta = download_french(cfg, root)
            print("French files:", ", ".join(m["file"] for m in meta.values()))
        if a.what in ("market", "all"):
            import pandas as pd

            m = cfg["market"]
            e = download_prices(list(m["etfs"]), m["etf_start"], root / m["directory"] / "etfs")
            print(f"ETFs: {len(e['succeeded'])} ok, failed: {e['failed']}")
            uni = pd.read_csv(root / m["stock_universe_file"], comment="#")
            s = download_prices(list(uni["ticker"]), m["stock_start"], root / m["directory"] / "stocks")
            print(f"stocks: {len(s['succeeded'])} ok, failed: {list(s['failed'])}")
        return 0
    ctx = pipeline.run(a.mode, a.size, a.stages, a.regenerate)
    for k, v in ctx.meta.get("stages", {}).items():
        print(f"{k}: {v.get('status', 'done')} ({v.get('runtime_s', '-')} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
