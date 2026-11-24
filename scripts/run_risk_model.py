"""Stages D/E: risk model, OOS risk validation, optimisation, backtest, attribution."""

from __future__ import annotations

import sys

from factors.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "--stages", "risk_and_portfolio", *sys.argv[1:]]))
