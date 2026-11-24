"""Download and cache style-ETF and stock prices via yfinance (needs internet)."""

from __future__ import annotations

import sys

from factors.cli import main

if __name__ == "__main__":
    sys.exit(main(["download", "market", *sys.argv[1:]]))
