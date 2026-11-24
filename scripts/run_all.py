"""Run every stage.  Default: offline SYNTHETIC fast mode.  Never reads the hold-out.

    python scripts/run_all.py                      # synthetic, fast
    python scripts/run_all.py --size research      # synthetic, larger
    python scripts/run_all.py --mode real          # needs cached French + yfinance data
"""

from __future__ import annotations

import sys

from factors.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
