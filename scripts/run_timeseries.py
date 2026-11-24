"""Stage A: CAPM/FF3/Carhart/FF5 regressions, rolling betas, GRS. Options: --mode, --size."""

from __future__ import annotations

import sys

from factors.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "--stages", "timeseries", *sys.argv[1:]]))
