"""Stage C: Fama-MacBeth characteristic regressions."""

from __future__ import annotations

import sys

from factors.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "--stages", "fama_macbeth", *sys.argv[1:]]))
