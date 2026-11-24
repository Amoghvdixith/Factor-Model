"""Stage B: characteristics, quintile sorts and long-short factors."""

from __future__ import annotations

import sys

from factors.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", "--stages", "custom_factors", *sys.argv[1:]]))
