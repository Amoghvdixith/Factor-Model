"""Download and cache Kenneth French factor and portfolio files (needs internet)."""

from __future__ import annotations

import sys

from factors.cli import main

if __name__ == "__main__":
    sys.exit(main(["download", "french", *sys.argv[1:]]))
