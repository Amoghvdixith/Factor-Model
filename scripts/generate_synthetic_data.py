"""Generate the seeded SYNTHETIC dataset (use --size research for the larger one)."""

from __future__ import annotations

import sys

from factors.cli import main

if __name__ == "__main__":
    sys.exit(main(["generate", *sys.argv[1:]]))
