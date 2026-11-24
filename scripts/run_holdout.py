"""Evaluate the pre-selected strategies on the PROTECTED hold-out (synthetic).

Not part of run_all.py.  Requires --confirm; a second evaluation also needs --rerun.
Use it once, after all research decisions are final.
"""

from __future__ import annotations

import argparse
import sys

from factors.pipeline import HoldoutLocked, run_holdout

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--size", choices=["fast", "research"], default="fast")
    a = ap.parse_args()
    try:
        res = run_holdout(a.confirm, a.size, a.rerun)
    except HoldoutLocked as exc:
        print(f"REFUSED: {exc}")
        sys.exit(2)
    print(res["tearsheet"].to_string(index=False))
