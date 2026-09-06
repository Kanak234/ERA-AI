#!/usr/bin/env python3
"""ERA AI — command-line health check.

    python scripts/health_check.py

Exit code 0 if ready (green/amber), 1 if red.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config          # noqa: E402
from app.health import run_all, overall     # noqa: E402


COLOR = {"green": "\033[92m", "amber": "\033[93m", "red": "\033[91m"}
RESET = "\033[0m"
MARK = {"green": "\u2714", "amber": "\u26a0", "red": "\u2716"}


def main() -> int:
    cfg = load_config()
    print(f"\n  ERA AI \u2014 {cfg.get('app.subtitle', '')}")
    print(f"  owner: {cfg.get('app.owner', '')}\n")

    checks = run_all(cfg)
    for c in checks:
        col = COLOR.get(c.level, "")
        print(f"  {col}{MARK.get(c.level,'?')}{RESET}  {c.name:<14} {c.detail}")

    status = overall(checks)
    col = COLOR.get(status, "")
    print(f"\n  Overall: {col}{status.upper()}{RESET}\n")
    return 0 if status != "red" else 1


if __name__ == "__main__":
    raise SystemExit(main())
