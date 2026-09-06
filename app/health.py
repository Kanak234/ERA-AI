"""ERA AI — health checks (spec Section 22).

Reusable readiness checks. Returns structured results so both the CLI
(scripts/health_check.py) and the GUI setup panel can share them.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.config import AppConfig
from app.ollama_client import OllamaClient
from app.registry import load_registry


@dataclass
class Check:
    name: str
    ok: bool
    level: str          # "green" | "amber" | "red"
    detail: str


def run_all(cfg: AppConfig) -> List[Check]:
    checks: List[Check] = []

    # 1. Python version
    py_ok = sys.version_info >= (3, 10)
    checks.append(Check(
        "Python", py_ok, "green" if py_ok else "red",
        f"{sys.version_info.major}.{sys.version_info.minor} "
        f"({'ok' if py_ok else 'need 3.10+'})",
    ))

    # 2. Directories
    try:
        for key in ("data", "conversations", "memory", "recordings", "logs", "exports"):
            cfg.dir(key)
        checks.append(Check("Directories", True, "green", "instance folders ready"))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("Directories", False, "red", str(e)))

    # 3. Ollama running
    client = OllamaClient(cfg.ollama_host, cfg.request_timeout)
    up = client.is_up()
    checks.append(Check(
        "Ollama", up, "green" if up else "red",
        f"version {client.version()}" if up else f"not reachable at {cfg.ollama_host}",
    ))

    # 4. Models present
    if up:
        try:
            installed = client.list_models()
            reg = load_registry(cfg.root, installed)
            avail = reg.available_workers()
            missing = [w.label for w in reg.workers if not w.installed]
            if avail:
                level = "green" if not missing else "amber"
                detail = f"{len(avail)} ready"
                if missing:
                    detail += f"; missing: {', '.join(missing)}"
                checks.append(Check("Models", True, level, detail))
            else:
                checks.append(Check(
                    "Models", False, "red",
                    "no declared workers installed (run `ollama pull ...`)",
                ))
        except Exception as e:  # noqa: BLE001
            checks.append(Check("Models", False, "red", str(e)))
    else:
        checks.append(Check("Models", False, "amber", "skipped (Ollama down)"))

    return checks


def overall(checks: List[Check]) -> str:
    if any(c.level == "red" for c in checks):
        return "red"
    if any(c.level == "amber" for c in checks):
        return "amber"
    return "green"
