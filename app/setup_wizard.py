"""ERA AI — first-run setup wizard (spec Section 28).

Detects hardware, checks Ollama, lists missing workers with the exact
pull commands, and auto-tunes how many models may be hot at once based
on available RAM. Logic is headless-testable; the GUI wraps it in a panel.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import psutil

from app.config import AppConfig
from app.ollama_client import OllamaClient
from app.registry import load_registry


@dataclass
class HardwareProfile:
    total_ram_gb: float
    cpu_count: int
    has_gpu: bool
    recommended_max_hot: int


def detect_hardware() -> HardwareProfile:
    total = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    has_gpu = _has_nvidia_gpu()
    # conservative: ~7 GB per hot 7-8B model, leave headroom
    max_hot = max(1, int((total - 4) // 7))
    return HardwareProfile(total, psutil.cpu_count() or 1, has_gpu, max_hot)


def _has_nvidia_gpu() -> bool:
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return False
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def plan_setup(cfg: AppConfig) -> Dict:
    hw = detect_hardware()
    client = OllamaClient(cfg.ollama_host, cfg.request_timeout)
    ollama_up = client.is_up()

    installed = client.list_models() if ollama_up else []
    reg = load_registry(cfg.root, installed)

    missing = [
        {"label": w.label, "pull": f"ollama pull {w.ollama_tag.split(':')[0]}"}
        for w in reg.workers if not w.installed
    ]
    ready = [w.label for w in reg.available_workers()]

    return {
        "hardware": asdict(hw),
        "ollama_running": ollama_up,
        "workers_ready": ready,
        "workers_missing": missing,
        "recommended_max_hot_models": hw.recommended_max_hot,
        "gpu_note": ("GPU detected — fine-tuning and parallel inference are "
                     "viable." if hw.has_gpu else
                     "No GPU detected — inference is fine; fine-tuning will be "
                     "slow. Prefer phi4-mini as the trainable target."),
    }


def write_profile(cfg: AppConfig, plan: Dict) -> Path:
    p = cfg.dir("data") / "hardware_profile.json"
    p.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return p
