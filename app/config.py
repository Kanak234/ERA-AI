"""ERA AI — configuration loader.

Loads config/settings.yaml, resolves all paths relative to the app
root (portable), and exposes a single AppConfig object. Modules should
receive this object rather than reading files/env directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


APP_ROOT = Path(__file__).resolve().parent.parent


def _deep_get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class AppConfig:
    root: Path
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience accessors -------------------------------------
    def get(self, path: str, default=None):
        return _deep_get(self.raw, path, default)

    @property
    def ollama_host(self) -> str:
        return os.environ.get("ERA_OLLAMA_HOST") or self.get(
            "ollama.host", "http://localhost:11434"
        )

    @property
    def request_timeout(self) -> int:
        return int(self.get("ollama.request_timeout", 300))

    @property
    def default_keep_alive(self) -> str:
        return str(self.get("ollama.default_keep_alive", "5m"))

    @property
    def max_hot_models(self) -> int:
        return int(self.get("models.max_hot_models", 1))

    @property
    def ram_safety_margin_mb(self) -> int:
        return int(self.get("models.ram_safety_margin_mb", 2048))

    @property
    def allow_true_parallel(self) -> bool:
        return bool(self.get("models.allow_true_parallel", False))

    @property
    def auto_route(self) -> bool:
        return bool(self.get("routing.auto_route", True))

    @property
    def temperature(self) -> float:
        return float(self.get("generation.temperature", 0.7))

    @property
    def max_tokens(self) -> int:
        return int(self.get("generation.max_tokens", 2048))

    @property
    def local_only_mode(self) -> bool:
        return bool(self.get("privacy.local_only_mode", True))

    @property
    def language(self) -> str:
        return str(self.get("app.language", "en"))

    @property
    def theme(self) -> str:
        return str(self.get("app.theme", "dark"))

    # ---- resolved directories (portable) ---------------------------
    def _data_root(self) -> Path:
        return self.root / self.get("paths.data_dir", "instance")

    def dir(self, key: str) -> Path:
        """Return a resolved data directory, creating it if needed."""
        mapping = {
            "data": self._data_root(),
            "conversations": self._data_root() / self.get("paths.conversations_dir", "conversations"),
            "memory": self._data_root() / self.get("paths.memory_dir", "memory"),
            "recordings": self._data_root() / self.get("paths.recordings_dir", "recordings"),
            "logs": self._data_root() / self.get("paths.logs_dir", "logs"),
            "exports": self._data_root() / self.get("paths.exports_dir", "exports"),
        }
        p = mapping.get(key, self._data_root() / key)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(root: Path | None = None) -> AppConfig:
    root = root or APP_ROOT
    settings_path = root / "config" / "settings.yaml"
    raw: Dict[str, Any] = {}
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    return AppConfig(root=root, raw=raw)
