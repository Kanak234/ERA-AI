"""ERA AI — model registry.

Loads era/models.json and cross-references each declared worker with the
models actually installed in Ollama. Workers are addressed by `id`, never
by display label, so labels (NEED-x / VED-x / anything) can change freely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Worker:
    id: str
    label: str
    ollama_tag: str
    provider: str
    license: str
    specialties: List[str]
    approx_ram_gb: float
    enabled: bool
    system_prompt_file: str = ""
    installed: bool = False          # filled in at runtime
    system_prompt: str = ""          # loaded from file if present

    @property
    def available(self) -> bool:
        return self.enabled and self.installed


@dataclass
class Registry:
    workers: List[Worker] = field(default_factory=list)
    identity: Dict[str, str] = field(default_factory=dict)

    def by_id(self, wid: str) -> Optional[Worker]:
        return next((w for w in self.workers if w.id == wid), None)

    def available_workers(self) -> List[Worker]:
        return [w for w in self.workers if w.available]

    def match_specialty(self, tag: str) -> Optional[Worker]:
        for w in self.available_workers():
            if tag in w.specialties:
                return w
        return None


def load_registry(root: Path, installed_tags: List[str]) -> Registry:
    path = root / "era" / "models.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    installed = set(installed_tags)
    workers: List[Worker] = []
    for w in data.get("workers", []):
        worker = Worker(
            id=w["id"],
            label=w.get("label", w["id"]),
            ollama_tag=w["ollama_tag"],
            provider=w.get("provider", "unknown"),
            license=w.get("license", "unverified"),
            specialties=list(w.get("specialties", [])),
            approx_ram_gb=float(w.get("approx_ram_gb", 0)),
            enabled=bool(w.get("enabled", True)),
            system_prompt_file=w.get("system_prompt_file", ""),
        )
        worker.installed = worker.ollama_tag in installed

        # Load an optional per-worker system prompt (identity injection)
        if worker.system_prompt_file:
            sp = root / worker.system_prompt_file
            if sp.exists():
                worker.system_prompt = sp.read_text(encoding="utf-8").strip()

        workers.append(worker)

    return Registry(workers=workers, identity=data.get("identity", {}))
