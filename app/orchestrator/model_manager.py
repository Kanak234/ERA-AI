"""ERA AI — hardware-aware model manager (spec Section 23).

On 16 GB RAM the four workers cannot be co-resident. This manager keeps
at most `max_hot_models` models loaded, unloading the least-recently-used
before bringing a new one up. It serialises heavy model swaps so two
requests never trigger two simultaneous loads.
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Dict, List, Optional

import psutil

from app.ollama_client import OllamaClient


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


class ModelManager:
    def __init__(self, client: OllamaClient, max_hot_models: int = 1,
                 ram_safety_margin_mb: int = 2048, keep_alive: str = "5m"):
        self.client = client
        self.max_hot_models = max(1, max_hot_models)
        self.ram_safety_margin_mb = ram_safety_margin_mb
        self.keep_alive = keep_alive

        self._states: Dict[str, ModelState] = {}
        self._last_used: Dict[str, float] = {}
        self._lock = threading.RLock()

    # ---- hardware ---------------------------------------------------
    @staticmethod
    def total_ram_gb() -> float:
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)

    @staticmethod
    def available_ram_gb() -> float:
        return round(psutil.virtual_memory().available / (1024 ** 3), 1)

    def has_headroom_for(self, approx_ram_gb: float) -> bool:
        margin = self.ram_safety_margin_mb / 1024
        return self.available_ram_gb() >= (approx_ram_gb + margin)

    # ---- state ------------------------------------------------------
    def state(self, tag: str) -> ModelState:
        with self._lock:
            return self._states.get(tag, ModelState.UNLOADED)

    def states(self) -> Dict[str, ModelState]:
        with self._lock:
            # refresh from Ollama's view of what is loaded
            loaded = set(self.client.loaded_models())
            for tag in list(self._states.keys()):
                if tag not in loaded and self._states[tag] in (
                    ModelState.READY, ModelState.BUSY
                ):
                    self._states[tag] = ModelState.UNLOADED
            return dict(self._states)

    def _set(self, tag: str, st: ModelState) -> None:
        with self._lock:
            self._states[tag] = st

    # ---- lifecycle --------------------------------------------------
    def _lru_hot(self) -> List[str]:
        loaded = set(self.client.loaded_models())
        hot = [t for t in loaded]
        hot.sort(key=lambda t: self._last_used.get(t, 0))  # oldest first
        return hot

    def ensure_capacity(self, incoming_tag: str) -> None:
        """Unload LRU models until there is room for one more hot model."""
        with self._lock:
            hot = self._lru_hot()
            hot = [t for t in hot if t != incoming_tag]
            while len(hot) >= self.max_hot_models:
                victim = hot.pop(0)
                self._set(victim, ModelState.UNLOADED)
                self.client.unload(victim)
                time.sleep(0.2)

    def acquire(self, tag: str) -> None:
        """Mark a model as about to be used; free capacity if needed."""
        self.ensure_capacity(tag)
        self._set(tag, ModelState.LOADING)
        self._last_used[tag] = time.time()

    def mark_ready(self, tag: str) -> None:
        self._set(tag, ModelState.READY)
        self._last_used[tag] = time.time()

    def mark_busy(self, tag: str) -> None:
        self._set(tag, ModelState.BUSY)

    def mark_error(self, tag: str) -> None:
        self._set(tag, ModelState.ERROR)

    def release(self, tag: str) -> None:
        self._set(tag, ModelState.READY)
        self._last_used[tag] = time.time()
