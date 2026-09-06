"""ERA AI — observability (spec Section 18 + 35).

Structured JSONL request logs and in-memory local metrics. No telemetry
leaves the machine. Conversation content is NOT logged unless explicitly
enabled in settings.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class RequestTrace:
    request_id: str
    task_type: str = ""
    workers: List[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""

    def finish(self, success: bool = True, error: str = ""):
        self.latency_ms = round((time.time() - self.started) * 1000, 1)
        self.success = success
        self.error = error
        return self


class Observability:
    def __init__(self, logs_dir: Path, log_content: bool = False):
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_content = log_content
        self.log_file = logs_dir / "requests.jsonl"

        # in-memory metrics
        self.total = 0
        self.errors = 0
        self.latencies: List[float] = []
        self.per_worker: Dict[str, int] = defaultdict(int)
        self.load_events = 0

    def record(self, trace: RequestTrace):
        self.total += 1
        if not trace.success:
            self.errors += 1
        self.latencies.append(trace.latency_ms)
        for w in trace.workers:
            self.per_worker[w] += 1

        entry = {
            "ts": round(trace.started, 3),
            "request_id": trace.request_id,
            "task_type": trace.task_type,
            "workers": trace.workers,
            "latency_ms": trace.latency_ms,
            "success": trace.success,
            "error": trace.error,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def note_load(self):
        self.load_events += 1

    def summary(self) -> Dict:
        n = len(self.latencies)
        avg = round(sum(self.latencies) / n, 1) if n else 0.0
        return {
            "requests": self.total,
            "errors": self.errors,
            "error_rate": round(self.errors / self.total, 3) if self.total else 0.0,
            "avg_latency_ms": avg,
            "per_worker": dict(self.per_worker),
            "model_load_events": self.load_events,
        }
