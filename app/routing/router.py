"""ERA AI — Tier-1 router (spec Section 6 + patch A2).

Fast, zero-model routing. Classifies a request from keywords/heuristics
and maps it to a worker via the registry's specialties. Never depends on
an LLM being loaded. Tier-2 (LLM judge) can be layered on later for
ambiguous cases; if that model is missing, we silently stay on Tier-1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.registry import Registry, Worker


# specialty -> signal keywords
SIGNALS = {
    "code": [
        r"\bcode\b", r"\bpython\b", r"\bjava(script)?\b", r"\bc\+\+\b",
        r"\bbug\b", r"\bdebug\b", r"\berror\b", r"\bfunction\b",
        r"\bcompile\b", r"\bscript\b", r"\bapi\b", r"\bregex\b",
        r"\brefactor\b", r"```",
    ],
    "reasoning": [
        r"\bwhy\b", r"\bprove\b", r"\bstep by step\b", r"\banaly[sz]e\b",
        r"\bplan\b", r"\bstrateg", r"\bcompare\b", r"\btrade-?off\b",
        r"\bderive\b", r"\bmath\b", r"\bcalcul", r"\bsolve\b",
    ],
    "summarize": [
        r"\bsummar", r"\btl;?dr\b", r"\bshort(en)?\b", r"\bclassif",
        r"\bin one line\b", r"\bbrief\b",
    ],
    "writing": [
        r"\bwrite\b", r"\bdraft\b", r"\bemail\b", r"\bessay\b",
        r"\brewrite\b", r"\bstory\b", r"\bpoem\b", r"\btranslate\b",
    ],
}

# rough complexity signals
COMPLEX_SIGNALS = [
    r"\bstep by step\b", r"\barchitecture\b", r"\bdesign\b",
    r"\bmulti[- ]?step\b", r"\boptimi[sz]e\b", r"\bprove\b",
]


@dataclass
class RouteDecision:
    task_type: str
    complexity: int
    worker_ids: List[str] = field(default_factory=list)
    parallel: bool = False
    verification_required: bool = False
    reason: str = ""


class Router:
    def __init__(self, registry: Registry, default_specialty: str = "general"):
        self.registry = registry
        self.default_specialty = default_specialty

    def _score(self, text: str, patterns: List[str]) -> int:
        return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

    def classify(self, text: str) -> RouteDecision:
        scores = {spec: self._score(text, pats) for spec, pats in SIGNALS.items()}
        best_spec = max(scores, key=scores.get)
        best_score = scores[best_spec]

        # nothing matched -> general conversation
        if best_score == 0:
            best_spec = "general"

        complexity = min(10, self._score(text, COMPLEX_SIGNALS) * 3 + len(text) // 400)

        worker = self._pick_worker(best_spec)
        worker_ids = [worker.id] if worker else []

        # heavy reasoning/coding -> flag verification (used later in debate)
        verification = complexity >= 6 and best_spec in ("reasoning", "code")

        return RouteDecision(
            task_type=best_spec,
            complexity=complexity,
            worker_ids=worker_ids,
            parallel=False,
            verification_required=verification,
            reason=self._explain(best_spec, worker),
        )

    def _pick_worker(self, specialty: str) -> Optional[Worker]:
        # exact specialty match first
        w = self.registry.match_specialty(specialty)
        if w:
            return w
        # fall back through a sensible chain
        for fallback in ("general", "chat", "reasoning", "fast"):
            w = self.registry.match_specialty(fallback)
            if w:
                return w
        # last resort: any available worker
        avail = self.registry.available_workers()
        return avail[0] if avail else None

    def _explain(self, specialty: str, worker: Optional[Worker]) -> str:
        if not worker:
            return "No worker available."
        return f"Detected '{specialty}' task \u2192 routed to {worker.label}."
