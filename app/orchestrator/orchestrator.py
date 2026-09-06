"""ERA AI — orchestrator (spec Section 7).

Runs the ERA loop: route -> consult one or more workers -> optionally
synthesise a final answer. On low RAM everything runs SEQUENTIALLY
(one model hot at a time via the ModelManager). Private reasoning
(<think>...</think>) is stripped before display and before synthesis.

Status is reported through a callback so the GUI can show concise
lines without exposing chain-of-thought.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from app.ollama_client import OllamaClient, OllamaError
from app.orchestrator.model_manager import ModelManager
from app.registry import Registry, Worker
from app.routing.router import RouteDecision, Router


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    return THINK_RE.sub("", text).strip()


StatusFn = Callable[[str], None]


@dataclass
class WorkerReply:
    worker: Worker
    text: str
    error: str = ""


@dataclass
class EraResult:
    decision: RouteDecision
    replies: List[WorkerReply] = field(default_factory=list)
    final: str = ""
    synthesized: bool = False


class Orchestrator:
    def __init__(self, client: OllamaClient, manager: ModelManager,
                 registry: Registry, router: Router, cfg):
        self.client = client
        self.manager = manager
        self.registry = registry
        self.router = router
        self.cfg = cfg

    # ---- low-level single call (non-streaming, collected) ----------
    def _run_worker(self, worker: Worker, messages: List[Dict[str, str]],
                   status: StatusFn) -> WorkerReply:
        tag = worker.ollama_tag
        try:
            status(f"Consulting {worker.label} \u2026")
            self.manager.acquire(tag)
            self.manager.mark_busy(tag)
            buf = []
            for piece in self.client.chat_stream(
                model=tag, messages=messages,
                keep_alive=self.cfg.default_keep_alive,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            ):
                buf.append(piece)
            self.manager.release(tag)
            return WorkerReply(worker, strip_reasoning("".join(buf)))
        except OllamaError as e:
            self.manager.mark_error(tag)
            return WorkerReply(worker, "", str(e))

    # ---- planning --------------------------------------------------
    def plan(self, text: str, forced_worker_id: Optional[str] = None) -> RouteDecision:
        if forced_worker_id:
            w = self.registry.by_id(forced_worker_id)
            return RouteDecision(
                task_type="manual", complexity=0,
                worker_ids=[w.id] if w else [],
                reason=f"Using {w.label}." if w else "Worker not found.",
            )
        return self.router.classify(text)

    def _debate_workers(self, decision: RouteDecision) -> List[Worker]:
        """For hard tasks, add a second perspective (sequentially)."""
        chosen = [self.registry.by_id(wid) for wid in decision.worker_ids]
        chosen = [w for w in chosen if w]
        if decision.verification_required:
            # add a reasoning worker if not already present
            reasoner = self.registry.match_specialty("reasoning")
            if reasoner and reasoner not in chosen:
                chosen.append(reasoner)
        return chosen[:2]  # cap at 2 on modest hardware

    # ---- main entry ------------------------------------------------
    def run(self, system: str, history: List[Dict[str, str]],
           decision: RouteDecision, status: StatusFn) -> EraResult:
        result = EraResult(decision=decision)
        workers = self._debate_workers(decision)
        if not workers:
            result.final = ("No model is available. Start Ollama or pull a "
                            "model, then try again.")
            return result

        messages = [{"role": "system", "content": system}] + history

        # sequential consult (RAM-safe)
        for w in workers:
            reply = self._run_worker(w, messages, status)
            result.replies.append(reply)

        good = [r for r in result.replies if r.text and not r.error]
        if not good:
            errs = "; ".join(r.error for r in result.replies if r.error)
            result.final = f"\u26a0 All workers failed: {errs or 'no output'}"
            return result

        # single worker -> its answer is final
        if len(good) == 1:
            result.final = good[0].text
            return result

        # multiple -> synthesise
        status("Comparing responses \u2026")
        result.final = self._synthesize(history, good, status)
        result.synthesized = True
        return result

    def _synthesize(self, history: List[Dict[str, str]],
                   replies: List[WorkerReply], status: StatusFn) -> str:
        status("Generating final answer \u2026")
        # pick a synthesiser: prefer general/writing worker
        synth = (self.registry.match_specialty("general")
                 or self.registry.match_specialty("reasoning")
                 or replies[0].worker)

        user_q = next((m["content"] for m in reversed(history)
                       if m["role"] == "user"), "")
        blocks = "\n\n".join(
            f"[{r.worker.label}]\n{r.text}" for r in replies
        )
        synth_prompt = [
            {"role": "system", "content":
                "You are ERA AI's synthesiser. Combine the expert answers "
                "below into one clear, correct final answer. Resolve "
                "conflicts, drop redundancy, keep it concise. Do not mention "
                "the experts by name."},
            {"role": "user", "content":
                f"Question:\n{user_q}\n\nExpert answers:\n{blocks}\n\n"
                "Final answer:"},
        ]
        reply = self._run_worker(synth, synth_prompt, status)
        return reply.text or replies[0].text
