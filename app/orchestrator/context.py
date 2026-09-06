"""ERA AI — conversation context management (spec Section 32).

Keeps prompts within a model's context window. When history is long,
older turns are folded into a rolling summary instead of blind
truncation. Token counting is a fast heuristic (~4 chars/token) so it
needs no tokenizer dependency.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def messages_tokens(messages: List[Dict[str, str]]) -> int:
    return sum(approx_tokens(m.get("content", "")) + 4 for m in messages)


class ContextManager:
    def __init__(self, max_context_tokens: int = 8192, reserve_for_reply: int = 1024):
        self.max_context_tokens = max_context_tokens
        self.reserve_for_reply = reserve_for_reply

    @property
    def budget(self) -> int:
        return self.max_context_tokens - self.reserve_for_reply

    def fit(
        self,
        system: str,
        history: List[Dict[str, str]],
        summarize: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> List[Dict[str, str]]:
        """Return a message list that fits the budget.

        Strategy: always keep the system message and the most recent
        turns. If older turns overflow, summarise them (if a summariser
        is provided) into a single system note; otherwise drop them.
        """
        sys_msg = {"role": "system", "content": system}
        budget = self.budget - messages_tokens([sys_msg])

        kept: List[Dict[str, str]] = []
        overflow: List[Dict[str, str]] = []
        running = 0
        # walk from newest to oldest, keep what fits
        for msg in reversed(history):
            t = approx_tokens(msg.get("content", "")) + 4
            if running + t <= budget:
                kept.append(msg)
                running += t
            else:
                overflow.append(msg)

        kept.reverse()
        overflow.reverse()

        result = [sys_msg]
        if overflow:
            if summarize:
                summary = summarize(overflow)
                result.append({
                    "role": "system",
                    "content": f"[Earlier conversation summary]\n{summary}",
                })
            # else: silently dropped
        result.extend(kept)
        return result
