"""ERA AI — dataset preparation (spec Section 13 + 14 + 26).

Turns dropped files into a clean instruction dataset:
ingest -> normalise -> dedup -> train/val/test split -> provenance
metadata. Never trains silently; this only PREPARES data. The user
reviews before training starts.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Pair:
    instruction: str
    input: str
    output: str

    def key(self) -> str:
        raw = (self.instruction + "\x00" + self.input + "\x00" + self.output)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class Provenance:
    dataset_name: str
    source: str
    license: str
    author: str = "Kanak Prabhakar"
    purpose: str = ""
    collection_date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    modification_date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))


def _load_jsonl(path: Path) -> List[Pair]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "messages" in obj:  # chat format
            msgs = obj["messages"]
            u = next((m["content"] for m in msgs if m["role"] == "user"), "")
            a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if u and a:
                pairs.append(Pair(instruction=u, input="", output=a))
        elif "instruction" in obj and "output" in obj:
            pairs.append(Pair(obj["instruction"], obj.get("input", ""), obj["output"]))
        elif "prompt" in obj and "response" in obj:
            pairs.append(Pair(obj["prompt"], "", obj["response"]))
    return pairs


def _load_plaintext(path: Path) -> List[Pair]:
    """Very light heuristic: treat 'Q: ... A: ...' blocks, else skip.
    Plain prose can't become instruction pairs automatically — the UI
    prompts the user to add them. We only auto-parse explicit Q/A."""
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs = []
    blocks = text.split("\n\n")
    for b in blocks:
        if "Q:" in b and "A:" in b:
            q = b.split("Q:", 1)[1].split("A:", 1)[0].strip()
            a = b.split("A:", 1)[1].strip()
            if q and a:
                pairs.append(Pair(q, "", a))
    return pairs


def ingest(paths: List[str]) -> List[Pair]:
    pairs: List[Pair] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        if path.suffix.lower() in (".jsonl", ".json"):
            pairs.extend(_load_jsonl(path))
        else:
            pairs.extend(_load_plaintext(path))
    return pairs


def dedup(pairs: List[Pair]) -> List[Pair]:
    seen = set()
    out = []
    for p in pairs:
        k = p.key()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def split(pairs: List[Pair], val: float = 0.1, test: float = 0.1):
    n = len(pairs)
    n_test = int(n * test)
    n_val = int(n * val)
    return {
        "train": pairs[: n - n_val - n_test],
        "val": pairs[n - n_val - n_test: n - n_test],
        "test": pairs[n - n_test:],
    }


def prepare(paths: List[str], out_dir: Path, prov: Provenance) -> Dict:
    """Full pipeline. Writes train/val/test jsonl + metadata. Returns stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = ingest(paths)
    clean = dedup(raw)
    parts = split(clean)

    for name, rows in parts.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    meta = {
        "provenance": asdict(prov),
        "counts": {
            "ingested": len(raw),
            "after_dedup": len(clean),
            "duplicates_removed": len(raw) - len(clean),
            "train": len(parts["train"]),
            "val": len(parts["val"]),
            "test": len(parts["test"]),
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta
