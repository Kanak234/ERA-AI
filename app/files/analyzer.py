"""ERA AI — file analysis (spec Section 11).

Reads a file, extracts text safely, and builds an "analyse this" prompt
that the orchestrator routes to the right worker. Heavy parsers (pdf,
docx) are optional imports so the core stays light.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CODE_EXT = {".py", ".c", ".cpp", ".h", ".hpp", ".java", ".js", ".ts",
            ".html", ".css", ".yaml", ".yml", ".sh", ".rs", ".go"}
TEXT_EXT = {".txt", ".md", ".log", ".ini", ".cfg", ".toml"}
MAX_CHARS = 20000


@dataclass
class FileAnalysis:
    path: str
    kind: str
    chars: int
    text: str
    error: str = ""

    def to_prompt(self, instruction: str = "Analyse this file.") -> str:
        return (f"{instruction}\n\nFile: {Path(self.path).name} "
                f"(type: {self.kind})\n\n```\n{self.text}\n```")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[pdf support not installed: pip install pypdf]"
    reader = PdfReader(str(path))
    return "\n".join((pg.extract_text() or "") for pg in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        return "[docx support not installed: pip install python-docx]"
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)


def _read_csv(path: Path) -> str:
    lines = _read_text(path).splitlines()
    head = lines[:50]
    more = f"\n... ({len(lines)-50} more rows)" if len(lines) > 50 else ""
    return "\n".join(head) + more


def _read_json(path: Path) -> str:
    try:
        obj = json.loads(_read_text(path))
        return json.dumps(obj, indent=2)[:MAX_CHARS]
    except json.JSONDecodeError as e:
        return f"[invalid JSON: {e}]"


def analyze_file(path: str) -> FileAnalysis:
    p = Path(path)
    if not p.exists():
        return FileAnalysis(path, "missing", 0, "", "file not found")
    ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            text, kind = _read_pdf(p), "pdf"
        elif ext == ".docx":
            text, kind = _read_docx(p), "docx"
        elif ext == ".csv" or ext == ".tsv":
            text, kind = _read_csv(p), "tabular"
        elif ext == ".json":
            text, kind = _read_json(p), "json"
        elif ext in CODE_EXT:
            text, kind = _read_text(p), f"code{ext}"
        elif ext in TEXT_EXT or ext == "":
            text, kind = _read_text(p), "text"
        else:
            text, kind = _read_text(p), "text"
    except Exception as e:  # noqa: BLE001
        return FileAnalysis(path, ext, 0, "", str(e))

    text = text[:MAX_CHARS]
    return FileAnalysis(path, kind, len(text), text)


def route_hint(analysis: FileAnalysis) -> str:
    """Suggest a specialty for the router based on file kind."""
    if analysis.kind.startswith("code"):
        return "code"
    if analysis.kind in ("tabular", "json"):
        return "reasoning"
    return "summarize"
