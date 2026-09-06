"""ERA AI — i18n (spec Section 33). Simple dict-based, en + hi."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class I18N:
    def __init__(self, locales_dir: Path, lang: str = "en"):
        self.locales_dir = locales_dir
        self.lang = lang
        self._data: Dict[str, str] = {}
        self._fallback: Dict[str, str] = {}
        self.load(lang)

    def load(self, lang: str):
        self.lang = lang
        self._fallback = self._read("en")
        self._data = self._read(lang) if lang != "en" else self._fallback

    def _read(self, lang: str) -> Dict[str, str]:
        p = self.locales_dir / f"{lang}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def t(self, key: str) -> str:
        return self._data.get(key) or self._fallback.get(key) or key
