"""ERA AI — Ollama client.

Thin wrapper over the local Ollama HTTP API. Supports listing installed
models, inspecting metadata, checking what is loaded, streaming chat,
and forcing a model to unload (keep_alive=0).

Everything is local by design (http://localhost:11434).
"""
from __future__ import annotations

import json
from typing import Callable, Dict, Iterator, List, Optional

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 300):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ---- health ----------------------------------------------------
    def is_up(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/version", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def version(self) -> Optional[str]:
        try:
            r = requests.get(f"{self.host}/api/version", timeout=5)
            r.raise_for_status()
            return r.json().get("version")
        except requests.RequestException:
            return None

    # ---- model discovery -------------------------------------------
    def list_models(self) -> List[str]:
        """Return installed model tags, e.g. ['phi4-mini:latest', ...]."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            data = r.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except requests.RequestException as e:
            raise OllamaError(f"Could not list models: {e}") from e

    def show(self, model: str) -> Dict:
        """Return metadata for a model (used to verify provider/license)."""
        try:
            r = requests.post(
                f"{self.host}/api/show", json={"name": model}, timeout=15
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise OllamaError(f"Could not show {model}: {e}") from e

    def loaded_models(self) -> List[str]:
        """Return tags of models currently resident in memory."""
        try:
            r = requests.get(f"{self.host}/api/ps", timeout=10)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
        except requests.RequestException:
            return []

    # ---- lifecycle -------------------------------------------------
    def unload(self, model: str) -> None:
        """Force a model out of memory by sending keep_alive=0."""
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
        except requests.RequestException:
            pass  # best-effort

    # ---- chat ------------------------------------------------------
    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        keep_alive: str = "5m",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Iterator[str]:
        """Yield content chunks from Ollama /api/chat as they arrive.

        stop_flag: optional callable returning True to cancel mid-stream.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if stop_flag and stop_flag():
                        break
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        raise OllamaError(obj["error"])
                    chunk = obj.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if obj.get("done"):
                        break
        except requests.RequestException as e:
            raise OllamaError(f"Chat request failed: {e}") from e
