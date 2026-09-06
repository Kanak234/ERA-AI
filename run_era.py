#!/usr/bin/env python3
"""ERA AI — entry point.

    python run_era.py

Boots the config, connects to Ollama, builds the model registry from
what is actually installed, and launches the desktop GUI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import load_config                       # noqa: E402
from app.ollama_client import OllamaClient               # noqa: E402
from app.orchestrator.model_manager import ModelManager  # noqa: E402
from app.registry import load_registry                   # noqa: E402
from app.routing.router import Router                     # noqa: E402


def main() -> int:
    cfg = load_config()

    # ensure instance folders exist (portable layout)
    for key in ("data", "conversations", "memory", "recordings", "logs", "exports"):
        cfg.dir(key)

    client = OllamaClient(cfg.ollama_host, cfg.request_timeout)

    installed = client.list_models() if client.is_up() else []
    if not client.is_up():
        print("\n  \u26a0  Ollama is not reachable at", cfg.ollama_host)
        print("     Start it with:  ollama serve")
        print("     The GUI will still open, but no worker will respond.\n")

    registry = load_registry(cfg.root, installed)
    manager = ModelManager(
        client,
        max_hot_models=cfg.max_hot_models,
        ram_safety_margin_mb=cfg.ram_safety_margin_mb,
        keep_alive=cfg.default_keep_alive,
    )
    router = Router(registry)

    # GUI import is deferred so health_check etc. don't need Qt installed
    from PySide6.QtWidgets import QApplication
    from app.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow(cfg, client, registry, manager, router)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
