"""Router + registry tests that run without Ollama."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.registry import Registry, Worker       # noqa: E402
from app.routing.router import Router            # noqa: E402


def _registry() -> Registry:
    workers = [
        Worker("worker-1", "Phi", "phi4-mini:latest", "MS", "x",
               ["fast", "summarize", "classify", "chat"], 3, True, installed=True),
        Worker("worker-2", "DeepSeek", "deepseek-r1:7b", "DS", "x",
               ["reasoning", "math", "planning"], 6, True, installed=True),
        Worker("worker-3", "Qwen", "qwen3-coder:latest", "Qwen", "x",
               ["code", "debug", "architecture"], 8, True, installed=True),
        Worker("worker-4", "Llama", "llama3.1:8b", "Meta", "x",
               ["general", "writing", "chat"], 6, True, installed=True),
    ]
    return Registry(workers=workers, identity={})


def test_code_routes_to_coder():
    r = Router(_registry())
    d = r.classify("Fix this python function, it throws an error")
    assert d.task_type == "code"
    assert d.worker_ids == ["worker-3"]


def test_reasoning_routes_to_deepseek():
    r = Router(_registry())
    d = r.classify("Analyze step by step why this plan fails and prove it")
    assert d.task_type == "reasoning"
    assert d.worker_ids == ["worker-2"]


def test_general_falls_back():
    r = Router(_registry())
    d = r.classify("hey how are you today")
    assert d.worker_ids  # some worker chosen
    assert d.task_type in ("general", "summarize", "writing", "reasoning", "code")


def test_unavailable_worker_excluded():
    reg = _registry()
    reg.workers[2].installed = False  # coder offline
    r = Router(reg)
    d = r.classify("write me some code")
    assert d.worker_ids and d.worker_ids[0] != "worker-3"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  \u2714 {name}")
    print("all router tests passed")
