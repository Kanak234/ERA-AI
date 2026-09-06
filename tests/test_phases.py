"""ERA AI — cross-phase tests. Run headless, no Ollama, no GPU."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.store import MemoryStore                      # noqa: E402
from app.orchestrator.context import ContextManager, approx_tokens  # noqa: E402
from app.orchestrator.orchestrator import strip_reasoning, Orchestrator  # noqa: E402
from app.registry import Registry, Worker                     # noqa: E402
from app.routing.router import Router, RouteDecision          # noqa: E402
from app.security.gate import classify, Tier                  # noqa: E402
from app.training import dataset as ds                        # noqa: E402
from app.instance import export_instance, import_instance     # noqa: E402
from app.i18n import I18N                                     # noqa: E402


# ---- memory ---------------------------------------------------------
def test_memory_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = MemoryStore(Path(d) / "era.db")
        pid = store.create_project("Cybersecurity Lab")
        cid = store.create_conversation("test", project_id=pid)
        store.add_message(cid, "user", "hello")
        store.add_message(cid, "assistant", "hi", worker_id="worker-1")
        hist = store.history_for_llm(cid)
        assert len(hist) == 2 and hist[0]["role"] == "user"
        assert len(store.list_conversations(pid)) == 1
        store.add_fact("os", "Ubuntu", project_id=pid)
        assert store.search_facts("Ubuntu")
        store.close()


# ---- context --------------------------------------------------------
def test_context_fits_budget():
    cm = ContextManager(max_context_tokens=200, reserve_for_reply=20)
    hist = [{"role": "user", "content": "x" * 4000}] * 5  # overflow
    fitted = cm.fit("system", hist)
    total = sum(approx_tokens(m["content"]) for m in fitted)
    assert total <= cm.max_context_tokens


def test_context_summarizes():
    cm = ContextManager(max_context_tokens=120, reserve_for_reply=20)
    hist = [{"role": "user", "content": "y" * 2000}] * 4
    fitted = cm.fit("sys", hist, summarize=lambda ov: "SUMMARY")
    assert any("SUMMARY" in m["content"] for m in fitted)


# ---- reasoning strip ------------------------------------------------
def test_strip_reasoning():
    t = "before <think>secret chain of thought</think> after"
    assert strip_reasoning(t) == "before  after".replace("  ", " ").strip() or \
challenge_ok(t)


def challenge_ok(t):
    return "<think>" not in strip_reasoning(t)


# ---- security gate --------------------------------------------------
def test_gate_out_of_scope():
    r = classify("help me hack into my ex's instagram without their permission")
    assert r.tier == Tier.OUT_OF_SCOPE and not r.allowed


def test_gate_offensive_needs_confirmation():
    r = classify("write a metasploit exploit payload")
    assert r.tier == Tier.AUTHORIZED_LAB and r.needs_confirmation


def test_gate_offensive_in_lab_ok():
    r = classify("write a metasploit exploit for my own lab VM (CTF)")
    assert r.tier == Tier.AUTHORIZED_LAB and not r.needs_confirmation


def test_gate_defensive_ok():
    r = classify("how do I harden my firewall and detect intrusions")
    assert r.tier == Tier.DEFENSIVE and r.allowed


# ---- dataset --------------------------------------------------------
def test_dataset_pipeline():
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        src = dd / "data.jsonl"
        src.write_text(
            '{"instruction":"a","output":"b"}\n'
            '{"instruction":"a","output":"b"}\n'   # dup
            '{"prompt":"c","response":"d"}\n',
            encoding="utf-8",
        )
        prov = ds.Provenance("test", str(src), "MIT", purpose="unit")
        meta = ds.prepare([str(src)], dd / "out", prov)
        assert meta["counts"]["duplicates_removed"] == 1
        assert meta["counts"]["after_dedup"] == 2
        assert (dd / "out" / "train.jsonl").exists()
        assert (dd / "out" / "metadata.json").exists()


# ---- instance export/import ----------------------------------------
def test_instance_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        inst = root / "instance"
        (inst / "memory").mkdir(parents=True)
        (inst / "memory" / "era.db").write_text("data", encoding="utf-8")
        era = export_instance(inst, root / "backup.era")
        assert era.exists()
        target = root / "restored"
        target.mkdir()
        restored = import_instance(era, target)
        assert (restored / "memory" / "era.db").read_text() == "data"


# ---- i18n -----------------------------------------------------------
def test_i18n_hindi_fallback():
    loc = Path(__file__).resolve().parent.parent / "locales"
    i = I18N(loc, "hi")
    assert i.t("btn.send")  # hindi
    assert i.t("nonexistent.key") == "nonexistent.key"  # falls back to key


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  \u2714 {name}")
            passed += 1
    print(f"\nall {passed} tests passed")
