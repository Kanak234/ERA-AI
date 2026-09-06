"""ERA AI — main window v2 (Phases 2-4 integrated).

Adds over Phase 1:
  - conversation persistence (SQLite memory store)
  - conversation list in the sidebar (load / new / delete)
  - ERA orchestrator with multi-worker debate + synthesis, run in a
    background thread, with a live activity panel
  - request tracing into local metrics
  - security gate pre-check with an inline authorization prompt

Streaming single-worker replies remain instant; multi-worker debate
runs sequentially (RAM-safe) and streams only the final answer.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QTextEdit, QVBoxLayout, QWidget,
)

from app.config import AppConfig
from app.gui.style import DARK
from app.memory.store import MemoryStore
from app.observability.metrics import Observability, RequestTrace, new_request_id
from app.ollama_client import OllamaClient
from app.orchestrator.model_manager import ModelManager
from app.orchestrator.orchestrator import Orchestrator
from app.registry import Registry, Worker
from app.routing.router import Router
from app.security.gate import classify as gate_classify, Tier


class EraThread(QThread):
    """Runs the orchestrator off the UI thread."""
    activity = Signal(str)
    done = Signal(str, list, bool)   # final, worker_labels, synthesized
    failed = Signal(str)

    def __init__(self, orch: Orchestrator, system: str,
                 history: List[Dict[str, str]], decision):
        super().__init__()
        self.orch = orch
        self.system = system
        self.history = history
        self.decision = decision

    def run(self):
        try:
            res = self.orch.run(self.system, self.history, self.decision,
                                self.activity.emit)
            labels = [r.worker.label for r in res.replies]
            self.done.emit(res.final, labels, res.synthesized)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class Bubble(QFrame):
    def __init__(self, role: str, tag_text: str, text: str = ""):
        super().__init__()
        self.setObjectName("BubbleUser" if role == "user" else "BubbleEra")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        tag = QLabel(tag_text)
        tag.setObjectName("RoleTagUser" if role == "user" else "RoleTag")
        self.body = QLabel(text)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(tag)
        lay.addWidget(self.body)

    def set_text(self, t: str):
        self.body.setText(t)


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, client: OllamaClient,
                 registry: Registry, manager: ModelManager, router: Router):
        super().__init__()
        self.cfg = cfg
        self.client = client
        self.registry = registry
        self.manager = manager
        self.router = router

        self.store = MemoryStore(cfg.dir("memory") / "era.db")
        self.metrics = Observability(cfg.dir("logs"),
                                     log_content=cfg.get("privacy.log_conversation_content", False))
        self.orch = Orchestrator(client, manager, registry, router, cfg)

        self.conv_id: Optional[str] = None
        self.thread: Optional[EraThread] = None
        self.pending_gate = None

        self.setWindowTitle("ERA AI")
        self.resize(1120, 740)
        self.setStyleSheet(DARK)
        self._build_ui()
        self._new_conversation()
        self._reload_conv_list()

    # ---- UI --------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._sidebar())
        root.addWidget(self._main_panel(), 1)
        self.setCentralWidget(central)

    def _sidebar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(250)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(16, 18, 16, 18)
        lay.setSpacing(6)

        name = QLabel("ERA AI")
        name.setObjectName("BrandName")
        sub = QLabel(self.cfg.get("app.subtitle", ""))
        sub.setObjectName("BrandSub")
        lay.addWidget(name)
        lay.addWidget(sub)
        lay.addSpacing(12)

        new_btn = QPushButton("+ New Conversation")
        new_btn.setObjectName("NavBtn")
        new_btn.clicked.connect(self._new_conversation)
        lay.addWidget(new_btn)

        self.conv_list = QListWidget()
        self.conv_list.setObjectName("ChatCanvas")
        self.conv_list.itemClicked.connect(self._open_conv)
        lay.addWidget(self.conv_list, 1)

        del_btn = QPushButton("Delete selected")
        del_btn.setObjectName("NavBtn")
        del_btn.clicked.connect(self._delete_conv)
        lay.addWidget(del_btn)

        owner = QLabel(f"by {self.cfg.get('app.owner','')}")
        owner.setObjectName("StatusLine")
        lay.addWidget(owner)
        return bar

    def _main_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        header = QWidget()
        header.setObjectName("Header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 12, 20, 12)
        title = QLabel("Personal Local Intelligence")
        title.setObjectName("HeaderTitle")
        hl.addWidget(title)
        hl.addStretch(1)
        self.model_box = QComboBox()
        self.model_box.addItem("Auto (ERA Router)", userData=None)
        for w in self.registry.workers:
            label = w.label if w.available else f"{w.label}  (unavailable)"
            self.model_box.addItem(label, userData=w.id)
            if not w.available:
                self.model_box.model().item(self.model_box.count() - 1).setEnabled(False)
        hl.addWidget(self.model_box)
        lay.addWidget(header)

        self.activity = QLabel("")
        self.activity.setObjectName("Activity")
        self.activity.setContentsMargins(20, 6, 20, 0)
        lay.addWidget(self.activity)

        self.status = QLabel(self._ready_status())
        self.status.setObjectName("StatusLine")
        self.status.setContentsMargins(20, 2, 20, 4)
        lay.addWidget(self.status)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("ChatCanvas")
        self.scroll.setWidgetResizable(True)
        self.chat_holder = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_holder)
        self.chat_layout.setContentsMargins(20, 16, 20, 16)
        self.chat_layout.setSpacing(12)
        self.chat_layout.addStretch(1)
        self.scroll.setWidget(self.chat_holder)
        lay.addWidget(self.scroll, 1)

        comp = QWidget()
        cl = QHBoxLayout(comp)
        cl.setContentsMargins(20, 8, 20, 16)
        self.composer = QTextEdit()
        self.composer.setObjectName("Composer")
        self.composer.setPlaceholderText("Ask ERA anything\u2026  (Ctrl+Enter to send)")
        self.composer.setFixedHeight(70)
        cl.addWidget(self.composer, 1)
        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("Send")
        self.send_btn.clicked.connect(self._on_send)
        cl.addWidget(self.send_btn)
        lay.addWidget(comp)
        return panel

    # ---- conversations ---------------------------------------------
    def _ready_status(self) -> str:
        n = len(self.registry.available_workers())
        ram = ModelManager.available_ram_gb()
        m = self.metrics.summary()
        return (f"{n} worker(s) \u00b7 {ram} GB free \u00b7 "
                f"{m['requests']} req \u00b7 avg {m['avg_latency_ms']} ms")

    def _clear_chat(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_bubble(self, role: str, tag: str, text: str = "") -> Bubble:
        b = Bubble(role, tag, text)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, b)
        QApplication.processEvents()
        self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum())
        return b

    def _new_conversation(self):
        self.conv_id = self.store.create_conversation("New Conversation")
        self._clear_chat()
        self.status.setText(self._ready_status())
        self._reload_conv_list()

    def _reload_conv_list(self):
        self.conv_list.clear()
        for c in self.store.list_conversations():
            it = QListWidgetItem(c["title"])
            it.setData(Qt.UserRole, c["id"])
            self.conv_list.addItem(it)

    def _open_conv(self, item: QListWidgetItem):
        self.conv_id = item.data(Qt.UserRole)
        self._clear_chat()
        for m in self.store.get_messages(self.conv_id):
            if m.role == "user":
                self._add_bubble("user", "YOU", m.content)
            elif m.role == "assistant":
                self._add_bubble("era", "ERA", m.content)

    def _delete_conv(self):
        item = self.conv_list.currentItem()
        if not item:
            return
        self.store.delete_conversation(item.data(Qt.UserRole))
        self._reload_conv_list()
        self._new_conversation()

    # ---- send ------------------------------------------------------
    def _build_system(self, worker: Optional[Worker]) -> str:
        identity = self.registry.identity.get("who_created_answer", "")
        base = ("You are ERA AI (Emergent Reasoning Architecture), a local AI "
                "system engineered by Kanak Prabhakar. " + identity)
        if worker and worker.system_prompt:
            base += "\n\n" + worker.system_prompt
        return base

    def _on_send(self):
        text = self.composer.toPlainText().strip()
        if not text or self.thread:
            return

        # security gate pre-check
        gate = gate_classify(text)
        if gate.tier == Tier.OUT_OF_SCOPE:
            self.composer.clear()
            self._add_bubble("user", "YOU", text)
            self._add_bubble("era", "ERA \u00b7 safety", gate.message)
            return
        if gate.needs_confirmation and self.pending_gate != text:
            ok = QMessageBox.question(
                self, "Authorized-lab confirmation", gate.message + "\n\nProceed?")
            if ok != QMessageBox.Yes:
                return
            self.pending_gate = text

        self.composer.clear()
        self._add_bubble("user", "YOU", text)
        self.store.add_message(self.conv_id, "user", text)

        # title from first message
        convs = {c["id"]: c for c in self.store.list_conversations()}
        if convs.get(self.conv_id, {}).get("title") == "New Conversation":
            self.store.rename_conversation(self.conv_id, text[:40])
            self._reload_conv_list()

        forced = self.model_box.currentData()
        decision = self.orch.plan(text, forced_worker_id=forced)
        if not decision.worker_ids:
            self._add_bubble("era", "ERA", "No model available. Start Ollama "
                             "or pull a model.")
            return

        worker = self.registry.by_id(decision.worker_ids[0])
        system = self._build_system(worker)
        history = self.store.history_for_llm(self.conv_id)

        self.trace = RequestTrace(new_request_id(), task_type=decision.task_type)
        self.status.setText(decision.reason)
        self.send_btn.setEnabled(False)

        self.thread = EraThread(self.orch, system, history, decision)
        self.thread.activity.connect(self.activity.setText)
        self.thread.done.connect(self._on_done)
        self.thread.failed.connect(self._on_failed)
        self.thread.start()

    def _on_done(self, final: str, labels: List[str], synthesized: bool):
        tag = "ERA"
        if labels:
            tag = "ERA \u00b7 " + (" + ".join(labels) if synthesized else labels[0])
        self._add_bubble("era", tag, final)
        self.store.add_message(self.conv_id, "assistant", final,
                               worker_id=",".join(labels))
        self.trace.workers = labels
        self.metrics.record(self.trace.finish(success=True))
        self._end()

    def _on_failed(self, msg: str):
        self._add_bubble("era", "ERA", f"\u26a0 {msg}")
        self.metrics.record(self.trace.finish(success=False, error=msg))
        self._end()

    def _end(self):
        if self.thread:
            self.thread.wait(2000)
        self.thread = None
        self.activity.setText("")
        self.status.setText(self._ready_status())
        self.send_btn.setEnabled(True)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and (e.modifiers() & Qt.ControlModifier):
            self._on_send()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.store.close()
        super().closeEvent(e)
