"""ERA AI — local memory store (spec Section 8 + 10).

SQLite only, local-only. Stores projects, conversations, messages,
durable facts and settings. No cloud, no telemetry.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def _uid() -> str:
    return uuid.uuid4().hex[:12]


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    config TEXT DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    worker_id TEXT DEFAULT '',
    created_at REAL NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_proj ON conversations(project_id);
"""


@dataclass
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    worker_id: str
    created_at: float


class MemoryStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---- projects --------------------------------------------------
    def create_project(self, name: str, description: str = "",
                       config: Optional[Dict] = None) -> str:
        pid = _uid()
        self.conn.execute(
            "INSERT INTO projects(id,name,description,config,created_at) VALUES(?,?,?,?,?)",
            (pid, name, description, json.dumps(config or {}), time.time()),
        )
        self.conn.commit()
        return pid

    def list_projects(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_project(self, pid: str) -> None:
        self.conn.execute("DELETE FROM projects WHERE id=?", (pid,))
        self.conn.commit()

    # ---- conversations ---------------------------------------------
    def create_conversation(self, title: str = "New Conversation",
                            project_id: Optional[str] = None) -> str:
        cid = _uid()
        now = time.time()
        self.conn.execute(
            "INSERT INTO conversations(id,project_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (cid, project_id, title, now, now),
        )
        self.conn.commit()
        return cid

    def rename_conversation(self, cid: str, title: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title, time.time(), cid),
        )
        self.conn.commit()

    def list_conversations(self, project_id: Optional[str] = None) -> List[Dict]:
        if project_id:
            rows = self.conn.execute(
                "SELECT * FROM conversations WHERE project_id=? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, cid: str) -> None:
        self.conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
        self.conn.commit()

    # ---- messages --------------------------------------------------
    def add_message(self, cid: str, role: str, content: str,
                   worker_id: str = "") -> str:
        mid = _uid()
        now = time.time()
        self.conn.execute(
            "INSERT INTO messages(id,conversation_id,role,content,worker_id,created_at) VALUES(?,?,?,?,?,?)",
            (mid, cid, role, content, worker_id, now),
        )
        self.conn.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?", (now, cid)
        )
        self.conn.commit()
        return mid

    def get_messages(self, cid: str) -> List[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
            (cid,),
        ).fetchall()
        return [Message(**dict(r)) for r in rows]

    def history_for_llm(self, cid: str) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content}
                for m in self.get_messages(cid)
                if m.role in ("user", "assistant")]

    # ---- facts (durable memory) ------------------------------------
    def add_fact(self, key: str, value: str,
                project_id: Optional[str] = None) -> str:
        fid = _uid()
        self.conn.execute(
            "INSERT INTO facts(id,project_id,key,value,created_at) VALUES(?,?,?,?,?)",
            (fid, project_id, key, value, time.time()),
        )
        self.conn.commit()
        return fid

    def search_facts(self, query: str,
                    project_id: Optional[str] = None) -> List[Dict]:
        like = f"%{query}%"
        if project_id:
            rows = self.conn.execute(
                "SELECT * FROM facts WHERE project_id=? AND (key LIKE ? OR value LIKE ?)",
                (project_id, like, like),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM facts WHERE key LIKE ? OR value LIKE ?",
                (like, like),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_fact(self, fid: str) -> None:
        self.conn.execute("DELETE FROM facts WHERE id=?", (fid,))
        self.conn.commit()

    def clear_all(self) -> None:
        for t in ("messages", "conversations", "facts"):
            self.conn.execute(f"DELETE FROM {t}")
        self.conn.commit()

    # ---- kv settings -----------------------------------------------
    def set_kv(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_kv(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default
