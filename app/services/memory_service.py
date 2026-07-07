"""
DataMind Agent — Memory & Context Layer
Stores conversation history, past analyses, user preferences,
and cross-session context so the AI remembers what it has seen before.
"""
from __future__ import annotations
import json, logging, sqlite3, uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)
MEMORY_DB = "./datamind_memory.db"


def _get_conn():
    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db():
    conn = _get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        industry    TEXT,
        query       TEXT,
        created_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS analysis_memory (
        id              TEXT PRIMARY KEY,
        session_id      TEXT NOT NULL,
        industry        TEXT,
        query           TEXT,
        key_findings    TEXT,
        anomalies       TEXT,
        recommendations TEXT,
        provider        TEXT,
        row_count       INTEGER,
        col_count       INTEGER,
        created_at      TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS user_context (
        session_id      TEXT PRIMARY KEY,
        industry        TEXT,
        last_query      TEXT,
        preferences     TEXT,
        data_schema     TEXT,
        updated_at      TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS rag_cache (
        id          TEXT PRIMARY KEY,
        query_hash  TEXT UNIQUE,
        industry    TEXT,
        context     TEXT,
        created_at  TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
    CREATE INDEX IF NOT EXISTS idx_mem_session ON analysis_memory(session_id);
    """)
    conn.commit()
    conn.close()
    logger.info("Memory database initialised")


class MemoryService:
    """
    Persistent memory layer for DataMind Agent.
    Stores conversation history, analysis results, and user context
    so the AI can reference previous findings and build on past work.
    """

    def save_message(self, session_id: str, role: str, content: str,
                     industry: str = None, query: str = None) -> str:
        msg_id = str(uuid.uuid4())
        conn = _get_conn()
        conn.execute(
            "INSERT INTO conversations (id,session_id,role,content,industry,query,created_at) VALUES (?,?,?,?,?,?,?)",
            (msg_id, session_id, role, content, industry, query, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()
        return msg_id

    def get_conversation_history(self, session_id: str, limit: int = 10) -> list[dict]:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT role, content, industry, query, created_at FROM conversations "
            "WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]

    def save_analysis_memory(self, session_id: str, result: dict) -> str:
        """Save key findings from an analysis to memory for future reference."""
        mem_id = str(uuid.uuid4())
        findings = result.get("insights", [])[:5]
        anomalies = [f for f in findings if f.get("severity") in ["critical","warning"]]
        recs = []
        narrative = result.get("narrative", "")
        if narrative:
            lines = narrative.split("\n")
            rec_lines = [l for l in lines if l.strip().startswith(("1.","2.","3.","•","-")) and len(l) > 15]
            recs = rec_lines[:5]
        conn = _get_conn()
        conn.execute(
            "INSERT INTO analysis_memory (id,session_id,industry,query,key_findings,anomalies,recommendations,provider,row_count,col_count,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (mem_id, session_id,
             result.get("industry"), result.get("query"),
             json.dumps(findings), json.dumps(anomalies), json.dumps(recs),
             result.get("provider"), result.get("row_count",0), result.get("col_count",0),
             datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()
        return mem_id

    def get_memory_context(self, session_id: str) -> str:
        """
        Build a context string from past analyses in this session.
        Injected into the LLM prompt so Claude remembers what it saw before.
        """
        conn = _get_conn()
        rows = conn.execute(
            "SELECT industry, query, key_findings, anomalies, recommendations, created_at "
            "FROM analysis_memory WHERE session_id=? ORDER BY created_at DESC LIMIT 5",
            (session_id,)
        ).fetchall()
        conn.close()
        if not rows:
            return ""
        lines = ["PREVIOUS ANALYSES IN THIS SESSION (for context):"]
        for r in rows:
            findings = json.loads(r["key_findings"] or "[]")
            anomalies = json.loads(r["anomalies"] or "[]")
            lines.append(f"\nAnalysis: {r['query']} ({r['industry']})")
            if anomalies:
                lines.append(f"  Issues found: {len(anomalies)} — "+"; ".join([a.get("title","") for a in anomalies[:3]]))
            if findings:
                lines.append(f"  Key findings: {len(findings)}")
        return "\n".join(lines)

    def save_user_context(self, session_id: str, industry: str, query: str,
                          schema: dict = None, preferences: dict = None):
        conn = _get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO user_context (session_id,industry,last_query,preferences,data_schema,updated_at) VALUES (?,?,?,?,?,?)",
            (session_id, industry, query,
             json.dumps(preferences or {}), json.dumps(schema or {}),
             datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()

    def get_user_context(self, session_id: str) -> Optional[dict]:
        conn = _get_conn()
        row = conn.execute("SELECT * FROM user_context WHERE session_id=?", (session_id,)).fetchone()
        conn.close()
        if not row:
            return None
        r = dict(row)
        r["preferences"] = json.loads(r["preferences"] or "{}")
        r["data_schema"] = json.loads(r["data_schema"] or "{}")
        return r

    def search_memory(self, query: str, session_id: str = None) -> list[dict]:
        """Search past analyses by keyword."""
        conn = _get_conn()
        q = f"%{query.lower()}%"
        if session_id:
            rows = conn.execute(
                "SELECT * FROM analysis_memory WHERE session_id=? AND (LOWER(query) LIKE ? OR LOWER(key_findings) LIKE ?) ORDER BY created_at DESC LIMIT 10",
                (session_id, q, q)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM analysis_memory WHERE LOWER(query) LIKE ? OR LOWER(key_findings) LIKE ? ORDER BY created_at DESC LIMIT 10",
                (q, q)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def clear_session(self, session_id: str):
        conn = _get_conn()
        conn.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM analysis_memory WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM user_context WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()


memory_service = MemoryService()
