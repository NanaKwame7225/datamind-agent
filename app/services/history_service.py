"""
DataMind Agent — Saved Analysis History

Stores each analysis a user runs so they can come back to it. Scoped strictly
by user_id, so nobody sees anyone else's work. Guests get history too (tied to
their guest id); it carries over automatically if they later register.

A saved analysis keeps enough to redisplay it — the query, industry, the
narrative, metrics, insights, and a small data preview — but NOT the full
dataset, to keep documents lightweight.
"""
from __future__ import annotations
import uuid, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PREVIEW_ROWS = 50          # cap stored preview so documents stay small
MAX_TITLE = 120


def now():
    return datetime.now(timezone.utc)


class HistoryService:

    async def _col(self):
        from app.database import connect
        db = await connect()
        return db.analyses if db is not None else None

    def _title(self, query: str, industry: str) -> str:
        q = (query or "").strip().replace("\n", " ")
        if not q:
            return f"{(industry or 'General').title()} analysis"
        return (q[:MAX_TITLE] + "…") if len(q) > MAX_TITLE else q

    async def save(self, user_id: str, payload: dict) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable — the database isn't configured."}
        if not user_id:
            return {"success": False, "error": "Not signed in."}

        result = payload.get("result") or {}
        doc = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": self._title(payload.get("query"), payload.get("industry")),
            "query": payload.get("query", ""),
            "industry": payload.get("industry", "general"),
            "provider": result.get("provider"),
            "narrative": result.get("narrative", ""),
            "metrics": result.get("metrics", []),
            "insights": result.get("insights", []),
            "row_count": payload.get("row_count"),
            "col_count": payload.get("col_count"),
            "columns": payload.get("columns", []),
            "data_preview": (payload.get("data_preview") or [])[:PREVIEW_ROWS],
            "source": payload.get("source"),
            "created_at": now(),
        }
        await col.insert_one(doc)
        return {"success": True, "id": doc["_id"], "title": doc["title"]}

    async def list(self, user_id: str, limit: int = 50, skip: int = 0) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable.", "items": []}
        if not user_id:
            return {"success": True, "items": []}
        limit = max(1, min(int(limit or 50), 100))
        cursor = col.find(
            {"user_id": user_id},
            {"narrative": 0, "data_preview": 0, "insights": 0},   # list view stays light
        ).sort("created_at", -1).skip(int(skip or 0)).limit(limit)
        items = []
        async for d in cursor:
            items.append({
                "id": d["_id"], "title": d.get("title"),
                "query": d.get("query"), "industry": d.get("industry"),
                "provider": d.get("provider"),
                "row_count": d.get("row_count"), "col_count": d.get("col_count"),
                "created_at": d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime) else d.get("created_at"),
            })
        total = await col.count_documents({"user_id": user_id})
        return {"success": True, "items": items, "total": total}

    async def get(self, user_id: str, analysis_id: str) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        d = await col.find_one({"_id": analysis_id, "user_id": user_id})
        if not d:
            return {"success": False, "error": "Analysis not found."}
        d["id"] = d.pop("_id")
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return {"success": True, "analysis": d}

    async def delete(self, user_id: str, analysis_id: str) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        res = await col.delete_one({"_id": analysis_id, "user_id": user_id})
        if res.deleted_count == 0:
            return {"success": False, "error": "Analysis not found."}
        return {"success": True, "deleted": analysis_id}

    async def rename(self, user_id: str, analysis_id: str, title: str) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        title = (title or "").strip()[:MAX_TITLE]
        if not title:
            return {"success": False, "error": "Title cannot be empty."}
        res = await col.update_one({"_id": analysis_id, "user_id": user_id},
                                   {"$set": {"title": title}})
        if res.matched_count == 0:
            return {"success": False, "error": "Analysis not found."}
        return {"success": True, "id": analysis_id, "title": title}


history_service = HistoryService()
