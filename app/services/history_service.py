"""
DataMind Agent — Saved Analysis History (with version timelines)

Each analysis document holds a `versions` array — a timeline of snapshots.
The top-level fields mirror the LATEST version so the list view stays fast
without reading every version.

Save behaviour (both automatic and manual):
  • Automatic  — saving the same query on the same-shaped data appends a new
                 version to the matching analysis, rather than creating a row.
  • Manual     — the caller can force a new analysis (save_as="new") or force a
                 version onto a specific analysis (save_as="version", target_id=…).

Everything is scoped by user_id, so nobody can touch anyone else's analyses
or versions.
"""
from __future__ import annotations
import uuid, logging, hashlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PREVIEW_ROWS = 50
MAX_TITLE = 120
MAX_VERSIONS = 50          # keep timelines bounded


def now():
    return datetime.now(timezone.utc)


class HistoryService:

    async def _col(self):
        from app.database import connect
        db = await connect()
        return db.analyses if db is not None else None

    def _scope(self, user_id: str, workspace_id: str = None) -> dict:
        """
        Build the ownership filter for a query.
          • no workspace, or a personal workspace → the user's own analyses that
            are NOT attached to any shared workspace (workspace_id null/absent)
          • a shared workspace → scope by workspace_id (all members share it)
        This keeps every existing personal analysis working with no migration,
        while making sure shared-workspace analyses never leak into a personal list.
        """
        from app.services.workspace_service import is_personal
        if workspace_id and not is_personal(workspace_id):
            return {"workspace_id": workspace_id}
        # Personal: mine, and not owned by a shared workspace
        return {"user_id": user_id, "workspace_id": None}

    def _title(self, query: str, industry: str) -> str:
        q = (query or "").strip().replace("\n", " ")
        if not q:
            return f"{(industry or 'General').title()} analysis"
        return (q[:MAX_TITLE] + "…") if len(q) > MAX_TITLE else q

    def _signature(self, query: str, columns: list) -> str:
        """A stable fingerprint of 'same question on same-shaped data'."""
        q = (query or "").strip().lower()
        cols = ",".join(sorted([str(c).lower() for c in (columns or [])]))
        return hashlib.sha1(f"{q}||{cols}".encode("utf-8")).hexdigest()

    def _make_version(self, payload: dict) -> dict:
        result = payload.get("result") or {}
        return {
            "version_id": str(uuid.uuid4()),
            "created_at": now(),
            "query": payload.get("query", ""),
            "provider": result.get("provider"),
            "narrative": result.get("narrative", ""),
            "metrics": result.get("metrics", []),
            "insights": result.get("insights", []),
            "row_count": payload.get("row_count"),
            "col_count": payload.get("col_count"),
            "columns": payload.get("columns", []),
            "data_preview": (payload.get("data_preview") or [])[:PREVIEW_ROWS],
            "source": payload.get("source"),
        }

    def _mirror(self, doc: dict, version: dict):
        """Copy the latest version's fields to the top level for the list view."""
        doc["query"] = version["query"]
        doc["provider"] = version["provider"]
        doc["narrative"] = version["narrative"]
        doc["metrics"] = version["metrics"]
        doc["insights"] = version["insights"]
        doc["row_count"] = version["row_count"]
        doc["col_count"] = version["col_count"]
        doc["columns"] = version["columns"]
        doc["data_preview"] = version["data_preview"]
        doc["source"] = version["source"]
        doc["updated_at"] = version["created_at"]

    # ── SAVE ─────────────────────────────────────────────────────────────────
    async def save(self, user_id: str, payload: dict,
                   save_as: str = "auto", target_id: str = None,
                   workspace_id: str = None) -> dict:
        """
        save_as:
          "auto"     → append a version if a matching analysis exists, else new
          "new"      → always create a new analysis
          "version"  → append a version to target_id (must belong to the scope)
        workspace_id: when a shared workspace, the analysis belongs to it and all
          members can see it; otherwise it's the user's personal analysis.
        """
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable — the database isn't configured."}
        if not user_id:
            return {"success": False, "error": "Not signed in."}

        scope = self._scope(user_id, workspace_id)
        version = self._make_version(payload)
        sig = self._signature(payload.get("query"), payload.get("columns"))

        # ── Explicit: add a version to a named analysis ──
        if save_as == "version" and target_id:
            doc = await col.find_one({"_id": target_id, **scope})
            if not doc:
                return {"success": False, "error": "That analysis was not found."}
            return await self._append_version(col, doc, version)

        # ── Automatic: does a matching analysis already exist in this scope? ──
        if save_as == "auto":
            match = await col.find_one({**scope, "signature": sig})
            if match:
                return await self._append_version(col, match, version, matched=True)

        # ── Otherwise create a brand-new analysis ──
        from app.services.workspace_service import is_personal
        doc = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "workspace_id": workspace_id if (workspace_id and not is_personal(workspace_id)) else None,
            "created_by": user_id,
            "title": self._title(payload.get("query"), payload.get("industry")),
            "industry": payload.get("industry", "general"),
            "signature": sig,
            "versions": [version],
            "version_count": 1,
            "created_at": now(),
        }
        self._mirror(doc, version)
        await col.insert_one(doc)
        return {"success": True, "id": doc["_id"], "title": doc["title"],
                "version_id": version["version_id"], "version_count": 1, "new_analysis": True}

    async def _append_version(self, col, doc, version, matched=False):
        versions = doc.get("versions", [])
        versions.append(version)
        if len(versions) > MAX_VERSIONS:
            versions = versions[-MAX_VERSIONS:]      # drop the oldest
        update = {"versions": versions, "version_count": len(versions)}
        # Mirror the new latest version to the top level
        tmp = dict(doc); self._mirror(tmp, version)
        for k in ("query", "provider", "narrative", "metrics", "insights",
                  "row_count", "col_count", "columns", "data_preview", "source", "updated_at"):
            update[k] = tmp[k]
        await col.update_one({"_id": doc["_id"], "user_id": doc["user_id"]}, {"$set": update})
        return {"success": True, "id": doc["_id"], "title": doc.get("title"),
                "version_id": version["version_id"], "version_count": len(versions),
                "new_version": True, "auto_matched": matched}

    # ── LIST ─────────────────────────────────────────────────────────────────
    async def list(self, user_id: str, limit: int = 50, skip: int = 0, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable.", "items": []}
        if not user_id:
            return {"success": True, "items": []}
        limit = max(1, min(int(limit or 50), 100))
        scope = self._scope(user_id, workspace_id)
        cursor = col.find(
            scope,
            {"narrative": 0, "data_preview": 0, "insights": 0, "versions": 0},
        ).sort("updated_at", -1).skip(int(skip or 0)).limit(limit)
        items = []
        async for d in cursor:
            ts = d.get("updated_at") or d.get("created_at")
            items.append({
                "id": d["_id"], "title": d.get("title"),
                "query": d.get("query"), "industry": d.get("industry"),
                "provider": d.get("provider"),
                "row_count": d.get("row_count"), "col_count": d.get("col_count"),
                "version_count": d.get("version_count", 1),
                "updated_at": ts.isoformat() if isinstance(ts, datetime) else ts,
                "created_at": d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime) else d.get("created_at"),
            })
        total = await col.count_documents(scope)
        return {"success": True, "items": items, "total": total}

    # ── GET (latest, or a specific version) ──────────────────────────────────
    async def get(self, user_id: str, analysis_id: str, version_id: str = None, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        d = await col.find_one({"_id": analysis_id, **self._scope(user_id, workspace_id)})
        if not d:
            return {"success": False, "error": "Analysis not found."}

        versions = d.get("versions", [])
        chosen = None
        if version_id:
            chosen = next((v for v in versions if v.get("version_id") == version_id), None)
            if not chosen:
                return {"success": False, "error": "That version was not found."}
        elif versions:
            chosen = versions[-1]

        out = {
            "id": d["_id"], "title": d.get("title"), "industry": d.get("industry"),
            "version_count": d.get("version_count", len(versions) or 1),
            "created_at": d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime) else d.get("created_at"),
        }
        src = chosen or d
        out.update({
            "query": src.get("query"), "provider": src.get("provider"),
            "narrative": src.get("narrative"), "metrics": src.get("metrics", []),
            "insights": src.get("insights", []), "row_count": src.get("row_count"),
            "col_count": src.get("col_count"), "columns": src.get("columns", []),
            "data_preview": src.get("data_preview", []), "source": src.get("source"),
            "version_id": (chosen or {}).get("version_id"),
        })
        return {"success": True, "analysis": out}

    # ── VERSION TIMELINE (metadata only, lightweight) ────────────────────────
    async def versions(self, user_id: str, analysis_id: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        d = await col.find_one({"_id": analysis_id, **self._scope(user_id, workspace_id)}, {"versions": 1, "title": 1})
        if not d:
            return {"success": False, "error": "Analysis not found."}
        vs = []
        for i, v in enumerate(d.get("versions", [])):
            ts = v.get("created_at")
            vs.append({
                "version_id": v.get("version_id"),
                "number": i + 1,
                "query": v.get("query"),
                "provider": v.get("provider"),
                "row_count": v.get("row_count"),
                "col_count": v.get("col_count"),
                "created_at": ts.isoformat() if isinstance(ts, datetime) else ts,
            })
        vs.reverse()   # newest first
        return {"success": True, "title": d.get("title"), "versions": vs, "total": len(vs)}

    # ── DELETE a whole analysis, or restore/delete one version ───────────────
    async def delete(self, user_id: str, analysis_id: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        res = await col.delete_one({"_id": analysis_id, **self._scope(user_id, workspace_id)})
        if res.deleted_count == 0:
            return {"success": False, "error": "Analysis not found."}
        return {"success": True, "deleted": analysis_id}

    async def delete_version(self, user_id: str, analysis_id: str, version_id: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        d = await col.find_one({"_id": analysis_id, **self._scope(user_id, workspace_id)})
        if not d:
            return {"success": False, "error": "Analysis not found."}
        versions = d.get("versions", [])
        if len(versions) <= 1:
            return {"success": False, "error": "Can't delete the only version — delete the analysis instead."}
        kept = [v for v in versions if v.get("version_id") != version_id]
        if len(kept) == len(versions):
            return {"success": False, "error": "That version was not found."}
        update = {"versions": kept, "version_count": len(kept)}
        tmp = dict(d); self._mirror(tmp, kept[-1])
        for k in ("query", "provider", "narrative", "metrics", "insights",
                  "row_count", "col_count", "columns", "data_preview", "source", "updated_at"):
            update[k] = tmp[k]
        await col.update_one({"_id": analysis_id, **self._scope(user_id, workspace_id)}, {"$set": update})
        return {"success": True, "version_count": len(kept)}

    async def restore_version(self, user_id: str, analysis_id: str, version_id: str, workspace_id: str = None) -> dict:
        """Make an older version the current one by copying it to the top of the timeline."""
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        d = await col.find_one({"_id": analysis_id, **self._scope(user_id, workspace_id)})
        if not d:
            return {"success": False, "error": "Analysis not found."}
        versions = d.get("versions", [])
        src = next((v for v in versions if v.get("version_id") == version_id), None)
        if not src:
            return {"success": False, "error": "That version was not found."}
        clone = dict(src)
        clone["version_id"] = str(uuid.uuid4())
        clone["created_at"] = now()
        clone["restored_from"] = version_id
        return await self._append_version(col, d, clone)

    async def rename(self, user_id: str, analysis_id: str, title: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "History is unavailable."}
        title = (title or "").strip()[:MAX_TITLE]
        if not title:
            return {"success": False, "error": "Title cannot be empty."}
        res = await col.update_one({"_id": analysis_id, **self._scope(user_id, workspace_id)},
                                   {"$set": {"title": title}})
        if res.matched_count == 0:
            return {"success": False, "error": "Analysis not found."}
        return {"success": True, "id": analysis_id, "title": title}


history_service = HistoryService()
