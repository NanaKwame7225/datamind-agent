"""
DataMind Agent — Notebooks

A notebook is a document of stacked cells. Each cell holds a question and the
multi-agent answer it produced. Cells run in order but are independent; you can
re-run, edit, reorder, or delete any cell.

Notebooks are scoped like analyses: personal by default, or shared when created
inside a shared workspace (workspace_id). Data for the notebook is stored once
at the notebook level (all cells analyse the same dataset).
"""
from __future__ import annotations
import uuid, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc)


class NotebookService:

    async def _col(self):
        from app.database import connect
        db = await connect()
        return db.notebooks if db is not None else None

    def _scope(self, user_id: str, workspace_id: str = None) -> dict:
        from app.services.workspace_service import is_personal
        if workspace_id and not is_personal(workspace_id):
            return {"workspace_id": workspace_id}
        return {"user_id": user_id}

    # ── Notebook CRUD ─────────────────────────────────────────────────────────
    async def create(self, user_id: str, title: str, data: list, columns: list = None,
                     industry: str = "general", workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks are unavailable — database not configured."}
        if not user_id:
            return {"success": False, "error": "Sign in to create a notebook."}
        if not data:
            return {"success": False, "error": "A notebook needs data. Upload data first."}

        from app.services.workspace_service import is_personal
        from app.services.agent_service import agent_service
        wid = workspace_id if (workspace_id and not is_personal(workspace_id)) else None
        # Precompute the data summary ONCE so every cell reuses it instead of
        # recomputing stats over thousands of rows on each run.
        data_summary = agent_service._data_summary(data[:5000], columns)
        doc = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "workspace_id": wid,
            "created_by": user_id,
            "title": (title or "Untitled notebook").strip()[:120],
            "industry": industry or "general",
            "data": data[:5000],
            "data_summary": data_summary,
            "row_count": len(data[:5000]),
            "columns": columns or (list(data[0].keys()) if data and isinstance(data[0], dict) else []),
            "cells": [],
            "created_at": now(),
            "updated_at": now(),
        }
        await col.insert_one(doc)
        return {"success": True, "id": doc["_id"], "notebook": self._meta(doc)}

    def _meta(self, d: dict) -> dict:
        return {
            "id": d["_id"], "title": d.get("title"),
            "industry": d.get("industry"), "row_count": d.get("row_count", 0),
            "col_count": len(d.get("columns") or []),
            "cell_count": len(d.get("cells") or []),
            "workspace_id": d.get("workspace_id"),
            "created_at": _iso(d.get("created_at")),
            "updated_at": _iso(d.get("updated_at")),
        }

    async def list(self, user_id: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks unavailable.", "items": []}
        if not user_id:
            return {"success": True, "items": []}
        scope = self._scope(user_id, workspace_id)
        items = []
        async for d in col.find(scope, {"data": 0, "cells": {"$slice": 0}}).sort("updated_at", -1):
            items.append(self._meta({**d, "cells": d.get("cells", [])}))
        return {"success": True, "items": items}

    async def get(self, user_id: str, notebook_id: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks unavailable."}
        d = await col.find_one({"_id": notebook_id, **self._scope(user_id, workspace_id)})
        if not d:
            return {"success": False, "error": "Notebook not found."}
        return {"success": True, "notebook": {
            **self._meta(d),
            "columns": d.get("columns", []),
            "cells": d.get("cells", []),
            "has_data": bool(d.get("data")),
        }}

    async def rename(self, user_id: str, notebook_id: str, title: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks unavailable."}
        title = (title or "").strip()[:120]
        if not title:
            return {"success": False, "error": "Title can't be empty."}
        res = await col.update_one({"_id": notebook_id, **self._scope(user_id, workspace_id)},
                                   {"$set": {"title": title, "updated_at": now()}})
        if res.matched_count == 0:
            return {"success": False, "error": "Notebook not found."}
        return {"success": True, "title": title}

    async def delete(self, user_id: str, notebook_id: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks unavailable."}
        res = await col.delete_one({"_id": notebook_id, **self._scope(user_id, workspace_id)})
        if res.deleted_count == 0:
            return {"success": False, "error": "Notebook not found."}
        return {"success": True, "deleted": notebook_id}

    # ── Cells ─────────────────────────────────────────────────────────────────
    async def add_cell(self, user_id: str, notebook_id: str, question: str,
                       workspace_id: str = None, deep: bool = False) -> dict:
        """Add a question cell and run the multi-agent analysis on the notebook's data."""
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks unavailable."}
        nb = await col.find_one({"_id": notebook_id, **self._scope(user_id, workspace_id)})
        if not nb:
            return {"success": False, "error": "Notebook not found."}
        question = (question or "").strip()
        if not question:
            return {"success": False, "error": "Ask a question for this cell."}

        from app.services.agent_service import agent_service
        runner = agent_service.analyze if deep else agent_service.analyze_fast
        result = await runner(
            question, nb.get("data") or [], nb.get("columns"), nb.get("industry", "general"),
            data_summary=nb.get("data_summary"))

        cell = {
            "id": str(uuid.uuid4()),
            "question": question,
            "answer": result.get("answer") if result.get("success") else None,
            "agents": result.get("agents", []),
            "emphasis": result.get("emphasis"),
            "provider": result.get("provider"),
            "mode": "deep" if deep else "fast",
            "ok": result.get("success", False),
            "error": result.get("error") if not result.get("success") else None,
            "created_at": now(),
        }
        await col.update_one({"_id": notebook_id},
            {"$push": {"cells": cell}, "$set": {"updated_at": now()}})
        return {"success": True, "cell": cell}

    async def rerun_cell(self, user_id: str, notebook_id: str, cell_id: str,
                         question: str = None, workspace_id: str = None, deep: bool = False) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks unavailable."}
        nb = await col.find_one({"_id": notebook_id, **self._scope(user_id, workspace_id)})
        if not nb:
            return {"success": False, "error": "Notebook not found."}
        cells = nb.get("cells", [])
        idx = next((i for i, c in enumerate(cells) if c["id"] == cell_id), None)
        if idx is None:
            return {"success": False, "error": "Cell not found."}
        q = (question or cells[idx]["question"]).strip()

        from app.services.agent_service import agent_service
        runner = agent_service.analyze if deep else agent_service.analyze_fast
        result = await runner(
            q, nb.get("data") or [], nb.get("columns"), nb.get("industry", "general"),
            data_summary=nb.get("data_summary"))
        cells[idx] = {
            **cells[idx], "question": q,
            "answer": result.get("answer") if result.get("success") else None,
            "agents": result.get("agents", []),
            "emphasis": result.get("emphasis"),
            "provider": result.get("provider"),
            "mode": "deep" if deep else "fast",
            "ok": result.get("success", False),
            "error": result.get("error") if not result.get("success") else None,
            "updated_at": now(),
        }
        await col.update_one({"_id": notebook_id},
            {"$set": {"cells": cells, "updated_at": now()}})
        return {"success": True, "cell": cells[idx]}

    async def delete_cell(self, user_id: str, notebook_id: str, cell_id: str,
                          workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks unavailable."}
        res = await col.update_one({"_id": notebook_id, **self._scope(user_id, workspace_id)},
            {"$pull": {"cells": {"id": cell_id}}, "$set": {"updated_at": now()}})
        if res.matched_count == 0:
            return {"success": False, "error": "Notebook not found."}
        return {"success": True, "deleted": cell_id}

    async def reorder_cells(self, user_id: str, notebook_id: str, order: list,
                            workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Notebooks unavailable."}
        nb = await col.find_one({"_id": notebook_id, **self._scope(user_id, workspace_id)})
        if not nb:
            return {"success": False, "error": "Notebook not found."}
        by_id = {c["id"]: c for c in nb.get("cells", [])}
        new_cells = [by_id[cid] for cid in order if cid in by_id]
        # keep any cells not mentioned (safety) appended at the end
        for c in nb.get("cells", []):
            if c["id"] not in order:
                new_cells.append(c)
        await col.update_one({"_id": notebook_id},
            {"$set": {"cells": new_cells, "updated_at": now()}})
        return {"success": True, "count": len(new_cells)}


def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v


notebook_service = NotebookService()
