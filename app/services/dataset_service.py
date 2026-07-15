"""
DataMind Agent — Server-side Dataset Store

The old flow sent every row to the browser and back again on each analysis, which
capped practical size at a few thousand rows. This service keeps the data on the
server: a file is parsed once, stored, and afterwards the browser only ever asks
for small aggregates (a preview, chart series, stats) computed here.

Storage: parquet on disk (compact, fast, typed) with metadata in MongoDB.
Parquet keeps a large frame small and loads far faster than JSON.

Scale note (honest): this lifts the ceiling to roughly hundreds of thousands of
rows — bounded by the host's RAM and disk, since pandas loads the frame to
compute. Millions-plus needs a warehouse (BigQuery/Postgres) pushing SQL down.
"""
from __future__ import annotations
import os, uuid, json, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STORE_DIR = os.environ.get("DATASET_DIR", "/tmp/datamind_datasets")
MAX_ROWS = int(os.environ.get("DATASET_MAX_ROWS", "1000000"))   # hard safety cap


def now():
    return datetime.now(timezone.utc)


def _ensure_dir():
    os.makedirs(STORE_DIR, exist_ok=True)


class DatasetService:

    async def _col(self):
        from app.database import connect
        db = await connect()
        return db.datasets if db is not None else None

    def _path(self, dataset_id: str) -> str:
        return os.path.join(STORE_DIR, f"{dataset_id}.parquet")

    # ── Store ─────────────────────────────────────────────────────────────────
    async def store(self, user_id: str, records: list, filename: str = None,
                    workspace_id: str = None) -> dict:
        """Persist a parsed dataset server-side; return an id + light metadata."""
        if not records:
            return {"success": False, "error": "No rows to store."}
        try:
            import pandas as pd
        except ImportError:
            return {"success": False, "error": "pandas unavailable on the server."}

        df = pd.DataFrame(records)
        truncated = False
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            truncated = True

        did = str(uuid.uuid4())
        _ensure_dir()
        try:
            df.to_parquet(self._path(did), index=False)
        except Exception as e:
            # parquet needs pyarrow; fall back to compressed pickle
            logger.warning(f"Parquet write failed ({e}); using pickle fallback.")
            try:
                df.to_pickle(self._path(did).replace(".parquet", ".pkl"))
            except Exception as e2:
                return {"success": False, "error": f"Could not store dataset: {e2}"}

        meta = {
            "_id": did,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "filename": filename or "dataset",
            "row_count": int(len(df)),
            "col_count": int(len(df.columns)),
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "truncated": truncated,
            "created_at": now(),
        }
        col = await self._col()
        if col is not None:
            await col.insert_one(meta)
        return {"success": True, "dataset_id": did, "meta": _public(meta),
                "preview": _json_safe(df.head(50)),
                "truncated": truncated, "max_rows": MAX_ROWS}

    # ── Load ──────────────────────────────────────────────────────────────────
    def _read(self, dataset_id: str):
        import pandas as pd
        p = self._path(dataset_id)
        if os.path.exists(p):
            return pd.read_parquet(p)
        pk = p.replace(".parquet", ".pkl")
        if os.path.exists(pk):
            return pd.read_pickle(pk)
        return None

    async def _meta(self, user_id: str, dataset_id: str, workspace_id: str = None):
        col = await self._col()
        if col is None:
            return None
        from app.services.workspace_service import is_personal
        scope = ({"workspace_id": workspace_id}
                 if workspace_id and not is_personal(workspace_id)
                 else {"user_id": user_id})
        return await col.find_one({"_id": dataset_id, **scope})

    async def get_frame(self, user_id: str, dataset_id: str, workspace_id: str = None):
        """Load the stored frame if the caller may see it. Returns (df, error)."""
        meta = await self._meta(user_id, dataset_id, workspace_id)
        if not meta:
            return None, "Dataset not found."
        df = self._read(dataset_id)
        if df is None:
            # Railway/Render filesystems are ephemeral — a redeploy wipes /tmp.
            # Be explicit rather than cryptic so the fix is obvious.
            return None, ("This dataset's file is no longer on the server (the app "
                          "restarted). Re-upload the file to restore it.")
        return df, None

    # ── Small, cheap reads the browser actually needs ─────────────────────────
    async def preview(self, user_id: str, dataset_id: str, limit: int = 50,
                      workspace_id: str = None) -> dict:
        df, err = await self.get_frame(user_id, dataset_id, workspace_id)
        if err:
            return {"success": False, "error": err}
        return {"success": True, "rows": _json_safe(df.head(min(limit, 500))),
                "row_count": int(len(df)), "columns": [str(c) for c in df.columns]}

    async def sample(self, user_id: str, dataset_id: str, n: int = 4000,
                     workspace_id: str = None) -> dict:
        """An evenly-spread sample — for the AI and for charting."""
        df, err = await self.get_frame(user_id, dataset_id, workspace_id)
        if err:
            return {"success": False, "error": err}
        if len(df) > n:
            step = max(1, len(df) // n)
            df = df.iloc[::step].head(n)
        return {"success": True, "rows": _json_safe(df), "sampled": True,
                "row_count": int(len(df))}

    async def list(self, user_id: str, workspace_id: str = None) -> dict:
        col = await self._col()
        if col is None:
            return {"success": True, "items": []}
        from app.services.workspace_service import is_personal
        scope = ({"workspace_id": workspace_id}
                 if workspace_id and not is_personal(workspace_id)
                 else {"user_id": user_id})
        items = []
        async for d in col.find(scope).sort("created_at", -1):
            items.append(_public(d))
        return {"success": True, "items": items}

    async def delete(self, user_id: str, dataset_id: str, workspace_id: str = None) -> dict:
        meta = await self._meta(user_id, dataset_id, workspace_id)
        if not meta:
            return {"success": False, "error": "Dataset not found."}
        for p in (self._path(dataset_id), self._path(dataset_id).replace(".parquet", ".pkl")):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        col = await self._col()
        if col is not None:
            await col.delete_one({"_id": dataset_id})
        return {"success": True, "deleted": dataset_id}

    # ── Aggregation: the whole point — compute here, send tiny results ────────
    async def aggregate(self, user_id: str, dataset_id: str, group_by: str,
                        metric: str, agg: str = "sum", limit: int = 20,
                        workspace_id: str = None) -> dict:
        df, err = await self.get_frame(user_id, dataset_id, workspace_id)
        if err:
            return {"success": False, "error": err}
        if group_by not in df.columns:
            return {"success": False, "error": f"Column '{group_by}' not found."}
        try:
            import pandas as pd
            if metric and metric in df.columns:
                s = df.groupby(group_by)[metric]
                out = getattr(s, agg)() if hasattr(s, agg) else s.sum()
            else:
                out = df.groupby(group_by).size()
            out = out.sort_values(ascending=False).head(limit)
            return {"success": True,
                    "labels": [str(i) for i in out.index.tolist()],
                    "values": [float(v) for v in out.values.tolist()],
                    "group_by": group_by, "metric": metric, "agg": agg}
        except Exception as e:
            return {"success": False, "error": f"Aggregation failed: {e}"}

    async def stats(self, user_id: str, dataset_id: str, workspace_id: str = None) -> dict:
        """Column stats computed server-side — the browser never sees the rows."""
        df, err = await self.get_frame(user_id, dataset_id, workspace_id)
        if err:
            return {"success": False, "error": err}
        try:
            num = df.select_dtypes(include="number")
            numeric = {}
            for c in num.columns:
                s = num[c]
                numeric[str(c)] = {
                    "min": _f(s.min()), "max": _f(s.max()),
                    "mean": _f(s.mean()), "median": _f(s.median()),
                    "std": _f(s.std()), "missing": int(df[c].isna().sum()),
                }
            cats = {}
            for c in df.columns:
                if c in num.columns:
                    continue
                nun = int(df[c].nunique(dropna=True))
                if 0 < nun <= 60:
                    vc = df[c].value_counts().head(8)
                    cats[str(c)] = {"unique": nun,
                                    "top": [[str(k), int(v)] for k, v in vc.items()]}
            return {"success": True, "row_count": int(len(df)),
                    "columns": [str(c) for c in df.columns],
                    "numeric": numeric, "categorical": cats}
        except Exception as e:
            return {"success": False, "error": f"Stats failed: {e}"}


def _f(v):
    try:
        import math
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    except Exception:
        return None

def _json_safe(df):
    """DataFrame -> list of dicts with NaN/Timestamp made JSON-safe."""
    return json.loads(df.to_json(orient="records", date_format="iso"))

def _public(m: dict) -> dict:
    return {"id": m["_id"], "filename": m.get("filename"),
            "row_count": m.get("row_count"), "col_count": m.get("col_count"),
            "columns": m.get("columns", []), "truncated": m.get("truncated", False),
            "created_at": m["created_at"].isoformat() if hasattr(m.get("created_at"), "isoformat") else m.get("created_at")}


dataset_service = DatasetService()
