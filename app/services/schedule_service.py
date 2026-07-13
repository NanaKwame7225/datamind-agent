"""
DataMind Agent — Scheduled Reports

A schedule stores everything needed to re-run an analysis unattended: the data
(stored inline, since a schedule must be self-contained), the query, the
industry, how often to run, and which channels to deliver on.

Cadence is expressed simply (daily / weekly / monthly + time), converted to a
cron expression internally so croniter can compute the next run. Every schedule
is scoped by user_id.

When due, a schedule:
  1. re-runs the analysis (reusing analysis_service),
  2. saves the result to the user's history (in-app channel, always),
  3. delivers via email and/or SMS if those channels are enabled.
"""
from __future__ import annotations
import uuid, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def now():
    return datetime.now(timezone.utc)


def _cron_for(freq: str, hour: int, minute: int, weekday: str, day: int) -> str:
    """Build a cron expression from friendly cadence fields."""
    freq = (freq or "weekly").lower()
    m, h = int(minute), int(hour)
    if freq == "daily":
        return f"{m} {h} * * *"
    if freq == "weekly":
        wd = WEEKDAYS.index(weekday.lower()) if weekday and weekday.lower() in WEEKDAYS else 0
        return f"{m} {h} * * {wd}"
    if freq == "monthly":
        d = max(1, min(int(day or 1), 28))     # 28 to stay valid every month
        return f"{m} {h} {d} * *"
    # hourly / fallback
    if freq == "hourly":
        return f"{m} * * * *"
    return f"{m} {h} * * 0"


def _next_run(cron: str, after: datetime = None) -> datetime | None:
    try:
        from croniter import croniter
        base = after or now()
        return croniter(cron, base).get_next(datetime)
    except Exception as e:
        logger.warning(f"Cron parse failed for '{cron}': {e}")
        return None


class ScheduleService:

    async def _col(self):
        from app.database import connect
        db = await connect()
        return db.schedules if db is not None else None

    # ── CRUD ─────────────────────────────────────────────────────────────────
    async def create(self, user_id: str, payload: dict) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Scheduling is unavailable — the database isn't configured."}
        if not user_id:
            return {"success": False, "error": "Not signed in."}

        data = payload.get("data") or []
        if not data:
            return {"success": False, "error": "Attach the data to schedule this report on."}
        query = (payload.get("query") or "").strip()
        if not query:
            return {"success": False, "error": "Add the question the report should answer."}

        channels = payload.get("channels") or {}
        email_to = (channels.get("email_to") or "").strip()
        sms_to = (channels.get("sms_to") or "").strip()

        cron = _cron_for(payload.get("frequency", "weekly"),
                         payload.get("hour", 8), payload.get("minute", 0),
                         payload.get("weekday", "monday"), payload.get("day", 1))
        nxt = _next_run(cron)

        doc = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "name": (payload.get("name") or query)[:120],
            "query": query,
            "industry": payload.get("industry", "general"),
            "data": data[:5000],                 # cap stored rows
            "row_count": len(data[:5000]),       # stored so list view can show it
            "columns": payload.get("columns", []),
            "frequency": payload.get("frequency", "weekly"),
            "hour": int(payload.get("hour", 8)),
            "minute": int(payload.get("minute", 0)),
            "weekday": payload.get("weekday", "monday"),
            "day": int(payload.get("day", 1)),
            "cron": cron,
            "channels": {
                "in_app": True,                  # always
                "email": bool(channels.get("email")) and bool(email_to),
                "email_to": email_to,
                "sms": bool(channels.get("sms")) and bool(sms_to),
                "sms_to": sms_to,
            },
            "active": True,
            "created_at": now(),
            "next_run": nxt,
            "last_run": None,
            "last_status": None,
            "run_count": 0,
        }
        await col.insert_one(doc)
        return {"success": True, "id": doc["_id"], "next_run": nxt.isoformat() if nxt else None,
                "schedule": self._public(doc)}

    def _public(self, d: dict) -> dict:
        return {
            "id": d["_id"], "name": d.get("name"), "query": d.get("query"),
            "industry": d.get("industry"), "frequency": d.get("frequency"),
            "hour": d.get("hour"), "minute": d.get("minute"),
            "weekday": d.get("weekday"), "day": d.get("day"),
            "channels": {k: v for k, v in (d.get("channels") or {}).items()},
            "active": d.get("active", True),
            "row_count": d.get("row_count", len(d.get("data") or [])),
            "next_run": _iso(d.get("next_run")),
            "last_run": _iso(d.get("last_run")),
            "last_status": d.get("last_status"),
            "run_count": d.get("run_count", 0),
            "created_at": _iso(d.get("created_at")),
        }

    async def list(self, user_id: str) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Scheduling unavailable.", "items": []}
        if not user_id:
            return {"success": True, "items": []}
        cursor = col.find({"user_id": user_id}, {"data": 0}).sort("created_at", -1)
        items = []
        async for d in cursor:
            items.append(self._public({**d, "data": [None] * 0}))
        return {"success": True, "items": items}

    async def toggle(self, user_id: str, sid: str, active: bool) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Scheduling unavailable."}
        d = await col.find_one({"_id": sid, "user_id": user_id})
        if not d:
            return {"success": False, "error": "Schedule not found."}
        upd = {"active": bool(active)}
        if active:
            upd["next_run"] = _next_run(d["cron"])
        await col.update_one({"_id": sid, "user_id": user_id}, {"$set": upd})
        return {"success": True, "active": bool(active),
                "next_run": _iso(upd.get("next_run"))}

    async def delete(self, user_id: str, sid: str) -> dict:
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Scheduling unavailable."}
        res = await col.delete_one({"_id": sid, "user_id": user_id})
        if res.deleted_count == 0:
            return {"success": False, "error": "Schedule not found."}
        return {"success": True, "deleted": sid}

    async def run_now(self, user_id: str, sid: str) -> dict:
        """Manually trigger a schedule (for the 'Run now' button)."""
        col = await self._col()
        if col is None:
            return {"success": False, "error": "Scheduling unavailable."}
        d = await col.find_one({"_id": sid, "user_id": user_id})
        if not d:
            return {"success": False, "error": "Schedule not found."}
        return await self._execute(col, d)

    # ── SCHEDULER TICK ───────────────────────────────────────────────────────
    async def run_due(self) -> dict:
        """Called by the background loop. Runs every schedule whose time passed."""
        col = await self._col()
        if col is None:
            return {"ran": 0}
        cutoff = now()
        ran = 0
        cursor = col.find({"active": True, "next_run": {"$lte": cutoff}})
        async for d in cursor:
            try:
                await self._execute(col, d)
                ran += 1
            except Exception as e:
                logger.error(f"Scheduled report {d['_id']} failed: {e}")
                await col.update_one({"_id": d["_id"]},
                    {"$set": {"last_status": f"error: {e}",
                              "next_run": _next_run(d["cron"])}})
        return {"ran": ran}

    async def _execute(self, col, d: dict) -> dict:
        """Run one report: analyse → save to history → deliver on channels."""
        # A schedule with no data can never produce a report. Pause it so it
        # stops retrying every minute, and report clearly.
        if not (d.get("data") or []):
            await col.update_one({"_id": d["_id"]}, {"$set": {
                "active": False,
                "last_status": "no data — schedule paused. Recreate it with data attached.",
                "last_run": now(),
            }})
            return {"success": False, "error": "This schedule has no data attached. It has been paused — delete it and create a new one after uploading data."}
        result = await self._analyse(d)
        chan = d.get("channels") or {}
        delivered = {"in_app": False, "email": None, "sms": None}

        # 1. In-app: always save to history
        try:
            from app.services.history_service import history_service
            await history_service.save(d["user_id"], {
                "query": d["query"], "industry": d.get("industry", "general"),
                "result": result, "columns": d.get("columns", []),
                "row_count": len(d.get("data") or []),
                "col_count": len(d.get("columns") or []),
                "data_preview": (d.get("data") or [])[:50],
                "source": "scheduled",
            }, save_as="auto")
            delivered["in_app"] = True
        except Exception as e:
            logger.error(f"History save for schedule failed: {e}")

        # 2. Email + SMS if enabled
        from app.services.notify_service import notify_service
        summary = _summarise(result, d)
        if chan.get("email") and chan.get("email_to"):
            html = _email_html(d, result)
            delivered["email"] = notify_service.send_email(
                chan["email_to"], f"DataMind report: {d.get('name')}", html)
        if chan.get("sms") and chan.get("sms_to"):
            delivered["sms"] = notify_service.send_sms(chan["sms_to"], summary)

        # Build an honest status that reports what SUCCEEDED, not just failures.
        ok_parts, fail_parts = [], []
        if delivered["in_app"]:
            ok_parts.append("saved to history")
        if chan.get("email"):
            if delivered["email"] and delivered["email"].get("success"):
                ok_parts.append("emailed")
            elif delivered["email"]:
                fail_parts.append("email needs a verified domain")
        if chan.get("sms"):
            if delivered["sms"] and delivered["sms"].get("success"):
                ok_parts.append("texted")
            elif delivered["sms"]:
                fail_parts.append("sms failed: " + str(delivered["sms"].get("error", ""))[:60])

        status = " · ".join(ok_parts) if ok_parts else "ran"
        if fail_parts:
            status += " (" + "; ".join(fail_parts) + ")"

        await col.update_one({"_id": d["_id"]}, {"$set": {
            "last_run": now(), "last_status": status,
            "next_run": _next_run(d["cron"]),
            "run_count": d.get("run_count", 0) + 1,
        }})
        return {"success": True, "status": status, "delivered": delivered}

    async def _analyse(self, d: dict) -> dict:
        """Reuse the analysis engine on the stored data."""
        try:
            import pandas as pd
            from app.services.analysis_service import analysis_service
            df = pd.DataFrame(d.get("data") or [])
            df, _ = analysis_service.clean_data(df)
            desc = analysis_service.describe(df)
            narrative = ""
            try:
                # Best-effort LLM narrative; fall back to a plain summary
                from app.services.llm_service import llm_service
                narrative = llm_service.analyse(df, d["query"], d.get("industry", "general")).get("narrative", "")
            except Exception:
                narrative = _plain_summary(df, d["query"])
            return {"narrative": narrative or _plain_summary(df, d["query"]),
                    "provider": "scheduled", "metrics": desc.get("metrics", []) if isinstance(desc, dict) else [],
                    "insights": []}
        except Exception as e:
            logger.error(f"Schedule analysis failed: {e}")
            return {"narrative": f"Could not analyse the scheduled data: {e}",
                    "provider": "scheduled", "metrics": [], "insights": []}


# ── helpers ───────────────────────────────────────────────────────────────────
def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v

def _plain_summary(df, query):
    try:
        rows, cols = df.shape
        return f"Report for “{query}”: {rows} rows across {cols} columns analysed on {now():%d %b %Y}."
    except Exception:
        return f"Report for “{query}”."

def _summarise(result: dict, d: dict) -> str:
    narr = (result.get("narrative") or "").strip()
    head = f"DataMind: {d.get('name')}\n"
    body = narr[:360] if narr else "Your scheduled report is ready. Open DataMind to view it."
    return head + body

def _email_html(d: dict, result: dict) -> str:
    narr = (result.get("narrative") or "").replace("\n", "<br>")
    name = d.get("name") or "Scheduled report"
    return f"""<div style="font-family:Inter,Arial,sans-serif;max-width:600px;margin:0 auto;color:#0f1a2e">
      <div style="background:#0b1221;padding:20px 24px;border-radius:12px 12px 0 0">
        <span style="color:#00c8be;font-weight:800;font-size:18px">DataMind</span>
        <span style="color:#7d93b3;font-size:13px"> · scheduled report</span>
      </div>
      <div style="border:1px solid #e2e8f0;border-top:none;border-radius:0 0 12px 12px;padding:24px">
        <h2 style="margin:0 0 4px;font-size:18px">{name}</h2>
        <p style="color:#64748b;font-size:12px;margin:0 0 16px">Question: {d.get('query')} · {now():%d %b %Y}</p>
        <div style="font-size:14px;line-height:1.6;color:#334155">{narr}</div>
        <p style="margin-top:22px;font-size:11px;color:#94a3b8">Sent automatically by DataMind for NkaySolutions.</p>
      </div></div>"""


schedule_service = ScheduleService()
