"""
DataMind Agent — Superadmin Router

Platform-wide oversight. Every route is gated behind require_superadmin, which
accepts EITHER:
  • a user whose role is "superadmin", or
  • the ADMIN_API_KEY passed as the X-Admin-Key header (break-glass access).

GET    /api/v1/admin/stats                — headline platform numbers
GET    /api/v1/admin/users                — list users (search, sort, paginate)
PATCH  /api/v1/admin/users/{uid}          — suspend / unsuspend / change role
DELETE /api/v1/admin/users/{uid}          — soft delete
POST   /api/v1/admin/users/{uid}/password — set a new password
GET    /api/v1/admin/usage                — LLM token/cost usage by user & provider
GET    /api/v1/admin/workspaces           — every workspace with owner + counts
GET    /api/v1/admin/health               — provider keys, DB, error counts
"""
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


def now():
    return datetime.now(timezone.utc)


# ── Gate ──────────────────────────────────────────────────────────────────────

async def require_superadmin(authorization: Optional[str] = Header(None),
                             x_admin_key: Optional[str] = Header(None)) -> dict:
    """
    Superadmin access. Either a superadmin-role user token, or the
    ADMIN_API_KEY header for emergency/bootstrap access.
    """
    admin_key = os.getenv("ADMIN_API_KEY")
    if admin_key and x_admin_key and x_admin_key == admin_key:
        return {"_id": "admin-key", "role": "superadmin", "via": "api_key"}

    from app.services.auth_service import decode_token
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in as an administrator.")
    payload = decode_token(authorization.split(" ", 1)[1].strip())
    if not payload:
        raise HTTPException(401, "Session expired. Sign in again.")
    if payload.get("role") != "superadmin":
        raise HTTPException(403, "Administrator access only.")
    return {"_id": payload.get("sub"), "role": "superadmin", "via": "token"}


async def _db():
    from app.database import connect
    db = await connect()
    if db is None:
        raise HTTPException(503, "Database unavailable.")
    return db


def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v


# ── Headline stats ────────────────────────────────────────────────────────────

@router.get("/stats")
async def platform_stats(admin: dict = Depends(require_superadmin)):
    db = await _db()
    since_24h = now() - timedelta(hours=24)
    since_7d = now() - timedelta(days=7)
    try:
        total_users = await db.users.count_documents({"is_guest": {"$ne": True}})
        guests = await db.users.count_documents({"is_guest": True})
        suspended = await db.users.count_documents({"suspended": True})
        new_7d = await db.users.count_documents({"created_at": {"$gte": since_7d}})
        workspaces = await db.workspaces.count_documents({})
        analyses = await db.analyses.count_documents({})
        analyses_24h = await db.analyses.count_documents({"created_at": {"$gte": since_24h}})
        schedules = await db.schedules.count_documents({})

        tokens_24h = 0
        try:
            cur = db.usage_log.aggregate([
                {"$match": {"at": {"$gte": since_24h}}},
                {"$group": {"_id": None, "t": {"$sum": "$tokens"}}},
            ])
            async for r in cur:
                tokens_24h = int(r.get("t") or 0)
        except Exception:
            pass

        return {"success": True, "stats": {
            "users": total_users, "guests": guests, "suspended": suspended,
            "new_users_7d": new_7d, "workspaces": workspaces,
            "analyses_total": analyses, "analyses_24h": analyses_24h,
            "schedules": schedules, "tokens_24h": tokens_24h,
        }}
    except Exception as e:
        logger.error(f"admin stats failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ── Users ─────────────────────────────────────────────────────────────────────

@router.get("/users")
async def list_users(admin: dict = Depends(require_superadmin),
                     q: Optional[str] = Query(None, description="search email or name"),
                     role: Optional[str] = None,
                     include_guests: bool = False,
                     sort: str = "created_at",
                     order: int = -1,
                     limit: int = Query(50, ge=1, le=200),
                     skip: int = Query(0, ge=0)):
    db = await _db()
    flt = {}
    if not include_guests:
        flt["is_guest"] = {"$ne": True}
    if role:
        flt["role"] = role
    if q:
        flt["$or"] = [{"email": {"$regex": q, "$options": "i"}},
                      {"name": {"$regex": q, "$options": "i"}}]
    sort_field = sort if sort in ("created_at", "email", "name", "role", "last_seen") else "created_at"
    try:
        total = await db.users.count_documents(flt)
        cur = db.users.find(flt).sort(sort_field, order).skip(skip).limit(limit)
        items = []
        async for u in cur:
            uid = u.get("_id")
            # Cheap per-user counts
            try:
                ws_count = await db.workspace_members.count_documents({"user_id": uid})
            except Exception:
                ws_count = 0
            try:
                an_count = await db.analyses.count_documents({"user_id": uid})
            except Exception:
                an_count = 0
            items.append({
                "id": uid,
                "email": u.get("email"),
                "name": u.get("name"),
                "role": u.get("role", "user"),
                "is_guest": u.get("is_guest", False),
                "suspended": bool(u.get("suspended")),
                "deleted": bool(u.get("deleted")),
                "created_at": _iso(u.get("created_at")),
                "last_seen": _iso(u.get("last_seen")),
                "workspaces": ws_count,
                "analyses": an_count,
            })
        return {"success": True, "total": total, "items": items,
                "limit": limit, "skip": skip}
    except Exception as e:
        logger.error(f"admin list_users failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


class UserPatch(BaseModel):
    suspended: Optional[bool] = None
    role: Optional[str] = None
    name: Optional[str] = None


@router.patch("/users/{uid}")
async def update_user(uid: str, body: UserPatch,
                      admin: dict = Depends(require_superadmin)):
    db = await _db()
    update = {}
    if body.suspended is not None:
        update["suspended"] = bool(body.suspended)
    if body.name is not None:
        update["name"] = str(body.name)[:120]
    if body.role is not None:
        if body.role not in ("user", "guest", "superadmin"):
            raise HTTPException(400, "Role must be user, guest, or superadmin.")
        update["role"] = body.role
    if not update:
        raise HTTPException(400, "Nothing to update.")
    update["updated_at"] = now()
    r = await db.users.update_one({"_id": uid}, {"$set": update})
    if r.matched_count == 0:
        raise HTTPException(404, "User not found.")
    await _audit(db, admin, "update_user", uid, update)
    return {"success": True, "updated": {k: v for k, v in update.items() if k != "updated_at"}}


@router.delete("/users/{uid}")
async def delete_user(uid: str, hard: bool = False,
                      admin: dict = Depends(require_superadmin)):
    """Soft delete by default (recoverable). hard=true removes the record."""
    db = await _db()
    if hard:
        r = await db.users.delete_one({"_id": uid})
        if r.deleted_count == 0:
            raise HTTPException(404, "User not found.")
        await db.workspace_members.delete_many({"user_id": uid})
        await _audit(db, admin, "hard_delete_user", uid, {})
        return {"success": True, "deleted": "permanent"}
    r = await db.users.update_one(
        {"_id": uid}, {"$set": {"deleted": True, "suspended": True, "deleted_at": now()}})
    if r.matched_count == 0:
        raise HTTPException(404, "User not found.")
    await _audit(db, admin, "soft_delete_user", uid, {})
    return {"success": True, "deleted": "soft"}


class PasswordReset(BaseModel):
    new_password: str


@router.post("/users/{uid}/password")
async def reset_password(uid: str, body: PasswordReset,
                         admin: dict = Depends(require_superadmin)):
    if not body.new_password or len(body.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    from app.services.auth_service import hash_password
    db = await _db()
    r = await db.users.update_one(
        {"_id": uid}, {"$set": {"password": hash_password(body.new_password),
                                "password_reset_at": now()}})
    if r.matched_count == 0:
        raise HTTPException(404, "User not found.")
    await _audit(db, admin, "reset_password", uid, {})
    return {"success": True, "message": "Password updated."}


# ── Usage & cost ──────────────────────────────────────────────────────────────

# Rough per-1k-token costs (USD) for a sane cost estimate. Adjust as pricing moves.
TOKEN_COST_PER_1K = {
    "Claude Sonnet 4": 0.009, "GPT-4o": 0.0075, "Gemini 2.0 Flash": 0.0005,
    "Llama 3.3 70B": 0.0006, "Mistral Large": 0.006, "Command R+": 0.005,
}


@router.get("/usage")
async def usage(admin: dict = Depends(require_superadmin),
                days: int = Query(30, ge=1, le=365)):
    db = await _db()
    since = now() - timedelta(days=days)
    out = {"by_user": [], "by_provider": [], "by_day": [],
           "totals": {"tokens": 0, "requests": 0, "estimated_cost_usd": 0.0}}
    try:
        # By user
        cur = db.usage_log.aggregate([
            {"$match": {"at": {"$gte": since}}},
            {"$group": {"_id": "$user_id", "tokens": {"$sum": "$tokens"},
                        "requests": {"$sum": 1}}},
            {"$sort": {"tokens": -1}}, {"$limit": 50},
        ])
        async for r in cur:
            uid = r["_id"]
            u = await db.users.find_one({"_id": uid}) if uid else None
            out["by_user"].append({
                "user_id": uid,
                "email": (u or {}).get("email") or ("guest" if uid else "unknown"),
                "name": (u or {}).get("name"),
                "tokens": int(r.get("tokens") or 0),
                "requests": int(r.get("requests") or 0),
            })

        # By provider (with cost estimate)
        cur = db.usage_log.aggregate([
            {"$match": {"at": {"$gte": since}}},
            {"$group": {"_id": "$provider", "tokens": {"$sum": "$tokens"},
                        "requests": {"$sum": 1}}},
            {"$sort": {"tokens": -1}},
        ])
        total_tokens = total_reqs = 0
        total_cost = 0.0
        async for r in cur:
            prov = r["_id"] or "unknown"
            tk = int(r.get("tokens") or 0)
            rq = int(r.get("requests") or 0)
            cost = round(tk / 1000.0 * TOKEN_COST_PER_1K.get(prov, 0.004), 4)
            total_tokens += tk
            total_reqs += rq
            total_cost += cost
            out["by_provider"].append({"provider": prov, "tokens": tk,
                                       "requests": rq, "estimated_cost_usd": cost})

        # By day
        cur = db.usage_log.aggregate([
            {"$match": {"at": {"$gte": since}}},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$at"}},
                        "tokens": {"$sum": "$tokens"}, "requests": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ])
        async for r in cur:
            out["by_day"].append({"day": r["_id"], "tokens": int(r.get("tokens") or 0),
                                  "requests": int(r.get("requests") or 0)})

        out["totals"] = {"tokens": total_tokens, "requests": total_reqs,
                         "estimated_cost_usd": round(total_cost, 2)}
        out["success"] = True
        out["note"] = ("Costs are estimates based on published per-token pricing; "
                       "check your provider dashboards for authoritative billing.")
        return out
    except Exception as e:
        logger.warning(f"admin usage: {e}")
        return {"success": True, **out,
                "note": "No usage recorded yet — usage logging starts with the next request."}


# ── Workspaces ────────────────────────────────────────────────────────────────

@router.get("/workspaces")
async def list_workspaces(admin: dict = Depends(require_superadmin),
                          limit: int = Query(100, ge=1, le=500),
                          skip: int = Query(0, ge=0)):
    db = await _db()
    try:
        total = await db.workspaces.count_documents({})
        cur = db.workspaces.find({}).sort("created_at", -1).skip(skip).limit(limit)
        items = []
        async for w in cur:
            wid = w.get("_id")
            owner = await db.users.find_one({"_id": w.get("owner_id")})
            members = await db.workspace_members.count_documents({"workspace_id": wid})
            try:
                analyses = await db.analyses.count_documents({"workspace_id": wid})
            except Exception:
                analyses = 0
            items.append({
                "id": wid, "name": w.get("name"),
                "owner_id": w.get("owner_id"),
                "owner_email": (owner or {}).get("email"),
                "members": members, "analyses": analyses,
                "created_at": _iso(w.get("created_at")),
            })
        return {"success": True, "total": total, "items": items}
    except Exception as e:
        logger.error(f"admin workspaces failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ── System health ─────────────────────────────────────────────────────────────

@router.get("/health")
async def system_health(admin: dict = Depends(require_superadmin)):
    providers = []
    for label, env in (("Claude Sonnet 4", "ANTHROPIC_API_KEY"),
                       ("GPT-4o", "OPENAI_API_KEY"),
                       ("Gemini 2.0 Flash", ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
                       ("Llama 3.3 70B (Groq)", "GROQ_API_KEY"),
                       ("Mistral Large", "MISTRAL_API_KEY"),
                       ("Command R+", "COHERE_API_KEY")):
        names = env if isinstance(env, tuple) else (env,)
        key = next((os.getenv(n) for n in names if os.getenv(n)), None)
        providers.append({"provider": label,
                          "configured": bool(key),
                          "env_var": names[0],
                          "key_hint": (key[:6] + "…" + key[-4:]) if key and len(key) > 12 else None})

    db_ok, db_err, collections = False, None, {}
    try:
        db = await _db()
        db_ok = True
        for c in ("users", "workspaces", "workspace_members", "analyses",
                  "schedules", "usage_log", "admin_audit"):
            try:
                collections[c] = await db[c].count_documents({})
            except Exception:
                collections[c] = None
    except Exception as e:
        db_err = str(e)

    errors_24h = 0
    try:
        db = await _db()
        errors_24h = await db.error_log.count_documents(
            {"at": {"$gte": now() - timedelta(hours=24)}})
    except Exception:
        pass

    return {"success": True,
            "providers": providers,
            "database": {"connected": db_ok, "error": db_err, "collections": collections},
            "errors_24h": errors_24h,
            "server_time": now().isoformat()}


# ── Audit trail ───────────────────────────────────────────────────────────────

async def _audit(db, admin: dict, action: str, target: str, detail: dict):
    """Record every administrative action. Never raises."""
    try:
        import uuid
        await db.admin_audit.insert_one({
            "_id": str(uuid.uuid4()),
            "at": now(),
            "admin_id": admin.get("_id"),
            "via": admin.get("via"),
            "action": action,
            "target": target,
            "detail": {k: str(v)[:200] for k, v in (detail or {}).items()},
        })
    except Exception as e:
        logger.debug(f"audit write skipped: {e}")


@router.get("/audit")
async def audit_log(admin: dict = Depends(require_superadmin),
                    limit: int = Query(100, ge=1, le=500)):
    db = await _db()
    items = []
    try:
        cur = db.admin_audit.find({}).sort("at", -1).limit(limit)
        async for a in cur:
            items.append({"at": _iso(a.get("at")), "admin_id": a.get("admin_id"),
                          "via": a.get("via"), "action": a.get("action"),
                          "target": a.get("target"), "detail": a.get("detail")})
    except Exception:
        pass
    return {"success": True, "items": items}
