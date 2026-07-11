"""
DataMind Agent — Auth Service

Email + password accounts with JWT, plus lightweight guest sessions.

Design choices:
  • Passwords hashed with bcrypt (passlib) — never stored in plain text.
  • JWT signed with a secret from settings/env; tokens carry the user id + role.
  • Guest users get a real token too, so the frontend treats everyone the same.
    A guest can later "claim" their account by registering, which upgrades the
    same record and keeps their saved history.
  • Every function degrades gracefully if the database isn't configured, so the
    app never hard-crashes just because Mongo is missing.
"""
from __future__ import annotations
import os, re, uuid, logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

JWT_ALGO = "HS256"
GUEST_DAYS = 30
USER_DAYS = 30

_pwd_ctx = None
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _secret() -> str:
    try:
        from config.settings import settings
        v = getattr(settings, "JWT_SECRET", None) or getattr(settings, "SECRET_KEY", None)
        if v:
            return v
    except Exception:
        pass
    # Fall back to env; last-resort default keeps dev working but is flagged.
    return os.environ.get("JWT_SECRET") or os.environ.get("SECRET_KEY") or "datamind-dev-secret-change-me"


def _ctx():
    global _pwd_ctx
    if _pwd_ctx is None:
        from passlib.context import CryptContext
        _pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return _pwd_ctx


def now() -> datetime:
    return datetime.now(timezone.utc)


# ── Password helpers ──────────────────────────────────────────────────────────
# bcrypt only hashes the first 72 bytes and raises on longer input, so we clamp
# before hashing AND before verifying, keeping the two consistent.
def _clamp(pw: str) -> bytes:
    return (pw or "").encode("utf-8")[:72]

def hash_password(pw: str) -> str:
    return _ctx().hash(_clamp(pw))

def verify_password(pw: str, hashed: str) -> bool:
    try:
        return _ctx().verify(_clamp(pw), hashed)
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────
def make_token(user_id: str, role: str = "user", days: int = USER_DAYS) -> str:
    from jose import jwt
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now(),
        "exp": now() + timedelta(days=days),
    }
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGO)

def decode_token(token: str) -> dict | None:
    from jose import jwt, JWTError
    try:
        return jwt.decode(token, _secret(), algorithms=[JWT_ALGO])
    except JWTError:
        return None


# ── Serialisation ─────────────────────────────────────────────────────────────
def _public(user: dict) -> dict:
    """Strip secrets before returning a user to the client."""
    if not user:
        return {}
    return {
        "id": user.get("_id"),
        "email": user.get("email"),
        "name": user.get("name"),
        "role": user.get("role", "user"),
        "is_guest": user.get("is_guest", False),
        "created_at": (user.get("created_at").isoformat()
                       if isinstance(user.get("created_at"), datetime) else user.get("created_at")),
    }


class AuthService:

    async def _users(self):
        from app.database import connect
        db = await connect()
        return db.users if db is not None else None

    # ── Registration ─────────────────────────────────────────────────────────
    async def register(self, email: str, password: str, name: str = "",
                       guest_id: str = None) -> dict:
        email = (email or "").strip().lower()
        if not EMAIL_RE.match(email):
            return {"success": False, "error": "Enter a valid email address."}
        if not password or len(password) < 8:
            return {"success": False, "error": "Password must be at least 8 characters."}

        users = await self._users()
        if users is None:
            return {"success": False, "error": "Accounts are unavailable — the database isn't configured.",
                    "hint": "Add MONGODB_URI in Railway to enable sign-up."}

        existing = await users.find_one({"email": email})
        if existing and not existing.get("is_guest"):
            return {"success": False, "error": "An account with that email already exists. Try logging in."}

        # If a guest is registering, upgrade their existing record so their
        # saved analyses carry over.
        if guest_id:
            guest = await users.find_one({"_id": guest_id, "is_guest": True})
            if guest:
                await users.update_one({"_id": guest_id}, {"$set": {
                    "email": email, "name": name or email.split("@")[0],
                    "password": hash_password(password),
                    "is_guest": False, "upgraded_at": now(),
                }})
                user = await users.find_one({"_id": guest_id})
                return {"success": True, "token": make_token(guest_id, user.get("role", "user")),
                        "user": _public(user), "upgraded": True}

        uid = str(uuid.uuid4())
        doc = {
            "_id": uid, "email": email, "name": name or email.split("@")[0],
            "password": hash_password(password), "role": "user",
            "is_guest": False, "created_at": now(),
        }
        await users.insert_one(doc)
        return {"success": True, "token": make_token(uid), "user": _public(doc)}

    # ── Login ────────────────────────────────────────────────────────────────
    async def login(self, email: str, password: str) -> dict:
        email = (email or "").strip().lower()
        users = await self._users()
        if users is None:
            return {"success": False, "error": "Login is unavailable — the database isn't configured."}

        user = await users.find_one({"email": email})
        if not user or user.get("is_guest") or not verify_password(password, user.get("password", "")):
            return {"success": False, "error": "Wrong email or password."}

        await users.update_one({"_id": user["_id"]}, {"$set": {"last_login": now()}})
        return {"success": True, "token": make_token(user["_id"], user.get("role", "user")),
                "user": _public(user)}

    # ── Guest session ────────────────────────────────────────────────────────
    async def guest(self) -> dict:
        """Create a throwaway guest account so the app works without sign-up."""
        users = await self._users()
        uid = "guest_" + uuid.uuid4().hex[:16]
        if users is None:
            # No DB: still hand back a token so the UI works, just unsaved.
            return {"success": True, "token": make_token(uid, "guest", GUEST_DAYS),
                    "user": {"id": uid, "name": "Guest", "is_guest": True, "role": "guest"},
                    "ephemeral": True}
        doc = {"_id": uid, "email": None, "name": "Guest", "password": None,
               "role": "guest", "is_guest": True, "created_at": now()}
        await users.insert_one(doc)
        return {"success": True, "token": make_token(uid, "guest", GUEST_DAYS),
                "user": _public(doc)}

    # ── Lookup by token ──────────────────────────────────────────────────────
    async def user_from_token(self, token: str) -> dict | None:
        payload = decode_token(token)
        if not payload:
            return None
        uid = payload.get("sub")
        users = await self._users()
        if users is None:
            # DB gone but token valid — reconstruct a minimal identity.
            return {"_id": uid, "role": payload.get("role", "user"),
                    "is_guest": payload.get("role") == "guest", "name": "Guest"}
        return await users.find_one({"_id": uid})

    async def me(self, token: str) -> dict:
        user = await self.user_from_token(token)
        if not user:
            return {"success": False, "error": "Session expired. Please log in again."}
        return {"success": True, "user": _public(user)}


auth_service = AuthService()
