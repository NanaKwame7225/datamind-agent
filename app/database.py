"""
DataMind Agent — MongoDB connection layer

A single shared Motor (async) client for the whole app. Reads the connection
string from settings.MONGODB_URI or the MONGODB_URI environment variable, so it
works whether or not config/settings.py has been updated.

If no URI is configured, the app still runs — auth and history simply report
"not configured" rather than crashing. This keeps guest mode and stateless
analysis working even before you add the database.
"""
from __future__ import annotations
import os, logging

logger = logging.getLogger(__name__)

_client = None
_db = None
_init_error = None


def _uri() -> str | None:
    # Prefer settings, fall back to env, so this works without editing settings.py
    try:
        from config.settings import settings
        v = getattr(settings, "MONGODB_URI", None) or getattr(settings, "MONGO_URI", None)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")


def _db_name() -> str:
    try:
        from config.settings import settings
        v = getattr(settings, "MONGODB_DB", None)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get("MONGODB_DB", "datamind_db")


def is_configured() -> bool:
    return bool(_uri())


async def connect():
    """Open the connection and confirm it works. Safe to call more than once."""
    global _client, _db, _init_error
    if _db is not None:
        return _db
    uri = _uri()
    if not uri:
        _init_error = "No MONGODB_URI configured."
        logger.warning("MongoDB not configured — auth and saved history are disabled. "
                       "Add MONGODB_URI in Railway to enable them.")
        return None
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=8000, uuidRepresentation="standard")
        # Force a round-trip so a bad URI fails loudly at startup, not mid-request
        await _client.admin.command("ping")
        _db = _client[_db_name()]
        await _ensure_indexes(_db)
        logger.info(f"MongoDB connected — database '{_db_name()}'")
        _init_error = None
        return _db
    except Exception as e:
        _init_error = str(e)
        logger.error(f"MongoDB connection failed: {e}")
        _client = None
        _db = None
        return None


async def _ensure_indexes(db):
    """Indexes that make lookups fast and enforce uniqueness."""
    try:
        await db.users.create_index("email", unique=True)
        await db.analyses.create_index([("user_id", 1), ("created_at", -1)])
        await db.analyses.create_index("created_at")
    except Exception as e:
        logger.warning(f"Index creation skipped: {e}")


def get_db():
    """Synchronous accessor for code paths that already know we're connected."""
    return _db


def status() -> dict:
    return {
        "configured": is_configured(),
        "connected": _db is not None,
        "database": _db_name() if _db is not None else None,
        "error": _init_error,
    }


async def close():
    global _client, _db
    if _client:
        _client.close()
    _client = None
    _db = None
