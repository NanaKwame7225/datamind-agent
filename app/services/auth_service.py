"""
DataMind Agent v2 — Authentication Service
JWT access + refresh tokens, bcrypt passwords, SQLite persistence.
Replace SQLite with PostgreSQL for production by changing DB_URL.
"""
from __future__ import annotations
import uuid, logging, sqlite3, json
from datetime import datetime, timedelta, timezone
from typing import Optional
from pathlib import Path

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.user import UserCreate, UserOut, PlanTier, PLAN_LIMITS
from config.settings import settings

logger = logging.getLogger(__name__)

ALGORITHM             = "HS256"
ACCESS_TOKEN_MINUTES  = 60 * 8      # 8 hours
REFRESH_TOKEN_DAYS    = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

DB_PATH = "./datamind_users.db"


# ── Database setup ─────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id            TEXT PRIMARY KEY,
        email         TEXT UNIQUE NOT NULL,
        hashed_pw     TEXT NOT NULL,
        full_name     TEXT NOT NULL,
        company       TEXT,
        industry      TEXT,
        plan          TEXT DEFAULT 'free',
        analyses_used INTEGER DEFAULT 0,
        is_active     INTEGER DEFAULT 1,
        created_at    TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS refresh_tokens (
        token      TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        revoked    INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS analyses (
        id               TEXT PRIMARY KEY,
        user_id          TEXT NOT NULL,
        title            TEXT,
        query            TEXT,
        industry         TEXT,
        provider         TEXT,
        model            TEXT,
        row_count        INTEGER DEFAULT 0,
        col_count        INTEGER DEFAULT 0,
        execution_ms     REAL DEFAULT 0,
        tokens_used      INTEGER DEFAULT 0,
        status           TEXT DEFAULT 'completed',
        has_finance      INTEGER DEFAULT 0,
        fraud_risk_score REAL,
        health_score     REAL,
        narrative        TEXT,
        metrics          TEXT,
        insights         TEXT,
        charts           TEXT,
        pipeline_steps   TEXT,
        preview          TEXT,
        finance_results  TEXT,
        created_at       TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS scheduled_reports (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        name        TEXT NOT NULL,
        industry    TEXT,
        query       TEXT,
        schedule    TEXT NOT NULL,
        recipients  TEXT,
        is_active   INTEGER DEFAULT 1,
        last_run    TEXT,
        next_run    TEXT,
        created_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS usage_log (
        id         TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        period     TEXT NOT NULL,
        event      TEXT,
        tokens     INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id);
    CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_usage_user_period ON usage_log(user_id, period);
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised")


# ── Password ───────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Token creation ─────────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    token = str(uuid.uuid4())
    expires = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_DAYS)
    conn = get_db()
    conn.execute(
        "INSERT INTO refresh_tokens (token, user_id, expires_at) VALUES (?,?,?)",
        (token, user_id, expires.isoformat())
    )
    conn.commit()
    conn.close()
    return token


def verify_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e}")


# ── User CRUD ──────────────────────────────────────────────────────────────────

def create_user(data: UserCreate) -> UserOut:
    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE email=?", (data.email,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO users (id,email,hashed_pw,full_name,company,industry,plan,analyses_used,is_active,created_at) "
        "VALUES (?,?,?,?,?,?,?,0,1,?)",
        (user_id, data.email, hash_password(data.password),
         data.full_name, data.company, data.industry, PlanTier.free.value, now)
    )
    conn.commit()
    conn.close()
    logger.info(f"New user created: {data.email}")
    return get_user_by_id(user_id)


def get_user_by_email(email: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[UserOut]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    row = dict(row)
    plan = PlanTier(row["plan"])
    return UserOut(
        id=row["id"], email=row["email"], full_name=row["full_name"],
        company=row["company"], industry=row["industry"],
        plan=plan, analyses_used=row["analyses_used"],
        analyses_limit=PLAN_LIMITS[plan]["analyses"],
        created_at=datetime.fromisoformat(row["created_at"]),
        is_active=bool(row["is_active"]),
    )


def authenticate_user(email: str, password: str) -> Optional[dict]:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["hashed_pw"]):
        return None
    return user


def increment_usage(user_id: str, tokens: int = 0):
    conn = get_db()
    conn.execute("UPDATE users SET analyses_used = analyses_used + 1 WHERE id=?", (user_id,))
    period = datetime.now(timezone.utc).strftime("%Y-%m")
    conn.execute(
        "INSERT INTO usage_log (id,user_id,period,event,tokens,created_at) VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()), user_id, period, "analysis", tokens, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def check_usage_limit(user_id: str) -> tuple[bool, int, int]:
    """Returns (can_run, used, limit)"""
    conn = get_db()
    row = conn.execute("SELECT analyses_used, plan FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return False, 0, 0
    plan = PlanTier(row["plan"])
    limit = PLAN_LIMITS[plan]["analyses"]
    used = row["analyses_used"]
    return used < limit, used, limit


# ── Analysis history ───────────────────────────────────────────────────────────

def save_analysis(user_id: str, result: dict, finance_result: dict = None) -> str:
    conn = get_db()
    aid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    query = result.get("query", "")
    title = query[:60] + ("..." if len(query) > 60 else "")
    has_finance = 1 if finance_result else 0
    fraud_score = finance_result.get("fraud", {}).get("risk_score") if finance_result else None
    health = finance_result.get("accounting", {}).get("health_score") if finance_result else None

    conn.execute("""
        INSERT INTO analyses
        (id,user_id,title,query,industry,provider,model,row_count,col_count,
         execution_ms,tokens_used,status,has_finance,fraud_risk_score,health_score,
         narrative,metrics,insights,charts,pipeline_steps,preview,finance_results,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        aid, user_id, title, query,
        result.get("industry","general"),
        result.get("provider","unknown"),
        result.get("model","unknown"),
        result.get("row_count", 0),
        result.get("col_count", 0),
        result.get("execution_ms", 0),
        result.get("tokens_used", 0),
        "completed",
        has_finance, fraud_score, health,
        result.get("narrative",""),
        json.dumps(result.get("metrics",[])),
        json.dumps(result.get("insights",[])),
        json.dumps([{k:v for k,v in c.items() if k != "data"} for c in result.get("charts",[])]),
        json.dumps(result.get("pipeline_steps",[])),
        json.dumps(result.get("raw_data_preview",[])),
        json.dumps(finance_result) if finance_result else None,
        now,
    ))
    conn.commit()
    conn.close()
    return aid


def get_analysis_history(user_id: str, limit: int = 20, offset: int = 0) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT id,title,query,industry,provider,model,row_count,col_count,"
        "execution_ms,tokens_used,status,has_finance,fraud_risk_score,health_score,created_at "
        "FROM analyses WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (user_id, limit, offset)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id: str, user_id: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM analyses WHERE id=? AND user_id=?", (analysis_id, user_id)
    ).fetchone()
    conn.close()
    if not row:
        return None
    r = dict(row)
    for field in ["metrics","insights","charts","pipeline_steps","preview","finance_results"]:
        try:
            r[field] = json.loads(r[field]) if r[field] else None
        except Exception:
            r[field] = None
    return r


def get_usage_stats(user_id: str) -> dict:
    conn = get_db()
    period = datetime.now(timezone.utc).strftime("%Y-%m")
    row = conn.execute("SELECT analyses_used, plan FROM users WHERE id=?", (user_id,)).fetchone()
    monthly = conn.execute(
        "SELECT COUNT(*) as cnt, SUM(tokens) as tok FROM usage_log WHERE user_id=? AND period=?",
        (user_id, period)
    ).fetchone()
    conn.close()
    if not row:
        return {}
    plan = PlanTier(row["plan"])
    limit = PLAN_LIMITS[plan]["analyses"]
    used = row["analyses_used"]
    return {
        "user_id": user_id, "period": period,
        "analyses_run": used, "limit": limit,
        "pct_used": round(used / max(limit, 1) * 100, 1),
        "monthly_analyses": monthly["cnt"] or 0,
        "monthly_tokens": monthly["tok"] or 0,
        "plan": plan.value,
    }


# ── FastAPI dependency ─────────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> UserOut:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    payload = verify_access_token(credentials.credentials)
    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return user


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Optional[UserOut]:
    """Returns user if logged in, None otherwise (for public endpoints)."""
    if not credentials:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


auth_service = type("AuthService", (), {
    "init_db": staticmethod(init_db),
    "create_user": staticmethod(create_user),
    "authenticate_user": staticmethod(authenticate_user),
    "create_access_token": staticmethod(create_access_token),
    "create_refresh_token": staticmethod(create_refresh_token),
    "get_user_by_id": staticmethod(get_user_by_id),
    "increment_usage": staticmethod(increment_usage),
    "check_usage_limit": staticmethod(check_usage_limit),
    "save_analysis": staticmethod(save_analysis),
    "get_analysis_history": staticmethod(get_analysis_history),
    "get_analysis_by_id": staticmethod(get_analysis_by_id),
    "get_usage_stats": staticmethod(get_usage_stats),
})()
