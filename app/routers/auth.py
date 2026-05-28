"""
DataMind Agent v2 — Auth Router
POST /api/v2/auth/register
POST /api/v2/auth/login
POST /api/v2/auth/refresh
GET  /api/v2/auth/me
PUT  /api/v2/auth/me
POST /api/v2/auth/logout
GET  /api/v2/auth/usage
"""
from fastapi import APIRouter, HTTPException, Depends
from app.models.user import UserCreate, UserLogin, UserOut, TokenResponse
from app.services.auth_service import (
    create_user, authenticate_user, create_access_token,
    create_refresh_token, get_current_user, get_usage_stats,
    get_db, get_user_by_id
)
import logging
from datetime import datetime, timezone

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(data: UserCreate):
    """Create a new account."""
    user = create_user(data)
    access_token  = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id)
    logger.info(f"New registration: {data.email}")
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=60 * 60 * 8,
        user=user,
    )


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin):
    """Log in with email and password."""
    user_dict = authenticate_user(data.email, data.password)
    if not user_dict:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    user = get_user_by_id(user_dict["id"])
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled. Contact support.")
    access_token  = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id)
    logger.info(f"Login: {data.email}")
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=60 * 60 * 8,
        user=user,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """Get a new access token using a refresh token."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM refresh_tokens WHERE token=? AND revoked=0", (refresh_token,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    row = dict(row)
    expires = datetime.fromisoformat(row["expires_at"])
    if expires < datetime.now(timezone.utc):
        conn.close()
        raise HTTPException(status_code=401, detail="Refresh token has expired. Please log in again.")
    # Revoke old refresh token (rotation)
    conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE token=?", (refresh_token,))
    conn.commit()
    conn.close()
    user = get_user_by_id(row["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    new_access  = create_access_token(user.id, user.email)
    new_refresh = create_refresh_token(user.id)
    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=60 * 60 * 8,
        user=user,
    )


@router.get("/me", response_model=UserOut)
async def get_me(current_user: UserOut = Depends(get_current_user)):
    """Get your profile."""
    return current_user


@router.put("/me", response_model=UserOut)
async def update_me(
    updates: dict,
    current_user: UserOut = Depends(get_current_user)
):
    """Update your profile (full_name, company, industry)."""
    allowed = {"full_name", "company", "industry"}
    safe = {k: v for k, v in updates.items() if k in allowed}
    if not safe:
        raise HTTPException(400, "No valid fields to update")
    conn = get_db()
    sets = ", ".join([f"{k}=?" for k in safe])
    conn.execute(f"UPDATE users SET {sets} WHERE id=?", list(safe.values()) + [current_user.id])
    conn.commit()
    conn.close()
    return get_user_by_id(current_user.id)


@router.post("/logout")
async def logout(
    refresh_token: str,
    current_user: UserOut = Depends(get_current_user)
):
    """Revoke refresh token on logout."""
    conn = get_db()
    conn.execute(
        "UPDATE refresh_tokens SET revoked=1 WHERE token=? AND user_id=?",
        (refresh_token, current_user.id)
    )
    conn.commit()
    conn.close()
    return {"message": "Logged out successfully"}


@router.get("/usage")
async def get_usage(current_user: UserOut = Depends(get_current_user)):
    """Get your usage statistics for this month."""
    return get_usage_stats(current_user.id)
