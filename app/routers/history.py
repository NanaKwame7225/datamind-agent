"""
DataMind Agent v2 — History Router
GET    /api/v2/history
GET    /api/v2/history/{id}
DELETE /api/v2/history/{id}
GET    /api/v2/history/stats
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from app.services.auth_service import (
    get_current_user, get_analysis_history,
    get_analysis_by_id, get_db
)
from app.models.user import UserOut
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("")
async def list_history(
    limit: int  = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    industry: str = Query(None),
    current_user: UserOut = Depends(get_current_user),
):
    """List your past analyses, newest first."""
    history = get_analysis_history(current_user.id, limit, offset)
    if industry:
        history = [h for h in history if h.get("industry") == industry]
    return {
        "items": history,
        "count": len(history),
        "offset": offset,
        "limit": limit,
    }


@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    """Retrieve a full saved analysis report."""
    result = get_analysis_by_id(analysis_id, current_user.id)
    if not result:
        raise HTTPException(404, "Analysis not found or does not belong to your account")
    return result


@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    """Delete a saved analysis."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM analyses WHERE id=? AND user_id=?",
        (analysis_id, current_user.id)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Analysis not found")
    conn.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
    conn.commit()
    conn.close()
    return {"message": "Analysis deleted"}


@router.get("/stats/summary")
async def history_stats(current_user: UserOut = Depends(get_current_user)):
    """Summary statistics across all your analyses."""
    conn = get_db()
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            AVG(execution_ms) as avg_ms,
            SUM(tokens_used) as total_tokens,
            COUNT(CASE WHEN has_finance=1 THEN 1 END) as finance_analyses,
            COUNT(CASE WHEN industry='finance' THEN 1 END) as finance_count,
            COUNT(CASE WHEN industry='education' THEN 1 END) as education_count,
            COUNT(CASE WHEN industry='healthcare' THEN 1 END) as healthcare_count,
            MIN(created_at) as first_analysis,
            MAX(created_at) as last_analysis
        FROM analyses WHERE user_id=?
    """, (current_user.id,)).fetchone()

    industry_breakdown = conn.execute("""
        SELECT industry, COUNT(*) as count
        FROM analyses WHERE user_id=?
        GROUP BY industry ORDER BY count DESC
    """, (current_user.id,)).fetchall()

    provider_breakdown = conn.execute("""
        SELECT provider, COUNT(*) as count
        FROM analyses WHERE user_id=?
        GROUP BY provider ORDER BY count DESC
    """, (current_user.id,)).fetchall()

    conn.close()

    return {
        "total_analyses": stats["total"],
        "avg_execution_ms": round(stats["avg_ms"] or 0, 1),
        "total_tokens_used": stats["total_tokens"] or 0,
        "finance_analyses": stats["finance_analyses"],
        "first_analysis": stats["first_analysis"],
        "last_analysis": stats["last_analysis"],
        "industry_breakdown": [dict(r) for r in industry_breakdown],
        "provider_breakdown": [dict(r) for r in provider_breakdown],
    }
