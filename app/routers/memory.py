"""
DataMind Agent — Memory Router
GET  /api/v1/memory/context/{session_id}
POST /api/v1/memory/save
DELETE /api/v1/memory/clear/{session_id}
GET  /api/v1/memory/search
"""
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


class SaveMemoryRequest(BaseModel):
    session_id: str
    role: str
    content: str
    industry: Optional[str] = None
    query: Optional[str] = None


class SaveAnalysisRequest(BaseModel):
    session_id: str
    result: dict


@router.get("/context/{session_id}")
async def get_context(session_id: str):
    """Get conversation history and memory context for a session."""
    try:
        from app.services.memory_service import memory_service
        history = memory_service.get_conversation_history(session_id)
        context = memory_service.get_memory_context(session_id)
        user_ctx = memory_service.get_user_context(session_id)
        return {
            "session_id": session_id,
            "history": history,
            "memory_context": context,
            "user_context": user_ctx,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/save")
async def save_message(request: SaveMemoryRequest):
    """Save a message to conversation memory."""
    try:
        from app.services.memory_service import memory_service
        msg_id = memory_service.save_message(
            request.session_id, request.role, request.content,
            request.industry, request.query
        )
        return {"message_id": msg_id, "saved": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/save-analysis")
async def save_analysis(request: SaveAnalysisRequest):
    """Save analysis results to memory for future reference."""
    try:
        from app.services.memory_service import memory_service
        mem_id = memory_service.save_analysis_memory(request.session_id, request.result)
        return {"memory_id": mem_id, "saved": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/clear/{session_id}")
async def clear_memory(session_id: str):
    """Clear all memory for a session."""
    try:
        from app.services.memory_service import memory_service
        memory_service.clear_session(session_id)
        return {"cleared": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/search")
async def search_memory(
    q: str = Query(...),
    session_id: Optional[str] = Query(None),
):
    """Search past analyses by keyword."""
    try:
        from app.services.memory_service import memory_service
        results = memory_service.search_memory(q, session_id)
        return {"query": q, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/benchmarks/{industry}")
async def get_benchmarks(industry: str):
    """Get industry benchmarks from the RAG knowledge base."""
    try:
        from app.services.rag_service import rag_service, BENCHMARKS
        context = rag_service.get_industry_context(industry)
        benchmarks = BENCHMARKS.get(industry, {})
        return {
            "industry": industry,
            "benchmarks": benchmarks,
            "context_block": context,
        }
    except Exception as e:
        raise HTTPException(500, str(e))
