"""
DataMind Agent — Transcription Router

POST /api/v1/transcribe        — multipart audio file  (field: "audio")
POST /api/v1/transcribe/base64 — JSON {audio, mime}     (base64 audio)
GET  /api/v1/transcribe/health — is voice fallback available?
"""
import logging, traceback
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


def _svc():
    from app.services.transcription_service import transcription_service
    return transcription_service


@router.post("")
@router.post("/")
async def transcribe(audio: UploadFile = File(...), mime: Optional[str] = Form(None)):
    """Transcribe an uploaded audio clip (multipart)."""
    try:
        data = await audio.read()
        content_mime = mime or audio.content_type or "audio/webm"
        return _svc().transcribe(data, content_mime)
    except Exception as e:
        logger.error(f"Transcription failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": f"Could not process the audio: {e}"}


class B64Audio(BaseModel):
    audio: str
    mime: Optional[str] = "audio/webm"


@router.post("/base64")
async def transcribe_base64(req: B64Audio):
    """Transcribe base64-encoded audio (JSON)."""
    try:
        return _svc().transcribe_base64(req.audio, req.mime or "audio/webm")
    except Exception as e:
        logger.error(f"Transcription (base64) failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": f"Could not process the audio: {e}"}


@router.get("/health")
async def health():
    """Report whether server-side voice transcription is available."""
    try:
        from config.settings import settings
        groq = bool(getattr(settings, "GROQ_API_KEY", None))
        google = bool(getattr(settings, "GOOGLE_API_KEY", None))
        return {
            "available": groq or google,
            "providers": {"groq_whisper": groq, "gemini": google},
            "note": ("Voice input works on all devices."
                     if (groq or google)
                     else "Add GROQ_API_KEY or GOOGLE_API_KEY to enable voice on browsers without the Web Speech API."),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}
