# api/analyse_routes.py — TOP SECTION (replace imports block)

from fastapi import APIRouter
from typing import Any
from pydantic import BaseModel
import time, statistics, os

router = APIRouter()

# ── Lazy AI clients (initialised on first request, not at import time) ──────
# This prevents Railway healthcheck failures if env vars load after process start
_claude_client = None
_gemini_model  = None

def get_claude():
    global _claude_client
    if _claude_client is None:
        import anthropic
        _claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _claude_client

def get_gemini():
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY",""))
        _gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    return _gemini_model
