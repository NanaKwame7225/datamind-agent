import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import logging
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DataMind Agent API", version="2.0.0", docs_url="/docs")

# CORS must be added BEFORE routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
from app.routers import analysis, pipeline, connectors, upload, export

try:
    from app.routers import finance
    app.include_router(finance.router, prefix="/api/v1/finance", tags=["Finance"])
    logger.info("Finance router loaded successfully")
except Exception as e:
    logger.error(f"Finance router failed: {e}", exc_info=True)

app.include_router(analysis.router,   prefix="/api/v1/analysis",   tags=["Analysis"])
app.include_router(pipeline.router,   prefix="/api/v1/pipeline",   tags=["Pipeline"])
app.include_router(connectors.router, prefix="/api/v1/connectors", tags=["Connectors"])
app.include_router(upload.router,     prefix="/api/v1/upload",     tags=["Upload"])
app.include_router(export.router,     prefix="/api/v1/export",     tags=["Export"])


# ── Chat endpoint (used by AI Copilot + Report Generator) ─────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system: str = ""

@app.post("/api/v1/chat", tags=["Chat"])
async def chat(req: ChatRequest):
    """
    AI chat pass-through. Tries providers in order:
      1. Groq   (GROQ_API_KEY)
      2. Anthropic (ANTHROPIC_API_KEY)
      3. OpenAI    (OPENAI_API_KEY)
    Returns: { "reply": "...", "provider": "groq|anthropic|openai" }
    """
    groq_key      = os.getenv("GROQ_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key    = os.getenv("OPENAI_API_KEY")

    messages_payload = [{"role": m.role, "content": m.content} for m in req.messages]

    # ── 1. Groq ────────────────────────────────────────────────────────────────
    if groq_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "max_tokens": 1000,
                        "messages": [
                            {"role": "system", "content": req.system},
                            *messages_payload,
                        ],
                    },
                )
                r.raise_for_status()
                data = r.json()
                reply = data["choices"][0]["message"]["content"]
                logger.info("Chat served via Groq")
                return {"reply": reply, "provider": "groq"}
        except Exception as e:
            logger.warning(f"Groq failed, trying next provider: {e}")

    # ── 2. Anthropic ───────────────────────────────────────────────────────────
    if anthropic_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1000,
                        "system": req.system,
                        "messages": messages_payload,
                    },
                )
                r.raise_for_status()
                data = r.json()
                reply = data["content"][0]["text"]
                logger.info("Chat served via Anthropic")
                return {"reply": reply, "provider": "anthropic"}
        except Exception as e:
            logger.warning(f"Anthropic failed, trying next provider: {e}")

    # ── 3. OpenAI ──────────────────────────────────────────────────────────────
    if openai_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o",
                        "max_tokens": 1000,
                        "messages": [
                            {"role": "system", "content": req.system},
                            *messages_payload,
                        ],
                    },
                )
                r.raise_for_status()
                data = r.json()
                reply = data["choices"][0]["message"]["content"]
                logger.info("Chat served via OpenAI")
                return {"reply": reply, "provider": "openai"}
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")

    # ── No keys configured ─────────────────────────────────────────────────────
    logger.error("No AI provider keys configured in Railway environment variables")
    return {
        "reply": "No AI provider keys are configured on the server. Add GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY to your Railway environment variables.",
        "provider": "none",
    }


# ── Core routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"service": "DataMind Agent", "version": "2.0.0", "status": "online"}

@app.get("/health")
async def health():
    providers = {
        "groq":      bool(os.getenv("GROQ_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai":    bool(os.getenv("OPENAI_API_KEY")),
    }
    active = [k for k, v in providers.items() if v]
    return {
        "status":    "healthy",
        "version":   "2.0.0",
        "providers": providers,
        "active":    active,
        "chat":      len(active) > 0,
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )
