"""
DataMind Agent — Voice Transcription Service

Turns a short audio clip into text so voice input works on EVERY device,
including browsers without the Web Speech API (Firefox, many Android WebViews,
in-app browsers). The frontend uses the browser API when present and falls
back to this endpoint everywhere else.

Provider chain (reuses keys already in Railway):
  1. Groq Whisper large-v3  — most accurate for speech-to-text, very fast
  2. Gemini 2.0 Flash        — native audio understanding, already configured

No new API key required if either GROQ_API_KEY or GOOGLE_API_KEY is set.
"""
from __future__ import annotations
import base64, io, logging

logger = logging.getLogger(__name__)

MAX_AUDIO_MB = 20                 # a spoken question is a few hundred KB
MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024


class TranscriptionService:

    def transcribe(self, audio_bytes: bytes, mime: str = "audio/webm") -> dict:
        """Transcribe audio bytes to text. Tries Groq, then Gemini."""
        if not audio_bytes:
            return {"success": False, "error": "No audio received."}
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return {"success": False,
                    "error": f"Audio clip is too long (max {MAX_AUDIO_MB}MB). Keep questions short."}

        errors = []

        text = self._try_groq(audio_bytes, mime, errors)
        if text is not None:
            return {"success": True, "text": text, "provider": "groq_whisper"}

        text = self._try_gemini(audio_bytes, mime, errors)
        if text is not None:
            return {"success": True, "text": text, "provider": "gemini"}

        detail = " ".join(errors) if errors else "No transcription provider is configured."
        return {
            "success": False,
            "error": "Could not transcribe the audio.",
            "hint": "Add GROQ_API_KEY or GOOGLE_API_KEY in Railway Variables to enable voice on all devices.",
            "detail": detail,
        }

    # ── Groq Whisper ──────────────────────────────────────────────────────────
    def _try_groq(self, audio_bytes: bytes, mime: str, errors: list) -> str | None:
        try:
            from config.settings import settings
            key = getattr(settings, "GROQ_API_KEY", None)
            if not key:
                return None
            try:
                from groq import Groq
            except ImportError:
                errors.append("groq package not installed.")
                return None

            client = Groq(api_key=key)
            ext = self._ext_for(mime)
            fileobj = (f"audio.{ext}", io.BytesIO(audio_bytes))
            resp = client.audio.transcriptions.create(
                file=fileobj,
                model="whisper-large-v3",
                response_format="text",
            )
            text = (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()
            return text or ""
        except Exception as e:
            logger.warning(f"Groq transcription failed: {e}")
            errors.append(f"Groq: {e}")
            return None

    # ── Gemini native audio ───────────────────────────────────────────────────
    def _try_gemini(self, audio_bytes: bytes, mime: str, errors: list) -> str | None:
        try:
            from config.settings import settings
            key = getattr(settings, "GOOGLE_API_KEY", None)
            if not key:
                return None
            try:
                import google.generativeai as genai
            except ImportError:
                errors.append("google-generativeai not installed.")
                return None

            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content([
                {"mime_type": self._clean_mime(mime), "data": audio_bytes},
                "Transcribe this audio to plain text. Return ONLY the words spoken, "
                "with no preamble, quotes, or commentary. If silent, return nothing.",
            ])
            text = (getattr(resp, "text", "") or "").strip()
            # Strip any accidental quotes the model wraps around the transcript
            if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
                text = text[1:-1].strip()
            return text
        except Exception as e:
            logger.warning(f"Gemini transcription failed: {e}")
            errors.append(f"Gemini: {e}")
            return None

    # ── helpers ───────────────────────────────────────────────────────────────
    def _clean_mime(self, mime: str) -> str:
        # "audio/webm;codecs=opus" -> "audio/webm"
        return (mime or "audio/webm").split(";")[0].strip()

    def _ext_for(self, mime: str) -> str:
        m = self._clean_mime(mime)
        return {
            "audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4",
            "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
            "audio/aac": "aac", "audio/flac": "flac", "audio/m4a": "m4a",
        }.get(m, "webm")

    def transcribe_base64(self, b64: str, mime: str = "audio/webm") -> dict:
        """Convenience for JSON payloads that carry base64 audio."""
        try:
            if "," in b64 and b64.strip().startswith("data:"):
                b64 = b64.split(",", 1)[1]        # strip data URL prefix
            audio = base64.b64decode(b64)
        except Exception as e:
            return {"success": False, "error": f"Invalid audio encoding: {e}"}
        return self.transcribe(audio, mime)


transcription_service = TranscriptionService()
