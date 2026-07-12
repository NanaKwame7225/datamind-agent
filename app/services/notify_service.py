"""
DataMind Agent — Notification Channels

Two delivery channels for scheduled reports, plus in-app which needs no sending
(the scheduler just saves to history). Each channel degrades gracefully: if it
isn't configured, it reports that rather than crashing the whole report run.

  • Email — Resend by default (one HTTP POST). Provider-agnostic: set
            EMAIL_PROVIDER=sendgrid|brevo to switch, or leave it on resend.
  • SMS   — Mnotify (already used across NkaySolutions apps).

No secrets are hard-coded; everything reads from settings/env.
"""
from __future__ import annotations
import os, logging

logger = logging.getLogger(__name__)


def _get(name: str, default=None):
    try:
        from config.settings import settings
        v = getattr(settings, name, None)
        if v is not None:
            return v
    except Exception:
        pass
    return os.environ.get(name, default)


class NotifyService:

    # ── EMAIL ─────────────────────────────────────────────────────────────────
    def email_configured(self) -> bool:
        return bool(_get("RESEND_API_KEY") or _get("SENDGRID_API_KEY") or _get("BREVO_API_KEY"))

    def send_email(self, to: str, subject: str, html: str, text: str = "") -> dict:
        provider = (_get("EMAIL_PROVIDER") or "resend").lower()
        frm = _get("EMAIL_FROM") or "DataMind <onboarding@resend.dev>"
        if not to:
            return {"success": False, "error": "No recipient email."}
        try:
            import requests
        except ImportError:
            return {"success": False, "error": "requests not installed."}

        try:
            if provider == "resend" or _get("RESEND_API_KEY"):
                key = _get("RESEND_API_KEY")
                if not key:
                    return {"success": False, "error": "Email not configured (no RESEND_API_KEY)."}
                r = requests.post("https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"from": frm, "to": [to], "subject": subject,
                          "html": html, "text": text or _strip(html)}, timeout=20)
                if r.status_code < 300:
                    return {"success": True, "provider": "resend", "id": r.json().get("id")}
                return {"success": False, "error": f"Resend {r.status_code}: {r.text[:180]}"}

            if provider == "sendgrid" or _get("SENDGRID_API_KEY"):
                key = _get("SENDGRID_API_KEY")
                r = requests.post("https://api.sendgrid.com/v3/mail/send",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"personalizations": [{"to": [{"email": to}]}],
                          "from": {"email": _clean_from(frm)}, "subject": subject,
                          "content": [{"type": "text/html", "value": html}]}, timeout=20)
                if r.status_code < 300:
                    return {"success": True, "provider": "sendgrid"}
                return {"success": False, "error": f"SendGrid {r.status_code}: {r.text[:180]}"}

            if provider == "brevo" or _get("BREVO_API_KEY"):
                key = _get("BREVO_API_KEY")
                r = requests.post("https://api.brevo.com/v3/smtp/email",
                    headers={"api-key": key, "Content-Type": "application/json"},
                    json={"sender": {"email": _clean_from(frm), "name": "DataMind"},
                          "to": [{"email": to}], "subject": subject, "htmlContent": html}, timeout=20)
                if r.status_code < 300:
                    return {"success": True, "provider": "brevo"}
                return {"success": False, "error": f"Brevo {r.status_code}: {r.text[:180]}"}

            return {"success": False, "error": "Email not configured."}
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return {"success": False, "error": str(e)}

    # ── SMS (Mnotify) ─────────────────────────────────────────────────────────
    def _mnotify_key(self):
        # Accept either name — MNOTIFY_KEY matches the other NkaySolutions apps.
        return _get("MNOTIFY_API_KEY") or _get("MNOTIFY_KEY")

    def sms_configured(self) -> bool:
        return bool(self._mnotify_key())

    def send_sms(self, to: str, message: str) -> dict:
        key = self._mnotify_key()
        sender = _get("MNOTIFY_SENDER_ID") or "NkaySolutions"
        if not key:
            return {"success": False, "error": "SMS not configured (no MNOTIFY_API_KEY)."}
        if not to:
            return {"success": False, "error": "No recipient phone number."}
        try:
            import requests
        except ImportError:
            return {"success": False, "error": "requests not installed."}
        try:
            # Mnotify quick-send endpoint
            r = requests.post(
                f"https://api.mnotify.com/api/sms/quick?key={key}",
                json={"recipient": [to], "sender": sender,
                      "message": message[:459]},   # ~3 SMS segments max
                timeout=20)
            if r.status_code < 300:
                body = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
                ok = str(body.get("status", "")).lower() in ("success", "ok") or r.status_code == 200
                return {"success": ok, "provider": "mnotify", "detail": body}
            return {"success": False, "error": f"Mnotify {r.status_code}: {r.text[:180]}"}
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
            return {"success": False, "error": str(e)}

    def status(self) -> dict:
        return {"email": self.email_configured(), "sms": self.sms_configured(),
                "email_provider": (_get("EMAIL_PROVIDER") or "resend")}


def _strip(html: str) -> str:
    import re
    return re.sub(r"<[^>]+>", " ", html or "").replace("&nbsp;", " ").strip()

def _clean_from(frm: str) -> str:
    # "DataMind <x@y.com>" -> "x@y.com"
    import re
    m = re.search(r"<([^>]+)>", frm or "")
    return m.group(1) if m else (frm or "")


notify_service = NotifyService()
