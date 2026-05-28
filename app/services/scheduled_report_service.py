"""
DataMind Agent v2 — Scheduled Reports Service
Cron-based scheduling, email delivery via SMTP.
"""
from __future__ import annotations
import uuid, logging, json, smtplib, asyncio
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from croniter import croniter

from app.services.auth_service import get_db
from config.settings import settings

logger = logging.getLogger(__name__)


class ScheduledReportService:

    def create_schedule(self, user_id: str, data: dict) -> dict:
        conn = get_db()
        sid = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        cron = croniter(data["schedule"], now)
        next_run = cron.get_next(datetime).isoformat()
        conn.execute("""
            INSERT INTO scheduled_reports
            (id,user_id,name,industry,query,schedule,recipients,is_active,next_run,created_at)
            VALUES (?,?,?,?,?,?,?,1,?,?)
        """, (
            sid, user_id, data["name"], data.get("industry","general"),
            data["query"], data["schedule"],
            json.dumps(data.get("recipients",[])),
            next_run, now.isoformat(),
        ))
        conn.commit()
        conn.close()
        return self.get_schedule(sid, user_id)

    def get_schedule(self, schedule_id: str, user_id: str) -> Optional[dict]:
        conn = get_db()
        row = conn.execute(
            "SELECT * FROM scheduled_reports WHERE id=? AND user_id=?",
            (schedule_id, user_id)
        ).fetchone()
        conn.close()
        if not row:
            return None
        r = dict(row)
        r["recipients"] = json.loads(r["recipients"] or "[]")
        return r

    def list_schedules(self, user_id: str) -> list[dict]:
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM scheduled_reports WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        conn.close()
        result = []
        for row in rows:
            r = dict(row)
            r["recipients"] = json.loads(r["recipients"] or "[]")
            result.append(r)
        return result

    def toggle_schedule(self, schedule_id: str, user_id: str, active: bool) -> dict:
        conn = get_db()
        conn.execute(
            "UPDATE scheduled_reports SET is_active=? WHERE id=? AND user_id=?",
            (1 if active else 0, schedule_id, user_id)
        )
        conn.commit()
        conn.close()
        return self.get_schedule(schedule_id, user_id)

    def delete_schedule(self, schedule_id: str, user_id: str):
        conn = get_db()
        conn.execute(
            "DELETE FROM scheduled_reports WHERE id=? AND user_id=?",
            (schedule_id, user_id)
        )
        conn.commit()
        conn.close()

    def get_due_schedules(self) -> list[dict]:
        """Return all schedules that are due to run."""
        conn = get_db()
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            "SELECT * FROM scheduled_reports WHERE is_active=1 AND (next_run IS NULL OR next_run <= ?)",
            (now,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def update_next_run(self, schedule_id: str, cron_expr: str):
        now = datetime.now(timezone.utc)
        cron = croniter(cron_expr, now)
        next_run = cron.get_next(datetime).isoformat()
        conn = get_db()
        conn.execute(
            "UPDATE scheduled_reports SET last_run=?, next_run=? WHERE id=?",
            (now.isoformat(), next_run, schedule_id)
        )
        conn.commit()
        conn.close()

    def build_email_html(self, analysis_result: dict, schedule: dict) -> str:
        """Build a professional HTML email from analysis results."""
        metrics = analysis_result.get("metrics", [])
        insights = analysis_result.get("insights", [])
        narrative = analysis_result.get("narrative", "")
        industry = analysis_result.get("industry", "General")
        query = analysis_result.get("query", "Scheduled analysis")
        now = datetime.now(timezone.utc).strftime("%B %d, %Y")

        severity_colors = {
            "critical": "#f04060",
            "warning": "#f0a020",
            "success": "#00d888",
            "info": "#3d7ef5",
        }

        metrics_html = "".join([f"""
            <td style="padding:12px;text-align:center;background:#1a2030;border-radius:8px;min-width:100px">
                <div style="font-size:10px;text-transform:uppercase;color:#6a85aa;margin-bottom:4px">{m.get('label','')}</div>
                <div style="font-family:Georgia,serif;font-size:22px;font-weight:700;color:#dce8f5">{m.get('value','')}</div>
            </td>
            """ for m in metrics[:4]])

        insights_html = "".join([f"""
            <div style="border-left:3px solid {severity_colors.get(i.get('severity','info'),'#3d7ef5')};
                        padding:10px 14px;margin-bottom:10px;background:#0f1a2e;border-radius:0 6px 6px 0">
                <div style="font-size:11px;font-weight:700;color:{severity_colors.get(i.get('severity','info'),'#3d7ef5')};
                             text-transform:uppercase;margin-bottom:4px">
                    {'Action Required' if i.get('severity')=='critical' else 'Needs Attention' if i.get('severity')=='warning' else 'Note'}
                </div>
                <div style="font-weight:600;color:#dce8f5;margin-bottom:3px;font-size:13px">{i.get('title','')}</div>
                <div style="color:#6a85aa;font-size:12px;line-height:1.6">{i.get('body','')[:200]}</div>
            </div>
            """ for i in insights[:5]])

        narrative_paras = "\n".join([
            f'<p style="margin:0 0 12px;color:#dce8f5;font-size:13px;line-height:1.8">{p}</p>'
            for p in narrative.split("\n") if p.strip()
        ])

        return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#080d1a;font-family:'DM Sans',Arial,sans-serif">
  <div style="max-width:640px;margin:0 auto;padding:24px 16px">

    <div style="background:linear-gradient(135deg,#00c8be,#3d7ef5);border-radius:12px;padding:24px;margin-bottom:20px;text-align:center">
      <div style="font-family:Georgia,serif;font-size:28px;font-weight:800;color:#000;margin-bottom:4px">DataMind Agent</div>
      <div style="font-size:13px;color:rgba(0,0,0,.7)">Scheduled Analysis Report</div>
    </div>

    <div style="background:#0b1221;border-radius:12px;padding:20px;margin-bottom:16px;border:1px solid rgba(0,200,190,.1)">
      <div style="font-size:11px;color:#6a85aa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">{now} · {industry.replace('_',' ').title()}</div>
      <div style="font-size:16px;font-weight:700;color:#dce8f5;margin-bottom:4px">{schedule.get('name','Scheduled Report')}</div>
      <div style="font-size:12px;color:#6a85aa">Query: {query}</div>
    </div>

    <div style="background:#0b1221;border-radius:12px;padding:20px;margin-bottom:16px;border:1px solid rgba(0,200,190,.1)">
      <div style="font-size:11px;color:#6a85aa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">Key Metrics</div>
      <table style="width:100%;border-collapse:separate;border-spacing:6px"><tr>{metrics_html}</tr></table>
    </div>

    {f'''<div style="background:#0b1221;border-radius:12px;padding:20px;margin-bottom:16px;border:1px solid rgba(0,200,190,.1)">
      <div style="font-size:11px;color:#6a85aa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">Key Findings</div>
      {insights_html}
    </div>''' if insights_html else ''}

    {f'''<div style="background:#0b1221;border-radius:12px;padding:20px;margin-bottom:16px;border-left:3px solid #00c8be;border-top:1px solid rgba(0,200,190,.1);border-right:1px solid rgba(0,200,190,.1);border-bottom:1px solid rgba(0,200,190,.1)">
      <div style="font-size:11px;color:#00c8be;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">AI Analysis & Recommendations</div>
      {narrative_paras}
    </div>''' if narrative else ''}

    <div style="text-align:center;padding:16px;color:#2a3d5c;font-size:11px">
      Sent by DataMind Agent · NkaySolutions · Unsubscribe
    </div>
  </div>
</body>
</html>"""

    def send_email(self, recipients: list[str], subject: str, html_body: str):
        """Send email via SMTP. Configure SMTP settings in .env"""
        if not settings.SMTP_HOST or not settings.SMTP_USER:
            logger.warning("SMTP not configured — skipping email send")
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = settings.SMTP_FROM or settings.SMTP_USER
            msg["To"]      = ", ".join(recipients)
            msg.attach(MIMEText(html_body, "html"))
            with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT or 465) as server:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.SMTP_USER, recipients, msg.as_string())
            logger.info(f"Email sent to {recipients}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


scheduled_report_service = ScheduledReportService()
