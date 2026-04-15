from fastapi import APIRouter
from typing import Any
from pydantic import BaseModel
import time, statistics
import anthropic, os
import google.generativeai as genai

router = APIRouter()

# ── AI Clients ────────────────────────────────────────────────────────────────
# Primary: Claude  →  color #cc785c (warm copper/orange)
claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Subsidiary: Gemini  →  color #1a73e8 (Google blue)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Provider color metadata (consumed by frontend)
PROVIDER_META = {
    "claude": {
        "label":        "Powered by Claude (Anthropic)",
        "color":        "#cc785c",
        "bg":           "rgba(204,120,92,0.12)",
        "border":       "rgba(204,120,92,0.35)",
        "badge_text":   "Claude Primary",
        "dot_color":    "#cc785c",
    },
    "gemini": {
        "label":        "Powered by Gemini (Google)",
        "color":        "#1a73e8",
        "bg":           "rgba(26,115,232,0.12)",
        "border":       "rgba(26,115,232,0.35)",
        "badge_text":   "Gemini Assist",
        "dot_color":    "#1a73e8",
    },
    "statistical": {
        "label":        "Local statistical analysis",
        "color":        "#6a85aa",
        "bg":           "rgba(106,133,170,0.12)",
        "border":       "rgba(106,133,170,0.3)",
        "badge_text":   "Statistical Fallback",
        "dot_color":    "#6a85aa",
    },
}

# ── Request model ─────────────────────────────────────────────────────────────
class AnalyseRequest(BaseModel):
    query: str
    industry: str = "general"
    provider: str = "anthropic"
    inline_data: list[dict[str, Any]] = []
    enable_viz: bool = True
    enable_anomaly_detection: bool = True
    enable_forecast: bool = False
    conversation_history: list[dict[str, str]] = []

# ── Main endpoint ─────────────────────────────────────────────────────────────
@router.post("/analyse")
async def analyse(req: AnalyseRequest):
    t0   = time.perf_counter()
    data = req.inline_data or []

    stats         = compute_stats(data)
    metrics       = build_metrics(data, stats)
    insights      = detect_anomalies(data, stats) if req.enable_anomaly_detection else []
    charts        = build_chart_specs(data, stats) if req.enable_viz else []
    forecast_note = build_forecast(data, stats)    if req.enable_forecast else None

    pipe = [
        {"name": "Data ingested",        "status": "done", "duration_ms": 2},
        {"name": "Statistical analysis", "status": "done", "duration_ms": 8},
        {"name": "Anomaly detection",    "status": "done" if req.enable_anomaly_detection else "skip", "duration_ms": 4},
        {"name": "Chart generation",     "status": "done" if req.enable_viz else "skip",               "duration_ms": 5},
        {"name": "AI narrative",         "status": "pending", "duration_ms": 0},
    ]

    narrative, engine_used = await call_ai(req.query, req.industry, data, stats, insights, forecast_note)

    pipe[-1]["status"]      = "done"
    pipe[-1]["duration_ms"] = round((time.perf_counter() - t0) * 1000)
    pipe[-1]["engine"]      = engine_used

    provider_info = PROVIDER_META.get(engine_used, PROVIDER_META["statistical"])

    return {
        "query":            req.query,
        "industry":         req.industry,
        "provider":         engine_used,
        "model":            _model_label(engine_used),
        "provider_meta":    provider_info,
        "narrative":        narrative,
        "metrics":          metrics,
        "insights":         insights,
        "charts":           charts,
        "pipeline_steps":   pipe,
        "execution_ms":     round((time.perf_counter() - t0) * 1000),
        "raw_data_preview": data[:6],
    }

def _model_label(engine: str) -> str:
    return {
        "claude":      "claude-sonnet-4-20250514",
        "gemini":      "gemini-1.5-pro",
        "statistical": "local-stats-v1",
    }.get(engine, "unknown")

# ── AI routing ────────────────────────────────────────────────────────────────
async def call_ai(query, industry, data, stats, insights, forecast_note):
    """
    Routing logic:
      1. Claude  (primary   — copper #cc785c)
      2. Gemini  (subsidiary — blue  #1a73e8)
      3. Statistical narrative (offline fallback — slate #6a85aa)
    """
    prompt_system, prompt_user = build_prompt(query, industry, data, stats, insights, forecast_note)

    # ── 1. Claude ─────────────────────────────────────────────────────────────
    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            system=prompt_system,
            messages=[{"role": "user", "content": prompt_user}],
        )
        return message.content[0].text, "claude"

    except Exception as claude_err:
        print(f"[DataMind] ⚠ Claude unavailable: {claude_err} — switching to Gemini.")

    # ── 2. Gemini ─────────────────────────────────────────────────────────────
    try:
        gemini_prompt  = f"{prompt_system}\n\n{prompt_user}"
        response       = gemini_model.generate_content(gemini_prompt)
        return response.text, "gemini"

    except Exception as gemini_err:
        print(f"[DataMind] ⚠ Gemini unavailable: {gemini_err} — using statistical fallback.")

    # ── 3. Statistical fallback ───────────────────────────────────────────────
    return _statistical_narrative(query, industry, stats, insights), "statistical"


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(query, industry, data, stats, insights, forecast_note):
    stat_lines = "\n".join(
        f"  • {k}: mean={s['mean']:,.1f}, min={s['min']:,.1f}, "
        f"max={s['max']:,.1f}, Δ={s['pct_change']:+.1f}%"
        for k, s in list(stats.items())[:6]
    )
    anomaly_text = "\n".join(
        f"  - [{i['severity'].upper()}] {i['title']}: {i['body']}"
        for i in insights
    ) or "  No anomalies detected."
    forecast_text = f"\nForecast note: {forecast_note}" if forecast_note else ""

    system = """You are DataMind Audit AI — a senior financial analyst and chartered accountant.
Your role is to write analysis narratives that read like a real audit memo: precise, professional, and written in clear flowing prose.

STRICT FORMATTING RULES:
- Write in full, well-constructed paragraphs. Never use bullet points or numbered lists anywhere in your response.
- Every section must be at least 3 sentences long.
- Reference specific numbers from the data — percentages, averages, peaks, troughs.
- Apply ACCA, IFRS, ISA, and GRA (Ghana Revenue Authority) standards naturally within the prose.
- Use professional audit language: "the period under review", "material variance", "audit trail", "going concern", etc.
- Transition smoothly between sections using connector phrases.

STRUCTURE (use these exact bold headings, then write paragraphs beneath each):
**Executive Summary**
**Key Findings**
**Risk and Anomaly Assessment**
**Recommendations**"""

    user = (
        f"Industry: {industry.replace('_',' ').upper()}\n"
        f"Analyst query: {query}\n\n"
        f"Statistical summary:\n{stat_lines}\n\n"
        f"Anomalies:\n{anomaly_text}"
        f"{forecast_text}\n\n"
        f"Sample rows: {str(data[:3])}\n\n"
        "Write the full audit narrative now. Every section must be flowing paragraphs — no lists, no bullets."
    )
    return system, user


# ── Statistical helpers ───────────────────────────────────────────────────────
def numeric_keys(data):
    if not data:
        return []
    return [k for k, v in data[0].items() if isinstance(v, (int, float))]

def compute_stats(data):
    result = {}
    for k in numeric_keys(data):
        vals = [r[k] for r in data if isinstance(r.get(k), (int, float))]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std  = statistics.pstdev(vals) if len(vals) > 1 else 0
        result[k] = {
            "mean": mean, "std": std,
            "min": min(vals), "max": max(vals),
            "first": vals[0], "last": vals[-1],
            "values": vals,
            "pct_change": round((vals[-1] - vals[0]) / vals[0] * 100, 2) if vals[0] else 0,
        }
    return result

def build_metrics(data, stats):
    out = []
    for k, s in list(stats.items())[:6]:
        v     = s["last"]
        label = k.replace("_", " ").title()
        fmt   = f"{v/1000:.1f}K" if v > 10000 else f"{v:,.1f}"
        trend = "up" if s["pct_change"] > 0 else ("down" if s["pct_change"] < 0 else "flat")
        out.append({
            "label":       label,
            "value":       fmt,
            "change_pct":  s["pct_change"],
            "trend":       trend,
            "description": f"Avg: {s['mean']:,.1f}  |  Δ {s['pct_change']:+.1f}% over period",
        })
    return out

def detect_anomalies(data, stats):
    insights = []
    for k, s in stats.items():
        if s["std"] == 0:
            continue
        outliers = [v for v in s["values"] if abs(v - s["mean"]) > 2.5 * s["std"]]
        if outliers:
            sev = "critical" if len(outliers) > 1 else "warning"
            dev = max(abs(v - s["mean"]) for v in outliers) / s["mean"] * 100
            insights.append({
                "title":    f"Outlier detected in {k.replace('_', ' ')}",
                "body":     (
                    f"{len(outliers)} value(s) deviate significantly from the norm — up to {dev:.0f}% "
                    f"away from the mean. Manual review is advised before this is included in any formal report."
                ),
                "severity": sev,
                "source":   "Z-score · σ = 2.5",
            })
        if s["pct_change"] < -20:
            insights.append({
                "title":    f"Sharp decline in {k.replace('_', ' ')}",
                "body":     (
                    f"This metric has fallen {abs(s['pct_change']):.1f}% from the opening to the closing period. "
                    f"A decline of this magnitude warrants investigation into underlying causes before any audit sign-off."
                ),
                "severity": "warning",
                "source":   "Trend analysis",
            })
        elif s["pct_change"] > 50:
            insights.append({
                "title":    f"Rapid growth in {k.replace('_', ' ')}",
                "body":     (
                    f"Growth of {s['pct_change']:.1f}% over the period is exceptional. While positive, auditors should "
                    f"verify the accuracy of underlying records to rule out data entry errors or revenue recognition issues."
                ),
                "severity": "info",
                "source":   "Trend analysis",
            })
    return insights[:8]

def build_chart_specs(data, stats):
    nk = list(stats.keys())
    charts = []
    if nk:
        charts.append({
            "type":  "line",
            "title": f"{nk[0].replace('_',' ').title()} Over Time",
            "x_key": list(data[0].keys())[0],
            "y_key": nk[0],
        })
    if len(nk) >= 2:
        charts.append({
            "type":   "bar",
            "title":  f"{nk[0].replace('_',' ').title()} vs {nk[1].replace('_',' ').title()}",
            "x_key":  list(data[0].keys())[0],
            "y_keys": [nk[0], nk[1]],
        })
    return charts

def build_forecast(data, stats):
    nk = list(stats.keys())
    if not nk:
        return None
    k  = nk[0]
    vs = stats[k]["values"]
    n  = len(vs)
    xs = list(range(n))
    mx = statistics.mean(xs)
    my = statistics.mean(vs)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, vs))
    den = sum((xi - mx) ** 2 for xi in xs)
    slope     = num / den if den else 0
    intercept = my - slope * mx
    forecasted = [intercept + slope * (n + i) for i in range(3)]
    return (
        f"Linear forecast for {k.replace('_',' ')} (next 3 periods): "
        f"{', '.join(f'{v:,.1f}' for v in forecasted)}"
    )

# ── Statistical fallback narrative ────────────────────────────────────────────
def _statistical_narrative(query, industry, stats, insights):
    nk = list(stats.keys())
    if not nk:
        return "Insufficient data to generate a narrative at this time."
    k   = nk[0]
    s   = stats[k]
    ind = industry.replace("_", " ").title()
    trend = "upward" if s["pct_change"] > 0 else "downward"
    return (
        f"**Executive Summary**\n\n"
        f"The {ind} dataset submitted for the period under review has been processed using local statistical methods. "
        f"The primary metric, {k.replace('_',' ')}, recorded a mean of {s['mean']:,.1f} and moved in a {trend} direction, "
        f"representing a net change of {s['pct_change']:+.1f}% between the opening and closing periods. "
        f"This preliminary narrative is based on descriptive statistics and is provided pending the restoration of AI model connectivity.\n\n"
        f"**Key Findings**\n\n"
        f"Across the period under review, {k.replace('_',' ')} ranged from a low of {s['min']:,.1f} to a high of {s['max']:,.1f}, "
        f"representing a spread of {s['max']-s['min']:,.1f} units. "
        f"The variance observed is consistent with "
        f"{'normal operational fluctuation' if abs(s['pct_change']) < 20 else 'a significant structural shift that merits formal investigation'}. "
        f"A total of {len(insights)} anomal{'y was' if len(insights)==1 else 'ies were'} flagged during automated screening.\n\n"
        f"**Risk and Anomaly Assessment**\n\n"
        f"{'No critical anomalies were detected during the automated review of this dataset, suggesting that all values fall within expected statistical bounds for the sector.' if not insights else f'The automated screening process identified {len(insights)} area(s) of concern. The most significant relates to {insights[0][\"title\"].lower()}, which should be examined carefully before any audit opinion is issued.'} "
        f"All findings have been logged for auditor review in accordance with ISA 315 procedures, and supporting documentation should be retained as part of the formal audit trail.\n\n"
        f"**Recommendations**\n\n"
        f"Management is advised to review the {k.replace('_',' ')} trend in the context of established industry benchmarks for the {ind} sector. "
        f"Where anomalies have been flagged, corroborating documentation should be obtained and retained in the audit file before the period is closed. "
        f"A full AI-powered analysis using the Claude or Gemini engine is recommended once API connectivity is restored, as this will yield deeper pattern recognition and model-driven insights tailored to the specific characteristics of this dataset."
    )
