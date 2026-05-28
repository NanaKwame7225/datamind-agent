"""
DataMind Agent — Finance Router
POST /api/v1/finance/tax
POST /api/v1/finance/accounting
POST /api/v1/finance/fraud
POST /api/v1/finance/full   ← runs all three + AI narrative
"""
import time, logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from app.services.tax_service import tax_service
from app.services.accounting_service import accounting_service
from app.services.fraud_service import fraud_service
from app.services.llm_service import llm_service
from app.services.analysis_service import analysis_service
from app.models.schemas import LLMProvider

router = APIRouter()
logger = logging.getLogger(__name__)


class FinanceRequest(BaseModel):
    data: list[dict]
    query: Optional[str] = "Analyse this financial data comprehensively"
    provider: str = "anthropic"
    run_tax: bool = True
    run_accounting: bool = True
    run_fraud: bool = True
    generate_narrative: bool = True


class FinanceResponse(BaseModel):
    query: str
    execution_ms: float
    tax: Optional[dict] = None
    accounting: Optional[dict] = None
    fraud: Optional[dict] = None
    narrative: Optional[str] = None
    provider_used: Optional[str] = None
    tokens_used: int = 0
    combined_risk_score: float = 0
    combined_health_score: float = 0
    priority_actions: list[dict] = []


@router.post("/tax")
async def analyse_tax(request: FinanceRequest):
    if not request.data:
        raise HTTPException(400, "Provide financial data as a list of records")
    df = pd.DataFrame(request.data)
    df, _ = analysis_service.clean_data(df)
    result = tax_service.analyse(df)
    return result


@router.post("/accounting")
async def analyse_accounting(request: FinanceRequest):
    if not request.data:
        raise HTTPException(400, "Provide financial data as a list of records")
    df = pd.DataFrame(request.data)
    df, _ = analysis_service.clean_data(df)
    result = accounting_service.analyse(df)
    return result


@router.post("/fraud")
async def analyse_fraud(request: FinanceRequest):
    if not request.data:
        raise HTTPException(400, "Provide transaction data as a list of records")
    df = pd.DataFrame(request.data)
    df, _ = analysis_service.clean_data(df)
    result = fraud_service.analyse(df)
    return result


@router.post("/full", response_model=FinanceResponse)
async def full_finance_analysis(request: FinanceRequest):
    """
    Run all three finance modules (tax, accounting, fraud) in parallel,
    then generate a unified AI narrative from the combined results.
    """
    t0 = time.perf_counter()

    if not request.data:
        raise HTTPException(400, "Provide financial data")

    df = pd.DataFrame(request.data)
    df, cleaning_report = analysis_service.clean_data(df)

    tax_result        = None
    accounting_result = None
    fraud_result      = None

    if request.run_tax:
        try:
            tax_result = tax_service.analyse(df)
            logger.info(f"Tax analysis: {len(tax_result.get('findings',[]))} findings")
        except Exception as e:
            logger.warning(f"Tax analysis failed: {e}")
            tax_result = {"error": str(e), "findings": [], "metrics": []}

    if request.run_accounting:
        try:
            accounting_result = accounting_service.analyse(df)
            logger.info(f"Accounting analysis: {len(accounting_result.get('findings',[]))} findings")
        except Exception as e:
            logger.warning(f"Accounting analysis failed: {e}")
            accounting_result = {"error": str(e), "findings": [], "metrics": [], "health_score": 0}

    if request.run_fraud:
        try:
            fraud_result = fraud_service.analyse(df)
            logger.info(f"Fraud analysis: {len(fraud_result.get('findings',[]))} findings, risk={fraud_result.get('risk_score',0)}")
        except Exception as e:
            logger.warning(f"Fraud analysis failed: {e}")
            fraud_result = {"error": str(e), "findings": [], "metrics": [], "risk_score": 0}

    # ── Combined scores ───────────────────────────────────────────────────────
    fraud_risk   = fraud_result.get("risk_score", 0) if fraud_result else 0
    acct_health  = accounting_result.get("health_score", 100) if accounting_result else 100
    combined_risk    = round(fraud_risk, 1)
    combined_health  = round(acct_health, 1)

    # ── Priority actions (merged from all modules) ────────────────────────────
    all_recs = []
    for module, res in [("Tax", tax_result), ("Accounting", accounting_result), ("Fraud", fraud_result)]:
        if res:
            for r in res.get("recommendations", []):
                r["module"] = module
                all_recs.append(r)
    priority_actions = sorted(all_recs, key=lambda x: x.get("priority", 99))[:6]

    # ── AI narrative ──────────────────────────────────────────────────────────
    narrative    = None
    tokens_used  = 0
    provider_used = request.provider

    if request.generate_narrative:
        context_parts = [
            f"FINANCIAL ANALYSIS RESULTS FOR: {request.query}",
            f"Dataset: {len(df)} rows × {len(df.columns)} columns",
            f"Columns: {list(df.columns)[:15]}",
            "",
        ]

        if tax_result and not tax_result.get("error"):
            context_parts.append("TAX ANALYSIS FINDINGS:")
            for f in tax_result.get("findings", [])[:4]:
                context_parts.append(f"  [{f.get('severity','info').upper()}] {f.get('title','')}: {f.get('body','')[:200]}")
            for m in tax_result.get("metrics", [])[:3]:
                context_parts.append(f"  Metric — {m.get('label')}: {m.get('value')} (benchmark: {m.get('benchmark','')})")
            context_parts.append("")

        if accounting_result and not accounting_result.get("error"):
            context_parts.append(f"ACCOUNTING ANALYSIS FINDINGS (health score: {accounting_result.get('health_score','?')}/100):")
            for f in accounting_result.get("findings", [])[:4]:
                context_parts.append(f"  [{f.get('severity','info').upper()}] {f.get('title','')}: {f.get('body','')[:200]}")
            for m in accounting_result.get("metrics", [])[:4]:
                context_parts.append(f"  Ratio — {m.get('label')}: {m.get('value')} (benchmark: {m.get('benchmark','')})")
            context_parts.append("")

        if fraud_result and not fraud_result.get("error"):
            context_parts.append(f"FRAUD DETECTION FINDINGS (risk score: {fraud_result.get('risk_score','?')}/100, level: {fraud_result.get('risk_level','?')}):")
            for f in fraud_result.get("findings", [])[:4]:
                if f.get("severity") in ["critical","warning"]:
                    context_parts.append(f"  [{f.get('severity','').upper()}] {f.get('title','')}: {f.get('body','')[:200]}")
            context_parts.append(f"  Flags raised: {fraud_result.get('flags', [])}")
            context_parts.append("")

        if priority_actions:
            context_parts.append("PRE-RANKED PRIORITY ACTIONS:")
            for i, a in enumerate(priority_actions[:5], 1):
                context_parts.append(f"  #{i} [{a.get('module','')}] {a.get('action','')}: {a.get('reason','')}")

        messages = [{"role": "user", "content": "\n".join(context_parts)}]

        try:
            provider_enum = LLMProvider(request.provider)
        except ValueError:
            provider_enum = LLMProvider.anthropic

        try:
            narrative, tokens_used, provider_used = await llm_service.chat(
                messages=messages,
                industry="finance",
                provider=provider_enum,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            narrative = _build_fallback_narrative(tax_result, accounting_result, fraud_result, priority_actions)
            tokens_used = 0

    return FinanceResponse(
        query=request.query,
        execution_ms=round((time.perf_counter() - t0) * 1000, 1),
        tax=tax_result,
        accounting=accounting_result,
        fraud=fraud_result,
        narrative=narrative,
        provider_used=provider_used,
        tokens_used=tokens_used,
        combined_risk_score=combined_risk,
        combined_health_score=combined_health,
        priority_actions=priority_actions,
    )


def _build_fallback_narrative(tax, acct, fraud, actions) -> str:
    parts = ["COMPREHENSIVE FINANCIAL ANALYSIS\n"]
    if tax and not tax.get("error"):
        critical = [f for f in tax.get("findings",[]) if f.get("severity") == "critical"]
        if critical:
            parts.append(f"TAX: {len(critical)} critical issue(s) found — " + "; ".join([f["title"] for f in critical[:2]]))
        for m in tax.get("metrics",[])[:2]:
            parts.append(f"  {m.get('label')}: {m.get('value')}")
    if acct and not acct.get("error"):
        parts.append(f"\nACCOUNTING: Health score {acct.get('health_score','?')}/100")
        for m in acct.get("metrics",[])[:3]:
            parts.append(f"  {m.get('label')}: {m.get('value')}")
    if fraud and not fraud.get("error"):
        parts.append(f"\nFRAUD RISK: {fraud.get('risk_level','?')} ({fraud.get('risk_score','?')}/100)")
        flags = fraud.get("flags", [])
        if flags:
            parts.append(f"  Flags: {', '.join(flags)}")
    if actions:
        parts.append("\nPRIORITY ACTIONS:")
        for a in actions[:4]:
            parts.append(f"  {a.get('priority','?')}. [{a.get('module','')}] {a.get('action','')}")
    return "\n".join(parts)
