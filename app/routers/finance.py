"""
DataMind Agent — Finance Router
POST /api/v1/finance/tax
POST /api/v1/finance/accounting
POST /api/v1/finance/fraud
POST /api/v1/finance/full
"""
import time, logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

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


@router.get("/ping")
async def ping():
    return {"status": "finance router alive"}


@router.post("/tax")
async def analyse_tax(request: FinanceRequest):
    if not request.data:
        raise HTTPException(400, "Provide financial data")
    try:
        from app.services.tax_service import tax_service
        from app.services.analysis_service import analysis_service
        df = pd.DataFrame(request.data)
        df, _ = analysis_service.clean_data(df)
        return tax_service.analyse(df)
    except Exception as e:
        logger.error(f"Tax analysis error: {e}", exc_info=True)
        raise HTTPException(500, f"Tax analysis failed: {str(e)}")


@router.post("/accounting")
async def analyse_accounting(request: FinanceRequest):
    if not request.data:
        raise HTTPException(400, "Provide financial data")
    try:
        from app.services.accounting_service import accounting_service
        from app.services.analysis_service import analysis_service
        df = pd.DataFrame(request.data)
        df, _ = analysis_service.clean_data(df)
        return accounting_service.analyse(df)
    except Exception as e:
        logger.error(f"Accounting analysis error: {e}", exc_info=True)
        raise HTTPException(500, f"Accounting analysis failed: {str(e)}")


@router.post("/fraud")
async def analyse_fraud(request: FinanceRequest):
    if not request.data:
        raise HTTPException(400, "Provide transaction data")
    try:
        from app.services.fraud_service import fraud_service
        from app.services.analysis_service import analysis_service
        df = pd.DataFrame(request.data)
        df, _ = analysis_service.clean_data(df)
        return fraud_service.analyse(df)
    except Exception as e:
        logger.error(f"Fraud analysis error: {e}", exc_info=True)
        raise HTTPException(500, f"Fraud analysis failed: {str(e)}")


@router.post("/full")
async def full_finance_analysis(request: FinanceRequest):
    t0 = time.perf_counter()
    if not request.data:
        raise HTTPException(400, "Provide financial data")

    # Import inside function to prevent startup import failures
    try:
        from app.services.tax_service import tax_service
        from app.services.accounting_service import accounting_service
        from app.services.fraud_service import fraud_service
        from app.services.analysis_service import analysis_service
        from app.services.llm_service import llm_service
    except Exception as e:
        logger.error(f"Service import failed: {e}", exc_info=True)
        raise HTTPException(500, f"Service import failed: {str(e)}")

    df = pd.DataFrame(request.data)
    try:
        df, _ = analysis_service.clean_data(df)
    except Exception as e:
        logger.warning(f"Cleaning failed, using raw data: {e}")

    tax_result = accounting_result = fraud_result = None

    if request.run_tax:
        try:
            tax_result = tax_service.analyse(df)
        except Exception as e:
            logger.warning(f"Tax failed: {e}")
            tax_result = {"error": str(e), "findings": [], "metrics": [], "recommendations": []}

    if request.run_accounting:
        try:
            accounting_result = accounting_service.analyse(df)
        except Exception as e:
            logger.warning(f"Accounting failed: {e}")
            accounting_result = {"error": str(e), "findings": [], "metrics": [], "health_score": 0, "recommendations": []}

    if request.run_fraud:
        try:
            fraud_result = fraud_service.analyse(df)
        except Exception as e:
            logger.warning(f"Fraud failed: {e}")
            fraud_result = {"error": str(e), "findings": [], "metrics": [], "risk_score": 0, "risk_level": "Unknown", "flags": [], "recommendations": []}

    # Merge priority actions
    all_recs = []
    for module, res in [("Tax", tax_result), ("Accounting", accounting_result), ("Fraud", fraud_result)]:
        if res:
            for r in res.get("recommendations", []):
                r["module"] = module
                all_recs.append(r)
    priority_actions = sorted(all_recs, key=lambda x: x.get("priority", 99))[:6]

    # AI narrative
    narrative = None
    tokens_used = 0
    provider_used = request.provider

    if request.generate_narrative:
        context_parts = [
            f"COMPREHENSIVE FINANCIAL ANALYSIS",
            f"Query: {request.query}",
            f"Dataset: {len(df)} rows x {len(df.columns)} columns",
            f"Columns: {list(df.columns)[:15]}",
            "",
        ]
        if tax_result and not tax_result.get("error"):
            context_parts.append("TAX FINDINGS:")
            for f in tax_result.get("findings", [])[:3]:
                context_parts.append(f"  [{f.get('severity','info').upper()}] {f.get('title','')} — {f.get('body','')[:180]}")
            for m in tax_result.get("metrics", [])[:3]:
                context_parts.append(f"  {m.get('label')}: {m.get('value')} (benchmark: {m.get('benchmark','')})")
            context_parts.append("")

        if accounting_result and not accounting_result.get("error"):
            context_parts.append(f"ACCOUNTING FINDINGS (health: {accounting_result.get('health_score','?')}/100):")
            for f in accounting_result.get("findings", [])[:3]:
                context_parts.append(f"  [{f.get('severity','info').upper()}] {f.get('title','')} — {f.get('body','')[:180]}")
            for m in accounting_result.get("metrics", [])[:4]:
                context_parts.append(f"  {m.get('label')}: {m.get('value')} (benchmark: {m.get('benchmark','')})")
            context_parts.append("")

        if fraud_result and not fraud_result.get("error"):
            context_parts.append(f"FRAUD DETECTION (risk: {fraud_result.get('risk_score','?')}/100 — {fraud_result.get('risk_level','?')}):")
            for f in fraud_result.get("findings", [])[:3]:
                if f.get("severity") in ["critical", "warning"]:
                    context_parts.append(f"  [{f.get('severity','').upper()}] {f.get('title','')} — {f.get('body','')[:180]}")
            context_parts.append(f"  Flags raised: {fraud_result.get('flags', [])}")
            context_parts.append("")

        if priority_actions:
            context_parts.append("PRIORITY ACTIONS:")
            for a in priority_actions[:5]:
                context_parts.append(f"  #{a.get('priority','?')} [{a.get('module','')}] {a.get('action','')} — {a.get('reason','')}")

        messages = [{"role": "user", "content": "\n".join(context_parts)}]

        try:
            result = await llm_service.chat(
                messages=messages,
                industry="finance",
                max_tokens=2000,
            )
            if isinstance(result, tuple):
                if len(result) == 3:
                    narrative, tokens_used, provider_used = result
                elif len(result) == 2:
                    narrative, tokens_used = result
            else:
                narrative = str(result)
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            narrative = _fallback_narrative(tax_result, accounting_result, fraud_result, priority_actions)

    return {
        "query": request.query,
        "execution_ms": round((time.perf_counter() - t0) * 1000, 1),
        "tax": tax_result,
        "accounting": accounting_result,
        "fraud": fraud_result,
        "narrative": narrative,
        "provider_used": provider_used,
        "tokens_used": tokens_used,
        "combined_risk_score": round(fraud_result.get("risk_score", 0) if fraud_result else 0, 1),
        "combined_health_score": round(accounting_result.get("health_score", 100) if accounting_result else 100, 1),
        "priority_actions": priority_actions,
    }


def _fallback_narrative(tax, acct, fraud, actions) -> str:
    parts = ["COMPREHENSIVE FINANCIAL ANALYSIS\n"]
    if tax and not tax.get("error"):
        critical = [f for f in tax.get("findings", []) if f.get("severity") == "critical"]
        if critical:
            parts.append(f"TAX: {len(critical)} critical issue(s) — " + "; ".join([f["title"] for f in critical[:2]]))
        for m in tax.get("metrics", [])[:2]:
            parts.append(f"  {m.get('label')}: {m.get('value')}")
    if acct and not acct.get("error"):
        parts.append(f"\nACCOUNTING: Health score {acct.get('health_score', '?')}/100")
        for m in acct.get("metrics", [])[:3]:
            parts.append(f"  {m.get('label')}: {m.get('value')}")
    if fraud and not fraud.get("error"):
        parts.append(f"\nFRAUD RISK: {fraud.get('risk_level', '?')} ({fraud.get('risk_score', '?')}/100)")
        if fraud.get("flags"):
            parts.append(f"  Flags: {', '.join(fraud['flags'])}")
    if actions:
        parts.append("\nPRIORITY ACTIONS:")
        for a in actions[:4]:
            parts.append(f"  {a.get('priority', '?')}. [{a.get('module', '')}] {a.get('action', '')}")
    return "\n".join(parts)
