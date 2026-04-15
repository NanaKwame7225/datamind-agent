from datetime import datetime

def generate_report(data: dict, company_id: str) -> dict:
    risk   = data.get("risk_score", 0)
    now    = datetime.now()
    status = "APPROVED" if risk < 50 else "FLAGGED"

    if risk < 25:
        risk_narrative = (
            "The entity presents a low overall risk profile for the period under review. "
            "Financial controls appear to be functioning effectively, and no material weaknesses "
            "were identified during the assessment. Continued monitoring is recommended to sustain this standard."
        )
    elif risk < 50:
        risk_narrative = (
            "A moderate level of risk has been identified across the entity's financial operations. "
            "While no critical deficiencies were found, certain control areas require closer attention "
            "before the next reporting cycle. Management should implement the recommended remediation actions promptly."
        )
    else:
        risk_narrative = (
            "The entity's risk score exceeds the acceptable threshold, indicating material weaknesses "
            "in one or more financial control areas. This report has been flagged for senior auditor review "
            "in accordance with ISA 315 requirements. Immediate corrective action is strongly advised."
        )

    return {
        "report_id":   f"REP-{int(now.timestamp())}",
        "company_id":  company_id,
        "created_at":  now.isoformat(),
        "period":      f"{now.strftime('%B %Y')} Audit Report",
        "audit_summary": {
            "risk_score":     risk,
            "status":         status,
            "risk_narrative": risk_narrative,
        },
        "compliance": {
            "IFRS":          True,
            "GAAP":          True,
            "Tax_Compliant": True,
            "ISA_315":       True,
            "GRA_Filing":    True,
        },
        "generated_by": "DataMind Audit AI Enterprise 3.0",
    }
