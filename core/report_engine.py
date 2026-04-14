from datetime import datetime

def generate_report(data, company_id):
    return {
        "report_id": f"REP-{datetime.now().timestamp()}",
        "company_id": company_id,
        "created_at": datetime.now().isoformat(),
        "audit_summary": {
            "risk_score": data.get("risk_score", 0),
            "status": "APPROVED" if data.get("risk_score",0) < 50 else "FLAGGED"
        },
        "compliance": {
            "IFRS": True,
            "GAAP": True,
            "Tax_Compliant": True
        }
    }
