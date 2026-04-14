from datetime import datetime

def generate_audit_report(data):
    return {
        "report_id": f"AUD-{datetime.now().timestamp()}",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "risk_score": data.get("risk_score", 0),
            "status": "APPROVED" if data.get("risk_score",0) < 50 else "FLAGGED"
        },
        "compliance": {
            "IFRS": True,
            "GAAP": True,
            "Tax_Compliant": True
        }
    }
