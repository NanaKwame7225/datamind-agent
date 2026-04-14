def generate_audit_opinion(findings):
    risk_count = len([f for f in findings if f["severity"] == "HIGH"])

    if risk_count > 5:
        return {
            "opinion": "ADVERSE OPINION",
            "reason": "High material misstatements detected"
        }

    elif risk_count > 2:
        return {
            "opinion": "QUALIFIED OPINION",
            "reason": "Some risks identified"
        }

    else:
        return {
            "opinion": "UNQUALIFIED OPINION",
            "reason": "Financials appear reasonable"
        }
