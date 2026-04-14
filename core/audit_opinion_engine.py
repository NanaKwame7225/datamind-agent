def generate_audit_opinion(findings):
    high_risk = len([f for f in findings if f.get("severity") == "HIGH"])

    if high_risk > 5:
        return {"opinion": "ADVERSE OPINION"}

    elif high_risk > 2:
        return {"opinion": "QUALIFIED OPINION"}

    return {"opinion": "UNQUALIFIED OPINION"}
