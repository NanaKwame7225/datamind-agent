from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# GENERATE AUDIT REPORT PDF
# =========================

def generate_audit_report(filename, audit_data):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AUDIT REPORT", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Findings: {audit_data['findings']}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Risk Level: {audit_data['risk']}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Conclusion: {audit_data['conclusion']}", styles["Normal"]))

    doc.build(content)

    return {"status": "Report generated", "file": filename}
