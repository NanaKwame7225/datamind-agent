"""
DataMind Agent — Document Export Service
Generates professional Word (.docx) and PDF reports.
Times New Roman, 12pt body, 14pt headings, bulleted paragraphs.
"""
from __future__ import annotations
import io, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── AUTHOR SIGNATURE — appears on every generated document ────────────────────
SIGNATURE = {
    "name": "Nana Kwame Asomani-Appah",
    "phone1": "+233 55 702 3768",
    "phone2": "+233 53 835 0574",
    "linkedin": "https://www.linkedin.com/in/nana-kwame-asomani",
    "org": "NkaySolutions",
    "location": "Accra, Ghana",
}


class DocumentExportService:
    """
    Exports analysis reports as professional documents.
    Word: python-docx. PDF: reportlab.
    Times New Roman throughout, 12pt body, 14pt headings.
    """

    # ── WORD EXPORT ───────────────────────────────────────────────────────────

    def build_word_report(self, result: dict, finance: dict = None, chart_images: list = None) -> bytes:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.style import WD_STYLE_TYPE

        doc = Document()

        # Base style — Times New Roman 12pt
        style = doc.styles['Normal']
        style.font.name = 'Times New Roman'
        style.font.size = Pt(12)

        industry = result.get("industry", "General").replace("_", " ").title()
        query = result.get("query", "Data Analysis")
        date_str = datetime.now(timezone.utc).strftime("%d %B %Y")

        # ── TITLE ──
        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.add_run("DataMind Agent")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0a, 0x4d, 0x4a)

        subtitle = doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = subtitle.add_run(f"{industry} Analysis Report")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.font.bold = True

        meta = doc.add_paragraph()
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = meta.add_run(f"{date_str}  |  {SIGNATURE['name']}  |  {SIGNATURE['org']}, {SIGNATURE['location']}")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        doc.add_paragraph()

        # ── QUERY ──
        self._add_heading(doc, "Analysis Question")
        self._add_body(doc, query)

        # ── METRICS ──
        metrics = result.get("metrics", [])
        if metrics:
            self._add_heading(doc, "Key Metrics")
            for m in metrics:
                change = f" ({m.get('change_pct','')}% vs previous)" if m.get('change_pct') is not None else ""
                self._add_bullet(doc, f"{m.get('label','')}: {m.get('value','')}{change}")

        # ── AI NARRATIVE ──
        narrative = result.get("narrative", "")
        if narrative:
            self._add_heading(doc, "Analysis & Recommendations")
            self._render_narrative_word(doc, narrative)

        # ── FINDINGS ──
        insights = result.get("insights", [])
        if insights:
            self._add_heading(doc, "Key Findings")
            for i in insights:
                sev = i.get("severity", "info").upper()
                conf = f" (Confidence: {int(i.get('confidence',0)*100)}%)" if i.get('confidence') else ""
                p = doc.add_paragraph(style='List Bullet')
                run = p.add_run(f"[{sev}] {i.get('title','')}{conf}")
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                run.font.bold = True
                body_p = doc.add_paragraph()
                body_p.paragraph_format.left_indent = Inches(0.5)
                run = body_p.add_run(i.get('body',''))
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                if i.get('source'):
                    src_p = doc.add_paragraph()
                    src_p.paragraph_format.left_indent = Inches(0.5)
                    run = src_p.add_run(f"Method: {i.get('source')}")
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(10)
                    run.font.italic = True
                    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        # ── CHART IMAGES ──
        if chart_images:
            self._add_charts_word(doc, chart_images)

        # ── FINANCE MODULES ──
        if finance:
            self._add_finance_word(doc, finance)

        # ── SIGNATURE BLOCK ──
        self._add_signature_word(doc)

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf.read()

    def _add_signature_word(self, doc):
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement

        doc.add_paragraph()

        # Horizontal rule
        hr = doc.add_paragraph()
        pPr = hr._p.get_or_add_pPr()
        pbdr = OxmlElement('w:pBdr')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '4')
        bottom.set(qn('w:space'), '1')
        bottom.set(qn('w:color'), 'cccccc')
        pbdr.append(bottom)
        pPr.append(pbdr)

        # Prepared by
        p = doc.add_paragraph()
        r = p.add_run("Prepared by")
        r.font.name = 'Times New Roman'; r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        p = doc.add_paragraph()
        r = p.add_run(SIGNATURE["name"])
        r.font.name = 'Times New Roman'; r.font.size = Pt(12); r.font.bold = True
        r.font.color.rgb = RGBColor(0x0a, 0x4d, 0x4a)

        for line in [
            f"Telephone: {SIGNATURE['phone1']}  |  {SIGNATURE['phone2']}",
            f"LinkedIn: {SIGNATURE['linkedin']}",
        ]:
            p = doc.add_paragraph()
            r = p.add_run(line)
            r.font.name = 'Times New Roman'; r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

        doc.add_paragraph()
        footer = doc.add_paragraph()
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("Generated by DataMind Agent — AI Data Analysis Platform  |  NkaySolutions, Accra, Ghana")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9)
        run.font.italic = True
        run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    def _add_heading(self, doc, text):
        from docx.shared import Pt, RGBColor
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0a, 0x4d, 0x4a)
        # bottom border
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        pPr = p._p.get_or_add_pPr()
        pbdr = OxmlElement('w:pBdr')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '6')
        bottom.set(qn('w:space'), '2')
        bottom.set(qn('w:color'), '0a4d4a')
        pbdr.append(bottom)
        pPr.append(pbdr)

    def _add_body(self, doc, text):
        from docx.shared import Pt
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)

    def _add_bullet(self, doc, text):
        from docx.shared import Pt
        p = doc.add_paragraph(style='List Bullet')
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)

    def _render_narrative_word(self, doc, narrative):
        from docx.shared import Pt, RGBColor
        # Clean JSON leakage
        import re
        narrative = re.sub(r'^\s*\{"narrative"\s*:\s*"', '', narrative)
        narrative = re.sub(r'",\s*"metrics"\s*:\s*\[.*$', '', narrative, flags=re.DOTALL)
        narrative = narrative.replace('\\n', '\n').replace('\\"', '"')

        for line in narrative.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Markdown headings
            if line.startswith('## '):
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(8)
                run = p.add_run(line[3:].strip())
                run.font.name = 'Times New Roman'
                run.font.size = Pt(14)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0x0a, 0x4d, 0x4a)
            elif line.startswith('### '):
                p = doc.add_paragraph()
                run = p.add_run(line[4:].strip())
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                run.font.bold = True
            # Bulleted / numbered lines become bullet paragraphs
            elif line.startswith(('- ', '• ', '* ')) or re.match(r'^\d+\.\s', line):
                clean = re.sub(r'^[-•*]\s*', '', line)
                clean = re.sub(r'^\d+\.\s*', '', clean)
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
                p = doc.add_paragraph(style='List Bullet')
                run = p.add_run(clean)
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
            else:
                # Regular paragraph with bold support
                p = doc.add_paragraph()
                parts = re.split(r'(\*\*.+?\*\*)', line)
                for part in parts:
                    if part.startswith('**') and part.endswith('**'):
                        run = p.add_run(part[2:-2])
                        run.font.bold = True
                    else:
                        run = p.add_run(part)
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)

    def _add_charts_word(self, doc, chart_images):
        import base64, io
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        self._add_heading(doc, "Dashboard Visualisations")
        for ci in chart_images[:8]:
            if not ci or not ci.get("image"):
                continue
            title = ci.get("title", "")
            if title:
                p = doc.add_paragraph()
                run = p.add_run(title)
                run.font.name = 'Times New Roman'; run.font.size = Pt(12); run.font.bold = True
            try:
                img_bytes = base64.b64decode(ci["image"].split(",")[-1])
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(io.BytesIO(img_bytes), width=Inches(6.0))
            except Exception:
                pass
            if ci.get("subtitle"):
                sp = doc.add_paragraph()
                r = sp.add_run(ci["subtitle"])
                r.font.name = 'Times New Roman'; r.font.size = Pt(10); r.font.italic = True
                r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    def _add_finance_word(self, doc, finance):
        from docx.shared import Pt
        tax = finance.get("tax")
        acct = finance.get("accounting")
        fraud = finance.get("fraud")

        if tax and not tax.get("error"):
            self._add_heading(doc, "Tax Analysis")
            for m in tax.get("metrics", []):
                self._add_bullet(doc, f"{m.get('label','')}: {m.get('value','')} (benchmark: {m.get('benchmark','')})")
            for f in tax.get("findings", [])[:5]:
                self._add_bullet(doc, f"{f.get('title','')} — {f.get('body','')}")

        if acct and not acct.get("error"):
            self._add_heading(doc, "Accounting Analysis")
            self._add_body(doc, f"Balance Sheet Health Score: {acct.get('health_score','?')}/100")
            for m in acct.get("metrics", []):
                self._add_bullet(doc, f"{m.get('label','')}: {m.get('value','')}")

        if fraud and not fraud.get("error"):
            self._add_heading(doc, "Fraud Detection")
            self._add_body(doc, f"Risk Score: {fraud.get('risk_score','?')}/100 ({fraud.get('risk_level','?')} risk)")
            for f in fraud.get("findings", [])[:5]:
                if f.get("severity") in ["critical", "warning"]:
                    self._add_bullet(doc, f"{f.get('title','')} — {f.get('body','')}")

        actions = finance.get("priority_actions", [])
        if actions:
            self._add_heading(doc, "Priority Actions")
            for a in actions:
                self._add_bullet(doc, f"[{a.get('module','')}] {a.get('action','')} — {a.get('reason','')}")

    # ── PDF EXPORT ────────────────────────────────────────────────────────────

    def build_pdf_report(self, result: dict, finance: dict = None, chart_images: list = None) -> bytes:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
        import re

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter,
                                topMargin=0.8*inch, bottomMargin=0.8*inch,
                                leftMargin=1*inch, rightMargin=1*inch)

        # Styles — Times New Roman
        title_style = ParagraphStyle('Title', fontName='Times-Bold', fontSize=20,
                                     textColor=HexColor('#0a4d4a'), alignment=TA_CENTER, spaceAfter=4)
        subtitle_style = ParagraphStyle('Subtitle', fontName='Times-Bold', fontSize=14,
                                        alignment=TA_CENTER, spaceAfter=4)
        meta_style = ParagraphStyle('Meta', fontName='Times-Roman', fontSize=10,
                                    textColor=HexColor('#666666'), alignment=TA_CENTER, spaceAfter=16)
        heading_style = ParagraphStyle('Heading', fontName='Times-Bold', fontSize=14,
                                       textColor=HexColor('#0a4d4a'), spaceBefore=14, spaceAfter=6,
                                       borderWidth=0, borderColor=HexColor('#0a4d4a'), borderPadding=2)
        subhead_style = ParagraphStyle('Subhead', fontName='Times-Bold', fontSize=12,
                                       spaceBefore=8, spaceAfter=3)
        body_style = ParagraphStyle('Body', fontName='Times-Roman', fontSize=12,
                                    spaceAfter=6, leading=16, alignment=TA_LEFT)
        bullet_style = ParagraphStyle('Bullet', fontName='Times-Roman', fontSize=12,
                                      leftIndent=20, spaceAfter=4, leading=15)
        src_style = ParagraphStyle('Source', fontName='Times-Italic', fontSize=10,
                                   textColor=HexColor('#666666'), leftIndent=20, spaceAfter=6)
        footer_style = ParagraphStyle('Footer', fontName='Times-Italic', fontSize=9,
                                      textColor=HexColor('#999999'), alignment=TA_CENTER, spaceBefore=20)

        story = []
        industry = result.get("industry", "General").replace("_", " ").title()
        query = result.get("query", "Data Analysis")
        date_str = datetime.now(timezone.utc).strftime("%d %B %Y")

        # Title
        story.append(Paragraph("DataMind Agent", title_style))
        story.append(Paragraph(f"{industry} Analysis Report", subtitle_style))
        story.append(Paragraph(f"{date_str}  |  {SIGNATURE['name']}  |  {SIGNATURE['org']}, {SIGNATURE['location']}", meta_style))

        # Query
        story.append(Paragraph("Analysis Question", heading_style))
        story.append(Paragraph(self._esc(query), body_style))

        # Metrics
        metrics = result.get("metrics", [])
        if metrics:
            story.append(Paragraph("Key Metrics", heading_style))
            items = []
            for m in metrics:
                change = f" ({m.get('change_pct','')}% vs previous)" if m.get('change_pct') is not None else ""
                items.append(ListItem(Paragraph(f"<b>{self._esc(str(m.get('label','')))}</b>: {self._esc(str(m.get('value','')))}{change}", bullet_style)))
            story.append(ListFlowable(items, bulletType='bullet', start='•'))

        # Narrative
        narrative = result.get("narrative", "")
        if narrative:
            story.append(Paragraph("Analysis &amp; Recommendations", heading_style))
            self._render_narrative_pdf(story, narrative, body_style, heading_style, subhead_style, bullet_style)

        # Findings
        insights = result.get("insights", [])
        if insights:
            story.append(Paragraph("Key Findings", heading_style))
            for i in insights:
                sev = i.get("severity", "info").upper()
                conf = f" (Confidence: {int(i.get('confidence',0)*100)}%)" if i.get('confidence') else ""
                story.append(Paragraph(f"<b>[{sev}] {self._esc(i.get('title',''))}</b>{conf}", subhead_style))
                story.append(Paragraph(self._esc(i.get('body','')), bullet_style))
                if i.get('source'):
                    story.append(Paragraph(f"Method: {self._esc(i.get('source'))}", src_style))

        # Chart images
        if chart_images:
            self._add_charts_pdf(story, chart_images, heading_style, subhead_style, src_style)

        # Finance
        if finance:
            self._add_finance_pdf(story, finance, heading_style, body_style, bullet_style)

        # ── SIGNATURE BLOCK ──
        from reportlab.platypus import HRFlowable
        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(width="100%", thickness=0.7, color=HexColor('#cccccc'), spaceAfter=8))
        sig_label = ParagraphStyle('SigLabel', fontName='Times-Roman', fontSize=10,
                                   textColor=HexColor('#666666'), spaceAfter=2)
        sig_name = ParagraphStyle('SigName', fontName='Times-Bold', fontSize=12,
                                  textColor=HexColor('#0a4d4a'), spaceAfter=3)
        sig_line = ParagraphStyle('SigLine', fontName='Times-Roman', fontSize=10,
                                  textColor=HexColor('#333333'), spaceAfter=2)
        story.append(Paragraph("Prepared by", sig_label))
        story.append(Paragraph(self._esc(SIGNATURE["name"]), sig_name))
        story.append(Paragraph(f"Telephone: {SIGNATURE['phone1']}  |  {SIGNATURE['phone2']}", sig_line))
        story.append(Paragraph(f'LinkedIn: <link href="{SIGNATURE["linkedin"]}" color="#0a4d4a">{SIGNATURE["linkedin"]}</link>', sig_line))

        story.append(Paragraph(
            f"Generated by DataMind Agent — AI Data Analysis Platform  |  {SIGNATURE['org']}, {SIGNATURE['location']}",
            footer_style))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    def _render_narrative_pdf(self, story, narrative, body, heading, subhead, bullet):
        from reportlab.platypus import Paragraph, ListFlowable, ListItem
        import re
        narrative = re.sub(r'^\s*\{"narrative"\s*:\s*"', '', narrative)
        narrative = re.sub(r'",\s*"metrics"\s*:\s*\[.*$', '', narrative, flags=re.DOTALL)
        narrative = narrative.replace('\\n', '\n').replace('\\"', '"')

        bullet_buffer = []
        def flush_bullets():
            if bullet_buffer:
                story.append(ListFlowable([ListItem(Paragraph(b, bullet)) for b in bullet_buffer], bulletType='bullet', start='•'))
                bullet_buffer.clear()

        for line in narrative.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('## '):
                flush_bullets()
                story.append(Paragraph(self._esc(line[3:].strip()), heading))
            elif line.startswith('### '):
                flush_bullets()
                story.append(Paragraph(self._esc(line[4:].strip()), subhead))
            elif line.startswith(('- ', '• ', '* ')) or re.match(r'^\d+\.\s', line):
                clean = re.sub(r'^[-•*]\s*', '', line)
                clean = re.sub(r'^\d+\.\s*', '', clean)
                clean = self._esc(clean)
                clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', clean)
                bullet_buffer.append(clean)
            else:
                flush_bullets()
                text = self._esc(line)
                text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
                story.append(Paragraph(text, body))
        flush_bullets()

    def _add_charts_pdf(self, story, chart_images, heading, subhead, src):
        import base64, io
        from reportlab.platypus import Paragraph, Spacer, Image
        from reportlab.lib.units import inch
        story.append(Paragraph("Dashboard Visualisations", heading))
        for ci in chart_images[:8]:
            if not ci or not ci.get("image"):
                continue
            if ci.get("title"):
                story.append(Paragraph(self._esc(ci["title"]), subhead))
            try:
                img_bytes = base64.b64decode(ci["image"].split(",")[-1])
                img = Image(io.BytesIO(img_bytes), width=6*inch, height=3*inch, kind='proportional')
                story.append(img)
            except Exception:
                pass
            if ci.get("subtitle"):
                story.append(Paragraph(self._esc(ci["subtitle"]), src))
            story.append(Spacer(1, 0.15*inch))

    def _add_finance_pdf(self, story, finance, heading, body, bullet):
        from reportlab.platypus import Paragraph, ListFlowable, ListItem
        tax = finance.get("tax")
        acct = finance.get("accounting")
        fraud = finance.get("fraud")

        if tax and not tax.get("error"):
            story.append(Paragraph("Tax Analysis", heading))
            items = [ListItem(Paragraph(f"<b>{self._esc(str(m.get('label','')))}</b>: {self._esc(str(m.get('value','')))} (benchmark: {self._esc(str(m.get('benchmark','')))})", bullet)) for m in tax.get("metrics", [])]
            if items:
                story.append(ListFlowable(items, bulletType='bullet', start='•'))

        if acct and not acct.get("error"):
            story.append(Paragraph("Accounting Analysis", heading))
            story.append(Paragraph(f"Balance Sheet Health Score: {acct.get('health_score','?')}/100", body))
            items = [ListItem(Paragraph(f"<b>{self._esc(str(m.get('label','')))}</b>: {self._esc(str(m.get('value','')))}", bullet)) for m in acct.get("metrics", [])]
            if items:
                story.append(ListFlowable(items, bulletType='bullet', start='•'))

        if fraud and not fraud.get("error"):
            story.append(Paragraph("Fraud Detection", heading))
            story.append(Paragraph(f"Risk Score: {fraud.get('risk_score','?')}/100 ({fraud.get('risk_level','?')} risk)", body))

        actions = finance.get("priority_actions", [])
        if actions:
            story.append(Paragraph("Priority Actions", heading))
            items = [ListItem(Paragraph(f"<b>[{self._esc(str(a.get('module','')))}]</b> {self._esc(str(a.get('action','')))} — {self._esc(str(a.get('reason','')))}", bullet)) for a in actions]
            story.append(ListFlowable(items, bulletType='bullet', start='•'))

    def _esc(self, text):
        if not isinstance(text, str):
            text = str(text)
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


document_service = DocumentExportService()
