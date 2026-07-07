"""
DataMind Agent — RAG Knowledge Base
Industry benchmarks, domain knowledge, and regulatory context
retrieved and injected into every AI analysis for grounded recommendations.
"""
from __future__ import annotations
import json, logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── INDUSTRY BENCHMARK DATABASE ───────────────────────────────────────────────
BENCHMARKS = {
    "finance": {
        "gross_margin": {"good": ">40%", "acceptable": "20-40%", "poor": "<20%", "context": "Varies heavily by industry — SaaS 70%+, manufacturing 20-30%"},
        "net_margin": {"good": ">15%", "acceptable": "5-15%", "poor": "<5%", "context": "S&P 500 average ~11%"},
        "ebitda_margin": {"good": ">20%", "acceptable": "10-20%", "poor": "<10%", "context": "Typical healthy business: 15-25%"},
        "current_ratio": {"good": ">2.0", "acceptable": "1.5-2.0", "poor": "<1.0", "context": "Below 1.0 means current liabilities exceed current assets — liquidity risk"},
        "dso": {"good": "<30 days", "acceptable": "30-45 days", "poor": ">60 days", "context": "Days Sales Outstanding — lower = cash collected faster"},
        "dpo": {"good": "30-60 days", "acceptable": "60-90 days", "poor": ">90 days or <15 days", "context": "Days Payable Outstanding — optimize without straining suppliers"},
        "etr": {"good": "20-25%", "acceptable": "25-30%", "poor": ">35% or <15%", "context": "Effective Tax Rate — <15% may signal aggressive tax avoidance risk"},
        "revenue_growth": {"good": ">15% YoY", "acceptable": "5-15%", "poor": "<5% or negative", "context": "Inflation-adjusted; emerging markets should see higher growth"},
        "debt_to_equity": {"good": "<1.0", "acceptable": "1.0-2.0", "poor": ">3.0", "context": "Higher leverage magnifies risk in downturns"},
        "return_on_equity": {"good": ">15%", "acceptable": "8-15%", "poor": "<8%", "context": "Warren Buffett benchmark: >15% sustained ROE"},
    },
    "education": {
        "pass_rate": {"good": ">85%", "acceptable": "70-85%", "poor": "<70%", "context": "National average in Ghana ~65-70% — aim for 80%+"},
        "dropout_rate": {"good": "<5%", "acceptable": "5-10%", "poor": ">15%", "context": "High dropout signals financial, academic, or welfare issues"},
        "fee_collection_rate": {"good": ">90%", "acceptable": "75-90%", "poor": "<70%", "context": "Low collection risks operational cash flow"},
        "teacher_student_ratio": {"good": "<25:1", "acceptable": "25-35:1", "poor": ">40:1", "context": "UNESCO recommends <30:1 for quality education"},
        "avg_score": {"good": ">75%", "acceptable": "60-75%", "poor": "<60%", "context": "Below 60% indicates systemic teaching or assessment issues"},
    },
    "healthcare": {
        "readmission_rate": {"good": "<10%", "acceptable": "10-15%", "poor": ">20%", "context": "US national benchmark ~15% — high readmission signals discharge quality issues"},
        "avg_los": {"good": "<4 days", "acceptable": "4-6 days", "poor": ">8 days", "context": "Average length of stay — longer LOS = higher cost and bed pressure"},
        "bed_occupancy": {"good": "75-85%", "acceptable": "65-75%", "poor": ">90% or <60%", "context": ">90% risks surge capacity; <60% is financially unsustainable"},
        "cost_per_patient": {"good": "Trending down", "acceptable": "Stable", "poor": "Rising >5% above inflation", "context": "Benchmark against national average for your facility type"},
    },
    "supply_chain": {
        "otif": {"good": ">95%", "acceptable": "90-95%", "poor": "<85%", "context": "On-Time In-Full — industry gold standard is 98%+"},
        "inventory_turnover": {"good": ">8x/year", "acceptable": "4-8x", "poor": "<4x", "context": "Higher turnover = less cash tied up in stock"},
        "lead_time": {"good": "<3 days", "acceptable": "3-7 days", "poor": ">14 days", "context": "Benchmark varies by product type and geography"},
        "stockout_rate": {"good": "<1%", "acceptable": "1-3%", "poor": ">5%", "context": "Each stockout erodes customer trust and sales"},
        "supplier_reliability": {"good": ">98%", "acceptable": "95-98%", "poor": "<90%", "context": "Unreliable suppliers cascade into delivery failures"},
    },
    "mining": {
        "oee": {"good": ">80%", "acceptable": "70-80%", "poor": "<65%", "context": "World-class mining OEE: 85%+"},
        "recovery_rate": {"good": ">90%", "acceptable": "85-90%", "poor": "<80%", "context": "Each 1% improvement in recovery can add millions in recovered value"},
        "trifr": {"good": "<1.0", "acceptable": "1.0-3.0", "poor": ">5.0", "context": "Total Recordable Injury Frequency Rate — zero harm target"},
        "cost_per_tonne": {"good": "Trending down", "acceptable": "Stable", "poor": "Rising >CPI", "context": "Benchmark against global cost curve for your ore type"},
        "uptime_pct": {"good": ">92%", "acceptable": "85-92%", "poor": "<80%", "context": "Equipment availability — unplanned downtime is most expensive"},
    },
    "retail": {
        "gross_margin": {"good": ">35%", "acceptable": "20-35%", "poor": "<15%", "context": "Grocery: 20-25%; Fashion: 50-60%; Electronics: 10-15%"},
        "inventory_turnover": {"good": ">6x", "acceptable": "4-6x", "poor": "<3x", "context": "Slow turnover risks obsolescence and cash lockup"},
        "customer_retention": {"good": ">70%", "acceptable": "50-70%", "poor": "<40%", "context": "Acquiring a new customer costs 5-7x more than retaining one"},
        "basket_size": {"good": "Growing >5%", "acceptable": "Stable", "poor": "Declining", "context": "Declining basket signals pricing, assortment, or experience issues"},
        "return_rate": {"good": "<5%", "acceptable": "5-15%", "poor": ">20%", "context": "High returns erode margin and signal product or expectation issues"},
    },
    "petroleum": {
        "lifting_cost": {"good": "<$15/boe", "acceptable": "$15-25/boe", "poor": ">$30/boe", "context": "Competitive operators target sub-$20/boe lifting cost"},
        "uptime_pct": {"good": ">95%", "acceptable": "90-95%", "poor": "<85%", "context": "Each 1% downtime on a 10,000 boe/day field costs ~$2.5M/year at $70 oil"},
        "water_cut_pct": {"good": "<20%", "acceptable": "20-40%", "poor": ">60%", "context": "Rising water cut signals reservoir depletion or water breakthrough"},
        "rrr": {"good": ">1.0", "acceptable": "0.8-1.0", "poor": "<0.8", "context": "Reserve Replacement Ratio — must exceed 1.0 to maintain reserves base"},
    },
    "manufacturing": {
        "oee": {"good": ">85%", "acceptable": "75-85%", "poor": "<65%", "context": "World-class OEE: 85%+ (Availability × Performance × Quality)"},
        "defects_ppm": {"good": "<100 PPM", "acceptable": "100-500 PPM", "poor": ">1000 PPM", "context": "Six Sigma target: 3.4 PPM defects"},
        "scrap_rate": {"good": "<1%", "acceptable": "1-3%", "poor": ">5%", "context": "Each 1% scrap rate represents direct material cost loss"},
        "first_pass_yield": {"good": ">97%", "acceptable": "90-97%", "poor": "<85%", "context": "Low FPY = hidden factory cost and delay"},
        "mttr": {"good": "<2 hours", "acceptable": "2-8 hours", "poor": ">24 hours", "context": "Mean Time To Repair — speed of maintenance response"},
    },
    "ngo": {
        "cost_per_beneficiary": {"good": "<$50", "acceptable": "$50-150", "poor": ">$300", "context": "Highly context-dependent — GiveWell top charities typically <$100"},
        "donor_retention": {"good": ">60%", "acceptable": "40-60%", "poor": "<30%", "context": "Industry average: 45% — acquiring new donors costs 10x more"},
        "budget_utilisation": {"good": "85-95%", "acceptable": "75-85%", "poor": "<70% or >98%", "context": ">98% risks insufficient reserves; <70% signals delivery challenges"},
        "programme_overhead_ratio": {"good": "<15%", "acceptable": "15-25%", "poor": ">30%", "context": "Charity Navigator benchmark: admin costs should be <25% of total spend"},
    },
    "procurement": {
        "savings_rate": {"good": ">8%", "acceptable": "5-8%", "poor": "<3%", "context": "Best-in-class procurement saves 8-12% of managed spend annually"},
        "maverick_spend": {"good": "<5%", "acceptable": "5-15%", "poor": ">20%", "context": "Off-contract spending loses negotiated savings and creates risk"},
        "po_cycle_time": {"good": "<3 days", "acceptable": "3-7 days", "poor": ">14 days", "context": "Long PO cycle slows business and frustrates stakeholders"},
        "supplier_concentration": {"good": "Top 10 < 50%", "acceptable": "Top 10 50-70%", "poor": "Top 3 > 50%", "context": "High concentration creates supply chain risk"},
    },
    "agriculture": {
        "yield_gap": {"good": "<10% vs potential", "acceptable": "10-25% gap", "poor": ">40% gap", "context": "Yield gap = difference between actual and potential yield given inputs"},
        "input_efficiency": {"good": ">3:1 return", "acceptable": "2-3:1", "poor": "<1.5:1", "context": "Revenue per unit of input spend"},
        "water_use_efficiency": {"good": "Improving >5%/year", "acceptable": "Stable", "poor": "Worsening", "context": "Water scarcity will increasingly drive agricultural economics"},
    },
    "general": {
        "growth_rate": {"good": ">10% YoY", "acceptable": "3-10%", "poor": "<3%", "context": "Inflation-adjusted real growth"},
        "efficiency_ratio": {"good": "Improving", "acceptable": "Stable", "poor": "Declining", "context": "Output per unit of input"},
    }
}

# ── DOMAIN KNOWLEDGE ──────────────────────────────────────────────────────────
DOMAIN_KNOWLEDGE = {
    "finance": [
        "Benford's Law: In legitimate financial datasets, ~30% of numbers start with 1, ~17% with 2, ~12% with 3. Deviation from this pattern is a fraud signal.",
        "Effective Tax Rate: ETR = Tax Expense / Pre-Tax Income. An ETR significantly below the statutory rate (25% in Ghana) warrants investigation.",
        "Working Capital: Current Assets - Current Liabilities. Negative working capital means the business cannot pay short-term debts from current assets.",
        "Days Sales Outstanding rising above 60 days in Ghana indicates potential client payment stress or weak collections.",
        "Revenue recognition: Sudden revenue spikes in specific periods (month-end, quarter-end) can indicate premature recognition.",
    ],
    "education": [
        "Dropout rates above 10% in Ghana typically correlate with inability to pay fees, long commute, or poor academic performance.",
        "Term 3 historically shows highest performance as exams approach — Term 1 reflects fresh enrolment and adjustment period.",
        "Gender disaggregated analysis is critical — female dropout rates often exceed male rates in rural Ghana.",
    ],
    "healthcare": [
        "WHO recommends 1 doctor per 1,000 population as a minimum. Ghana's ratio is approximately 1:7,000.",
        "Malaria accounts for ~30% of OPD visits in Ghana — seasonal spikes (May-July, September-November) are expected.",
        "Readmission within 30 days often signals inadequate discharge planning or insufficient post-discharge support.",
    ],
    "mining": [
        "Ghana is Africa's largest gold producer. Average all-in sustaining cost (AISC) for Ghanaian operations is ~$1,100-1,400/oz.",
        "Grade decline is the most common cause of cost increases — model head grade trends carefully.",
        "Safety: Ghana's Minerals Commission requires TRIFR reporting. Industry target is <1.0 per million hours worked.",
    ],
    "general": [
        "Correlation does not imply causation — a statistical relationship between two variables does not mean one causes the other.",
        "Simpson's Paradox: An aggregate trend can reverse when data is broken into subgroups — always check segment-level findings.",
        "P-value < 0.05 means there is less than 5% probability the result is due to chance — not that the effect is large or important.",
        "Confidence intervals are more informative than p-values — they show the range of plausible values for the true effect.",
    ]
}


class RAGService:
    """
    Retrieval-Augmented Generation knowledge base.
    Retrieves relevant benchmarks and domain knowledge
    and injects them into the LLM prompt for grounded analysis.
    """

    def get_industry_context(self, industry: str, metrics_found: list[str] = None) -> str:
        """Build a context block of benchmarks and domain knowledge for the AI."""
        benchmarks = BENCHMARKS.get(industry, BENCHMARKS.get("general", {}))
        knowledge = DOMAIN_KNOWLEDGE.get(industry, []) + DOMAIN_KNOWLEDGE.get("general", [])

        lines = [
            f"INDUSTRY BENCHMARKS & DOMAIN KNOWLEDGE ({industry.upper()}):",
            "Use these to contextualise findings — compare metrics to these benchmarks in your analysis.",
            "",
        ]

        # If we know which metrics are in the data, prioritise those benchmarks
        if metrics_found:
            relevant_benchmarks = {}
            for metric_name in metrics_found:
                for bench_key, bench_val in benchmarks.items():
                    if bench_key.lower() in metric_name.lower() or any(w in metric_name.lower() for w in bench_key.split('_')):
                        relevant_benchmarks[bench_key] = bench_val
            if not relevant_benchmarks:
                relevant_benchmarks = benchmarks
        else:
            relevant_benchmarks = benchmarks

        lines.append("KEY BENCHMARKS:")
        for metric, thresholds in list(relevant_benchmarks.items())[:8]:
            lines.append(f"  {metric.replace('_',' ').upper()}: Good={thresholds['good']} | Acceptable={thresholds['acceptable']} | Poor={thresholds['poor']}")
            if thresholds.get('context'):
                lines.append(f"    Context: {thresholds['context']}")

        lines.append("\nDOMAIN KNOWLEDGE:")
        for fact in knowledge[:5]:
            lines.append(f"  • {fact}")

        return "\n".join(lines)

    def get_benchmark_for_metric(self, metric_name: str, value: float, industry: str) -> dict:
        """Assess where a specific metric value falls against benchmarks."""
        benchmarks = BENCHMARKS.get(industry, BENCHMARKS.get("general", {}))
        for bench_key, thresholds in benchmarks.items():
            if bench_key.lower() in metric_name.lower():
                return {
                    "metric": metric_name,
                    "value": value,
                    "benchmark": thresholds,
                    "industry": industry,
                    "context": thresholds.get("context", ""),
                }
        return {"metric": metric_name, "value": value, "benchmark": None, "industry": industry}

    def get_regulatory_flags(self, industry: str, findings: list[dict]) -> list[str]:
        """Flag findings that may have regulatory or compliance implications."""
        flags = []
        regulatory_map = {
            "finance": {
                "etr_low": ("ETR below 15%", "May trigger tax authority scrutiny under Ghana Revenue Authority guidelines"),
                "anomaly_high": ("Large unexplained revenue spikes", "May require disclosure under GRA reporting requirements"),
                "fraud_flag": ("Fraud indicators detected", "Mandatory reporting obligation may apply under Ghana's Anti-Money Laundering Act"),
            },
            "healthcare": {
                "readmission_high": ("High readmission rate", "Ghana Health Service quality indicators require investigation above 15%"),
            },
            "mining": {
                "safety_incident": ("Safety incidents detected", "Ghana Minerals Commission requires TRIFR reporting and incident investigation within 24 hours"),
                "grade_variance": ("Significant grade variance", "Mineral reserve estimation may need updating — SEC/JORC reporting obligation"),
            },
        }
        industry_flags = regulatory_map.get(industry, {})
        for finding in findings:
            sev = finding.get("severity", "")
            if sev in ["critical", "warning"]:
                for flag_key, (trigger, explanation) in industry_flags.items():
                    if any(w in finding.get("title","").lower() for w in trigger.lower().split()):
                        flags.append(f"REGULATORY NOTE: {explanation}")
        return list(set(flags))[:3]

    def search_knowledge(self, query: str, industry: str = "general") -> list[str]:
        """Search the knowledge base for relevant facts."""
        q = query.lower()
        results = []
        all_knowledge = DOMAIN_KNOWLEDGE.get(industry, []) + DOMAIN_KNOWLEDGE.get("general", [])
        for fact in all_knowledge:
            if any(word in fact.lower() for word in q.split() if len(word) > 3):
                results.append(fact)
        return results[:5]


rag_service = RAGService()
