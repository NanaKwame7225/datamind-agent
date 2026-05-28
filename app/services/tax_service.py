"""
DataMind Agent — Tax Analysis Service
Effective tax rate, tax burden, deferred tax, jurisdiction breakdown,
tax efficiency ratios, period-over-period tax analysis.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class TaxAnalysisService:

    def analyse(self, df: pd.DataFrame) -> dict:
        """
        Full tax analysis pipeline.
        Auto-detects tax, income, and revenue columns.
        Returns grounded findings with evidence and confidence scores.
        """
        result = {
            "module": "tax",
            "findings": [],
            "metrics": [],
            "evidence": {},
            "recommendations": [],
            "confidence": {},
        }

        cols = {c.lower(): c for c in df.columns}

        # ── Auto-detect columns ───────────────────────────────────────────────
        tax_col        = self._find_col(cols, ["tax","tax_expense","income_tax","tax_paid","taxes"])
        income_col     = self._find_col(cols, ["profit","net_income","ebt","earnings_before_tax","pre_tax_income","income_before_tax","ebitda"])
        revenue_col    = self._find_col(cols, ["revenue","sales","turnover","gross_revenue","total_revenue"])
        period_col     = self._find_col(cols, ["month","quarter","period","year","date","week"])
        region_col     = self._find_col(cols, ["region","country","jurisdiction","location","territory"])
        ebitda_col     = self._find_col(cols, ["ebitda","operating_income","operating_profit","ebit"])

        result["evidence"]["columns_detected"] = {
            "tax": tax_col, "income": income_col, "revenue": revenue_col,
            "period": period_col, "region": region_col, "ebitda": ebitda_col,
        }

        if not tax_col:
            result["findings"].append({
                "type": "missing_data",
                "title": "No tax column detected",
                "body": "Could not find a tax expense column. Expected columns named: tax, tax_expense, income_tax, tax_paid.",
                "severity": "warning",
                "confidence": 1.0,
            })
            return result

        tax = df[tax_col].dropna()
        n   = len(tax)

        # ── 1. EFFECTIVE TAX RATE ─────────────────────────────────────────────
        if income_col:
            income = df[income_col]
            valid  = df[[tax_col, income_col]].dropna()
            valid  = valid[valid[income_col] != 0]
            if len(valid) > 0:
                etr_series = (valid[tax_col] / valid[income_col] * 100).round(2)
                mean_etr   = round(float(etr_series.mean()), 2)
                median_etr = round(float(etr_series.median()), 2)
                max_etr    = round(float(etr_series.max()), 2)
                min_etr    = round(float(etr_series.min()), 2)
                std_etr    = round(float(etr_series.std()), 2)

                # Flag if ETR is abnormally high or low
                severity = "info"
                flag = ""
                if mean_etr > 35:
                    severity = "critical"
                    flag = f"Mean ETR of {mean_etr}% exceeds typical corporate rate of 25-35%. Possible tax inefficiency or one-off charges."
                elif mean_etr < 10:
                    severity = "warning"
                    flag = f"Mean ETR of {mean_etr}% is unusually low. Review for deferred tax assets, tax credits, or data completeness."
                elif mean_etr < 0:
                    severity = "critical"
                    flag = f"Negative ETR detected — tax benefit exceeds tax expense. Confirm data integrity."

                result["metrics"].append({
                    "label": "Mean effective tax rate",
                    "value": f"{mean_etr}%",
                    "benchmark": "Typical: 20–30%",
                    "trend": "up" if mean_etr > 30 else "down" if mean_etr < 15 else "flat",
                })
                result["evidence"]["effective_tax_rate"] = {
                    "mean_etr_pct": mean_etr,
                    "median_etr_pct": median_etr,
                    "max_etr_pct": max_etr,
                    "min_etr_pct": min_etr,
                    "std_etr_pct": std_etr,
                    "n_periods": len(valid),
                    "etr_by_period": etr_series.tolist(),
                    "high_variability": bool(std_etr > 5),
                }
                result["confidence"]["effective_tax_rate"] = round(min(0.97, 0.7 + len(valid) / 100), 2)

                body = (
                    f"Effective Tax Rate (ETR) computed across {len(valid)} periods. "
                    f"Mean ETR: {mean_etr}% · Median: {median_etr}% · "
                    f"Range: {min_etr}% to {max_etr}% · Std dev: {std_etr}%. "
                )
                if flag:
                    body += flag
                if std_etr > 5:
                    body += f" High variability (σ={std_etr}%) suggests inconsistent tax positions across periods — investigate."

                result["findings"].append({
                    "type": "effective_tax_rate",
                    "title": f"Effective Tax Rate: {mean_etr}% average",
                    "body": body,
                    "severity": severity,
                    "confidence": result["confidence"]["effective_tax_rate"],
                    "impact_score": round(min(10, abs(mean_etr - 25) / 2.5), 2),
                })

        # ── 2. TAX BURDEN AS % OF REVENUE ────────────────────────────────────
        if revenue_col:
            rev = df[revenue_col]
            valid = df[[tax_col, revenue_col]].dropna()
            valid = valid[valid[revenue_col] > 0]
            if len(valid) > 0:
                burden = (valid[tax_col] / valid[revenue_col] * 100).round(2)
                mean_burden = round(float(burden.mean()), 2)
                result["metrics"].append({
                    "label": "Tax burden (% of revenue)",
                    "value": f"{mean_burden}%",
                    "benchmark": "Typical: 5–15%",
                    "trend": "up" if mean_burden > 15 else "flat",
                })
                result["evidence"]["tax_burden"] = {
                    "mean_pct": mean_burden,
                    "max_pct": round(float(burden.max()), 2),
                    "min_pct": round(float(burden.min()), 2),
                    "n_periods": len(valid),
                }
                result["findings"].append({
                    "type": "tax_burden",
                    "title": f"Tax burden on revenue: {mean_burden}%",
                    "body": (
                        f"Tax represents {mean_burden}% of revenue on average across {len(valid)} periods "
                        f"(range: {round(float(burden.min()),2)}% to {round(float(burden.max()),2)}%). "
                        f"{'This exceeds the typical 5-15% range — review tax planning opportunities.' if mean_burden > 15 else 'Within typical range.'}"
                    ),
                    "severity": "warning" if mean_burden > 15 else "info",
                    "confidence": 0.90,
                    "impact_score": round(min(10, mean_burden / 1.5), 2),
                })

        # ── 3. PERIOD-OVER-PERIOD TAX TREND ──────────────────────────────────
        if period_col and len(tax) >= 3:
            tax_trend = df[[period_col, tax_col]].dropna()
            if len(tax_trend) >= 3:
                tax_vals = tax_trend[tax_col].values
                from scipy import stats as scipy_stats
                x = np.arange(len(tax_vals))
                slope, _, r, p, _ = scipy_stats.linregress(x, tax_vals)
                change_pct = round(float((tax_vals[-1] - tax_vals[0]) / max(abs(tax_vals[0]), 1e-10) * 100), 2)
                result["evidence"]["tax_trend"] = {
                    "slope_per_period": round(float(slope), 4),
                    "total_change_pct": change_pct,
                    "r_squared": round(float(r**2), 4),
                    "p_value": round(float(p), 4),
                    "significant": bool(p < 0.05),
                    "first_value": round(float(tax_vals[0]), 2),
                    "last_value": round(float(tax_vals[-1]), 2),
                    "n_periods": len(tax_vals),
                }
                result["findings"].append({
                    "type": "tax_trend",
                    "title": f"Tax expense {'rising' if slope > 0 else 'falling'}: {change_pct}% over {len(tax_vals)} periods",
                    "body": (
                        f"Tax expense has changed {change_pct:+.1f}% from {round(float(tax_vals[0]),2)} to "
                        f"{round(float(tax_vals[-1]),2)} over {len(tax_vals)} periods. "
                        f"Regression R² = {round(float(r**2),4)}, p = {round(float(p),4)} "
                        f"({'statistically significant' if p < 0.05 else 'not yet significant'}). "
                        f"{'Rising tax expense warrants review of tax planning strategy.' if slope > 0 and change_pct > 10 else ''}"
                    ),
                    "severity": "warning" if abs(change_pct) > 20 else "info",
                    "confidence": round(min(0.95, 0.6 + r**2 * 0.3), 2),
                    "impact_score": round(min(10, abs(change_pct) / 10), 2),
                })

        # ── 4. JURISDICTION / REGION TAX BREAKDOWN ────────────────────────────
        if region_col and tax_col:
            reg_tax = df[[region_col, tax_col]].dropna()
            if len(reg_tax) > 1 and reg_tax[region_col].nunique() > 1:
                breakdown = []
                overall_mean = reg_tax[tax_col].mean()
                for region, grp in reg_tax.groupby(region_col):
                    grp_mean = float(grp[tax_col].mean())
                    deviation = (grp_mean - overall_mean) / max(abs(overall_mean), 1e-10) * 100
                    breakdown.append({
                        "jurisdiction": str(region),
                        "count": int(len(grp)),
                        "mean_tax": round(grp_mean, 2),
                        "total_tax": round(float(grp[tax_col].sum()), 2),
                        "deviation_from_avg_pct": round(float(deviation), 2),
                        "rank": 0,
                    })
                breakdown.sort(key=lambda x: x["mean_tax"], reverse=True)
                for i, b in enumerate(breakdown):
                    b["rank"] = i + 1
                result["evidence"]["jurisdiction_breakdown"] = breakdown
                highest = breakdown[0]
                lowest  = breakdown[-1]
                result["findings"].append({
                    "type": "jurisdiction_breakdown",
                    "title": f"Tax varies significantly by {region_col.replace('_',' ')}",
                    "body": (
                        f"Highest tax: {highest['jurisdiction']} (mean {highest['mean_tax']}, "
                        f"{highest['deviation_from_avg_pct']:+.1f}% vs average). "
                        f"Lowest tax: {lowest['jurisdiction']} (mean {lowest['mean_tax']}, "
                        f"{lowest['deviation_from_avg_pct']:+.1f}% vs average). "
                        f"Differential of {round(highest['mean_tax'] - lowest['mean_tax'], 2)} "
                        f"between highest and lowest jurisdiction. "
                        f"Review whether transfer pricing or jurisdiction mix can be optimised."
                    ),
                    "severity": "info",
                    "confidence": 0.88,
                    "impact_score": 6.0,
                })

        # ── 5. DEFERRED TAX PROXY ────────────────────────────────────────────
        if income_col and tax_col:
            valid = df[[tax_col, income_col]].dropna()
            if len(valid) >= 4:
                rolling_mean_etr = (valid[tax_col] / valid[income_col].replace(0, np.nan)).rolling(3).mean()
                current_etr = float((valid[tax_col] / valid[income_col].replace(0, np.nan)).iloc[-1]) if len(valid) > 0 else None
                avg_etr = float(rolling_mean_etr.dropna().mean()) if not rolling_mean_etr.dropna().empty else None
                if current_etr and avg_etr and abs(current_etr - avg_etr) > 0.05:
                    direction = "higher" if current_etr > avg_etr else "lower"
                    result["findings"].append({
                        "type": "deferred_tax_signal",
                        "title": f"Deferred tax signal — current ETR {direction} than 3-period average",
                        "body": (
                            f"Current period ETR ({round(current_etr*100,1)}%) is {direction} than the "
                            f"3-period rolling average ({round(avg_etr*100,1)}%). "
                            f"Difference: {round(abs(current_etr-avg_etr)*100,1)} percentage points. "
                            f"This may indicate deferred tax assets or liabilities — confirm with tax team."
                        ),
                        "severity": "warning",
                        "confidence": 0.72,
                        "impact_score": 5.0,
                    })

        # ── 6. TAX EFFICIENCY SCORE ───────────────────────────────────────────
        efficiency_score = 100
        deductions = 0
        etr_ev = result["evidence"].get("effective_tax_rate", {})
        mean_etr = etr_ev.get("mean_etr_pct")
        if mean_etr:
            if mean_etr > 35:   efficiency_score -= 20; deductions += 1
            elif mean_etr > 30: efficiency_score -= 10; deductions += 1
            if etr_ev.get("high_variability"): efficiency_score -= 10; deductions += 1
        burden_ev = result["evidence"].get("tax_burden", {})
        if burden_ev.get("mean_pct", 0) > 15: efficiency_score -= 10; deductions += 1
        result["metrics"].append({
            "label": "Tax efficiency score",
            "value": f"{efficiency_score}/100",
            "benchmark": "Target: >75",
            "trend": "up" if efficiency_score > 75 else "down",
        })
        result["evidence"]["tax_efficiency_score"] = {
            "score": efficiency_score,
            "deductions_applied": deductions,
            "interpretation": (
                "Excellent" if efficiency_score >= 85 else
                "Good" if efficiency_score >= 70 else
                "Needs improvement" if efficiency_score >= 50 else
                "Poor — immediate review recommended"
            ),
        }

        # ── 7. RECOMMENDATIONS ────────────────────────────────────────────────
        recs = []
        mean_etr_val = result["evidence"].get("effective_tax_rate", {}).get("mean_etr_pct", 0)
        if mean_etr_val and mean_etr_val > 30:
            recs.append({
                "priority": 1,
                "action": "Review tax planning strategy",
                "reason": f"ETR of {mean_etr_val}% exceeds typical 20-30% range",
                "expected_impact": "Potential 3-8% reduction in ETR through planning",
            })
        if result["evidence"].get("jurisdiction_breakdown"):
            recs.append({
                "priority": 2,
                "action": "Optimise jurisdiction mix",
                "reason": "Significant tax variation detected across regions",
                "expected_impact": "Align operations with lower-tax jurisdictions where commercially viable",
            })
        if result["evidence"].get("effective_tax_rate", {}).get("high_variability"):
            recs.append({
                "priority": 3,
                "action": "Stabilise tax provisioning process",
                "reason": f"ETR standard deviation of {result['evidence']['effective_tax_rate']['std_etr_pct']}% indicates inconsistent provisions",
                "expected_impact": "More predictable financial reporting and reduced restatement risk",
            })
        result["recommendations"] = recs

        return result

    def _find_col(self, cols_lower: dict, keywords: list) -> Optional[str]:
        for kw in keywords:
            if kw in cols_lower:
                return cols_lower[kw]
        for kw in keywords:
            for col_lower, col_orig in cols_lower.items():
                if kw in col_lower:
                    return col_orig
        return None


tax_service = TaxAnalysisService()
