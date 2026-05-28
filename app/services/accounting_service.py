"""
DataMind Agent — Accounting Analysis Service
AR aging, DSO, DPO, working capital, liquidity ratios,
profitability margins, balance sheet health, EBITDA analysis.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Optional

logger = logging.getLogger(__name__)


class AccountingAnalysisService:

    def analyse(self, df: pd.DataFrame) -> dict:
        result = {
            "module": "accounting",
            "findings": [],
            "metrics": [],
            "ratios": {},
            "evidence": {},
            "recommendations": [],
            "health_score": 100,
        }

        cols = {c.lower(): c for c in df.columns}

        # ── Auto-detect columns ───────────────────────────────────────────────
        revenue_col    = self._find(cols, ["revenue","sales","turnover","gross_revenue"])
        cogs_col       = self._find(cols, ["cogs","cost_of_goods","cost_of_sales","direct_costs"])
        gross_profit_col = self._find(cols, ["gross_profit","gross_margin_value"])
        operating_col  = self._find(cols, ["operating_income","operating_profit","ebit"])
        net_income_col = self._find(cols, ["net_income","net_profit","profit_after_tax","pat"])
        ebitda_col     = self._find(cols, ["ebitda"])
        total_assets_col = self._find(cols, ["total_assets","assets"])
        current_assets_col = self._find(cols, ["current_assets"])
        current_liab_col = self._find(cols, ["current_liabilities","current_liabs"])
        total_debt_col = self._find(cols, ["total_debt","debt","borrowings","total_liabilities"])
        equity_col     = self._find(cols, ["equity","shareholders_equity","net_assets","book_value"])
        ar_col         = self._find(cols, ["accounts_receivable","receivables","ar","debtors"])
        ap_col         = self._find(cols, ["accounts_payable","payables","ap","creditors"])
        inventory_col  = self._find(cols, ["inventory","stock","inventories"])
        cash_col       = self._find(cols, ["cash","cash_and_equivalents","cash_balance"])
        period_col     = self._find(cols, ["month","quarter","period","year","date","week"])

        result["evidence"]["columns_detected"] = {k: v for k, v in {
            "revenue": revenue_col, "cogs": cogs_col, "gross_profit": gross_profit_col,
            "operating_income": operating_col, "net_income": net_income_col,
            "total_assets": total_assets_col, "current_assets": current_assets_col,
            "current_liabilities": current_liab_col, "total_debt": total_debt_col,
            "equity": equity_col, "accounts_receivable": ar_col,
            "accounts_payable": ap_col, "inventory": inventory_col, "cash": cash_col,
        }.items() if v is not None}

        health_deductions = 0

        # ── 1. PROFITABILITY MARGINS ──────────────────────────────────────────
        if revenue_col:
            rev = df[revenue_col].dropna()
            rev_mean = float(rev.mean())
            result["metrics"].append({"label": "Avg Revenue", "value": f"{round(rev_mean/1000,1)}K" if rev_mean > 1000 else str(round(rev_mean,2))})

            # Gross margin
            if cogs_col:
                valid = df[[revenue_col, cogs_col]].dropna()
                valid = valid[valid[revenue_col] > 0]
                gm = ((valid[revenue_col] - valid[cogs_col]) / valid[revenue_col] * 100).round(2)
                mean_gm = round(float(gm.mean()), 2)
                result["ratios"]["gross_margin_pct"] = {"mean": mean_gm, "std": round(float(gm.std()), 2), "trend_values": gm.tolist()}
                result["metrics"].append({"label": "Gross margin", "value": f"{mean_gm}%", "benchmark": "Target: >40%", "trend": "up" if mean_gm > 40 else "down"})
                if mean_gm < 20:
                    health_deductions += 15
                    result["findings"].append(self._finding("gross_margin", f"Low gross margin: {mean_gm}%",
                        f"Gross margin averages {mean_gm}% across {len(valid)} periods (range: {round(float(gm.min()),1)}% to {round(float(gm.max()),1)}%). "
                        f"Below 20% indicates cost pressure or pricing weakness. Industry benchmark typically 30-60%.",
                        "critical" if mean_gm < 10 else "warning", 0.92, 8.0))

            # Net margin
            if net_income_col:
                valid = df[[revenue_col, net_income_col]].dropna()
                valid = valid[valid[revenue_col] > 0]
                nm = (valid[net_income_col] / valid[revenue_col] * 100).round(2)
                mean_nm = round(float(nm.mean()), 2)
                result["ratios"]["net_margin_pct"] = {"mean": mean_nm, "trend_values": nm.tolist()}
                result["metrics"].append({"label": "Net profit margin", "value": f"{mean_nm}%", "benchmark": "Target: >10%", "trend": "up" if mean_nm > 10 else "down"})
                if mean_nm < 0:
                    health_deductions += 25
                    result["findings"].append(self._finding("net_margin", f"Negative net margin: {mean_nm}%",
                        f"Business is losing money — net margin of {mean_nm}% means expenses exceed revenue. "
                        f"Over {len(valid)} periods, {int((nm < 0).sum())} periods were loss-making ({round(float((nm<0).mean()*100),1)}% of all periods).",
                        "critical", 0.97, 10.0))
                elif mean_nm < 5:
                    health_deductions += 10
                    result["findings"].append(self._finding("net_margin_low", f"Thin net margin: {mean_nm}%",
                        f"Net margin of {mean_nm}% is below the 10% target. Very thin margins leave no buffer for cost increases.",
                        "warning", 0.90, 6.0))

            # EBITDA margin
            if ebitda_col:
                valid = df[[revenue_col, ebitda_col]].dropna()
                valid = valid[valid[revenue_col] > 0]
                em = (valid[ebitda_col] / valid[revenue_col] * 100).round(2)
                mean_em = round(float(em.mean()), 2)
                result["ratios"]["ebitda_margin_pct"] = {"mean": mean_em}
                result["metrics"].append({"label": "EBITDA margin", "value": f"{mean_em}%", "benchmark": "Target: >15%"})

        # ── 2. LIQUIDITY RATIOS ───────────────────────────────────────────────
        if current_assets_col and current_liab_col:
            valid = df[[current_assets_col, current_liab_col]].dropna()
            valid = valid[valid[current_liab_col] > 0]
            if len(valid) > 0:
                cr = (valid[current_assets_col] / valid[current_liab_col]).round(4)
                mean_cr = round(float(cr.mean()), 3)
                result["ratios"]["current_ratio"] = {"mean": mean_cr, "values": cr.tolist()}
                result["metrics"].append({"label": "Current ratio", "value": str(mean_cr), "benchmark": "Target: >1.5", "trend": "up" if mean_cr > 1.5 else "down"})
                if mean_cr < 1.0:
                    health_deductions += 20
                    result["findings"].append(self._finding("liquidity_critical",
                        f"Liquidity risk: current ratio {mean_cr}",
                        f"Current ratio of {mean_cr} means current liabilities exceed current assets. "
                        f"The business may struggle to meet short-term obligations. {int((cr<1).sum())} of {len(cr)} periods are below 1.0.",
                        "critical", 0.95, 9.0))
                elif mean_cr < 1.5:
                    health_deductions += 8
                    result["findings"].append(self._finding("liquidity_low",
                        f"Low liquidity: current ratio {mean_cr}",
                        f"Current ratio of {mean_cr} is below the 1.5 safety threshold. Build cash buffer.",
                        "warning", 0.90, 6.0))

                # Quick ratio (if inventory available)
                if inventory_col:
                    valid2 = df[[current_assets_col, inventory_col, current_liab_col]].dropna()
                    valid2 = valid2[valid2[current_liab_col] > 0]
                    qr = ((valid2[current_assets_col] - valid2[inventory_col]) / valid2[current_liab_col]).round(4)
                    mean_qr = round(float(qr.mean()), 3)
                    result["ratios"]["quick_ratio"] = {"mean": mean_qr}
                    result["metrics"].append({"label": "Quick ratio", "value": str(mean_qr), "benchmark": "Target: >1.0"})

        # ── 3. WORKING CAPITAL ────────────────────────────────────────────────
        if current_assets_col and current_liab_col:
            valid = df[[current_assets_col, current_liab_col]].dropna()
            wc = (valid[current_assets_col] - valid[current_liab_col])
            mean_wc = round(float(wc.mean()), 2)
            neg_periods = int((wc < 0).sum())
            result["ratios"]["working_capital"] = {"mean": mean_wc, "negative_periods": neg_periods}
            result["metrics"].append({"label": "Avg working capital", "value": f"{round(mean_wc/1000,1)}K" if abs(mean_wc) > 1000 else str(round(mean_wc,2))})
            if neg_periods > 0:
                result["findings"].append(self._finding("working_capital_negative",
                    f"Negative working capital in {neg_periods} periods",
                    f"Working capital was negative in {neg_periods} out of {len(wc)} periods ({round(neg_periods/len(wc)*100,1)}%). "
                    f"This signals short-term cash flow stress. Minimum working capital: {round(float(wc.min()),2)}.",
                    "critical" if neg_periods > len(wc) / 2 else "warning", 0.93, 8.5))

        # ── 4. DSO — DAYS SALES OUTSTANDING ──────────────────────────────────
        if ar_col and revenue_col:
            valid = df[[ar_col, revenue_col]].dropna()
            valid = valid[valid[revenue_col] > 0]
            if len(valid) > 0:
                daily_rev = valid[revenue_col] / 30
                dso = (valid[ar_col] / daily_rev).round(1)
                mean_dso = round(float(dso.mean()), 1)
                result["ratios"]["dso_days"] = {"mean": mean_dso, "max": round(float(dso.max()),1), "min": round(float(dso.min()),1)}
                result["metrics"].append({"label": "Days Sales Outstanding", "value": f"{mean_dso} days", "benchmark": "Target: <45 days"})
                if mean_dso > 60:
                    health_deductions += 12
                    result["findings"].append(self._finding("dso_high",
                        f"Customers taking too long to pay: DSO = {mean_dso} days",
                        f"On average it takes {mean_dso} days to collect payment from customers (range: {round(float(dso.min()),1)} to {round(float(dso.max()),1)} days). "
                        f"Benchmark is 30-45 days. Extended DSO locks up cash and increases bad debt risk. "
                        f"{int((dso > 60).sum())} of {len(dso)} periods exceeded 60 days.",
                        "warning" if mean_dso < 90 else "critical", 0.89, 7.0))

        # ── 5. DPO — DAYS PAYABLE OUTSTANDING ────────────────────────────────
        if ap_col and cogs_col:
            valid = df[[ap_col, cogs_col]].dropna()
            valid = valid[valid[cogs_col] > 0]
            if len(valid) > 0:
                daily_cogs = valid[cogs_col] / 30
                dpo = (valid[ap_col] / daily_cogs).round(1)
                mean_dpo = round(float(dpo.mean()), 1)
                result["ratios"]["dpo_days"] = {"mean": mean_dpo}
                result["metrics"].append({"label": "Days Payable Outstanding", "value": f"{mean_dpo} days", "benchmark": "Target: 30-60 days"})

        # ── 6. LEVERAGE / DEBT RATIOS ─────────────────────────────────────────
        if total_debt_col and equity_col:
            valid = df[[total_debt_col, equity_col]].dropna()
            valid = valid[valid[equity_col] != 0]
            if len(valid) > 0:
                de = (valid[total_debt_col] / valid[equity_col]).round(4)
                mean_de = round(float(de.mean()), 3)
                result["ratios"]["debt_to_equity"] = {"mean": mean_de}
                result["metrics"].append({"label": "Debt-to-equity", "value": str(mean_de), "benchmark": "Target: <2.0"})
                if mean_de > 3.0:
                    health_deductions += 15
                    result["findings"].append(self._finding("leverage_high",
                        f"High leverage: debt/equity = {mean_de}",
                        f"Debt is {mean_de}x equity on average. High leverage amplifies losses and increases insolvency risk. "
                        f"Consider debt reduction or equity injection.",
                        "critical", 0.92, 8.0))

        if total_assets_col and total_debt_col:
            valid = df[[total_assets_col, total_debt_col]].dropna()
            valid = valid[valid[total_assets_col] > 0]
            if len(valid) > 0:
                da = (valid[total_debt_col] / valid[total_assets_col]).round(4)
                mean_da = round(float(da.mean()), 3)
                result["ratios"]["debt_to_assets"] = {"mean": mean_da}
                result["metrics"].append({"label": "Debt-to-assets", "value": str(mean_da), "benchmark": "Target: <0.5"})

        # ── 7. RETURN ON EQUITY / RETURN ON ASSETS ────────────────────────────
        if net_income_col and equity_col:
            valid = df[[net_income_col, equity_col]].dropna()
            valid = valid[valid[equity_col] > 0]
            if len(valid) > 0:
                roe = (valid[net_income_col] / valid[equity_col] * 100).round(2)
                mean_roe = round(float(roe.mean()), 2)
                result["ratios"]["roe_pct"] = {"mean": mean_roe}
                result["metrics"].append({"label": "Return on equity", "value": f"{mean_roe}%", "benchmark": "Target: >15%"})

        if net_income_col and total_assets_col:
            valid = df[[net_income_col, total_assets_col]].dropna()
            valid = valid[valid[total_assets_col] > 0]
            if len(valid) > 0:
                roa = (valid[net_income_col] / valid[total_assets_col] * 100).round(2)
                mean_roa = round(float(roa.mean()), 2)
                result["ratios"]["roa_pct"] = {"mean": mean_roa}
                result["metrics"].append({"label": "Return on assets", "value": f"{mean_roa}%", "benchmark": "Target: >5%"})

        # ── 8. BALANCE SHEET HEALTH SCORE ─────────────────────────────────────
        health_score = max(0, 100 - health_deductions)
        result["health_score"] = health_score
        result["metrics"].append({
            "label": "Balance sheet health",
            "value": f"{health_score}/100",
            "benchmark": "Target: >70",
            "trend": "up" if health_score > 70 else "down",
        })
        result["evidence"]["health_score"] = {
            "score": health_score,
            "deductions": health_deductions,
            "grade": ("A" if health_score >= 85 else "B" if health_score >= 70 else
                      "C" if health_score >= 55 else "D" if health_score >= 40 else "F"),
            "interpretation": (
                "Excellent financial health" if health_score >= 85 else
                "Good with minor concerns" if health_score >= 70 else
                "Moderate — several areas need attention" if health_score >= 55 else
                "Poor — immediate management action required"
            ),
        }

        # ── 9. RECOMMENDATIONS ────────────────────────────────────────────────
        recs = []
        ratios = result["ratios"]
        if ratios.get("dso_days", {}).get("mean", 0) > 45:
            recs.append({"priority": 1, "action": "Tighten accounts receivable collection",
                "reason": f"DSO of {ratios['dso_days']['mean']} days — customers paying too slowly",
                "metric": "Reduce DSO below 45 days"})
        if ratios.get("current_ratio", {}).get("mean", 2) < 1.5:
            recs.append({"priority": 2, "action": "Build cash reserves",
                "reason": f"Current ratio of {ratios.get('current_ratio',{}).get('mean','?')} is below safe threshold",
                "metric": "Target current ratio above 1.5"})
        if ratios.get("gross_margin_pct", {}).get("mean", 50) < 30:
            recs.append({"priority": 3, "action": "Review pricing and cost of goods",
                "reason": f"Gross margin of {ratios.get('gross_margin_pct',{}).get('mean','?')}% is below target",
                "metric": "Target gross margin above 30%"})
        result["recommendations"] = recs

        return result

    def _finding(self, type_: str, title: str, body: str, severity: str, confidence: float, impact: float) -> dict:
        return {"type": type_, "title": title, "body": body,
                "severity": severity, "confidence": confidence, "impact_score": impact}

    def _find(self, cols_lower: dict, keywords: list) -> Optional[str]:
        for kw in keywords:
            if kw in cols_lower:
                return cols_lower[kw]
        for kw in keywords:
            for cl, co in cols_lower.items():
                if kw in cl:
                    return co
        return None


accounting_service = AccountingAnalysisService()
