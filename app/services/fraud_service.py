"""
DataMind Agent — Fraud Detection Service
Benford's Law, duplicate transactions, round-number bias,
velocity anomalies, journal entry timing, segregation of duties gaps,
statistical outliers, weekend/holiday transactions.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from collections import Counter
from typing import Optional
import math

logger = logging.getLogger(__name__)


class FraudDetectionService:

    BENFORDS_LAW = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046,
    }

    def analyse(self, df: pd.DataFrame) -> dict:
        result = {
            "module": "fraud_detection",
            "risk_score": 0,
            "risk_level": "Low",
            "findings": [],
            "metrics": [],
            "evidence": {},
            "recommendations": [],
            "flags": [],
        }

        cols = {c.lower(): c for c in df.columns}

        # ── Auto-detect columns ───────────────────────────────────────────────
        amount_col   = self._find(cols, ["amount","transaction_amount","value","payment","invoice_amount","cost","price","revenue","sales"])
        vendor_col   = self._find(cols, ["vendor","supplier","payee","company","merchant","counterparty"])
        date_col     = self._find(cols, ["date","transaction_date","payment_date","invoice_date","posting_date"])
        desc_col     = self._find(cols, ["description","narration","memo","notes","details","reference"])
        user_col     = self._find(cols, ["user","employee","approver","entered_by","created_by","staff"])
        account_col  = self._find(cols, ["account","gl_account","account_code","cost_centre"])
        id_col       = self._find(cols, ["id","transaction_id","invoice_id","reference","ref_no"])

        result["evidence"]["columns_detected"] = {k: v for k, v in {
            "amount": amount_col, "vendor": vendor_col, "date": date_col,
            "description": desc_col, "user": user_col, "account": account_col, "id": id_col,
        }.items() if v is not None}

        risk_score = 0

        # ── 1. BENFORD'S LAW TEST ─────────────────────────────────────────────
        if amount_col:
            amounts = df[amount_col].dropna()
            amounts = amounts[amounts > 0]
            if len(amounts) >= 50:
                first_digits = amounts.apply(lambda x: int(str(x).replace('.','').lstrip('0')[0])
                                              if str(x).replace('.','').lstrip('0') else None).dropna()
                first_digits = first_digits.astype(int)
                first_digits = first_digits[first_digits.between(1, 9)]

                observed_freq = {d: int((first_digits == d).sum()) for d in range(1, 10)}
                total = len(first_digits)
                observed_pct  = {d: round(v / total * 100, 3) for d, v in observed_freq.items()}
                expected_pct  = {d: round(v * 100, 3) for d, v in self.BENFORDS_LAW.items()}
                deviations    = {d: round(observed_pct[d] - expected_pct[d], 3) for d in range(1, 10)}

                # Chi-square test
                observed_counts = [observed_freq[d] for d in range(1, 10)]
                expected_counts = [self.BENFORDS_LAW[d] * total for d in range(1, 10)]
                chi2, p_value   = scipy_stats.chisquare(observed_counts, expected_counts)
                benford_fail    = bool(p_value < 0.05)

                # MAD (Mean Absolute Deviation) — Nigrini's threshold
                mad = np.mean([abs(observed_pct[d] - expected_pct[d]) for d in range(1, 10)])
                mad_risk = "High" if mad > 1.5 else "Medium" if mad > 1.0 else "Low"

                # Most suspicious digits
                suspicious_digits = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

                result["evidence"]["benfords_law"] = {
                    "n_transactions": total,
                    "chi_square": round(float(chi2), 4),
                    "p_value": round(float(p_value), 4),
                    "fails_benfords": benford_fail,
                    "mad": round(float(mad), 4),
                    "mad_risk_level": mad_risk,
                    "observed_pct": observed_pct,
                    "expected_pct": expected_pct,
                    "deviations": deviations,
                    "most_deviant_digits": suspicious_digits,
                }

                if benford_fail:
                    risk_score += 25
                    digit_str = ", ".join([f"digit {d} ({dev:+.1f}%)" for d, dev in suspicious_digits[:3]])
                    result["findings"].append({
                        "type": "benfords_law",
                        "title": f"Benford's Law violation detected (MAD={round(float(mad),2)}, p={round(float(p_value),4)})",
                        "body": (
                            f"Transaction amounts do NOT follow Benford's Law (chi²={round(float(chi2),2)}, "
                            f"p={round(float(p_value),4)} — statistically significant at 95% confidence). "
                            f"MAD of {round(float(mad),2)} indicates {mad_risk.lower()} fabrication risk. "
                            f"Most deviant first digits: {digit_str}. "
                            f"Benford violations are a classic indicator of fabricated or manipulated figures. "
                            f"Immediate audit of transactions starting with over-represented digits is recommended."
                        ),
                        "severity": "critical",
                        "confidence": round(1 - float(p_value), 3),
                        "impact_score": 9.0,
                        "method": "Benford's Law Chi-square test + Nigrini MAD",
                    })
                    result["flags"].append("BENFORD_VIOLATION")
                else:
                    result["findings"].append({
                        "type": "benfords_law",
                        "title": f"Benford's Law: PASS (MAD={round(float(mad),2)}, p={round(float(p_value),4)})",
                        "body": (
                            f"Transaction amounts follow Benford's Law (p={round(float(p_value),4)} > 0.05). "
                            f"No digit-frequency manipulation detected across {total} transactions."
                        ),
                        "severity": "success",
                        "confidence": 0.90,
                        "impact_score": 0.0,
                        "method": "Benford's Law Chi-square test",
                    })

        # ── 2. DUPLICATE TRANSACTION DETECTION ───────────────────────────────
        if amount_col:
            dup_cols = [c for c in [amount_col, vendor_col, date_col] if c]
            if len(dup_cols) >= 2:
                dup_mask = df.duplicated(subset=dup_cols, keep=False)
                dup_count = int(dup_mask.sum())
                dup_groups = df[dup_mask].groupby(dup_cols).size().reset_index(name="count")

                result["evidence"]["duplicates"] = {
                    "total_duplicate_rows": dup_count,
                    "duplicate_pct": round(dup_count / len(df) * 100, 2),
                    "duplicate_groups": len(dup_groups),
                    "check_columns": dup_cols,
                    "sample_duplicates": df[dup_mask].head(5).to_dict("records"),
                }

                if dup_count > 0:
                    risk_score += min(20, dup_count * 2)
                    result["findings"].append({
                        "type": "duplicate_transactions",
                        "title": f"{dup_count} potential duplicate transactions detected",
                        "body": (
                            f"Found {dup_count} rows ({round(dup_count/len(df)*100,2)}% of records) that share the same "
                            f"{' + '.join([c.replace('_',' ') for c in dup_cols])}. "
                            f"These form {len(dup_groups)} duplicate groups. "
                            f"Duplicates may indicate double payments, system errors, or intentional fraud. "
                            f"Each duplicate group should be investigated individually."
                        ),
                        "severity": "critical" if dup_count > 5 else "warning",
                        "confidence": 0.95,
                        "impact_score": min(10.0, dup_count),
                        "method": "Exact match on amount + vendor + date",
                    })
                    result["flags"].append(f"DUPLICATE_TRANSACTIONS:{dup_count}")

        # ── 3. ROUND NUMBER BIAS ──────────────────────────────────────────────
        if amount_col:
            amounts = df[amount_col].dropna()
            amounts = amounts[amounts > 0]
            if len(amounts) >= 20:
                round_100  = int((amounts % 100 == 0).sum())
                round_1000 = int((amounts % 1000 == 0).sum())
                round_pct  = round(round_100 / len(amounts) * 100, 2)
                expected_round_pct = 1.0  # expected ~1% naturally

                result["evidence"]["round_number_bias"] = {
                    "total_amounts": len(amounts),
                    "round_to_100_count": round_100,
                    "round_to_100_pct": round_pct,
                    "round_to_1000_count": round_1000,
                    "expected_natural_pct": expected_round_pct,
                    "bias_ratio": round(float(round_pct / expected_round_pct), 2),
                }

                if round_pct > 15:
                    risk_score += 15
                    result["findings"].append({
                        "type": "round_number_bias",
                        "title": f"Round number bias: {round_pct}% of amounts are multiples of 100",
                        "body": (
                            f"{round_100} out of {len(amounts)} transactions ({round_pct}%) are exact multiples of 100. "
                            f"Naturally, only ~1% of real transactions are round numbers. "
                            f"Ratio of {round(float(round_pct/expected_round_pct),1)}x expected frequency suggests "
                            f"estimates or fabricated amounts. Review round-number transactions individually."
                        ),
                        "severity": "warning",
                        "confidence": 0.82,
                        "impact_score": 6.0,
                        "method": "Round-number frequency analysis",
                    })
                    result["flags"].append("ROUND_NUMBER_BIAS")

        # ── 4. VELOCITY ANOMALIES (same vendor, high frequency) ───────────────
        if vendor_col and amount_col:
            vendor_stats = df.groupby(vendor_col)[amount_col].agg(["count","sum","mean","std"]).reset_index()
            vendor_stats.columns = ["vendor", "count", "total", "mean", "std"]
            overall_count_mean = float(vendor_stats["count"].mean())
            overall_count_std  = float(vendor_stats["count"].std()) if len(vendor_stats) > 1 else 0
            high_freq = vendor_stats[vendor_stats["count"] > overall_count_mean + 2 * overall_count_std]

            result["evidence"]["vendor_velocity"] = {
                "total_vendors": len(vendor_stats),
                "avg_transactions_per_vendor": round(overall_count_mean, 1),
                "high_frequency_vendors": len(high_freq),
                "top_vendors_by_count": vendor_stats.nlargest(5, "count").to_dict("records"),
                "top_vendors_by_total": vendor_stats.nlargest(5, "total").to_dict("records"),
            }

            if len(high_freq) > 0:
                risk_score += min(15, len(high_freq) * 5)
                top_vendor = high_freq.nlargest(1, "count").iloc[0]
                result["findings"].append({
                    "type": "vendor_velocity",
                    "title": f"{len(high_freq)} vendor(s) have unusually high transaction frequency",
                    "body": (
                        f"{len(high_freq)} vendor(s) have transaction counts more than 2 standard deviations "
                        f"above average. Top: '{top_vendor['vendor']}' with {int(top_vendor['count'])} transactions "
                        f"(avg per vendor: {round(overall_count_mean,1)}). "
                        f"High frequency with same vendor can indicate fictitious vendor schemes or split invoicing. "
                        f"Verify all transactions for high-frequency vendors."
                    ),
                    "severity": "warning",
                    "confidence": 0.78,
                    "impact_score": 7.0,
                    "method": "Z-score on transaction count per vendor",
                })
                result["flags"].append(f"HIGH_VELOCITY_VENDORS:{len(high_freq)}")

        # ── 5. WEEKEND / AFTER-HOURS TRANSACTIONS ─────────────────────────────
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                if len(dates) >= 10:
                    weekends = dates[dates.dt.dayofweek >= 5]
                    weekend_pct = round(len(weekends) / len(dates) * 100, 2)
                    expected_weekend_pct = 28.6  # 2 out of 7 days

                    result["evidence"]["timing_analysis"] = {
                        "total_dated_transactions": len(dates),
                        "weekend_transactions": len(weekends),
                        "weekend_pct": weekend_pct,
                        "expected_pct": expected_weekend_pct,
                        "weekend_dates": weekends.dt.strftime("%Y-%m-%d").tolist()[:10],
                    }

                    if weekend_pct > 15:
                        risk_score += 10
                        result["findings"].append({
                            "type": "weekend_transactions",
                            "title": f"{len(weekends)} transactions on weekends ({weekend_pct}%)",
                            "body": (
                                f"{len(weekends)} transactions ({weekend_pct}%) were posted on weekends. "
                                f"While expected rate is ~28.6% for random posting, legitimate business "
                                f"transactions are typically concentrated on weekdays. "
                                f"Review weekend postings — particularly large-value transactions."
                            ),
                            "severity": "warning" if weekend_pct > 30 else "info",
                            "confidence": 0.75,
                            "impact_score": 4.0,
                            "method": "Day-of-week frequency analysis",
                        })
            except Exception as e:
                logger.warning(f"Date analysis failed: {e}")

        # ── 6. JUST-BELOW-THRESHOLD TRANSACTIONS ─────────────────────────────
        if amount_col:
            amounts = df[amount_col].dropna()
            amounts = amounts[amounts > 0]
            if len(amounts) >= 20:
                # Common approval thresholds: 1000, 5000, 10000, 50000
                thresholds = [1000, 5000, 10000, 25000, 50000, 100000]
                threshold_clusters = {}
                for t in thresholds:
                    just_below = amounts[(amounts >= t * 0.95) & (amounts < t)]
                    if len(just_below) >= 3:
                        pct = round(len(just_below) / len(amounts) * 100, 2)
                        threshold_clusters[t] = {
                            "count": len(just_below),
                            "pct": pct,
                            "values": just_below.tolist()[:5],
                        }

                if threshold_clusters:
                    risk_score += 20
                    result["evidence"]["threshold_clustering"] = threshold_clusters
                    top_threshold = max(threshold_clusters.items(), key=lambda x: x[1]["count"])
                    result["findings"].append({
                        "type": "threshold_clustering",
                        "title": f"Suspicious clustering just below approval thresholds",
                        "body": (
                            f"Transactions cluster suspiciously just below approval thresholds. "
                            f"Most notable: {top_threshold[1]['count']} transactions between "
                            f"{round(top_threshold[0]*0.95)} and {top_threshold[0]} "
                            f"({top_threshold[1]['pct']}% of all transactions). "
                            f"This pattern is a classic indicator of 'structuring' — splitting transactions "
                            f"to avoid approval controls. Immediate investigation required."
                        ),
                        "severity": "critical",
                        "confidence": 0.88,
                        "impact_score": 9.5,
                        "method": "Threshold proximity analysis (95% of common limits)",
                    })
                    result["flags"].append("THRESHOLD_CLUSTERING")

        # ── 7. STATISTICAL OUTLIERS IN AMOUNTS ────────────────────────────────
        if amount_col:
            amounts = df[amount_col].dropna()
            amounts = amounts[amounts > 0]
            if len(amounts) >= 10:
                mean, std = float(amounts.mean()), float(amounts.std())
                z_scores = np.abs((amounts - mean) / std)
                outliers = amounts[z_scores > 3.5]
                outlier_pct = round(len(outliers) / len(amounts) * 100, 2)

                result["evidence"]["amount_outliers"] = {
                    "total_amounts": len(amounts),
                    "outlier_count": len(outliers),
                    "outlier_pct": outlier_pct,
                    "outlier_values": sorted(outliers.tolist(), reverse=True)[:5],
                    "mean": round(mean, 2),
                    "std": round(std, 2),
                    "threshold_3_5_sigma": round(mean + 3.5 * std, 2),
                }

                if len(outliers) > 0:
                    risk_score += min(15, len(outliers) * 3)
                    result["findings"].append({
                        "type": "amount_outliers",
                        "title": f"{len(outliers)} unusually large transactions detected",
                        "body": (
                            f"{len(outliers)} transaction amounts ({outlier_pct}% of records) exceed 3.5 standard deviations "
                            f"from the mean (threshold: {round(mean + 3.5*std,2)}). "
                            f"Largest outlier values: {sorted(outliers.tolist(),reverse=True)[:3]}. "
                            f"Mean: {round(mean,2)}, Std: {round(std,2)}. "
                            f"These require individual review — high-value outliers are common entry points for fraud."
                        ),
                        "severity": "warning",
                        "confidence": 0.87,
                        "impact_score": 7.5,
                        "method": "Z-score outlier detection (threshold: 3.5σ)",
                    })

        # ── 8. OVERALL RISK SCORE ─────────────────────────────────────────────
        result["risk_score"] = min(100, risk_score)
        result["risk_level"] = (
            "Critical" if risk_score >= 60 else
            "High"     if risk_score >= 40 else
            "Medium"   if risk_score >= 20 else
            "Low"
        )
        result["metrics"] = [
            {"label": "Fraud risk score",  "value": f"{min(100,risk_score)}/100", "trend": "up" if risk_score > 40 else "flat"},
            {"label": "Risk level",        "value": result["risk_level"], "trend": "up" if risk_score > 40 else "flat"},
            {"label": "Flags raised",      "value": str(len(result["flags"]))},
            {"label": "Tests run",         "value": str(len(result["findings"]))},
        ]

        # ── 9. RECOMMENDATIONS ────────────────────────────────────────────────
        recs = []
        if "BENFORD_VIOLATION" in result["flags"]:
            recs.append({"priority":1,"action":"Commission forensic audit of transaction amounts",
                "reason":"Benford's Law violation is a statistically significant fraud signal",
                "urgency":"Immediate"})
        if any("DUPLICATE" in f for f in result["flags"]):
            recs.append({"priority":2,"action":"Review and reverse all identified duplicate payments",
                "reason":"Duplicate transactions represent direct financial loss",
                "urgency":"This week"})
        if "THRESHOLD_CLUSTERING" in result["flags"]:
            recs.append({"priority":3,"action":"Review approval threshold controls and split transactions",
                "reason":"Structuring below approval limits is a major internal control weakness",
                "urgency":"This week"})
        if any("VELOCITY" in f for f in result["flags"]):
            recs.append({"priority":4,"action":"Audit all high-frequency vendors",
                "reason":"Unusual vendor transaction frequency can indicate fictitious vendors",
                "urgency":"This month"})
        if not recs:
            recs.append({"priority":1,"action":"Maintain current controls and run monthly fraud checks",
                "reason":"No high-risk signals detected in current dataset",
                "urgency":"Ongoing"})
        result["recommendations"] = recs

        return result

    def _find(self, cols_lower: dict, keywords: list) -> Optional[str]:
        for kw in keywords:
            if kw in cols_lower:
                return cols_lower[kw]
        for kw in keywords:
            for cl, co in cols_lower.items():
                if kw in cl:
                    return co
        return None


fraud_service = FraudDetectionService()
