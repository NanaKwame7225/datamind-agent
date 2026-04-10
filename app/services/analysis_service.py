from __future__ import annotations
import logging, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class EliteDataAnalysisService:
    """
    Elite-grade data analysis service.
    Every claim is grounded in actual data computation.
    Every finding includes evidence, impact rank, confidence score, and uncertainty.
    """

    # ── DATA CLEANING ─────────────────────────────────────────────────────────

    def clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        original_shape = df.shape
        report = {
            "original_rows": original_shape[0],
            "original_cols": original_shape[1],
            "steps": [],
            "evidence": {},
        }
        df = df.copy()

        # Step 1 — Strip whitespace with exact counts
        str_cols = df.select_dtypes(include="object").columns.tolist()
        whitespace_found = {}
        for col in str_cols:
            before = df[col].astype(str)
            df[col] = before.str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
            changed = int((before != df[col].astype(str)).sum())
            if changed > 0:
                whitespace_found[col] = changed
        report["steps"].append({
            "step": "Whitespace strip",
            "library": "Pandas",
            "affected_columns": len(str_cols),
            "cells_cleaned": sum(whitespace_found.values()),
            "evidence": whitespace_found,
        })

        # Step 2 — Type inference with success rates
        converted = []
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(20).tolist()
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                converted.append({"col": col, "to": "datetime", "sample": str(sample[:3])})
                continue
            except Exception:
                pass
            try:
                numeric_attempt = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce"
                )
                success_rate = numeric_attempt.notna().mean()
                if success_rate > 0.8:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce"
                    )
                    converted.append({"col": col, "to": "numeric",
                                      "success_rate_pct": round(success_rate * 100, 1),
                                      "failed_values": int((numeric_attempt.isna() & df[col].notna()).sum())})
                else:
                    report["evidence"][f"{col}_type_issue"] = {
                        "issue": f"{round((1-success_rate)*100,1)}% of values in '{col}' are non-numeric",
                        "failed_count": int(numeric_attempt.isna().sum()),
                        "total": len(df[col]),
                        "sample_bad_values": df[col][numeric_attempt.isna()].head(5).tolist(),
                        "severity": "high" if success_rate < 0.5 else "medium",
                    }
            except Exception:
                pass
        report["steps"].append({
            "step": "Type inference",
            "library": "Pandas",
            "conversions": converted,
            "columns_with_issues": len(report["evidence"]),
        })

        # Step 3 — Duplicate removal with exact counts and sample
        dup_mask = df.duplicated()
        dupes_count = int(dup_mask.sum())
        dup_sample = df[dup_mask].head(3).to_dict("records") if dupes_count > 0 else []
        df = df.drop_duplicates()
        report["steps"].append({
            "step": "Duplicate removal",
            "library": "Pandas",
            "rows_removed": dupes_count,
            "pct_of_total": round(dupes_count / max(len(df), 1) * 100, 2),
            "sample_duplicates": dup_sample,
        })

        # Step 4 — Imputation with exact stats
        numeric_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        imputed = {}
        for col in numeric_cols:
            n_missing = int(df[col].isnull().sum())
            if n_missing > 0:
                pct_missing = round(n_missing / len(df) * 100, 2)
                median_val = df[col].median()
                mean_val = df[col].mean()
                df[col] = df[col].fillna(median_val)
                imputed[col] = {
                    "missing_count": n_missing,
                    "missing_pct": pct_missing,
                    "filled_with": "median",
                    "median_value": round(float(median_val), 4),
                    "mean_value": round(float(mean_val), 4),
                    "severity": "high" if pct_missing > 20 else "medium" if pct_missing > 5 else "low",
                }
        for col in cat_cols:
            n_missing = int(df[col].isnull().sum())
            if n_missing > 0:
                pct_missing = round(n_missing / len(df) * 100, 2)
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                imputed[col] = {
                    "missing_count": n_missing,
                    "missing_pct": pct_missing,
                    "filled_with": f"mode ({mode_val})",
                    "severity": "high" if pct_missing > 20 else "medium" if pct_missing > 5 else "low",
                }
        report["steps"].append({
            "step": "Missing value imputation",
            "library": "Pandas + NumPy",
            "columns_imputed": imputed,
            "total_cells_filled": sum(v["missing_count"] for v in imputed.values()),
        })

        # Step 5 — Winsorisation with exact impact
        winsorised = {}
        for col in df.select_dtypes(include="number").columns:
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue
            lower, upper = mean - 3 * std, mean + 3 * std
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                outlier_vals = df[col][outlier_mask].tolist()
                df[col] = df[col].clip(lower=lower, upper=upper)
                winsorised[col] = {
                    "capped_count": n_outliers,
                    "pct_affected": round(n_outliers / len(df) * 100, 2),
                    "cap_range": [round(float(lower), 2), round(float(upper), 2)],
                    "extreme_values_found": [round(float(v), 2) for v in outlier_vals[:5]],
                }
        report["steps"].append({
            "step": "Outlier winsorisation",
            "library": "Pandas + NumPy",
            "columns_winsorised": winsorised,
        })

        # Step 6 — Skewness with measured values
        transformed = {}
        for col in df.select_dtypes(include="number").columns:
            try:
                skewness = float(df[col].skew())
                if abs(skewness) > 1.5 and df[col].min() > 0:
                    df[col + "_log"] = np.log1p(df[col])
                    transformed[col] = {
                        "original_skewness": round(skewness, 3),
                        "new_column": col + "_log",
                        "direction": "right-skewed" if skewness > 0 else "left-skewed",
                    }
            except Exception:
                pass
        report["steps"].append({
            "step": "Skewness correction",
            "library": "NumPy + SciPy",
            "columns_transformed": transformed,
        })

        report["final_rows"] = len(df)
        report["final_cols"] = len(df.columns)
        report["rows_removed_total"] = original_shape[0] - len(df)
        return df, report

    # ── ELITE ANALYSIS ────────────────────────────────────────────────────────

    def elite_analyse(self, df: pd.DataFrame, query: str, industry: str) -> dict:
        """
        Full elite analysis pipeline.
        Returns grounded findings with evidence, impact ranks, confidence scores,
        segmentation breakdowns, and self-audit uncertainty flags.
        """
        result = {
            "query": query,
            "industry": industry,
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": list(df.columns),
            "findings": [],
            "impact_ranking": [],
            "segmentation": {},
            "distributions": {},
            "correlations": {},
            "uncertainty": [],
            "self_audit": [],
            "data_grounding": {},
        }

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # ── 1. DISTRIBUTIONS (actual data, not claims) ────────────────────────
        for col in numeric_cols[:6]:
            series = df[col].dropna()
            if len(series) < 2:
                continue
            skewness = float(series.skew())
            kurtosis = float(series.kurt())
            _, p_normal = stats.normaltest(series) if len(series) > 8 else (None, None)
            result["distributions"][col] = {
                "count": int(series.count()),
                "missing": int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().mean() * 100, 2),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "p25": round(float(series.quantile(0.25)), 4),
                "p75": round(float(series.quantile(0.75)), 4),
                "p95": round(float(series.quantile(0.95)), 4),
                "skewness": round(skewness, 3),
                "kurtosis": round(kurtosis, 3),
                "is_normal": bool(p_normal > 0.05) if p_normal is not None else None,
                "cv_pct": round(float(series.std() / series.mean() * 100), 2) if series.mean() != 0 else None,
                "distribution_shape": (
                    "heavily right-skewed" if skewness > 2 else
                    "right-skewed" if skewness > 1 else
                    "roughly symmetric" if abs(skewness) < 0.5 else
                    "left-skewed" if skewness < -1 else "slight left skew"
                ),
            }

        # ── 2. ANOMALY DETECTION with evidence ───────────────────────────────
        anomaly_impacts = {}
        for col in numeric_cols[:5]:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            mean, std = series.mean(), series.std()
            if std == 0:
                continue
            z_scores = np.abs((series - mean) / std)
            anomaly_mask = z_scores > 3
            anomaly_count = int(anomaly_mask.sum())
            anomaly_pct = round(anomaly_mask.mean() * 100, 2)

            if anomaly_count > 0:
                anomaly_vals = series[anomaly_mask].tolist()
                anomaly_indices = series[anomaly_mask].index.tolist()

                # Impact: how much do anomalies affect the mean?
                mean_with = series.mean()
                mean_without = series[~anomaly_mask].mean()
                impact_on_mean_pct = abs(mean_with - mean_without) / max(abs(mean_without), 1e-10) * 100

                # Try to get context rows for anomalies
                context_rows = []
                for idx in anomaly_indices[:3]:
                    try:
                        row = df.loc[idx].to_dict()
                        context_rows.append({k: round(float(v), 2) if isinstance(v, float) else v
                                             for k, v in row.items()})
                    except Exception:
                        pass

                confidence = min(0.99, 0.7 + (anomaly_pct / 100) * 0.3) if anomaly_pct < 10 else 0.95
                impact_score = min(10, (impact_on_mean_pct / 10) + (anomaly_pct / 2))

                anomaly_impacts[col] = impact_score
                result["findings"].append({
                    "type": "anomaly",
                    "column": col,
                    "title": f"Unusual values in {col.replace('_', ' ')}",
                    "evidence": {
                        "anomaly_count": anomaly_count,
                        "anomaly_pct": anomaly_pct,
                        "anomaly_values": [round(float(v), 2) for v in anomaly_vals[:5]],
                        "normal_range": [round(float(mean - 3*std), 2), round(float(mean + 3*std), 2)],
                        "impact_on_mean_pct": round(impact_on_mean_pct, 2),
                        "context_rows": context_rows,
                        "z_score_max": round(float(z_scores.max()), 2),
                    },
                    "impact_score": round(impact_score, 2),
                    "confidence": round(confidence, 2),
                    "severity": "critical" if anomaly_pct > 5 else "warning",
                    "method": "Z-score (threshold: 3σ)",
                })

        # ── 3. TREND ANALYSIS with statistical validation ────────────────────
        time_col = self._detect_time_column(df)
        for col in numeric_cols[:4]:
            series = df[col].dropna().reset_index(drop=True)
            if len(series) < 4:
                continue
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            r_squared = r_value ** 2
            trend_pct = (slope * len(series)) / max(abs(series.mean()), 1e-10) * 100
            confidence_in_trend = 1 - p_value if p_value is not None else 0.5

            result["findings"].append({
                "type": "trend",
                "column": col,
                "title": f"{'Upward' if slope > 0 else 'Downward'} trend in {col.replace('_', ' ')}",
                "evidence": {
                    "slope_per_period": round(float(slope), 4),
                    "total_change_pct": round(float(trend_pct), 2),
                    "r_squared": round(float(r_squared), 4),
                    "p_value": round(float(p_value), 4) if p_value else None,
                    "statistically_significant": bool(p_value < 0.05) if p_value else None,
                    "first_value": round(float(series.iloc[0]), 2),
                    "last_value": round(float(series.iloc[-1]), 2),
                    "period_count": len(series),
                    "std_error": round(float(std_err), 4),
                },
                "impact_score": round(min(10, abs(trend_pct) / 10), 2),
                "confidence": round(min(0.99, confidence_in_trend), 2),
                "severity": "critical" if abs(trend_pct) > 30 else "warning" if abs(trend_pct) > 10 else "info",
                "method": "Linear regression (OLS)",
            })

        # ── 4. CORRELATIONS with strength classification ─────────────────────
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            strong_pairs = []
            for i, c1 in enumerate(numeric_cols):
                for c2 in numeric_cols[i+1:]:
                    r = float(corr_matrix.loc[c1, c2])
                    if abs(r) > 0.5 and not np.isnan(r):
                        n = df[[c1, c2]].dropna().shape[0]
                        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2) if abs(r) < 1 else np.inf
                        p_val = float(2 * stats.t.sf(abs(t_stat), df=n-2))
                        strong_pairs.append({
                            "col1": c1, "col2": c2,
                            "correlation": round(r, 4),
                            "strength": "very strong" if abs(r) > 0.8 else "strong" if abs(r) > 0.6 else "moderate",
                            "direction": "positive" if r > 0 else "negative",
                            "p_value": round(p_val, 4),
                            "n_observations": n,
                            "significant": bool(p_val < 0.05),
                            "interpretation": f"When {c1.replace('_',' ')} goes up, {c2.replace('_',' ')} tends to {'go up too' if r > 0 else 'go down'}",
                        })
            result["correlations"] = sorted(strong_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:6]

        # ── 5. SEGMENTATION by categorical columns ────────────────────────────
        for cat_col in cat_cols[:3]:
            unique_vals = df[cat_col].nunique()
            if unique_vals < 2 or unique_vals > 30:
                continue
            seg = {}
            for num_col in numeric_cols[:3]:
                seg_stats = []
                overall_mean = df[num_col].mean()
                overall_std = df[num_col].std()
                for group_val, group_df in df.groupby(cat_col):
                    grp_series = group_df[num_col].dropna()
                    if len(grp_series) < 2:
                        continue
                    grp_mean = float(grp_series.mean())
                    deviation_from_overall = (grp_mean - overall_mean) / max(abs(overall_mean), 1e-10) * 100
                    # T-test vs rest
                    rest = df[df[cat_col] != group_val][num_col].dropna()
                    if len(rest) > 1:
                        t_stat, p_val = stats.ttest_ind(grp_series, rest, equal_var=False)
                        significant = bool(p_val < 0.05)
                    else:
                        p_val, significant = 1.0, False

                    seg_stats.append({
                        "segment": str(group_val),
                        "count": int(len(grp_series)),
                        "pct_of_total": round(len(grp_series) / len(df) * 100, 1),
                        "mean": round(grp_mean, 4),
                        "median": round(float(grp_series.median()), 4),
                        "std": round(float(grp_series.std()), 4),
                        "deviation_from_overall_pct": round(deviation_from_overall, 2),
                        "statistically_different": significant,
                        "p_value": round(float(p_val), 4),
                        "rank": 0,
                    })
                seg_stats.sort(key=lambda x: x["mean"], reverse=True)
                for i, s in enumerate(seg_stats):
                    s["rank"] = i + 1
                seg[num_col] = seg_stats

            if seg:
                result["segmentation"][cat_col] = {
                    "unique_segments": int(unique_vals),
                    "metrics": seg,
                }

        # ── 6. IMPACT RANKING ─────────────────────────────────────────────────
        impact_scores = {}
        for f in result["findings"]:
            col = f["column"]
            if col not in impact_scores or f["impact_score"] > impact_scores[col]["score"]:
                impact_scores[col] = {
                    "column": col,
                    "score": f["impact_score"],
                    "confidence": f["confidence"],
                    "primary_issue": f["type"],
                    "reason": f['title'],
                }
        result["impact_ranking"] = sorted(impact_scores.values(), key=lambda x: x["score"], reverse=True)[:5]

        # ── 7. UNCERTAINTY SCORING ────────────────────────────────────────────
        uncertainties = []
        n_rows = len(df)
        if n_rows < 30:
            uncertainties.append({
                "issue": "Small sample size",
                "detail": f"Only {n_rows} rows of data. Statistical conclusions may not generalise — patterns could be coincidental.",
                "affected_analyses": ["trend analysis", "correlations", "anomaly detection"],
                "confidence_adjustment": -0.2,
                "recommendation": "Collect more data (minimum 50-100 rows for reliable analysis).",
            })
        if n_rows < 100:
            uncertainties.append({
                "issue": "Limited data volume",
                "detail": f"{n_rows} rows is below the recommended minimum of 100 for robust statistical analysis.",
                "affected_analyses": ["segmentation", "forecasting"],
                "confidence_adjustment": -0.1,
                "recommendation": "Analysis is indicative — validate conclusions with more data.",
            })
        for col, info in result["distributions"].items():
            if info["missing_pct"] > 20:
                uncertainties.append({
                    "issue": f"High missing data in {col.replace('_',' ')}",
                    "detail": f"{info['missing_pct']}% of values are missing ({info['missing']} out of {info['count'] + info['missing']} records). Imputation may have introduced bias.",
                    "affected_analyses": [f"any analysis involving {col}"],
                    "confidence_adjustment": -0.15,
                    "recommendation": f"Investigate why {col.replace('_',' ')} is missing. Check if absence is systematic.",
                })
        if not cat_cols:
            uncertainties.append({
                "issue": "No segmentation variables available",
                "detail": "Dataset has no categorical columns (e.g. region, department, product). Cannot segment results — insights are averages that may hide variation.",
                "affected_analyses": ["segmentation analysis"],
                "confidence_adjustment": -0.05,
                "recommendation": "Add categorical identifiers (region, category, team) to enable segment-level insights.",
            })
        result["uncertainty"] = uncertainties

        # ── 8. SELF-AUDIT ─────────────────────────────────────────────────────
        self_audit = []
        self_audit.append({
            "question": "What assumptions am I making?",
            "assumptions": [
                "Data provided is a representative sample of the full population",
                "Missing values are missing at random (not systematically missing for a reason)",
                "The time ordering of rows reflects actual chronological sequence" if not time_col else f"Column '{time_col}' correctly represents time ordering",
                "Relationships found are correlational — not necessarily causal",
            ],
        })
        self_audit.append({
            "question": "What could I be wrong about?",
            "risks": [
                f"Anomalies flagged could be legitimate outliers (real peaks/troughs) not errors — context needed from business",
                f"Trends found may be seasonal patterns misread as directional — need more periods to confirm",
                f"Correlations found between variables may be driven by a third hidden variable",
                f"Segment differences flagged may reflect different data quality per segment, not true performance differences",
            ],
        })
        self_audit.append({
            "question": "What information would improve this analysis?",
            "missing_context": [
                "Industry benchmarks to compare these numbers against",
                "Business targets or KPI thresholds",
                "External factors (market events, seasonal periods) that could explain spikes",
                "Definition of each column — what exactly is being measured",
            ],
        })
        result["self_audit"] = self_audit

        # ── 9. DATA GROUNDING SUMMARY ─────────────────────────────────────────
        result["data_grounding"] = {
            "total_rows_analysed": n_rows,
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(cat_cols),
            "total_data_points": n_rows * len(df.columns),
            "total_anomalies_found": sum(f.get("evidence", {}).get("anomaly_count", 0) for f in result["findings"] if f["type"] == "anomaly"),
            "total_correlations_found": len(result["correlations"]),
            "segments_analysed": sum(len(v["metrics"]) for v in result["segmentation"].values()),
            "highest_confidence_finding": max((f["confidence"] for f in result["findings"]), default=0),
            "lowest_confidence_finding": min((f["confidence"] for f in result["findings"]), default=0),
        }

        return result

    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        time_keywords = ["date", "month", "week", "quarter", "period", "year", "time"]
        for col in df.columns:
            if any(kw in col.lower() for kw in time_keywords):
                return col
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return None

    # ── EXISTING METHODS (unchanged) ─────────────────────────────────────────

    def describe(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include="number")
        categorical = df.select_dtypes(include=["object", "category"])
        result = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "numeric_stats": {},
            "categorical_stats": {},
        }
        if not numeric.empty:
            result["numeric_stats"] = numeric.describe().round(4).to_dict()
        for col in categorical.columns:
            result["categorical_stats"][col] = {
                "unique": int(df[col].nunique()),
                "top": str(df[col].mode()[0]) if not df[col].mode().empty else None,
            }
        return result

    def quality_report(self, df: pd.DataFrame) -> dict:
        total = df.shape[0] * df.shape[1]
        missing = int(df.isnull().sum().sum())
        dupes = int(df.duplicated().sum())
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        outliers = {}
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            n = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            if n > 0:
                outliers[col] = n
        completeness = round((1 - missing / max(total, 1)) * 100, 2)
        return {
            "completeness_pct": completeness,
            "missing_cells": missing,
            "duplicate_rows": dupes,
            "outlier_columns": outliers,
            "overall_score": round(completeness * 0.5 + (1 - dupes / max(len(df), 1)) * 100 * 0.3 + (1 - len(outliers) / max(len(numeric_cols), 1)) * 100 * 0.2, 1),
        }

    def detect_anomalies(self, df: pd.DataFrame, column: str, method: str = "zscore") -> dict:
        series = df[column].dropna()
        if method == "zscore":
            z = np.abs((series - series.mean()) / series.std())
            mask = z > 3
        elif method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1, 1))
            mask = pd.Series(preds == -1, index=series.index)
        else:
            mask = pd.Series(False, index=series.index)
        return {"method": method, "column": column, "total_records": len(series),
                "anomaly_count": int(mask.sum()), "anomaly_pct": round(mask.mean() * 100, 2)}

    def forecast_arima(self, series: pd.Series, periods: int = 12) -> dict:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        adf = adfuller(series.dropna())
        d = 0 if adf[1] < 0.05 else 1
        model = ARIMA(series.dropna(), order=(1, d, 1)).fit()
        forecast = model.forecast(steps=periods)
        ci = model.get_forecast(steps=periods).conf_int()
        return {"model": f"ARIMA(1,{d},1)", "aic": round(model.aic, 2),
                "forecast": [round(float(v), 4) for v in forecast],
                "lower_bound": [round(float(v), 4) for v in ci.iloc[:, 0]],
                "upper_bound": [round(float(v), 4) for v in ci.iloc[:, 1]]}

    def cluster(self, df: pd.DataFrame, features: list, n_clusters: int = 4) -> dict:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        X = df[features].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        df2 = X.copy()
        df2["_cluster"] = labels
        clusters = {}
        for i in range(n_clusters):
            grp = df2[df2["_cluster"] == i][features]
            clusters[f"cluster_{i}"] = {"size": int((labels == i).sum()), "means": grp.mean().round(4).to_dict()}
        return {"model": "K-Means", "n_clusters": n_clusters,
                "silhouette_score": round(float(sil), 4), "clusters": clusters}

    def forecast_xgboost(self, df, target, features):
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        X = df[features].fillna(0)
        y = df[target].ffill()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        return {"model": "XGBoost", "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
                "r2": round(float(r2_score(y_test, y_pred)), 4),
                "feature_importances": dict(zip(features, model.feature_importances_.round(4).tolist()))}

    def df_to_records(self, df: pd.DataFrame, limit: int = 50) -> list:
        return json.loads(df.head(limit).to_json(orient="records"))


analysis_service = EliteDataAnalysisService()
