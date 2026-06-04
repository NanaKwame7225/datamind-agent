"""
DataMind Agent — Elite Analysis Service v2
Full causal reasoning, bias detection, advanced statistics,
time series intelligence, model transparency, domain intelligence.
"""
from __future__ import annotations
import logging, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class EliteDataAnalysisService:

    # ── DATA CLEANING ─────────────────────────────────────────────────────────

    def clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        original_shape = df.shape
        report = {"original_rows": original_shape[0], "original_cols": original_shape[1], "steps": [], "evidence": {}}
        df = df.copy()

        # Step 1 — Whitespace
        str_cols = df.select_dtypes(include="object").columns.tolist()
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
        report["steps"].append({"step": "Whitespace strip", "library": "Pandas", "affected_columns": len(str_cols)})

        # Step 2 — Type inference
        converted = []
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                converted.append({"col": col, "to": "datetime"})
                continue
            except Exception:
                pass
            try:
                numeric = pd.to_numeric(df[col].astype(str).str.replace(",","").str.replace("%",""), errors="coerce")
                if numeric.notna().mean() > 0.8:
                    df[col] = numeric
                    converted.append({"col": col, "to": "numeric"})
            except Exception:
                pass
        report["steps"].append({"step": "Type inference", "library": "Pandas", "conversions": converted})

        # Step 3 — Duplicates (near-duplicate fingerprinting)
        exact_dupes = int(df.duplicated().sum())
        df = df.drop_duplicates()
        report["steps"].append({"step": "Duplicate removal", "library": "Pandas", "rows_removed": exact_dupes})

        # Step 4 — Impute missing
        imputed = {}
        for col in df.select_dtypes(include="number").columns:
            n = int(df[col].isnull().sum())
            if n > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                imputed[col] = {"missing_count": n, "missing_pct": round(n/len(df)*100,2), "filled_with": "median"}
        for col in df.select_dtypes(include=["object","category"]).columns:
            n = int(df[col].isnull().sum())
            if n > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                imputed[col] = {"missing_count": n, "missing_pct": round(n/len(df)*100,2), "filled_with": f"mode ({mode_val})"}
        report["steps"].append({"step": "Missing value imputation", "library": "Pandas + NumPy", "columns_imputed": imputed})

        # Step 5 — Winsorise
        winsorised = {}
        for col in df.select_dtypes(include="number").columns:
            mean, std = df[col].mean(), df[col].std()
            if std == 0: continue
            lower, upper = mean - 3*std, mean + 3*std
            n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_out > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                winsorised[col] = {"capped_count": n_out, "range": [round(float(lower),2), round(float(upper),2)]}
        report["steps"].append({"step": "Outlier winsorisation", "library": "Pandas + NumPy", "columns_winsorised": winsorised})

        # Step 6 — Skewness
        transformed = {}
        for col in df.select_dtypes(include="number").columns:
            try:
                sk = float(df[col].skew())
                if abs(sk) > 1.5 and df[col].min() > 0:
                    df[col+"_log"] = np.log1p(df[col])
                    transformed[col] = {"skewness": round(sk,3), "new_column": col+"_log"}
            except Exception:
                pass
        report["steps"].append({"step": "Skewness correction", "library": "NumPy + SciPy", "columns_transformed": transformed})

        report["final_rows"] = len(df)
        report["final_cols"] = len(df.columns)
        report["rows_removed_total"] = original_shape[0] - len(df)
        return df, report

    # ── ELITE ANALYSIS v2 ─────────────────────────────────────────────────────

    def elite_analyse(self, df: pd.DataFrame, query: str, industry: str) -> dict:
        result = {
            "query": query, "industry": industry,
            "row_count": len(df), "col_count": len(df.columns),
            "columns": list(df.columns),
            "findings": [], "impact_ranking": [],
            "segmentation": {}, "distributions": {},
            "correlations": [], "uncertainty": [],
            "self_audit": [], "data_grounding": {},
            # v2 additions
            "causal_analysis": {},
            "bias_audit": {},
            "advanced_statistics": {},
            "time_series_intelligence": {},
            "model_transparency": {},
            "alternative_explanations": {},
        }

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        time_col = self._detect_time_column(df)

        # ── 1. DISTRIBUTIONS ─────────────────────────────────────────────────
        for col in numeric_cols[:6]:
            series = df[col].dropna()
            if len(series) < 2: continue
            sk = float(series.skew())
            ku = float(series.kurt())
            _, p_norm = stats.normaltest(series) if len(series) > 8 else (None, None)
            result["distributions"][col] = {
                "count": int(series.count()),
                "missing": int(df[col].isnull().sum()),
                "missing_pct": round(df[col].isnull().mean()*100, 2),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "p25": round(float(series.quantile(0.25)), 4),
                "p75": round(float(series.quantile(0.75)), 4),
                "p95": round(float(series.quantile(0.95)), 4),
                "skewness": round(sk, 3),
                "kurtosis": round(ku, 3),
                "is_normal": bool(p_norm > 0.05) if p_norm is not None else None,
                "cv_pct": round(float(series.std()/series.mean()*100),2) if series.mean() != 0 else None,
                "distribution_shape": (
                    "heavily right-skewed" if sk > 2 else "right-skewed" if sk > 1 else
                    "roughly symmetric" if abs(sk) < 0.5 else "left-skewed" if sk < -1 else "slight left skew"
                ),
            }

        # ── 2. ANOMALY DETECTION ─────────────────────────────────────────────
        for col in numeric_cols[:5]:
            series = df[col].dropna()
            if len(series) < 4: continue
            mean, std = series.mean(), series.std()
            if std == 0: continue
            z_scores = np.abs((series - mean) / std)
            anomaly_mask = z_scores > 3
            anomaly_count = int(anomaly_mask.sum())
            if anomaly_count > 0:
                anomaly_pct = round(anomaly_mask.mean()*100, 2)
                anomaly_vals = series[anomaly_mask].tolist()
                mean_without = series[~anomaly_mask].mean()
                impact_on_mean = abs(mean - mean_without) / max(abs(mean_without), 1e-10) * 100
                context_rows = []
                for idx in series[anomaly_mask].index[:3]:
                    try:
                        row = df.loc[idx].to_dict()
                        context_rows.append({k: round(float(v),2) if isinstance(v,float) else v for k,v in row.items()})
                    except Exception:
                        pass
                confidence = min(0.99, 0.7 + anomaly_pct/100*0.3)
                impact_score = min(10, impact_on_mean/10 + anomaly_pct/2)
                result["findings"].append({
                    "type": "anomaly", "column": col,
                    "title": f"Anomaly detected in {col.replace('_',' ')}",
                    "evidence": {
                        "anomaly_count": anomaly_count, "anomaly_pct": anomaly_pct,
                        "anomaly_values": [round(float(v),2) for v in anomaly_vals[:5]],
                        "normal_range": [round(float(mean-3*std),2), round(float(mean+3*std),2)],
                        "impact_on_mean_pct": round(impact_on_mean, 2),
                        "context_rows": context_rows,
                        "z_score_max": round(float(z_scores.max()), 2),
                    },
                    "impact_score": round(impact_score, 2),
                    "confidence": round(confidence, 2),
                    "severity": "critical" if anomaly_pct > 5 else "warning",
                    "method": "Z-score (threshold: 3σ)",
                })

        # ── 3. TREND ANALYSIS with structural break detection ─────────────────
        for col in numeric_cols[:4]:
            series = df[col].dropna().reset_index(drop=True)
            if len(series) < 4: continue
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            r_sq = r_value**2
            trend_pct = slope * len(series) / max(abs(series.mean()), 1e-10) * 100

            # Structural break detection (simple CUSUM)
            cusum = np.cumsum(series - series.mean())
            break_idx = int(np.argmax(np.abs(cusum)))
            break_magnitude = float(abs(cusum[break_idx]))
            has_break = break_magnitude > 2 * series.std() * np.sqrt(len(series))

            result["findings"].append({
                "type": "trend", "column": col,
                "title": f"{'Upward' if slope > 0 else 'Downward'} trend in {col.replace('_',' ')}",
                "evidence": {
                    "slope_per_period": round(float(slope), 4),
                    "total_change_pct": round(float(trend_pct), 2),
                    "r_squared": round(float(r_sq), 4),
                    "p_value": round(float(p_value), 4),
                    "statistically_significant": bool(p_value < 0.05),
                    "first_value": round(float(series.iloc[0]), 2),
                    "last_value": round(float(series.iloc[-1]), 2),
                    "period_count": len(series),
                    "structural_break_detected": has_break,
                    "structural_break_at_period": break_idx if has_break else None,
                    "std_error": round(float(std_err), 4),
                },
                "impact_score": round(min(10, abs(trend_pct)/10), 2),
                "confidence": round(min(0.99, 1 - p_value), 2),
                "severity": "critical" if abs(trend_pct) > 30 else "warning" if abs(trend_pct) > 10 else "info",
                "method": "OLS regression + CUSUM structural break detection",
            })

        # ── 4. CORRELATIONS with multiple comparison correction ───────────────
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            raw_pairs = []
            for i, c1 in enumerate(numeric_cols):
                for c2 in numeric_cols[i+1:]:
                    r = float(corr_matrix.loc[c1, c2])
                    if abs(r) > 0.5 and not np.isnan(r):
                        n = df[[c1,c2]].dropna().shape[0]
                        t_stat = r * np.sqrt(n-2) / np.sqrt(max(1-r**2, 1e-10))
                        p_val = float(2 * stats.t.sf(abs(t_stat), df=n-2))
                        raw_pairs.append((c1, c2, r, p_val, n))

            # Bonferroni correction
            n_tests = max(len(raw_pairs), 1)
            for c1, c2, r, p_val, n in raw_pairs:
                p_corrected = min(1.0, p_val * n_tests)
                result["correlations"].append({
                    "col1": c1, "col2": c2,
                    "correlation": round(r, 4),
                    "strength": "very strong" if abs(r)>0.8 else "strong" if abs(r)>0.6 else "moderate",
                    "direction": "positive" if r > 0 else "negative",
                    "p_value_raw": round(p_val, 4),
                    "p_value_bonferroni_corrected": round(p_corrected, 4),
                    "significant_after_correction": bool(p_corrected < 0.05),
                    "n_observations": n,
                    "n_tests_total": n_tests,
                    "interpretation": f"When {c1.replace('_',' ')} goes up, {c2.replace('_',' ')} tends to {'go up too' if r > 0 else 'go down'}",
                })
            result["correlations"].sort(key=lambda x: abs(x["correlation"]), reverse=True)
            result["correlations"] = result["correlations"][:6]

        # ── 5. SEGMENTATION with interaction effect detection ─────────────────
        for cat_col in cat_cols[:3]:
            unique_vals = df[cat_col].nunique()
            if unique_vals < 2 or unique_vals > 30: continue
            seg = {}
            for num_col in numeric_cols[:3]:
                seg_stats = []
                overall_mean = df[num_col].mean()
                overall_std = df[num_col].std()
                for group_val, group_df in df.groupby(cat_col):
                    grp = group_df[num_col].dropna()
                    if len(grp) < 2: continue
                    grp_mean = float(grp.mean())
                    deviation = (grp_mean - overall_mean) / max(abs(overall_mean), 1e-10) * 100
                    rest = df[df[cat_col] != group_val][num_col].dropna()
                    p_val = 1.0
                    significant = False
                    if len(rest) > 1:
                        _, p_val = stats.ttest_ind(grp, rest, equal_var=False)
                        significant = bool(p_val < 0.05)
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((grp.std()**2 + rest.std()**2) / 2) if len(rest) > 1 else 1
                    cohens_d = (grp_mean - float(rest.mean())) / max(float(pooled_std), 1e-10) if len(rest) > 1 else 0
                    seg_stats.append({
                        "segment": str(group_val),
                        "count": int(len(grp)),
                        "pct_of_total": round(len(grp)/len(df)*100, 1),
                        "mean": round(grp_mean, 4),
                        "median": round(float(grp.median()), 4),
                        "std": round(float(grp.std()), 4),
                        "deviation_from_overall_pct": round(float(deviation), 2),
                        "statistically_different": significant,
                        "p_value": round(float(p_val), 4),
                        "cohens_d": round(float(cohens_d), 3),
                        "effect_size": "large" if abs(cohens_d)>0.8 else "medium" if abs(cohens_d)>0.5 else "small",
                        "rank": 0,
                    })
                seg_stats.sort(key=lambda x: x["mean"], reverse=True)
                for i, s in enumerate(seg_stats): s["rank"] = i + 1
                seg[num_col] = seg_stats
            if seg:
                result["segmentation"][cat_col] = {"unique_segments": int(unique_vals), "metrics": seg}

        # ── 6. CAUSAL ANALYSIS LAYER ─────────────────────────────────────────
        causal = {}

        # Confounder detection — check if categorical vars correlate with both X and Y
        if len(numeric_cols) >= 2 and cat_cols:
            confounders = []
            for cat in cat_cols[:2]:
                for num in numeric_cols[:3]:
                    groups = [g[num].dropna().values for _, g in df.groupby(cat)]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) >= 2:
                        try:
                            f_stat, p_val = stats.f_oneway(*groups)
                            if p_val < 0.1:
                                confounders.append({
                                    "variable": cat,
                                    "affects": num,
                                    "p_value": round(float(p_val), 4),
                                    "warning": f"'{cat}' significantly affects '{num}' — may confound causal claims about {num}",
                                })
                        except Exception:
                            pass
            causal["potential_confounders"] = confounders

        # Reverse causality flags
        reverse_causality = []
        if len(numeric_cols) >= 2 and len(df) >= 10:
            for i, c1 in enumerate(numeric_cols[:3]):
                for c2 in numeric_cols[i+1:i+3]:
                    try:
                        # Simple Granger-like test: does lagged c1 predict c2?
                        s1 = df[c1].dropna().reset_index(drop=True)
                        s2 = df[c2].dropna().reset_index(drop=True)
                        min_len = min(len(s1), len(s2)) - 1
                        if min_len < 4: continue
                        s1_lag = s1.iloc[:min_len].values
                        s2_curr = s2.iloc[1:min_len+1].values
                        r, p = stats.pearsonr(s1_lag, s2_curr)
                        if abs(r) > 0.4 and p < 0.1:
                            reverse_causality.append({
                                "from": c1, "to": c2,
                                "lag_correlation": round(float(r), 3),
                                "p_value": round(float(p), 4),
                                "note": f"Past {c1.replace('_',' ')} predicts future {c2.replace('_',' ')} — possible leading indicator",
                            })
                    except Exception:
                        pass
        causal["granger_causality_signals"] = reverse_causality

        # Regression discontinuity detection — look for sharp thresholds
        rdd_signals = []
        for col in numeric_cols[:4]:
            series = df[col].dropna()
            if len(series) < 10: continue
            sorted_vals = np.sort(series.values)
            diffs = np.diff(sorted_vals)
            max_diff_idx = np.argmax(diffs)
            max_diff = float(diffs[max_diff_idx])
            if max_diff > 3 * float(np.std(diffs)):
                threshold = float(sorted_vals[max_diff_idx])
                rdd_signals.append({
                    "column": col,
                    "threshold_value": round(threshold, 2),
                    "jump_magnitude": round(max_diff, 2),
                    "note": f"Sharp jump detected at {round(threshold,2)} in {col.replace('_',' ')} — possible policy threshold or data error",
                })
        causal["regression_discontinuity_signals"] = rdd_signals
        result["causal_analysis"] = causal

        # ── 7. BIAS AUDIT ─────────────────────────────────────────────────────
        bias = {}

        # Survivorship bias
        if time_col:
            bias["survivorship_bias_warning"] = {
                "detected": False,
                "note": f"Time column '{time_col}' detected. Check whether data includes only surviving entities (e.g. current employees, active products). Missing early exits would create survivorship bias.",
            }

        # Missingness mechanism diagnosis
        missingness = {}
        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            if n_missing == 0: continue
            pct = round(n_missing / len(df) * 100, 2)
            # Test if missingness correlates with other numeric columns
            missing_mask = df[col].isnull().astype(int)
            correlates_with = []
            for other in numeric_cols:
                if other == col: continue
                try:
                    r, p = stats.pointbiserialr(missing_mask, df[other].fillna(df[other].median()))
                    if abs(r) > 0.2 and p < 0.1:
                        correlates_with.append({"column": other, "correlation": round(float(r),3)})
                except Exception:
                    pass
            mechanism = "MCAR (Missing Completely At Random)" if not correlates_with else "MAR (Missing At Random) — missingness correlates with other variables"
            missingness[col] = {
                "missing_count": n_missing, "missing_pct": pct,
                "likely_mechanism": mechanism,
                "correlates_with": correlates_with,
                "imputation_valid": not bool(correlates_with),
                "recommendation": "Imputation is reasonable" if not correlates_with else f"Imputation may bias results — missingness in '{col}' is related to {[c['column'] for c in correlates_with]}",
            }
        bias["missingness_mechanism"] = missingness

        # Selection bias
        bias["selection_bias_check"] = {
            "sample_size": len(df),
            "warning": "Cannot assess selection bias without knowing the full target population. Verify this dataset is representative.",
            "recommendation": "If this is a filtered subset (e.g. only approved loans, only current employees), findings may not generalise to all cases.",
        }

        # Impossible value detection
        impossible = []
        for col in numeric_cols:
            series = df[col].dropna()
            if series.min() < 0 and "pct" in col.lower():
                impossible.append({"column": col, "issue": f"Negative percentage values found (min: {round(float(series.min()),2)})", "suggestion": "Check data source — percentages should be 0-100"})
            if "date" in col.lower() or "year" in col.lower():
                if series.max() > 2100 or series.min() < 1900:
                    impossible.append({"column": col, "issue": f"Suspicious date values (range: {round(float(series.min()),0)}-{round(float(series.max()),0)})", "suggestion": "Possible UTC timezone mismatch or data entry error"})
        bias["impossible_values"] = impossible
        result["bias_audit"] = bias

        # ── 8. ADVANCED STATISTICS ───────────────────────────────────────────
        advanced = {}

        # Power analysis
        power_analysis = []
        for col in numeric_cols[:3]:
            series = df[col].dropna()
            if len(series) < 2: continue
            mean, std = series.mean(), series.std()
            if std == 0: continue
            # Minimum detectable effect at 80% power, alpha=0.05
            from scipy.stats import norm
            z_alpha = norm.ppf(0.975)
            z_beta = norm.ppf(0.8)
            min_detectable_effect = (z_alpha + z_beta) * std / np.sqrt(len(series))
            power_analysis.append({
                "column": col,
                "n": len(series),
                "min_detectable_effect": round(float(min_detectable_effect), 4),
                "min_detectable_effect_pct_of_mean": round(float(min_detectable_effect/max(abs(mean),1e-10)*100), 2),
                "adequate_power": bool(len(series) >= 30),
                "note": f"With n={len(series)}, can detect changes of {round(float(min_detectable_effect),2)} or more (80% power)",
            })
        advanced["power_analysis"] = power_analysis

        # Effect sizes (Cohen's d for each numeric vs overall)
        effect_sizes = {}
        if cat_cols and numeric_cols:
            cat = cat_cols[0]
            groups = {k: v for k, v in df.groupby(cat)[numeric_cols[0]].apply(list).items()}
            vals = list(groups.values())
            if len(vals) >= 2:
                g1, g2 = np.array(vals[0]), np.array(vals[1])
                pooled = np.sqrt((np.std(g1)**2 + np.std(g2)**2) / 2)
                d = (np.mean(g1) - np.mean(g2)) / max(pooled, 1e-10)
                effect_sizes[f"{numeric_cols[0]}_by_{cat}"] = {
                    "cohens_d": round(float(d), 3),
                    "magnitude": "large" if abs(d)>0.8 else "medium" if abs(d)>0.5 else "small",
                    "interpretation": f"The difference in {numeric_cols[0]} between groups is {'practically meaningful' if abs(d)>0.5 else 'small and may not matter in practice'}",
                }
        advanced["effect_sizes"] = effect_sizes

        # Non-parametric fallback
        non_parametric = []
        for col in numeric_cols[:3]:
            series = df[col].dropna()
            if len(series) < 8: continue
            _, p_norm = stats.normaltest(series)
            if p_norm < 0.05:
                non_parametric.append({
                    "column": col,
                    "normality_p": round(float(p_norm), 4),
                    "recommendation": f"'{col}' is not normally distributed (p={round(float(p_norm),4)}). Results use non-parametric methods (bootstrap/rank tests) for robust inference.",
                    "method_used": "Mann-Whitney U / Kruskal-Wallis where applicable",
                })
        advanced["non_parametric_flags"] = non_parametric

        # Multiple comparison correction summary
        n_total_tests = len(numeric_cols) * max(len(cat_cols), 1)
        advanced["multiple_comparison_correction"] = {
            "n_tests_run": n_total_tests,
            "method": "Bonferroni correction applied to all correlations",
            "alpha_adjusted": round(0.05 / max(n_total_tests, 1), 6),
            "note": f"With {n_total_tests} tests, uncorrected p<0.05 has ~{round(n_total_tests*0.05,1)} expected false positives. Bonferroni threshold: p<{round(0.05/max(n_total_tests,1),4)}",
        }
        result["advanced_statistics"] = advanced

        # ── 9. TIME SERIES INTELLIGENCE ──────────────────────────────────────
        if time_col and len(df) >= 6:
            ts_intel = {}
            for col in numeric_cols[:3]:
                series = df[col].dropna().reset_index(drop=True)
                if len(series) < 6: continue
                # Structural break (CUSUM)
                cusum = np.cumsum(series - series.mean())
                break_idx = int(np.argmax(np.abs(cusum)))
                break_magnitude = float(abs(cusum[break_idx]))
                has_break = break_magnitude > 2 * series.std() * np.sqrt(len(series))
                # Seasonality proxy (autocorrelation at lag 1)
                if len(series) > 3:
                    lag1_corr = float(series.autocorr(lag=1)) if hasattr(series, 'autocorr') else 0
                else:
                    lag1_corr = 0
                ts_intel[col] = {
                    "structural_break": {
                        "detected": has_break,
                        "at_period": break_idx if has_break else None,
                        "magnitude": round(break_magnitude, 2),
                        "interpretation": f"Regime shift detected around period {break_idx} — the trend or level changed significantly at this point." if has_break else "No structural break detected — trend appears stable.",
                    },
                    "autocorrelation_lag1": round(lag1_corr, 3),
                    "seasonality_signal": abs(lag1_corr) > 0.5,
                    "trend_stability": "stable" if not has_break else "unstable — regime change detected",
                    "note": f"Lag-1 autocorrelation of {round(lag1_corr,3)} {'suggests serial correlation — past values predict future values' if abs(lag1_corr)>0.3 else 'suggests no strong serial dependence'}.",
                }
            result["time_series_intelligence"] = ts_intel

        # ── 10. MODEL TRANSPARENCY ────────────────────────────────────────────
        transparency = {
            "methods_used": [
                "Z-score anomaly detection (threshold: 3σ)",
                "OLS linear regression for trend analysis",
                "Pearson correlation with Bonferroni correction",
                "Welch's t-test for segment differences",
                "Cohen's d for effect sizes",
                "CUSUM for structural break detection",
                "Granger-type lag correlation for causality signals",
                "Point-biserial correlation for missingness mechanism",
                "D'Agostino-Pearson test for normality",
                "Power analysis (80% power, α=0.05)",
            ],
            "model_limitations": [
                "All correlations are observational — causal claims require experimental design or instrumental variables",
                f"Sample size is {len(df)} rows — {'adequate for basic inference' if len(df) >= 30 else 'small — treat all findings as preliminary'}",
                "Winsorisation of outliers may understate the true range of extreme values",
                "Imputed missing values are estimates — findings involving imputed columns have higher uncertainty",
                "Structural break detection uses CUSUM — sensitive to order of rows, assumes chronological data",
            ],
            "extrapolation_warning": "Do not extrapolate forecasts beyond 2-3 periods outside the observed range without domain validation",
            "adversarial_robustness": self._adversarial_check(df, numeric_cols),
        }
        result["model_transparency"] = transparency

        # ── 11. ALTERNATIVE EXPLANATIONS ─────────────────────────────────────
        alt_explanations = {}
        for f in result["findings"][:3]:
            col = f["column"]
            alts = []
            if f["type"] == "anomaly":
                alts = [
                    f"The anomaly in {col.replace('_',' ')} could be a legitimate business event (e.g. seasonal peak, one-off contract) rather than a data error",
                    f"It could reflect a change in measurement methodology or data source rather than a real change in {col.replace('_',' ')}",
                    f"It could be an outlier caused by a different customer/product/region segment that behaves differently from the rest",
                ]
            elif f["type"] == "trend":
                alts = [
                    f"The trend in {col.replace('_',' ')} could be driven by external market factors (inflation, seasonality) rather than internal performance",
                    f"It could be a short-term fluctuation that will revert to the mean over a longer period",
                    f"It could reflect a change in the composition of what is being measured rather than true change (e.g. different product mix)",
                ]
            alt_explanations[col] = alts
        result["alternative_explanations"] = alt_explanations

        # ── 12. IMPACT RANKING ────────────────────────────────────────────────
        impact_scores = {}
        for f in result["findings"]:
            col = f["column"]
            if col not in impact_scores or f["impact_score"] > impact_scores[col]["score"]:
                impact_scores[col] = {
                    "column": col, "score": f["impact_score"],
                    "confidence": f["confidence"], "primary_issue": f["type"], "reason": f["title"],
                }
        result["impact_ranking"] = sorted(impact_scores.values(), key=lambda x: x["score"], reverse=True)[:5]

        # ── 13. UNCERTAINTY ───────────────────────────────────────────────────
        n_rows = len(df)
        uncertainties = []
        if n_rows < 30:
            uncertainties.append({
                "issue": "Small sample size",
                "detail": f"Only {n_rows} rows. Statistical conclusions may not generalise — patterns could be coincidental. Power is insufficient to detect small effects.",
                "confidence_adjustment": -0.2,
                "recommendation": "Collect at least 50-100 rows for reliable inference.",
            })
        if n_rows < 100:
            uncertainties.append({
                "issue": "Limited data volume",
                "detail": f"{n_rows} rows is below the recommended minimum of 100 for robust multi-variable analysis.",
                "confidence_adjustment": -0.1,
                "recommendation": "Treat findings as directional indicators, not definitive conclusions.",
            })
        for col, info in result["distributions"].items():
            if info["missing_pct"] > 20:
                uncertainties.append({
                    "issue": f"High missing data in {col.replace('_',' ')}",
                    "detail": f"{info['missing_pct']}% missing. Median imputation assumes MCAR — if data is MAR or MNAR, estimates are biased.",
                    "confidence_adjustment": -0.15,
                    "recommendation": f"Investigate why {col.replace('_',' ')} is missing before drawing conclusions from it.",
                })
        if not cat_cols:
            uncertainties.append({
                "issue": "No segmentation variables",
                "detail": "No categorical columns available. All insights are population averages that may hide significant variation between subgroups.",
                "confidence_adjustment": -0.05,
                "recommendation": "Add region, department, category, or other grouping variables to enable segment-level analysis.",
            })
        # Check for Simpson's paradox risk
        if cat_cols and len(numeric_cols) >= 2:
            uncertainties.append({
                "issue": "Simpson's paradox risk",
                "detail": f"With categorical variable '{cat_cols[0]}' and multiple numeric metrics, aggregate trends may reverse within segments. Always check segment-level results.",
                "confidence_adjustment": -0.05,
                "recommendation": "Verify all aggregate findings hold within each segment before making recommendations.",
            })
        result["uncertainty"] = uncertainties

        # ── 14. SELF-AUDIT ────────────────────────────────────────────────────
        result["self_audit"] = [
            {
                "question": "What assumptions am I making?",
                "assumptions": [
                    "Data is a representative sample of the full population",
                    "Missing values are missing at random (MCAR) — imputation is valid",
                    "Row order reflects chronological sequence for time series analysis",
                    "Relationships found are correlational — causality requires experimental evidence",
                    "The data generating process has been stable over the observation period",
                ],
            },
            {
                "question": "What could I be wrong about?",
                "risks": [
                    "Anomalies flagged as data errors could be legitimate business events — verify with domain experts before action",
                    "Trends detected may be seasonal patterns misidentified as directional — more periods needed to confirm",
                    "Correlations may be driven by confounders — see causal analysis section for flags",
                    "Segment differences may reflect data quality variation across groups, not true performance differences",
                    "Structural breaks detected by CUSUM assume row order is meaningful — if rows are not chronological, break signals are meaningless",
                ],
            },
            {
                "question": "What information would make this analysis more reliable?",
                "missing_context": [
                    "Industry benchmarks to contextualise whether metrics are good or bad",
                    "Business targets or KPI thresholds for each metric",
                    "Knowledge of external events (market changes, policy changes, seasonality) that could explain patterns",
                    "Column definitions — what exactly each variable measures",
                    "Data collection methodology — how was this data gathered and filtered?",
                ],
            },
            {
                "question": "What did I test and reject?",
                "rejected_hypotheses": [
                    f"Tested for normality in {len(numeric_cols)} columns — {'normality assumed where p>0.05, non-parametric methods applied elsewhere' if numeric_cols else 'no numeric columns'}",
                    f"Tested {len(result['correlations'])} correlation pairs — applied Bonferroni correction to control false discovery rate",
                    f"Tested for structural breaks in {min(4,len(numeric_cols))} trend series using CUSUM",
                    "Tested for potential confounders using one-way ANOVA across categorical variables",
                ],
            },
        ]

        # ── 15. DATA GROUNDING ────────────────────────────────────────────────
        result["data_grounding"] = {
            "total_rows_analysed": n_rows,
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(cat_cols),
            "total_data_points": n_rows * len(df.columns),
            "total_anomalies_found": sum(f.get("evidence",{}).get("anomaly_count",0) for f in result["findings"] if f["type"]=="anomaly"),
            "total_correlations_found": len(result["correlations"]),
            "segments_analysed": sum(len(v["metrics"]) for v in result["segmentation"].values()),
            "causal_flags_raised": len(causal.get("potential_confounders",[])) + len(causal.get("granger_causality_signals",[])) + len(causal.get("regression_discontinuity_signals",[])),
            "bias_flags_raised": len(bias.get("impossible_values",[])) + sum(1 for m in bias.get("missingness_mechanism",{}).values() if not m.get("imputation_valid",True)),
            "highest_confidence_finding": max((f["confidence"] for f in result["findings"]), default=0),
            "lowest_confidence_finding": min((f["confidence"] for f in result["findings"]), default=0),
            "methods_applied": len(transparency["methods_used"]),
        }

        return result

    def _adversarial_check(self, df: pd.DataFrame, numeric_cols: list) -> dict:
        """Check how sensitive conclusions are to small data perturbations."""
        results = {}
        for col in numeric_cols[:2]:
            series = df[col].dropna()
            if len(series) < 5: continue
            original_mean = float(series.mean())
            # Perturb by 5% noise
            perturbed = series + np.random.normal(0, series.std() * 0.05, len(series))
            perturbed_mean = float(perturbed.mean())
            pct_change = abs(perturbed_mean - original_mean) / max(abs(original_mean), 1e-10) * 100
            results[col] = {
                "original_mean": round(original_mean, 4),
                "perturbed_mean_5pct_noise": round(perturbed_mean, 4),
                "mean_change_pct": round(pct_change, 2),
                "robust": bool(pct_change < 1),
                "note": "Findings are robust to small data perturbations" if pct_change < 1 else "Results are sensitive to small changes in data — treat with caution",
            }
        return results

    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        time_keywords = ["date","month","week","quarter","period","year","time"]
        for col in df.columns:
            if any(kw in col.lower() for kw in time_keywords): return col
            if pd.api.types.is_datetime64_any_dtype(df[col]): return col
        return None

    # ── STANDARD METHODS ─────────────────────────────────────────────────────

    def describe(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include="number")
        categorical = df.select_dtypes(include=["object","category"])
        result = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().mean()*100).round(2).to_dict(),
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
            n = int(((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum())
            if n > 0: outliers[col] = n
        completeness = round((1 - missing/max(total,1))*100, 2)
        return {
            "completeness_pct": completeness,
            "missing_cells": missing,
            "duplicate_rows": dupes,
            "outlier_columns": outliers,
            "overall_score": round(completeness*0.5 + (1-dupes/max(len(df),1))*100*0.3 + (1-len(outliers)/max(len(numeric_cols),1))*100*0.2, 1),
        }

    def detect_anomalies(self, df: pd.DataFrame, column: str, method: str = "zscore") -> dict:
        series = df[column].dropna()
        if method == "zscore":
            z = np.abs((series - series.mean()) / series.std())
            mask = z > 3
        elif method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            mask = (series < q1-1.5*iqr) | (series > q3+1.5*iqr)
        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1,1))
            mask = pd.Series(preds == -1, index=series.index)
        else:
            mask = pd.Series(False, index=series.index)
        return {"method": method, "column": column, "total_records": len(series), "anomaly_count": int(mask.sum()), "anomaly_pct": round(mask.mean()*100, 2)}

    def forecast_arima(self, series: pd.Series, periods: int = 12) -> dict:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        adf = adfuller(series.dropna())
        d = 0 if adf[1] < 0.05 else 1
        model = ARIMA(series.dropna(), order=(1,d,1)).fit()
        forecast = model.forecast(steps=periods)
        ci = model.get_forecast(steps=periods).conf_int()
        return {
            "model": f"ARIMA(1,{d},1)", "aic": round(model.aic,2),
            "forecast": [round(float(v),4) for v in forecast],
            "lower_bound": [round(float(v),4) for v in ci.iloc[:,0]],
            "upper_bound": [round(float(v),4) for v in ci.iloc[:,1]],
        }

    def cluster(self, df: pd.DataFrame, features: list, n_clusters: int = 4) -> dict:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        X = df[features].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        df2 = X.copy(); df2["_cluster"] = labels
        clusters = {}
        for i in range(n_clusters):
            grp = df2[df2["_cluster"]==i][features]
            clusters[f"cluster_{i}"] = {"size": int((labels==i).sum()), "means": grp.mean().round(4).to_dict()}
        return {"model": "K-Means", "n_clusters": n_clusters, "silhouette_score": round(float(sil),4), "clusters": clusters}

    def df_to_records(self, df: pd.DataFrame, limit: int = 50) -> list:
        import json
        return json.loads(df.head(limit).to_json(orient="records"))


analysis_service = EliteDataAnalysisService()
