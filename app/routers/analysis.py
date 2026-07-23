"""
DataMind Agent — Analysis Router (v3)

Fixes over v2:
  • Elite analysis step no longer silently fails on a missing key.
    Every field is read defensively, and the step reports real timing
    whether it succeeds or fails.
  • Correlation key corrected: 'significant_after_correction' (Bonferroni),
    with a fallback to the older 'significant' key.
  • RAG industry benchmarks are injected into the LLM prompt.
  • Memory layer stores each analysis and feeds prior context back in.
  • Causal flags, bias audit and advanced statistics surface as Insights.
  • A partial elite result still yields insights rather than nothing.

All routes require a signed-in user (guest sessions count). Auth is enforced
at the router level via the shared require_user dependency.
"""
import time, logging, json
from pydantic import BaseModel as _BaseModel
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import pandas as pd

from app.models.schemas import (
    AnalysisRequest, AnalysisResponse,
    Metric, Insight, ChartData, PipelineStep, LLMProvider
)
from app.services.llm_service import llm_service
from app.services.analysis_service import analysis_service
from app.services.viz_service import viz_service
from app.routers.auth import require_user

router = APIRouter(dependencies=[Depends(require_user)])
logger = logging.getLogger(__name__)


class ClusterRequest(_BaseModel):
    features: list[str]
    n_clusters: int = 4
    data: list[dict]


def _step(name, tool, status="done", ms=0.0, preview=None):
    return PipelineStep(name=name, tool=tool, status=status,
                        duration_ms=round(ms, 1), output_preview=preview)


def _g(d, *keys, default=None):
    """Safe nested get. _g(d, 'a', 'b') == d['a']['b'] or default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _count_segments(segmentation) -> int:
    """Count segments without assuming every entry has a 'metrics' key."""
    if not isinstance(segmentation, dict):
        return 0
    total = 0
    for v in segmentation.values():
        if isinstance(v, dict):
            m = v.get("metrics")
            if isinstance(m, dict):
                total += len(m)
            elif v.get("unique_segments"):
                total += int(v["unique_segments"])
    return total


# ── Insight builders — each is independently guarded ──────────────────────────

def _findings_to_insights(elite) -> list[Insight]:
    out = []
    for f in (elite.get("findings") or [])[:6]:
        try:
            ev = f.get("evidence", {}) or {}
            conf = float(f.get("confidence", 0.7))
            ftype = f.get("type", "")
            col = str(f.get("column", "")).replace("_", " ")

            if ftype == "anomaly":
                rng = ev.get("normal_range", ["?", "?"])
                lo, hi = (rng + ["?", "?"])[:2] if isinstance(rng, list) else ("?", "?")
                body = (
                    f"Detected {ev.get('anomaly_count','?')} anomalous records "
                    f"({ev.get('anomaly_pct','?')}% of data). "
                    f"Normal range: [{lo}, {hi}]. "
                    f"Anomalous values: {(ev.get('anomaly_values') or [])[:3]}. "
                    f"These inflate the average by {ev.get('impact_on_mean_pct','?')}%. "
                    f"Max Z-score: {ev.get('z_score_max','?')}σ."
                )
            elif ftype == "trend":
                sig = ("statistically significant" if ev.get("statistically_significant")
                       else "not yet statistically significant")
                brk = ""
                if ev.get("structural_break_detected"):
                    brk = f" A structural break was detected at period {ev.get('structural_break_at_period','?')}."
                body = (
                    f"Over {ev.get('period_count','?')} periods, {col} changed "
                    f"{ev.get('total_change_pct','?')}% (from {ev.get('first_value','?')} "
                    f"to {ev.get('last_value','?')}). R² = {ev.get('r_squared','?')} — "
                    f"trend is {sig} (p = {ev.get('p_value','?')}).{brk}"
                )
            else:
                body = f.get("body") or f.get("title", "")

            out.append(Insight(
                title=f.get("title", "Finding"),
                body=body,
                severity=f.get("severity", "info"),
                source=f"{f.get('method','Statistical analysis')} · Confidence: {round(conf*100)}%",
                confidence=conf,
            ))
        except Exception as e:
            logger.debug(f"Skipped a finding: {e}")
    return out


def _correlations_to_insights(elite) -> list[Insight]:
    out = []
    for c in (elite.get("correlations") or [])[:2]:
        try:
            # v2 used 'significant'; v3 uses 'significant_after_correction'
            sig = c.get("significant_after_correction", c.get("significant", False))
            if not sig:
                continue
            p_show = c.get("p_value_bonferroni_corrected", c.get("p_value_raw", c.get("p_value", "?")))
            out.append(Insight(
                title=(f"Strong relationship: {str(c.get('col1','')).replace('_',' ')} ↔ "
                       f"{str(c.get('col2','')).replace('_',' ')}"),
                body=(
                    f"Pearson r = {c.get('correlation','?')} "
                    f"({c.get('strength','?')} {c.get('direction','?')} correlation), "
                    f"p = {p_show} after Bonferroni correction, "
                    f"based on {c.get('n_observations','?')} observations. "
                    f"{c.get('interpretation','')}"
                ),
                severity="info",
                source="Pearson correlation · Bonferroni corrected · scipy.stats",
                confidence=0.90,
            ))
        except Exception as e:
            logger.debug(f"Skipped a correlation: {e}")
    return out


def _causal_to_insights(elite) -> list[Insight]:
    out = []
    causal = elite.get("causal_analysis") or {}
    try:
        for c in (causal.get("potential_confounders") or [])[:2]:
            out.append(Insight(
                title="Possible confounding variable",
                body=f"{c.get('warning','')} (p = {c.get('p_value','?')}). "
                     "A relationship you observe may be driven by this third variable rather than a direct effect.",
                severity="warning",
                source="Confounder detection · ANOVA",
                confidence=0.75,
            ))
        for g in (causal.get("granger_causality_signals") or [])[:1]:
            out.append(Insight(
                title="Lead-lag signal detected",
                body=f"{g.get('note','')} (lag correlation r = {g.get('lag_correlation','?')}, "
                     f"p = {g.get('p_value','?')}). This is suggestive of a lead-lag relationship, "
                     "not proof of causation.",
                severity="info",
                source="Granger-style lag analysis",
                confidence=0.65,
            ))
        for r in (causal.get("regression_discontinuity_signals") or [])[:1]:
            out.append(Insight(
                title="Discontinuity at a threshold",
                body=str(r.get("note", "")),
                severity="warning",
                source="Regression discontinuity screen",
                confidence=0.60,
            ))
    except Exception as e:
        logger.debug(f"Causal insights skipped: {e}")
    return out


def _bias_to_insights(elite) -> list[Insight]:
    out = []
    bias = elite.get("bias_audit") or {}
    try:
        for col, m in list((bias.get("missingness_mechanism") or {}).items())[:2]:
            out.append(Insight(
                title=f"Missing data in {col.replace('_',' ')} is not random",
                body=f"Likely mechanism: {m.get('likely_mechanism','?')}. {m.get('recommendation','')}",
                severity="warning",
                source="Missingness mechanism test · point-biserial",
                confidence=0.70,
            ))
        for iv in (bias.get("impossible_values") or [])[:2]:
            out.append(Insight(
                title="Impossible values present",
                body=f"{iv.get('issue','')} {iv.get('suggestion','')}",
                severity="critical",
                source="Range validation",
                confidence=0.95,
            ))
        surv = bias.get("survivorship_bias_warning") or {}
        if surv:
            out.append(Insight(
                title="Possible survivorship bias",
                body=str(surv.get("note", "")),
                severity="warning",
                source="Survivorship bias screen",
                confidence=0.65,
            ))
    except Exception as e:
        logger.debug(f"Bias insights skipped: {e}")
    return out


def _power_to_insights(elite) -> list[Insight]:
    out = []
    adv = elite.get("advanced_statistics") or {}
    try:
        for p in (adv.get("power_analysis") or [])[:1]:
            if p.get("adequate_power"):
                continue
            out.append(Insight(
                title="Sample size may be too small to detect real effects",
                body=(
                    f"With n = {p.get('n','?')}, the smallest effect detectable at 80% power "
                    f"is {p.get('min_detectable_effect','?')} "
                    f"({p.get('min_detectable_effect_pct_of_mean','?')}% of the mean). "
                    "Effects smaller than this could exist and go unseen."
                ),
                severity="warning",
                source="Power analysis · α=0.05, power=0.80",
                confidence=0.85,
            ))
    except Exception as e:
        logger.debug(f"Power insights skipped: {e}")
    return out


def _uncertainty_to_insights(elite) -> list[Insight]:
    out = []
    for u in (elite.get("uncertainty") or [])[:2]:
        try:
            out.append(Insight(
                title=f"Data limitation: {u.get('issue','')}",
                body=str(u.get("detail", "")),
                severity="warning",
                source="Self-audit",
                confidence=1.0,
            ))
        except Exception:
            pass
    return out


def _elite_metrics(elite) -> list[Metric]:
    out = []
    dg = elite.get("data_grounding") or {}
    try:
        if dg.get("total_rows_analysed") is not None:
            out.append(Metric(label="Rows analysed", value=str(dg["total_rows_analysed"])))
        out.append(Metric(label="Findings detected", value=str(len(elite.get("findings") or []))))
        if dg.get("segments_analysed") is not None:
            out.append(Metric(label="Segments checked", value=str(dg["segments_analysed"])))
        top = dg.get("highest_confidence_finding") or 0
        if top > 0:
            out.append(Metric(
                label="Top confidence",
                value=f"{round(top*100)}%",
                trend="up" if top > 0.8 else "flat",
            ))
    except Exception as e:
        logger.debug(f"Elite metrics skipped: {e}")
    return out


# ── Main endpoint ─────────────────────────────────────────────────────────────

def _deep_facts(df, max_chars: int = 11000, query: str = "") -> str:
    """
    Compute an EXACT, decision-grade fact sheet across the ENTIRE dataset.

    Nothing here is sampled — every figure is calculated over all rows in pandas.
    Only the *presentation* is bounded, so the prompt stays within context limits
    while the underlying maths covers everything.

    Produces:
      • Shape, column inventory and data-quality flags
      • Full distribution profile per numeric column (percentiles, skew, kurtosis, CV)
      • Concentration analysis (Pareto 80/20, top-10 share, Gini, Herfindahl)
      • Ranked leaders and laggards on the primary metric
      • Exact cross-tabs for every meaningful categorical dimension
      • Two-way cross-tabs where dimensions interact
      • Named outliers with their actual values and deviation
      • Missing-data patterns and co-missingness
      • Period-over-period movement where a time axis exists
    """
    import numpy as np
    import pandas as pd

    parts = []
    n_rows = len(df)
    num = df.select_dtypes(include="number")
    num_cols = list(num.columns)

    # ── Identify the primary metric (what "value" means for this data) ──
    def pick_metric():
        if not num_cols:
            return None
        skip = ("rank", "id", "index", "year", "no", "number", "code", "zip", "phone")
        cands = [c for c in num_cols if not any(s in str(c).lower() for s in skip)]
        if not cands:
            cands = list(num_cols)
        ql = (query or "").lower()
        named = [c for c in cands if str(c).lower().replace("_", " ") in ql or str(c).lower() in ql]
        whole = [c for c in cands
                 if any(w in str(c).lower()
                        for w in ("global", "total", "overall", "combined", "gross", "net",
                                  "value", "revenue", "sales", "amount"))]
        if named:
            return named[0]
        if whole:
            return whole[0]
        try:
            return num[cands].sum().idxmax()
        except Exception:
            return cands[0]

    metric = pick_metric()

    # ── Categorical dimensions worth breaking down by ──
    cats = []
    for c in df.columns:
        if c in num_cols:
            continue
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            nun = df[c].nunique(dropna=True)
            if 1 < nun <= max(80, n_rows // 3):
                cats.append((c, nun))
        except Exception:
            continue
    cats.sort(key=lambda t: t[1])
    cat_cols = [c for c, _ in cats]

    # ── 1. SHAPE & INVENTORY ──────────────────────────────────────────────
    parts.append(f"DATASET: {n_rows:,} rows x {len(df.columns)} columns. "
                 f"ALL figures below are computed over ALL {n_rows:,} rows — nothing is sampled.")
    parts.append(f"Numeric columns: {', '.join(map(str, num_cols)) or 'none'}")
    parts.append(f"Categorical columns: {', '.join(f'{c} ({n} distinct)' for c, n in cats[:10]) or 'none'}")
    if metric:
        parts.append(f"Primary metric identified: {metric}")

    # ── 2. DATA QUALITY (exact) ───────────────────────────────────────────
    try:
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss):
            parts.append("\nMISSING DATA (exact counts):")
            for c, m in miss.head(8).items():
                parts.append(f"  {c}: {int(m):,} missing ({m / n_rows * 100:.1f}%)")
            # Co-missingness — do columns go missing together?
            if len(miss) >= 2:
                a, b = miss.index[0], miss.index[1]
                both = int((df[a].isna() & df[b].isna()).sum())
                if both > 0:
                    parts.append(f"  Co-missing: {both:,} rows lack BOTH {a} and {b} "
                                 f"— suggests a systematic gap, not random omission.")
        dups = int(df.duplicated().sum())
        if dups:
            parts.append(f"  Exact duplicate rows: {dups:,} ({dups / n_rows * 100:.1f}%)")
    except Exception:
        pass

    # ── 3. DISTRIBUTION PROFILE (exact, per numeric column) ───────────────
    if num_cols:
        parts.append("\nDISTRIBUTION PROFILE (computed over every row):")
        for c in num_cols[:8]:
            try:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s) < 3:
                    continue
                p = s.quantile([.10, .25, .50, .75, .90, .99])
                mean, std = float(s.mean()), float(s.std())
                cv = (std / mean * 100) if mean else float("nan")
                skew = float(s.skew()) if len(s) > 2 else 0.0
                kurt = float(s.kurtosis()) if len(s) > 3 else 0.0
                shape = ("right-skewed (a few large values pull the mean up)" if skew > 1
                         else "left-skewed" if skew < -1 else "roughly symmetric")
                parts.append(
                    f"  {c}: n={len(s):,} sum={s.sum():,.0f} mean={mean:,.2f} median={s.median():,.2f}\n"
                    f"     P10={p[.10]:,.2f} P25={p[.25]:,.2f} P75={p[.75]:,.2f} P90={p[.90]:,.2f} P99={p[.99]:,.2f}\n"
                    f"     min={s.min():,.2f} max={s.max():,.2f} std={std:,.2f} CV={cv:.1f}% "
                    f"skew={skew:.2f} ({shape}) kurtosis={kurt:.2f}"
                )
            except Exception:
                continue

    # ── 4. CONCENTRATION / PARETO (exact) ─────────────────────────────────
    if metric is not None:
        try:
            s = pd.to_numeric(df[metric], errors="coerce").dropna()
            s = s[s > 0].sort_values(ascending=False)
            if len(s) > 4:
                total = float(s.sum())
                cum = s.cumsum() / total
                n80 = int((cum <= 0.80).sum()) + 1
                pct_rows_for_80 = n80 / len(s) * 100
                top10_share = float(s.head(max(1, len(s) // 10)).sum()) / total * 100
                top1 = float(s.iloc[0]) / total * 100
                # Gini (concentration; 0 = perfectly even, 1 = all in one record)
                arr = np.sort(s.values)
                idx = np.arange(1, len(arr) + 1)
                gini = float((2 * idx - len(arr) - 1).dot(arr) / (len(arr) * arr.sum()))
                # Herfindahl on shares
                shares = arr / arr.sum()
                hhi = float((shares ** 2).sum())
                parts.append(f"\nCONCENTRATION OF {str(metric).upper()} (exact, all rows):")
                parts.append(
                    f"  {n80:,} records ({pct_rows_for_80:.1f}% of rows) account for 80% of the total. "
                    f"Top decile holds {top10_share:.1f}%. Single largest record = {top1:.1f}% of total.")
                parts.append(
                    f"  Gini = {gini:.3f} ({'highly concentrated' if gini > .6 else 'moderately concentrated' if gini > .4 else 'fairly even'}), "
                    f"HHI = {hhi:.4f}. Total {metric} = {total:,.0f}.")
        except Exception:
            pass

    # ── 5. LEADERS & LAGGARDS (exact, from the whole file) ────────────────
    if metric is not None:
        try:
            label_cols = [c for c in df.columns if c not in num_cols][:2]
            show = [c for c in dict.fromkeys([*label_cols, metric]) if c in df.columns]
            top = df.nlargest(12, metric)[show]
            parts.append(f"\nTOP 12 RECORDS BY {str(metric).upper()} "
                         f"(ranked across ALL {n_rows:,} rows — authoritative for 'which is highest'):")
            parts.append(top.to_string(index=False, max_colwidth=30)[:1400])
            bot = df.nsmallest(5, metric)[show]
            parts.append(f"BOTTOM 5 BY {str(metric).upper()}:")
            parts.append(bot.to_string(index=False, max_colwidth=30)[:500])
        except Exception:
            pass

    # ── 6. EXACT BREAKDOWNS PER DIMENSION ─────────────────────────────────
    if metric is not None and cat_cols:
        for c in cat_cols[:4]:
            try:
                g = df.groupby(c)[metric].agg(["count", "sum", "mean"])
                g = g.sort_values("sum", ascending=False)
                gt = float(g["sum"].sum())
                parts.append(f"\nBY {str(c).upper()} — exact totals across all rows "
                             f"({df[c].nunique(dropna=True)} groups):")
                for name, row in g.head(10).iterrows():
                    share = row["sum"] / gt * 100 if gt else 0
                    parts.append(f"  {str(name)[:28]:<28} n={int(row['count']):>7,}  "
                                 f"{metric}={row['sum']:>14,.0f}  ({share:>5.1f}%)  avg={row['mean']:,.2f}")
                if len(g) > 10:
                    rest = g.iloc[10:]
                    parts.append(f"  ...and {len(rest)} more groups totalling "
                                 f"{rest['sum'].sum():,.0f} ({rest['sum'].sum()/gt*100:.1f}%)")
            except Exception:
                continue

    # ── 7. TWO-WAY CROSS-TAB (where two dimensions interact) ──────────────
    if metric is not None and len(cat_cols) >= 2:
        try:
            a, b = cat_cols[0], cat_cols[1]
            if df[a].nunique() <= 12 and df[b].nunique() <= 8:
                ct = pd.pivot_table(df, index=a, columns=b, values=metric,
                                    aggfunc="sum", fill_value=0)
                parts.append(f"\nCROSS-TAB: {a} x {b} (sum of {metric}, exact):")
                parts.append(ct.to_string(max_colwidth=14)[:1200])
        except Exception:
            pass

    # ── 8. NAMED OUTLIERS (actual records, not just counts) ───────────────
    if metric is not None:
        try:
            s = pd.to_numeric(df[metric], errors="coerce")
            mean, std = float(s.mean()), float(s.std())
            if std and std > 0:
                z = (s - mean) / std
                out = df.loc[z.abs() > 3].copy()
                if len(out):
                    out["_z"] = z.loc[out.index].round(2)
                    label_cols = [c for c in df.columns if c not in num_cols][:2]
                    show = [c for c in dict.fromkeys([*label_cols, metric, "_z"]) if c in out.columns]
                    out = out.reindex(out["_z"].abs().sort_values(ascending=False).index)
                    parts.append(f"\nOUTLIERS ON {str(metric).upper()} (|Z| > 3, exact): "
                                 f"{len(out):,} records, {len(out)/n_rows*100:.2f}% of data. "
                                 f"Normal range [{mean-3*std:,.0f}, {mean+3*std:,.0f}]. "
                                 f"Named records:")
                    parts.append(out.head(8)[show].to_string(index=False, max_colwidth=26)[:900])
                    infl = float(s.mean()) - float(s.drop(out.index).mean())
                    parts.append(f"  Removing these outliers shifts the mean by {infl:,.2f} "
                                 f"({infl/mean*100:+.1f}%) — they materially distort the average.")
        except Exception:
            pass

    # ── 9. TIME MOVEMENT (exact period-over-period) ───────────────────────
    try:
        tcol = next((c for c in df.columns
                     if any(w in str(c).lower()
                            for w in ("year", "date", "month", "period", "quarter", "week"))), None)
        if tcol is not None and metric is not None and df[tcol].nunique() > 1:
            g = df.groupby(tcol)[metric].sum().sort_index()
            if 1 < len(g) <= 60:
                parts.append(f"\n{str(metric).upper()} BY {str(tcol).upper()} (exact totals per period):")
                prev = None
                lines = []
                for k, v in g.items():
                    delta = f" ({(v-prev)/prev*100:+.1f}%)" if prev not in (None, 0) else ""
                    lines.append(f"  {str(k)[:12]:<12} {v:>14,.0f}{delta}")
                    prev = v
                parts.append("\n".join(lines[:24]))
                first, last = float(g.iloc[0]), float(g.iloc[-1])
                if first:
                    parts.append(f"  Overall movement: {first:,.0f} -> {last:,.0f} "
                                 f"({(last-first)/first*100:+.1f}%) across {len(g)} periods.")
    except Exception:
        pass

    # ── 10. CORRELATION HIGHLIGHTS (exact) ────────────────────────────────
    try:
        if len(num_cols) >= 2:
            corr = num[num_cols].corr(numeric_only=True)
            pairs = []
            seen = set()
            for a in corr.columns:
                for b in corr.columns:
                    if a == b or (b, a) in seen:
                        continue
                    seen.add((a, b))
                    r = corr.loc[a, b]
                    if pd.notna(r) and abs(r) >= 0.5:
                        pairs.append((abs(r), a, b, r))
            pairs.sort(reverse=True)
            if pairs:
                parts.append("\nSTRONGEST RELATIONSHIPS (Pearson r, computed on all rows):")
                for _, a, b, r in pairs[:6]:
                    parts.append(f"  {a} <-> {b}: r={r:+.3f} "
                                 f"({'strong' if abs(r) > .7 else 'moderate'} "
                                 f"{'positive' if r > 0 else 'negative'})")
    except Exception:
        pass

    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n[fact sheet truncated for length — all figures above are exact]"
    return out


# Backwards-compatible alias: the analyse endpoint calls _data_facts.
_data_facts = _deep_facts


@router.post("/analyse", response_model=AnalysisResponse)
async def analyse(req: AnalysisRequest):
    t0 = time.perf_counter()
    steps, metrics, insights, charts = [], [], [], []
    df = None
    elite_context = None
    session_id = getattr(req, "session_id", None) or "default"

    # ── STEP 1–2: Ingest and clean ────────────────────────────────────────────
    if req.inline_data:
        t = time.perf_counter()
        raw_df = pd.DataFrame(req.inline_data)

        tc = time.perf_counter()
        try:
            df, cleaning_report = analysis_service.clean_data(raw_df)
            cr_steps = cleaning_report.get("steps", [])
            imp = cr_steps[3].get("columns_imputed", {}) if len(cr_steps) > 3 else {}
            win = cr_steps[4].get("columns_winsorised", {}) if len(cr_steps) > 4 else {}
            dup = cr_steps[2].get("rows_removed", 0) if len(cr_steps) > 2 else 0
            type_issues = len(cleaning_report.get("evidence", {}))
            clean_preview = (f"Duplicates removed: {dup} · Missing filled: {len(imp)} columns · "
                             f"Outliers capped: {len(win)} columns · Type issues flagged: {type_issues}")
            clean_status = "done"
        except Exception as e:
            logger.error(f"Cleaning failed: {e}", exc_info=True)
            df = raw_df
            clean_preview = f"Cleaning skipped: {e}"
            clean_status = "error"

        steps.append(_step("Data received", "Pandas",
            ms=(time.perf_counter()-t)*1000,
            preview=f"{len(raw_df)} rows × {len(raw_df.columns)} columns ingested"))
        steps.append(_step("Data cleaning", "Pandas · NumPy · SciPy",
            status=clean_status, ms=(time.perf_counter()-tc)*1000, preview=clean_preview))

        # ── STEP 3: Elite deep analysis ───────────────────────────────────────
        te = time.perf_counter()
        try:
            elite_context = analysis_service.elite_analyse(df, req.query, req.industry.value)
            if not isinstance(elite_context, dict):
                raise ValueError(f"elite_analyse returned {type(elite_context).__name__}, expected dict")
        except Exception as e:
            logger.error(f"elite_analyse raised: {e}", exc_info=True)
            elite_context = None
            steps.append(_step("Elite analysis", "NumPy · SciPy · Scikit-learn",
                status="error", ms=(time.perf_counter()-te)*1000,
                preview=f"Failed: {e}"))

        if elite_context is not None:
            # Each builder is independently guarded, so one bad key
            # can no longer wipe out the whole step.
            insights.extend(_findings_to_insights(elite_context))
            insights.extend(_correlations_to_insights(elite_context))
            insights.extend(_causal_to_insights(elite_context))
            insights.extend(_bias_to_insights(elite_context))
            insights.extend(_power_to_insights(elite_context))
            insights.extend(_uncertainty_to_insights(elite_context))
            metrics.extend(_elite_metrics(elite_context))

            n_find = len(elite_context.get("findings") or [])
            n_corr = len(elite_context.get("correlations") or [])
            n_seg  = _count_segments(elite_context.get("segmentation"))
            n_causal = (len(_g(elite_context, "causal_analysis", "potential_confounders", default=[])) +
                        len(_g(elite_context, "causal_analysis", "granger_causality_signals", default=[])))
            n_bias = len(_g(elite_context, "bias_audit", "missingness_mechanism", default={}))

            steps.append(_step("Elite analysis", "NumPy · SciPy · Scikit-learn",
                status="done", ms=(time.perf_counter()-te)*1000,
                preview=(f"{n_find} findings · {n_corr} correlations · {n_seg} segments · "
                         f"{n_causal} causal flags · {n_bias} bias flags")))

        # ── STEP 4: Quality metrics ────────────────────────────────────────────
        t = time.perf_counter()
        try:
            quality = analysis_service.quality_report(df)
            metrics.insert(0, Metric(
                label="Data quality",
                value=f"{quality['overall_score']}%",
                trend="up" if quality["overall_score"] > 85 else "down",
            ))
            steps.append(_step("Quality scored", "Pandas",
                ms=(time.perf_counter()-t)*1000,
                preview=f"Score: {quality['overall_score']}% · Missing: {quality['missing_cells']} cells"))
        except Exception as e:
            logger.warning(f"Quality report failed: {e}")
            steps.append(_step("Quality scored", "Pandas", status="error",
                ms=(time.perf_counter()-t)*1000, preview=str(e)))
    else:
        steps.append(_step("Data received", "API", ms=1.0, preview="No dataset — AI analysis only"))

    # ── STEP 5: Forecast ──────────────────────────────────────────────────────
    if df is not None and req.enable_forecast:
        t = time.perf_counter()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols and len(df) >= 20:
            try:
                series = df[numeric_cols[0]].dropna()
                fr = analysis_service.forecast_arima(series, 12)
                metrics.append(Metric(
                    label="Next period forecast",
                    value=round(fr["forecast"][0], 2),
                    trend="up" if fr["forecast"][0] > float(series.iloc[-1]) else "down",
                ))
                fig = viz_service.plotly_forecast(
                    series.tolist()[-30:], fr["forecast"],
                    fr["lower_bound"], fr["upper_bound"],
                    f"{numeric_cols[0].replace('_',' ')} — Forecast")
                charts.append(ChartData(
                    chart_type="forecast",
                    title=f"{numeric_cols[0].replace('_',' ')} Forecast",
                    data=fig,
                    description=f"12-period ahead forecast using {fr['model']} · AIC: {fr['aic']}",
                ))
                steps.append(_step("Forecast generated", "Statsmodels ARIMA",
                    ms=(time.perf_counter()-t)*1000,
                    preview=f"Model: {fr['model']} · AIC: {fr['aic']}"))
            except Exception as e:
                logger.warning(f"Forecast failed: {e}")
                steps.append(_step("Forecast generated", "Statsmodels", status="error",
                    ms=(time.perf_counter()-t)*1000, preview=str(e)))

    # ── STEP 6: Charts ────────────────────────────────────────────────────────
    if df is not None and req.enable_viz:
        t = time.perf_counter()
        try:
            auto = viz_service.auto_chart(df)
            for c in auto:
                charts.append(ChartData(chart_type=c["type"], title=c["title"], data=c["figure"]))
            steps.append(_step("Charts built", "Plotly",
                ms=(time.perf_counter()-t)*1000,
                preview=f"{len(auto)} interactive charts generated"))
        except Exception as e:
            logger.warning(f"Viz failed: {e}")
            steps.append(_step("Charts built", "Plotly", status="error",
                ms=(time.perf_counter()-t)*1000, preview=str(e)))

    # ── STEP 7: Retrieve benchmarks + prior context (RAG + Memory) ────────────
    t = time.perf_counter()
    rag_block, memory_block = "", ""

    try:
        from app.services.rag_service import rag_service
        metric_names = list(df.columns) if df is not None else []
        rag_block = rag_service.get_industry_context(req.industry.value, metric_names)
    except Exception as e:
        logger.info(f"RAG context unavailable: {e}")

    try:
        from app.services.memory_service import memory_service
        memory_block = memory_service.get_memory_context(session_id)
    except Exception as e:
        logger.info(f"Memory context unavailable: {e}")

    if rag_block or memory_block:
        steps.append(_step("Knowledge retrieved", "RAG · Memory",
            ms=(time.perf_counter()-t)*1000,
            preview=(f"{'Benchmarks loaded' if rag_block else 'No benchmarks'} · "
                     f"{'Prior analyses recalled' if memory_block else 'No prior context'}")))

    # ── STEP 8: Elite AI narrative ────────────────────────────────────────────
    t = time.perf_counter()
    context_parts = [f"Industry: {req.industry.value}", f"User question: {req.query}"]

    if df is not None:
        try:
            desc = analysis_service.describe(df)
            context_parts.append(
                f"Dataset: {desc['shape']['rows']} rows × {desc['shape']['columns']} columns")
            context_parts.append(f"Columns available: {list(desc['dtypes'].keys())}")
        except Exception:
            context_parts.append(f"Dataset: {len(df)} rows × {len(df.columns)} columns")
            context_parts.append(f"Columns available: {list(df.columns)}")

        # Give the model the ACTUAL data — otherwise it can only speak in generalities.
        try:
            context_parts.append("\n" + _data_facts(df, query=req.query))
        except Exception as e:
            logger.warning(f"Data facts failed: {e}")

    if rag_block:
        context_parts.append("\n" + rag_block)
        context_parts.append(
            "Compare the findings against the benchmarks above. "
            "Where a metric falls outside the acceptable band, say so explicitly and cite the benchmark."
        )
    if memory_block:
        context_parts.append("\n" + memory_block)

    context_parts.append(
        "\n════════ HOW TO ANSWER ════════\n"
        f"Answer THIS exact question: \"{req.query}\"\n"
        "RULES:\n"
        "1. ANSWER THE QUESTION ASKED — do not just describe or list the data. If the "
        "question asks to classify, group, rank, or compare, then DO that classification "
        "and present the result. A list of rows is NOT an answer unless a list was asked "
        "for. Work out the answer, then show the evidence.\n"
        "1b. USE THE FULL FACT SHEET. Everything above is computed over EVERY row — "
        "percentiles, skew, concentration/Gini, exact group totals, cross-tabs, named "
        "outliers, and period movements. Draw on these specifics. If the distribution is "
        "skewed, say the mean is misleading and cite the median. If value is concentrated, "
        "quantify it ('43% of records hold 80% of value'). If outliers distort the average, "
        "name them and state the effect. Shallow answers that ignore this detail are wrong.\n"
        "2. THE 'TOP ROWS' BLOCK IS AUTHORITATIVE. It is computed from the ENTIRE dataset, "
        "not a sample. If asked which is highest/top/best/leading, name the first row of "
        "that block outright — never say the answer 'cannot be determined' or hedge about "
        "not seeing the full data. You have the leaders.\n"
        "2. SPECIFIC, NOT GENERAL. Name the actual rows, categories, and values from the "
        "data above. 'Shooter dominates with 7.07M across 2 titles, led by Asteroids "
        "(4.31M)' — never 'some genres performed well'.\n"
        "3. LEAD WITH THE ANSWER. The first sentence must state the conclusion the "
        "question is asking for, with a concrete figure.\n"
        "4. SHOW THE RECORDS. Support the answer with a markdown table of the relevant "
        "rows — grouped or ranked the way the question implies, with real values.\n"
        "5. CITE REAL NUMBERS everywhere — totals, deltas, percentages, counts computed "
        "from the data above.\n"
        "6. NO FILLER. Delete any sentence that would be true of any dataset.\n"
        "Be precise and evidence-bound. This is an elite analyst's answer to a specific "
        "question — not a data description."
    )

    context_msg = "\n".join(context_parts)
    messages = list(req.conversation_history) + [{"role": "user", "content": context_msg}]

    try:
        narrative, tokens, provider_used = await llm_service.chat(
            messages=messages,
            industry=req.industry.value,
            provider=req.provider,
            model=req.model,
            max_tokens=3000,
            elite_context=elite_context,
        )
        steps.append(_step("AI report written", req.provider.value,
            ms=(time.perf_counter()-t)*1000,
            preview=f"{tokens} tokens · {provider_used}"))
    except Exception as e:
        logger.error(f"LLM failed: {e}", exc_info=True)
        narrative = f"AI analysis unavailable: {e}"
        tokens = 0
        provider_used = req.provider.value
        steps.append(_step("AI report written", provider_used, status="error",
            ms=(time.perf_counter()-t)*1000, preview=str(e)))

    response = AnalysisResponse(
        query=req.query,
        industry=req.industry.value,
        provider=provider_used,
        model=req.model or provider_used,
        narrative=narrative,
        metrics=metrics,
        insights=insights,
        charts=charts,
        pipeline_steps=steps,
        raw_data_preview=(
            json.loads(df.head(6).to_json(orient="records")) if df is not None else None
        ),
        execution_ms=round((time.perf_counter()-t0)*1000, 1),
        tokens_used=tokens,
    )

    # ── STEP 9: Persist to memory (best effort) ───────────────────────────────
    try:
        from app.services.memory_service import memory_service
        memory_service.save_analysis_memory(session_id, {
            "industry": req.industry.value,
            "query": req.query,
            "provider": provider_used,
            "narrative": narrative,
            "insights": [i.model_dump() for i in insights],
            "row_count": len(df) if df is not None else 0,
            "col_count": len(df.columns) if df is not None else 0,
        })
    except Exception as e:
        logger.info(f"Could not persist to memory: {e}")

    return response


# ── Diagnostics ───────────────────────────────────────────────────────────────

@router.post("/elite-debug")
async def elite_debug(data: list[dict], query: str = "test", industry: str = "general"):
    """
    Run elite_analyse in isolation and report exactly what it returns
    or exactly how it fails. Use this when the Elite analysis pipeline
    step reports an error.
    """
    if not data:
        raise HTTPException(400, "Provide data")
    df = pd.DataFrame(data)
    t = time.perf_counter()
    try:
        elite = analysis_service.elite_analyse(df, query, industry)
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc().splitlines()[-8:],
            "duration_ms": round((time.perf_counter()-t)*1000, 1),
        }

    keys = list(elite.keys()) if isinstance(elite, dict) else []
    return {
        "success": True,
        "duration_ms": round((time.perf_counter()-t)*1000, 1),
        "returned_type": type(elite).__name__,
        "top_level_keys": keys,
        "counts": {
            "findings": len(elite.get("findings") or []),
            "correlations": len(elite.get("correlations") or []),
            "segments": _count_segments(elite.get("segmentation")),
            "confounders": len(_g(elite, "causal_analysis", "potential_confounders", default=[])),
            "bias_flags": len(_g(elite, "bias_audit", "missingness_mechanism", default={})),
            "uncertainty": len(elite.get("uncertainty") or []),
        },
        "correlation_keys": (list(elite["correlations"][0].keys())
                             if elite.get("correlations") else []),
        "finding_keys": (list(elite["findings"][0].keys())
                         if elite.get("findings") else []),
        "segmentation_sample": (list(elite["segmentation"].items())[:1]
                                if elite.get("segmentation") else []),
    }


@router.post("/describe")
async def describe_data(data: list[dict]):
    df = pd.DataFrame(data)
    return {"statistics": analysis_service.describe(df), "quality": analysis_service.quality_report(df)}


@router.post("/anomaly")
async def detect_anomaly(column: str, method: str = "zscore", data: list[dict] = []):
    if not data:
        raise HTTPException(400, "Provide data")
    df = pd.DataFrame(data)
    if column not in df.columns:
        raise HTTPException(400, f"Column not found. Available: {df.columns.tolist()}")
    return analysis_service.detect_anomalies(df, column, method)


@router.post("/cluster")
async def cluster_data(req: ClusterRequest):
    if not req.data:
        raise HTTPException(400, "Provide data")
    df = pd.DataFrame(req.data)
    missing = [f for f in req.features if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    return analysis_service.cluster(df, req.features, req.n_clusters)
