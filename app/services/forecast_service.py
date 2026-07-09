"""
DataMind Agent — Predictive Analytics Service
Time-series forecasting, driver analysis, and what-if scenarios.
Uses statsmodels (Holt-Winters, OLS) and scikit-learn, with honest
confidence intervals and explicit assumptions.
"""
from __future__ import annotations
import logging, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MIN_POINTS_TREND = 4
MIN_POINTS_HOLT = 6
MIN_POINTS_SEASONAL = 12


class ForecastService:
    """
    Produces forecasts that are honest about their own reliability.
    Every forecast carries a confidence interval, an R-squared, a method name,
    and a plain-English caveat about what could make it wrong.
    """

    def forecast(self, df: pd.DataFrame, periods: int = 3,
                 target_columns: list = None, time_column: str = None) -> dict:
        if df is None or df.empty:
            return {"success": False, "error": "No data to forecast"}

        time_col = time_column or self._detect_time_column(df)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if time_col in num_cols:
            num_cols.remove(time_col)
        targets = [c for c in (target_columns or num_cols) if c in num_cols][:6]

        if not targets:
            return {"success": False, "error": "No numeric columns available to forecast"}

        n = len(df)
        if n < MIN_POINTS_TREND:
            return {"success": False,
                    "error": f"Need at least {MIN_POINTS_TREND} periods to forecast. You have {n}.",
                    "hint": "Add more historical periods, then try again."}

        labels = self._period_labels(df, time_col, periods)
        forecasts = {}
        for col in targets:
            try:
                forecasts[col] = self._forecast_series(df[col].dropna().values, periods, n)
            except Exception as e:
                logger.warning(f"Forecast failed for {col}: {e}")
                forecasts[col] = {"error": str(e)}

        drivers = self._driver_analysis(df, targets, num_cols)
        overall_conf = self._overall_confidence(forecasts, n)

        return {
            "success": True,
            "periods_forecast": periods,
            "history_periods": n,
            "time_column": time_col,
            "future_labels": labels,
            "forecasts": forecasts,
            "drivers": drivers,
            "overall_confidence": overall_conf,
            "assumptions": self._assumptions(n, periods),
            "caveats": self._caveats(n, periods, forecasts),
        }

    # ── CORE FORECAST ─────────────────────────────────────────────────────────

    def _forecast_series(self, y: np.ndarray, periods: int, n: int) -> dict:
        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]
        n = len(y)
        if n < MIN_POINTS_TREND:
            raise ValueError(f"Only {n} valid points")

        method, preds, lower, upper, r2, resid_std = None, None, None, None, None, None

        # Holt-Winters if we have enough points and it converges
        if n >= MIN_POINTS_HOLT:
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                seasonal = "add" if n >= MIN_POINTS_SEASONAL else None
                sp = 12 if (seasonal and n >= 24) else (4 if seasonal else None)
                model = ExponentialSmoothing(
                    y, trend="add", seasonal=seasonal, seasonal_periods=sp,
                    initialization_method="estimated",
                ).fit(optimized=True)
                preds = np.asarray(model.forecast(periods), dtype=float)
                fitted = np.asarray(model.fittedvalues, dtype=float)
                resid = y - fitted
                resid_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
                ss_res = float(np.sum(resid ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                method = f"Holt-Winters exponential smoothing{' with seasonality' if seasonal else ''}"
            except Exception as e:
                logger.info(f"Holt-Winters unavailable, falling back to OLS: {e}")
                preds = None

        # OLS linear trend fallback
        if preds is None:
            x = np.arange(n, dtype=float)
            slope, intercept, r, p, se = stats.linregress(x, y)
            fx = np.arange(n, n + periods, dtype=float)
            preds = intercept + slope * fx
            fitted = intercept + slope * x
            resid = y - fitted
            resid_std = float(np.std(resid, ddof=1)) if n > 2 else 0.0
            r2 = float(r ** 2)
            method = f"OLS linear trend (slope={round(float(slope),4)}/period, p={round(float(p),4)})"

        # 95% prediction interval, widening with forecast horizon
        widen = np.sqrt(1 + (np.arange(1, periods + 1) / max(n, 1)))
        margin = 1.96 * (resid_std or 0.0) * widen
        lower = preds - margin
        upper = preds + margin

        # If the historical series is never negative, a negative bound is nonsense.
        clamped = False
        if float(np.min(y)) >= 0 and float(np.min(lower)) < 0:
            lower = np.maximum(lower, 0.0)
            clamped = True

        last = float(y[-1])
        first_pred = float(preds[0])
        last_pred = float(preds[-1])
        change_pct = ((last_pred - last) / last * 100) if last else 0.0

        return {
            "method": method,
            "r_squared": round(float(r2 or 0), 4),
            "residual_std": round(float(resid_std or 0), 4),
            "predictions": [round(float(v), 2) for v in preds],
            "lower_95": [round(float(v), 2) for v in lower],
            "upper_95": [round(float(v), 2) for v in upper],
            "last_actual": round(last, 2),
            "next_period": round(first_pred, 2),
            "final_period": round(last_pred, 2),
            "projected_change_pct": round(float(change_pct), 2),
            "direction": "rising" if last_pred > last else "falling" if last_pred < last else "flat",
            "reliability": self._reliability(r2 or 0, n),
            "interval_width_pct": round(float(np.mean(margin) / abs(last) * 100), 1) if last else None,
            "lower_bound_clamped_at_zero": clamped,
        }

    def _reliability(self, r2: float, n: int) -> dict:
        if n < MIN_POINTS_HOLT:
            level, note = "Low", f"Only {n} historical periods — too few for a dependable trend."
        elif r2 >= 0.7 and n >= 12:
            level, note = "High", "Strong fit on a reasonable history length."
        elif r2 >= 0.4:
            level, note = "Medium", "Moderate fit — treat the direction as more reliable than the exact figure."
        else:
            level, note = "Low", f"Weak fit (R²={round(r2,3)}) — the series is largely unpredictable from its own history."
        return {"level": level, "note": note, "r_squared": round(float(r2), 4), "n_periods": n}

    # ── DRIVER ANALYSIS ───────────────────────────────────────────────────────

    def _driver_analysis(self, df: pd.DataFrame, targets: list, num_cols: list) -> dict:
        """Which other variables most strongly move with each target."""
        out = {}
        for target in targets[:3]:
            others = [c for c in num_cols if c != target]
            if not others:
                continue
            rows = []
            for c in others:
                sub = df[[target, c]].dropna()
                if len(sub) < 4:
                    continue
                try:
                    r, p = stats.pearsonr(sub[target], sub[c])
                except Exception:
                    continue
                if np.isnan(r):
                    continue
                rows.append({
                    "variable": c,
                    "correlation": round(float(r), 4),
                    "p_value": round(float(p), 5),
                    "significant": bool(p < 0.05),
                    "strength": ("strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"),
                    "direction": "positive" if r > 0 else "negative",
                    "n": len(sub),
                })
            rows.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            if rows:
                out[target] = {
                    "top_drivers": rows[:4],
                    "caution": "Correlation is not causation. These variables move together; that does not prove one causes the other.",
                }
        return out

    # ── WHAT-IF SCENARIOS ─────────────────────────────────────────────────────

    def scenario(self, df: pd.DataFrame, target: str, driver: str, change_pct: float) -> dict:
        """
        Estimate the effect on `target` of changing `driver` by change_pct,
        using a simple OLS relationship. Explicitly flagged as correlational.
        """
        sub = df[[target, driver]].dropna()
        if len(sub) < 5:
            return {"success": False, "error": f"Need at least 5 paired observations; found {len(sub)}."}

        x = sub[driver].values.astype(float)
        y = sub[target].values.astype(float)
        slope, intercept, r, p, se = stats.linregress(x, y)

        current_x = float(x.mean())
        current_y = float(intercept + slope * current_x)
        new_x = current_x * (1 + change_pct / 100.0)
        new_y = float(intercept + slope * new_x)
        delta_pct = ((new_y - current_y) / current_y * 100) if current_y else 0.0

        return {
            "success": True,
            "target": target,
            "driver": driver,
            "driver_change_pct": change_pct,
            "current_target_estimate": round(current_y, 2),
            "projected_target": round(new_y, 2),
            "projected_change_pct": round(float(delta_pct), 2),
            "relationship": {
                "slope": round(float(slope), 6),
                "r_squared": round(float(r ** 2), 4),
                "p_value": round(float(p), 5),
                "significant": bool(p < 0.05),
                "n": len(sub),
            },
            "reliability": self._reliability(float(r ** 2), len(sub)),
            "warning": (
                "This is a correlational projection, not a causal guarantee. "
                f"It assumes the historical relationship between {driver} and {target} holds, "
                "that nothing else changes, and that the change stays within the observed data range."
            ),
            "out_of_range": bool(new_x < x.min() or new_x > x.max()),
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _detect_time_column(self, df: pd.DataFrame) -> str:
        keys = ("month", "week", "quarter", "period", "year", "term", "season", "date", "day")
        for c in df.columns:
            if any(k in str(c).lower() for k in keys):
                return c
        return df.columns[0]

    def _period_labels(self, df: pd.DataFrame, time_col: str, periods: int) -> list:
        """Generate sensible future labels: Jan..Dec, Q1..Q4, or Period N+1."""
        try:
            existing = df[time_col].astype(str).tolist()
        except Exception:
            existing = []

        MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        if existing and existing[-1][:3].title() in MONTHS:
            i = MONTHS.index(existing[-1][:3].title())
            return [MONTHS[(i + k) % 12] for k in range(1, periods + 1)]

        last = existing[-1] if existing else ""
        if last.upper().startswith("Q") and len(last) >= 2 and last[1].isdigit():
            q = int(last[1])
            return [f"Q{((q + k - 1) % 4) + 1}" for k in range(1, periods + 1)]

        start = len(existing)
        return [f"Period {start + k}" for k in range(1, periods + 1)]

    def _overall_confidence(self, forecasts: dict, n: int) -> dict:
        valid = [f for f in forecasts.values() if "error" not in f]
        if not valid:
            return {"level": "None", "note": "No forecast could be produced."}
        avg_r2 = float(np.mean([f["r_squared"] for f in valid]))
        return self._reliability(avg_r2, n)

    def _assumptions(self, n: int, periods: int) -> list:
        return [
            "The future follows the same pattern as the past.",
            "No structural break, policy change, or external shock occurs.",
            f"The {n} historical periods are representative of normal operations.",
            "Values in the data are accurate and consistently measured.",
            f"Forecast accuracy decays with horizon — period {periods} is far less certain than period 1.",
        ]

    def _caveats(self, n: int, periods: int, forecasts: dict) -> list:
        c = []
        if n < MIN_POINTS_HOLT:
            c.append(f"Only {n} periods of history. Forecasts below {MIN_POINTS_HOLT} periods are indicative at best.")
        if n < MIN_POINTS_SEASONAL:
            c.append(f"With fewer than {MIN_POINTS_SEASONAL} periods, seasonality cannot be detected or modelled.")
        if periods > n / 2:
            c.append(f"Forecasting {periods} periods from {n} of history is aggressive. Halve the horizon for a tighter interval.")
        weak = [k for k, v in forecasts.items() if "error" not in v and v["r_squared"] < 0.3]
        if weak:
            c.append(f"These series show weak historical structure and may be effectively unpredictable: {', '.join(weak)}.")
        clamped = [k for k, v in forecasts.items() if "error" not in v and v.get("lower_bound_clamped_at_zero")]
        if clamped:
            c.append(f"The uncertainty band for {', '.join(clamped)} extended below zero and was clamped. "
                     "That means the true uncertainty is wider than the interval suggests.")
        c.append("The 95% intervals show where the value is likely to land — not where it will land.")
        return c


forecast_service = ForecastService()
