from __future__ import annotations
import logging, json
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataAnalysisService:

    def clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        original_shape = df.shape
        report = {
            "original_rows": original_shape[0],
            "original_cols": original_shape[1],
            "steps": [],
        }
        df = df.copy()

        # Step 1 — Strip whitespace
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        report["steps"].append({"step": "Whitespace strip", "library": "Pandas", "affected_columns": len(str_cols)})

        # Step 2 — Infer types
        converted = []
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="raise")
                converted.append(f"{col} -> datetime")
                continue
            except Exception:
                pass
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="raise")
                converted.append(f"{col} -> numeric")
            except Exception:
                pass
        report["steps"].append({"step": "Type inference", "library": "Pandas", "conversions": converted})

        # Step 3 — Remove duplicates
        dupes_before = int(df.duplicated().sum())
        df = df.drop_duplicates()
        report["steps"].append({"step": "Duplicate removal", "library": "Pandas", "rows_removed": dupes_before})

        # Step 4 — Impute missing values
        numeric_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        imputed = {}
        for col in numeric_cols:
            n = int(df[col].isnull().sum())
            if n > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                imputed[col] = {"missing": n, "filled_with": f"median ({median_val:.4f})"}
        for col in cat_cols:
            n = int(df[col].isnull().sum())
            if n > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                imputed[col] = {"missing": n, "filled_with": f"mode ({mode_val})"}
        report["steps"].append({"step": "Missing value imputation", "library": "Pandas + NumPy", "columns_imputed": imputed})

        # Step 5 — Winsorise outliers
        winsorised = {}
        for col in df.select_dtypes(include="number").columns:
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue
            lower, upper = mean - 3 * std, mean + 3 * std
            n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_capped > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                winsorised[col] = {"capped_values": n_capped, "range": f"[{lower:.2f}, {upper:.2f}]"}
        report["steps"].append({"step": "Outlier winsorisation", "library": "Pandas + NumPy", "columns_winsorised": winsorised})

        # Step 6 — Log-transform skewed columns
        transformed = {}
        for col in df.select_dtypes(include="number").columns:
            try:
                skewness = float(df[col].skew())
                if abs(skewness) > 1.5 and df[col].min() > 0:
                    df[col + "_log"] = np.log1p(df[col])
                    transformed[col] = {"skewness": round(skewness, 3), "new_column": col + "_log"}
            except Exception:
                pass
        report["steps"].append({"step": "Skewness correction", "library": "NumPy + SciPy", "columns_transformed": transformed})

        report["final_rows"] = len(df)
        report["final_cols"] = len(df.columns)
        report["rows_removed_total"] = original_shape[0] - len(df)
        logger.info(f"Cleaning complete: {original_shape} -> {df.shape}")
        return df, report

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
            "overall_score": round(
                completeness * 0.5
                + (1 - dupes / max(len(df), 1)) * 100 * 0.3
                + (1 - len(outliers) / max(len(numeric_cols), 1)) * 100 * 0.2, 1
            ),
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
        return {
            "method": method,
            "column": column,
            "total_records": len(series),
            "anomaly_count": int(mask.sum()),
            "anomaly_pct": round(mask.mean() * 100, 2),
        }

    def forecast_arima(self, series: pd.Series, periods: int = 12) -> dict:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        adf = adfuller(series.dropna())
        d = 0 if adf[1] < 0.05 else 1
        model = ARIMA(series.dropna(), order=(1, d, 1)).fit()
        forecast = model.forecast(steps=periods)
        ci = model.get_forecast(steps=periods).conf_int()
        return {
            "model": f"ARIMA(1,{d},1)",
            "aic": round(model.aic, 2),
            "forecast": [round(float(v), 4) for v in forecast],
            "lower_bound": [round(float(v), 4) for v in ci.iloc[:, 0]],
            "upper_bound": [round(float(v), 4) for v in ci.iloc[:, 1]],
        }

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
        return {
            "model": "XGBoost",
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "feature_importances": dict(zip(features, model.feature_importances_.round(4).tolist())),
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
        df2 = X.copy()
        df2["_cluster"] = labels
        clusters = {}
        for i in range(n_clusters):
            grp = df2[df2["_cluster"] == i][features]
            clusters[f"cluster_{i}"] = {"size": int((labels == i).sum()), "means": grp.mean().round(4).to_dict()}
        return {"model": "K-Means", "n_clusters": n_clusters, "silhouette_score": round(float(sil), 4), "clusters": clusters}

    def df_to_records(self, df: pd.DataFrame, limit: int = 50) -> list:
        return json.loads(df.head(limit).to_json(orient="records"))


analysis_service = DataAnalysisService()
