from __future__ import annotations
import logging, json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DataAnalysisService:

    def describe(self, df: pd.DataFrame) -> dict:
        numeric = df.select_dtypes(include="number")
        categorical = df.select_dtypes(include=["object","category"])
        stats = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().mean()*100).round(2).to_dict(),
            "numeric_stats": {},
            "categorical_stats": {},
        }
        if not numeric.empty:
            stats["numeric_stats"] = numeric.describe().round(4).to_dict()
        for col in categorical.columns:
            stats["categorical_stats"][col] = {
                "unique": int(df[col].nunique()),
                "top": str(df[col].mode()[0]) if not df[col].mode().empty else None,
            }
        return stats

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
            if n > 0:
                outliers[col] = n
        completeness = round((1 - missing/max(total,1))*100, 2)
        return {
            "completeness_pct": completeness,
            "missing_cells": missing,
            "duplicate_rows": dupes,
            "outlier_columns": outliers,
            "overall_score": round(completeness*0.5 + (1-dupes/max(len(df),1))*100*0.3 +
                                   (1-len(outliers)/max(len(numeric_cols),1))*100*0.2, 1),
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
        return {
            "method": method, "column": column,
            "total_records": len(series),
            "anomaly_count": int(mask.sum()),
            "anomaly_pct": round(mask.mean()*100, 2),
        }

    def forecast_arima(self, series: pd.Series, periods: int = 12) -> dict:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        adf = adfuller(series.dropna())
        d = 0 if adf[1] < 0.05 else 1
        model = ARIMA(series.dropna(), order=(1,d,1)).fit()
        forecast = model.forecast(steps=periods)
        ci = model.get_forecast(steps=periods).conf_int()
        return {
            "model": f"ARIMA(1,{d},1)", "aic": round(model.aic, 2),
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
        df2 = X.copy()
        df2["_cluster"] = labels
        clusters = {}
        for i in range(n_clusters):
            grp = df2[df2["_cluster"]==i][features]
            clusters[f"cluster_{i}"] = {"size": int((labels==i).sum()),
                                         "means": grp.mean().round(4).to_dict()}
        return {"model": "K-Means", "n_clusters": n_clusters,
                "silhouette_score": round(float(sil),4), "clusters": clusters}

    def forecast_xgboost(self, df, target, features):
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        X = df[features].fillna(0)
        y = df[target].fillna(method="ffill")
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test,y_test)], verbose=False)
        y_pred = model.predict(X_test)
        return {"model": "XGBoost", "mae": round(float(mean_absolute_error(y_test,y_pred)),4),
                "r2": round(float(r2_score(y_test,y_pred)),4),
                "feature_importances": dict(zip(features, model.feature_importances_.round(4).tolist()))}

    def df_to_records(self, df: pd.DataFrame, limit: int = 50) -> list:
        return json.loads(df.head(limit).to_json(orient="records"))

analysis_service = DataAnalysisService()
