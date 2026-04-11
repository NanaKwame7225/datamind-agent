import numpy as np

class AnalyticsEngine:

    def analyze(self, df):

        numeric = df.select_dtypes(include=[np.number])

        if numeric.empty:
            return {"error": "no numeric data"}

        return {
            "mean": numeric.mean().to_dict(),
            "std": numeric.std().to_dict(),
            "correlation": numeric.corr().to_dict()
        }
