import pandas as pd

class DataProfiler:

    def profile(self, df: pd.DataFrame):
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_%": df.isnull().mean().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
