import numpy as np
import pandas as pd

class DataCleaner:

    def clean(self, df: pd.DataFrame):

        original_shape = df.shape

        df = df.replace(["NaN", "nan", "invalid", ""], np.nan)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        df = df.drop_duplicates()

        return df, {
            "original_shape": original_shape,
            "cleaned_shape": df.shape
        }
