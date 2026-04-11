class BusinessValidator:

    def validate(self, df):

        issues = []

        if {"units_sold", "unit_price", "revenue"}.issubset(df.columns):

            df["expected_revenue"] = df["units_sold"] * df["unit_price"]

            mismatch = df[
                (df["revenue"] - df["expected_revenue"]).abs() >
                0.1 * df["expected_revenue"]
            ]

            issues.append({
                "type": "revenue_mismatch",
                "count": len(mismatch)
            })

        return issues
