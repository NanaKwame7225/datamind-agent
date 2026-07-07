"""
DataMind Agent — SQL Execution Layer
Uses DuckDB to run SQL directly against uploaded data in memory.
No database server needed — blazing fast in-process SQL.
"""
from __future__ import annotations
import logging, json, time
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SQLExecutionService:
    """
    In-memory SQL engine using DuckDB.
    Lets users query their data with standard SQL —
    SELECT, WHERE, GROUP BY, JOIN, window functions, all supported.
    """

    def __init__(self):
        self._connections = {}

    def _get_conn(self):
        import duckdb
        return duckdb.connect(':memory:')

    def load_dataframe(self, df: pd.DataFrame, table_name: str = "data") -> dict:
        """Load a DataFrame as a SQL table and return schema info."""
        try:
            conn = self._get_conn()
            conn.register(table_name, df)
            schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
            row_count = len(df)
            conn.close()
            return {
                "success": True,
                "table_name": table_name,
                "row_count": row_count,
                "columns": schema.to_dict("records"),
                "sample_queries": self._generate_sample_queries(df, table_name),
            }
        except Exception as e:
            logger.error(f"SQL load error: {e}")
            return {"success": False, "error": str(e)}

    def execute_query(self, df: pd.DataFrame, query: str, table_name: str = "data") -> dict:
        """Execute a SQL query against the DataFrame and return results."""
        t0 = time.perf_counter()
        try:
            import duckdb
            conn = duckdb.connect(':memory:')
            conn.register(table_name, df)

            # Safety check — no destructive operations
            q_upper = query.strip().upper()
            if any(kw in q_upper for kw in ['DROP ', 'DELETE ', 'TRUNCATE ', 'INSERT ', 'UPDATE ', 'CREATE TABLE', 'ALTER ']):
                return {"success": False, "error": "Only SELECT queries are allowed. Destructive operations are not permitted."}

            result = conn.execute(query).fetchdf()
            conn.close()

            execution_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "success": True,
                "query": query,
                "row_count": len(result),
                "col_count": len(result.columns),
                "columns": list(result.columns),
                "data": json.loads(result.head(500).to_json(orient="records")),
                "execution_ms": execution_ms,
                "truncated": len(result) > 500,
            }
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "hint": self._get_error_hint(str(e)),
                "execution_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

    def natural_language_to_sql(self, question: str, df: pd.DataFrame, table_name: str = "data") -> dict:
        """
        Convert a natural language question into SQL.
        Used when users ask questions — auto-generates the right SQL query.
        """
        cols = list(df.columns)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        q = question.lower()

        # Pattern matching for common query types
        sql = None
        description = ""

        if any(w in q for w in ['top', 'highest', 'most', 'largest', 'biggest', 'best']):
            n = 10
            for word in q.split():
                try:
                    n = int(word)
                    break
                except:
                    pass
            col = num_cols[0] if num_cols else cols[0]
            for nc in num_cols:
                if any(w in nc.lower() for w in q.split() if len(w) > 3):
                    col = nc
                    break
            sql = f"SELECT * FROM {table_name} ORDER BY {col} DESC LIMIT {n}"
            description = f"Top {n} rows by {col}"

        elif any(w in q for w in ['bottom', 'lowest', 'least', 'smallest', 'worst']):
            n = 10
            col = num_cols[0] if num_cols else cols[0]
            sql = f"SELECT * FROM {table_name} ORDER BY {col} ASC LIMIT {n}"
            description = f"Bottom {n} rows by {col}"

        elif any(w in q for w in ['average', 'avg', 'mean', 'total', 'sum', 'count']):
            if num_cols:
                agg_parts = []
                for nc in num_cols[:6]:
                    if any(w in q for w in ['total', 'sum']):
                        agg_parts.append(f"SUM({nc}) as total_{nc}")
                    else:
                        agg_parts.append(f"AVG({nc}) as avg_{nc}, SUM({nc}) as total_{nc}")
                agg = ", ".join(agg_parts) if agg_parts else f"COUNT(*) as row_count"
                if cat_cols:
                    sql = f"SELECT {cat_cols[0]}, {agg} FROM {table_name} GROUP BY {cat_cols[0]} ORDER BY avg_{num_cols[0] if num_cols else ''} DESC"
                else:
                    sql = f"SELECT {agg} FROM {table_name}"
                description = f"Aggregated summary"

        elif any(w in q for w in ['where', 'filter', 'only', 'show']):
            sql = f"SELECT * FROM {table_name} LIMIT 100"
            description = "Filtered data"

        elif any(w in q for w in ['group', 'by', 'breakdown', 'split', 'segment']):
            if cat_cols and num_cols:
                aggs = ", ".join([f"AVG({nc}) as avg_{nc}" for nc in num_cols[:4]])
                sql = f"SELECT {cat_cols[0]}, COUNT(*) as count, {aggs} FROM {table_name} GROUP BY {cat_cols[0]} ORDER BY count DESC"
                description = f"Grouped by {cat_cols[0]}"

        elif any(w in q for w in ['anomal', 'outlier', 'unusual', 'spike', 'weird']):
            if num_cols:
                col = num_cols[0]
                sql = f"""
WITH stats AS (
  SELECT AVG({col}) as mean_val, STDDEV({col}) as std_val FROM {table_name}
)
SELECT t.*, ABS(t.{col} - s.mean_val) / NULLIF(s.std_val, 0) as z_score
FROM {table_name} t, stats s
WHERE ABS(t.{col} - s.mean_val) / NULLIF(s.std_val, 0) > 2.5
ORDER BY z_score DESC"""
                description = f"Anomaly detection on {col} (Z-score > 2.5)"

        elif any(w in q for w in ['correlat', 'relationship', 'related']):
            if len(num_cols) >= 2:
                sql = f"SELECT CORR({num_cols[0]}, {num_cols[1]}) as correlation, COUNT(*) as n FROM {table_name}"
                description = f"Correlation between {num_cols[0]} and {num_cols[1]}"

        elif any(w in q for w in ['trend', 'over time', 'growing', 'declining']):
            time_col = next((c for c in cols if any(t in c.lower() for t in ['date','month','week','quarter','period','year'])), cols[0])
            if num_cols:
                sql = f"SELECT {time_col}, {', '.join(num_cols[:4])} FROM {table_name} ORDER BY {time_col}"
                description = f"Trend over {time_col}"

        elif any(w in q for w in ['missing', 'null', 'empty', 'blank']):
            null_checks = " + ".join([f"(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in cols])
            sql = f"SELECT *, ({null_checks}) as null_count FROM {table_name} WHERE ({null_checks}) > 0"
            description = "Rows with missing values"

        if not sql:
            sql = f"SELECT * FROM {table_name} LIMIT 100"
            description = "All data (first 100 rows)"

        return {"sql": sql.strip(), "description": description, "table_name": table_name}

    def _generate_sample_queries(self, df: pd.DataFrame, table: str) -> list[str]:
        cols = list(df.columns)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        samples = [f"SELECT * FROM {table} LIMIT 10"]
        if num_cols:
            samples.append(f"SELECT AVG({num_cols[0]}), MIN({num_cols[0]}), MAX({num_cols[0]}) FROM {table}")
        if cat_cols and num_cols:
            samples.append(f"SELECT {cat_cols[0]}, AVG({num_cols[0]}) as avg_val, COUNT(*) as n FROM {table} GROUP BY {cat_cols[0]} ORDER BY avg_val DESC")
        if num_cols:
            samples.append(f"SELECT * FROM {table} ORDER BY {num_cols[0]} DESC LIMIT 5")
            if len(num_cols) >= 2:
                samples.append(f"SELECT CORR({num_cols[0]}, {num_cols[1]}) as correlation FROM {table}")
        return samples

    def _get_error_hint(self, error: str) -> str:
        if "column" in error.lower():
            return "Check column names — use the exact column names shown in the data preview."
        if "syntax" in error.lower():
            return "Check your SQL syntax. Example: SELECT col1, col2 FROM data WHERE col1 > 100"
        if "type" in error.lower():
            return "Type mismatch — make sure you are comparing numbers to numbers and text to text."
        return "Check your SQL query syntax and column names."

    def profile_table(self, df: pd.DataFrame, table_name: str = "data") -> dict:
        """Generate a full SQL-powered data profile."""
        try:
            import duckdb
            conn = duckdb.connect(':memory:')
            conn.register(table_name, df)
            num_cols = df.select_dtypes(include="number").columns.tolist()
            profiles = {}
            for col in num_cols[:8]:
                try:
                    r = conn.execute(f"""
                        SELECT
                            COUNT(*) as n,
                            COUNT({col}) as non_null,
                            AVG({col}) as mean,
                            MEDIAN({col}) as median,
                            STDDEV({col}) as std,
                            MIN({col}) as min_val,
                            MAX({col}) as max_val,
                            QUANTILE_CONT({col}, 0.25) as p25,
                            QUANTILE_CONT({col}, 0.75) as p75
                        FROM {table_name}
                    """).fetchdf()
                    profiles[col] = {k: round(float(v),4) if v is not None else None for k, v in r.iloc[0].to_dict().items()}
                except Exception:
                    pass
            conn.close()
            return {"success": True, "profiles": profiles, "row_count": len(df), "col_count": len(df.columns)}
        except Exception as e:
            return {"success": False, "error": str(e)}


sql_service = SQLExecutionService()
