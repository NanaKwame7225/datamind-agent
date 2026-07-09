"""
DataMind Agent — Universal File Parser Service
Parses Excel (.xlsx/.xls), CSV, JSON, Parquet, and PDF into clean DataFrames.
PDF parsing extracts tables using pdfplumber with a camelot-style fallback.
"""
from __future__ import annotations
import io, json, logging, re
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class FileParserService:
    """
    Converts any supported upload into a list-of-records DataFrame payload.
    Returns a consistent shape so the analysis pipeline never needs to care
    about the original file format.
    """

    SUPPORTED = {".csv", ".tsv", ".xlsx", ".xlsm", ".xls", ".json", ".parquet", ".pdf", ".txt"}

    # ── PUBLIC ENTRYPOINT ─────────────────────────────────────────────────────

    def parse(self, file_bytes: bytes, filename: str, sheet_name: str = None) -> dict:
        """
        Parse any supported file into records.
        Returns: {success, records, columns, row_count, sheets?, tables?, source_format}
        """
        ext = self._ext(filename)
        if ext not in self.SUPPORTED:
            return {"success": False, "error": f"Unsupported file type: {ext}. Supported: {', '.join(sorted(self.SUPPORTED))}"}

        try:
            if ext in (".xlsx", ".xlsm", ".xls"):
                return self._parse_excel(file_bytes, sheet_name)
            if ext == ".pdf":
                return self._parse_pdf(file_bytes)
            if ext in (".csv", ".tsv", ".txt"):
                return self._parse_csv(file_bytes, delimiter="\t" if ext == ".tsv" else None)
            if ext == ".json":
                return self._parse_json(file_bytes)
            if ext == ".parquet":
                return self._parse_parquet(file_bytes)
        except Exception as e:
            logger.error(f"Parse error for {filename}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "source_format": ext}

        return {"success": False, "error": "Unhandled file type"}

    def inspect(self, file_bytes: bytes, filename: str) -> dict:
        """
        Peek at a file without fully parsing it.
        For Excel: returns sheet names + row counts so the user can choose a sheet.
        For PDF: returns how many tables were detected on each page.
        """
        ext = self._ext(filename)
        try:
            if ext in (".xlsx", ".xlsm", ".xls"):
                xls = pd.ExcelFile(io.BytesIO(file_bytes))
                sheets = []
                for name in xls.sheet_names:
                    df = xls.parse(name, nrows=5)
                    full = xls.parse(name)
                    sheets.append({
                        "name": name,
                        "rows": len(full),
                        "columns": list(df.columns.astype(str)),
                        "preview": json.loads(df.head(5).to_json(orient="records")),
                    })
                return {"success": True, "type": "excel", "sheets": sheets, "sheet_count": len(sheets)}

            if ext == ".pdf":
                info = self._pdf_table_inventory(file_bytes)
                return {"success": True, "type": "pdf", **info}

            # csv/json/parquet: just parse and preview
            parsed = self.parse(file_bytes, filename)
            if not parsed.get("success"):
                return parsed
            return {
                "success": True,
                "type": ext.lstrip("."),
                "rows": parsed["row_count"],
                "columns": parsed["columns"],
                "preview": parsed["records"][:5],
            }
        except Exception as e:
            logger.error(f"Inspect error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ── EXCEL ─────────────────────────────────────────────────────────────────

    def _parse_excel(self, file_bytes: bytes, sheet_name: str = None) -> dict:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        sheets = xls.sheet_names

        # If no sheet requested, pick the one with the most rows (usually the data sheet)
        if sheet_name is None or sheet_name not in sheets:
            best, best_rows = sheets[0], -1
            for s in sheets:
                try:
                    n = len(xls.parse(s))
                    if n > best_rows:
                        best, best_rows = s, n
                except Exception:
                    continue
            sheet_name = best

        df = xls.parse(sheet_name)
        df = self._clean_frame(df)
        return {
            "success": True,
            "source_format": "excel",
            "sheet_used": sheet_name,
            "sheets": sheets,
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df),
            "col_count": len(df.columns),
        }

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _pdf_table_inventory(self, file_bytes: bytes) -> dict:
        """Report how many tables exist per page without extracting them fully."""
        try:
            import pdfplumber
        except ImportError:
            return {"error": "pdfplumber not installed", "pages": 0, "tables_found": 0}

        pages_info = []
        total = 0
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages[:20], 1):
                tables = page.find_tables()
                pages_info.append({"page": i, "tables": len(tables)})
                total += len(tables)
            page_count = len(pdf.pages)
        return {"pages": page_count, "tables_found": total, "per_page": pages_info}

    def _parse_pdf(self, file_bytes: bytes) -> dict:
        """
        Extract tables from a PDF. Uses the largest coherent table found,
        and returns all tables so the user can pick a different one.
        """
        try:
            import pdfplumber
        except ImportError:
            return {"success": False, "error": "PDF parsing requires pdfplumber. Add pdfplumber to requirements.txt."}

        all_tables = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for pnum, page in enumerate(pdf.pages[:30], 1):
                for tnum, table in enumerate(page.extract_tables(), 1):
                    if not table or len(table) < 2:
                        continue
                    df = self._table_to_frame(table)
                    if df is None or df.empty or len(df.columns) < 2:
                        continue
                    all_tables.append({
                        "page": pnum,
                        "table_index": tnum,
                        "rows": len(df),
                        "columns": list(df.columns.astype(str)),
                        "frame": df,
                    })

        if not all_tables:
            # Fall back to raw text extraction so the user at least gets something
            text = self._pdf_text(file_bytes)
            return {
                "success": False,
                "error": "No structured tables found in this PDF.",
                "source_format": "pdf",
                "extracted_text": text[:4000],
                "hint": "This PDF appears to be text or images rather than tables. Try exporting it to CSV or Excel first.",
            }

        # Pick the biggest table (rows x cols) as the primary dataset
        primary = max(all_tables, key=lambda t: t["rows"] * len(t["columns"]))
        df = self._clean_frame(primary["frame"])

        return {
            "success": True,
            "source_format": "pdf",
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df),
            "col_count": len(df.columns),
            "table_used": {"page": primary["page"], "table_index": primary["table_index"]},
            "tables_found": len(all_tables),
            "all_tables": [
                {"page": t["page"], "table_index": t["table_index"], "rows": t["rows"], "columns": t["columns"]}
                for t in all_tables
            ],
        }

    def _pdf_text(self, file_bytes: bytes) -> str:
        try:
            import pdfplumber
            out = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages[:10]:
                    out.append(page.extract_text() or "")
            return "\n".join(out)
        except Exception:
            return ""

    def _table_to_frame(self, table: list) -> Optional[pd.DataFrame]:
        """Convert a raw pdfplumber table (list of lists) into a DataFrame."""
        rows = [[(c or "").strip() for c in row] for row in table if row]
        if len(rows) < 2:
            return None
        header = rows[0]
        # Deduplicate / fill blank headers
        seen, clean_header = {}, []
        for i, h in enumerate(header):
            h = h or f"column_{i+1}"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            clean_header.append(h)
        body = [r for r in rows[1:] if any(c for c in r)]
        if not body:
            return None
        width = len(clean_header)
        body = [(r + [""] * width)[:width] for r in body]
        return pd.DataFrame(body, columns=clean_header)

    # ── CSV / JSON / PARQUET ──────────────────────────────────────────────────

    def _parse_csv(self, file_bytes: bytes, delimiter: str = None) -> dict:
        text = file_bytes.decode("utf-8", errors="replace")
        if delimiter is None:
            delimiter = self._sniff_delimiter(text)
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
        df = self._clean_frame(df)
        return {
            "success": True, "source_format": "csv",
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df), "col_count": len(df.columns),
            "delimiter_detected": delimiter,
        }

    def _parse_json(self, file_bytes: bytes) -> dict:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        if isinstance(data, dict):
            # Find the first list-of-dicts inside the object
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    data = v
                    break
            else:
                data = [data]
        df = pd.DataFrame(data)
        df = self._clean_frame(df)
        return {
            "success": True, "source_format": "json",
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df), "col_count": len(df.columns),
        }

    def _parse_parquet(self, file_bytes: bytes) -> dict:
        df = pd.read_parquet(io.BytesIO(file_bytes))
        df = self._clean_frame(df)
        return {
            "success": True, "source_format": "parquet",
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df), "col_count": len(df.columns),
        }

    # ── SHARED CLEANING ───────────────────────────────────────────────────────

    def _clean_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names, drop empty rows/cols, coerce numerics."""
        df = df.copy()
        # Drop fully-empty columns and rows
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        # Clean column names
        df.columns = [self._clean_col(str(c), i) for i, c in enumerate(df.columns)]
        # Coerce string columns that are actually numeric
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            try:
                coerced = pd.to_numeric(
                    df[col].astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("$", "", regex=False)
                        .str.replace("GHS", "", regex=False)
                        .str.replace("%", "", regex=False)
                        .str.replace("\u20b5", "", regex=False)
                        .str.strip(),
                    errors="coerce",
                )
            except Exception:
                continue
            # Only convert if most values parsed cleanly
            if coerced.notna().sum() >= max(1, int(0.8 * len(df))):
                df[col] = coerced
        return df.reset_index(drop=True)

    def _clean_col(self, name: str, idx: int) -> str:
        name = name.strip()
        if not name or name.lower().startswith("unnamed"):
            return f"column_{idx+1}"
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^\w]", "", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name.lower() or f"column_{idx+1}"

    def _to_records(self, df: pd.DataFrame) -> list:
        """Convert to JSON-safe records (NaN -> None, numpy types -> python)."""
        safe = df.where(pd.notna(df), None)
        return json.loads(safe.to_json(orient="records", date_format="iso"))

    def _sniff_delimiter(self, text: str) -> str:
        first = text.split("\n", 1)[0]
        counts = {d: first.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","

    def _ext(self, filename: str) -> str:
        return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


file_parser_service = FileParserService()"""
DataMind Agent — Universal File Parser Service
Parses Excel (.xlsx/.xls), CSV, JSON, Parquet, and PDF into clean DataFrames.
PDF parsing extracts tables using pdfplumber with a camelot-style fallback.
"""
from __future__ import annotations
import io, json, logging, re
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class FileParserService:
    """
    Converts any supported upload into a list-of-records DataFrame payload.
    Returns a consistent shape so the analysis pipeline never needs to care
    about the original file format.
    """

    SUPPORTED = {".csv", ".tsv", ".xlsx", ".xlsm", ".xls", ".json", ".parquet", ".pdf", ".txt"}

    # ── PUBLIC ENTRYPOINT ─────────────────────────────────────────────────────

    def parse(self, file_bytes: bytes, filename: str, sheet_name: str = None) -> dict:
        """
        Parse any supported file into records.
        Returns: {success, records, columns, row_count, sheets?, tables?, source_format}
        """
        ext = self._ext(filename)
        if ext not in self.SUPPORTED:
            return {"success": False, "error": f"Unsupported file type: {ext}. Supported: {', '.join(sorted(self.SUPPORTED))}"}

        try:
            if ext in (".xlsx", ".xlsm", ".xls"):
                return self._parse_excel(file_bytes, sheet_name)
            if ext == ".pdf":
                return self._parse_pdf(file_bytes)
            if ext in (".csv", ".tsv", ".txt"):
                return self._parse_csv(file_bytes, delimiter="\t" if ext == ".tsv" else None)
            if ext == ".json":
                return self._parse_json(file_bytes)
            if ext == ".parquet":
                return self._parse_parquet(file_bytes)
        except Exception as e:
            logger.error(f"Parse error for {filename}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "source_format": ext}

        return {"success": False, "error": "Unhandled file type"}

    def inspect(self, file_bytes: bytes, filename: str) -> dict:
        """
        Peek at a file without fully parsing it.
        For Excel: returns sheet names + row counts so the user can choose a sheet.
        For PDF: returns how many tables were detected on each page.
        """
        ext = self._ext(filename)
        try:
            if ext in (".xlsx", ".xlsm", ".xls"):
                xls = pd.ExcelFile(io.BytesIO(file_bytes))
                sheets = []
                for name in xls.sheet_names:
                    df = xls.parse(name, nrows=5)
                    full = xls.parse(name)
                    sheets.append({
                        "name": name,
                        "rows": len(full),
                        "columns": list(df.columns.astype(str)),
                        "preview": json.loads(df.head(5).to_json(orient="records")),
                    })
                return {"success": True, "type": "excel", "sheets": sheets, "sheet_count": len(sheets)}

            if ext == ".pdf":
                info = self._pdf_table_inventory(file_bytes)
                return {"success": True, "type": "pdf", **info}

            # csv/json/parquet: just parse and preview
            parsed = self.parse(file_bytes, filename)
            if not parsed.get("success"):
                return parsed
            return {
                "success": True,
                "type": ext.lstrip("."),
                "rows": parsed["row_count"],
                "columns": parsed["columns"],
                "preview": parsed["records"][:5],
            }
        except Exception as e:
            logger.error(f"Inspect error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ── EXCEL ─────────────────────────────────────────────────────────────────

    def _parse_excel(self, file_bytes: bytes, sheet_name: str = None) -> dict:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        sheets = xls.sheet_names

        # If no sheet requested, pick the one with the most rows (usually the data sheet)
        if sheet_name is None or sheet_name not in sheets:
            best, best_rows = sheets[0], -1
            for s in sheets:
                try:
                    n = len(xls.parse(s))
                    if n > best_rows:
                        best, best_rows = s, n
                except Exception:
                    continue
            sheet_name = best

        df = xls.parse(sheet_name)
        df = self._clean_frame(df)
        return {
            "success": True,
            "source_format": "excel",
            "sheet_used": sheet_name,
            "sheets": sheets,
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df),
            "col_count": len(df.columns),
        }

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _pdf_table_inventory(self, file_bytes: bytes) -> dict:
        """Report how many tables exist per page without extracting them fully."""
        try:
            import pdfplumber
        except ImportError:
            return {"error": "pdfplumber not installed", "pages": 0, "tables_found": 0}

        pages_info = []
        total = 0
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages[:20], 1):
                tables = page.find_tables()
                pages_info.append({"page": i, "tables": len(tables)})
                total += len(tables)
            page_count = len(pdf.pages)
        return {"pages": page_count, "tables_found": total, "per_page": pages_info}

    def _parse_pdf(self, file_bytes: bytes) -> dict:
        """
        Extract tables from a PDF. Uses the largest coherent table found,
        and returns all tables so the user can pick a different one.
        """
        try:
            import pdfplumber
        except ImportError:
            return {"success": False, "error": "PDF parsing requires pdfplumber. Add pdfplumber to requirements.txt."}

        all_tables = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for pnum, page in enumerate(pdf.pages[:30], 1):
                for tnum, table in enumerate(page.extract_tables(), 1):
                    if not table or len(table) < 2:
                        continue
                    df = self._table_to_frame(table)
                    if df is None or df.empty or len(df.columns) < 2:
                        continue
                    all_tables.append({
                        "page": pnum,
                        "table_index": tnum,
                        "rows": len(df),
                        "columns": list(df.columns.astype(str)),
                        "frame": df,
                    })

        if not all_tables:
            # Fall back to raw text extraction so the user at least gets something
            text = self._pdf_text(file_bytes)
            return {
                "success": False,
                "error": "No structured tables found in this PDF.",
                "source_format": "pdf",
                "extracted_text": text[:4000],
                "hint": "This PDF appears to be text or images rather than tables. Try exporting it to CSV or Excel first.",
            }

        # Pick the biggest table (rows x cols) as the primary dataset
        primary = max(all_tables, key=lambda t: t["rows"] * len(t["columns"]))
        df = self._clean_frame(primary["frame"])

        return {
            "success": True,
            "source_format": "pdf",
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df),
            "col_count": len(df.columns),
            "table_used": {"page": primary["page"], "table_index": primary["table_index"]},
            "tables_found": len(all_tables),
            "all_tables": [
                {"page": t["page"], "table_index": t["table_index"], "rows": t["rows"], "columns": t["columns"]}
                for t in all_tables
            ],
        }

    def _pdf_text(self, file_bytes: bytes) -> str:
        try:
            import pdfplumber
            out = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages[:10]:
                    out.append(page.extract_text() or "")
            return "\n".join(out)
        except Exception:
            return ""

    def _table_to_frame(self, table: list) -> Optional[pd.DataFrame]:
        """Convert a raw pdfplumber table (list of lists) into a DataFrame."""
        rows = [[(c or "").strip() for c in row] for row in table if row]
        if len(rows) < 2:
            return None
        header = rows[0]
        # Deduplicate / fill blank headers
        seen, clean_header = {}, []
        for i, h in enumerate(header):
            h = h or f"column_{i+1}"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            clean_header.append(h)
        body = [r for r in rows[1:] if any(c for c in r)]
        if not body:
            return None
        width = len(clean_header)
        body = [(r + [""] * width)[:width] for r in body]
        return pd.DataFrame(body, columns=clean_header)

    # ── CSV / JSON / PARQUET ──────────────────────────────────────────────────

    def _parse_csv(self, file_bytes: bytes, delimiter: str = None) -> dict:
        text = file_bytes.decode("utf-8", errors="replace")
        if delimiter is None:
            delimiter = self._sniff_delimiter(text)
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
        df = self._clean_frame(df)
        return {
            "success": True, "source_format": "csv",
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df), "col_count": len(df.columns),
            "delimiter_detected": delimiter,
        }

    def _parse_json(self, file_bytes: bytes) -> dict:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        if isinstance(data, dict):
            # Find the first list-of-dicts inside the object
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    data = v
                    break
            else:
                data = [data]
        df = pd.DataFrame(data)
        df = self._clean_frame(df)
        return {
            "success": True, "source_format": "json",
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df), "col_count": len(df.columns),
        }

    def _parse_parquet(self, file_bytes: bytes) -> dict:
        df = pd.read_parquet(io.BytesIO(file_bytes))
        df = self._clean_frame(df)
        return {
            "success": True, "source_format": "parquet",
            "records": self._to_records(df),
            "columns": list(df.columns.astype(str)),
            "row_count": len(df), "col_count": len(df.columns),
        }

    # ── SHARED CLEANING ───────────────────────────────────────────────────────

    def _clean_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names, drop empty rows/cols, coerce numerics."""
        df = df.copy()
        # Drop fully-empty columns and rows
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        # Clean column names
        df.columns = [self._clean_col(str(c), i) for i, c in enumerate(df.columns)]
        # Coerce string columns that are actually numeric
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            try:
                coerced = pd.to_numeric(
                    df[col].astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("$", "", regex=False)
                        .str.replace("GHS", "", regex=False)
                        .str.replace("%", "", regex=False)
                        .str.replace("\u20b5", "", regex=False)
                        .str.strip(),
                    errors="coerce",
                )
            except Exception:
                continue
            # Only convert if most values parsed cleanly
            if coerced.notna().sum() >= max(1, int(0.8 * len(df))):
                df[col] = coerced
        return df.reset_index(drop=True)

    def _clean_col(self, name: str, idx: int) -> str:
        name = name.strip()
        if not name or name.lower().startswith("unnamed"):
            return f"column_{idx+1}"
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^\w]", "", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name.lower() or f"column_{idx+1}"

    def _to_records(self, df: pd.DataFrame) -> list:
        """Convert to JSON-safe records (NaN -> None, numpy types -> python)."""
        safe = df.where(pd.notna(df), None)
        return json.loads(safe.to_json(orient="records", date_format="iso"))

    def _sniff_delimiter(self, text: str) -> str:
        first = text.split("\n", 1)[0]
        counts = {d: first.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","

    def _ext(self, filename: str) -> str:
        return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


file_parser_service = FileParserService()
