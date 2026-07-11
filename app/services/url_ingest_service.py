"""
DataMind Agent — URL Ingestion Service

Fetches a dataset from a link and hands the bytes to the existing file parser.
No OAuth. Works with anything served over plain HTTP(S):

  • Google Sheets  (share link -> CSV export)
  • Dropbox        (share link -> direct download)
  • OneDrive       (share link -> direct download)
  • GitHub         (blob page   -> raw file)
  • Google Drive   (file link   -> direct download, public files only)
  • S3 / any URL   (used as-is)

The URL normalisers are the whole point: a "view" link is not a "download"
link, and fetching one gives you an HTML page instead of your data.
"""
from __future__ import annotations
import io, re, logging
from urllib.parse import urlparse, parse_qs, urlencode, quote

logger = logging.getLogger(__name__)

MAX_MB = 100
TIMEOUT = 30
UA = "DataMind-Agent/1.0 (data analysis tool)"

# Content types that mean "you got an HTML page, not a file"
HTML_TYPES = ("text/html", "application/xhtml")


class URLIngestService:

    # ── URL NORMALISERS ───────────────────────────────────────────────────────

    def normalise(self, url: str) -> dict:
        """
        Turn a human-facing share link into a direct download link.
        Returns {url, source, sheet_hint, note}
        """
        url = (url or "").strip()
        if not url:
            raise ValueError("No URL provided")
        if not re.match(r"^https?://", url, re.I):
            url = "https://" + url

        p = urlparse(url)
        host = (p.netloc or "").lower()

        # ── Google Sheets ──
        # https://docs.google.com/spreadsheets/d/<ID>/edit#gid=<GID>
        m = re.search(r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
        if m:
            doc_id = m.group(1)
            gid = None
            # gid can live in the fragment (#gid=0) or the query (?gid=0)
            frag_gid = re.search(r"[#&?]gid=(\d+)", url)
            if frag_gid:
                gid = frag_gid.group(1)
            export = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
            if gid:
                export += f"&gid={gid}"
            return {
                "url": export, "source": "google_sheets", "sheet_hint": gid,
                "note": "Sheet must be shared as 'Anyone with the link can view'.",
                "filename": f"google_sheet_{doc_id[:8]}.csv",
            }

        # ── Google Drive file ──
        # https://drive.google.com/file/d/<ID>/view
        m = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
        if m:
            fid = m.group(1)
            return {
                "url": f"https://drive.google.com/uc?export=download&id={fid}",
                "source": "google_drive", "sheet_hint": None,
                "note": "File must be shared as 'Anyone with the link can view'.",
                "filename": f"drive_file_{fid[:8]}",
            }
        # https://drive.google.com/open?id=<ID>
        if "drive.google.com" in host:
            qid = parse_qs(p.query).get("id", [None])[0]
            if qid:
                return {
                    "url": f"https://drive.google.com/uc?export=download&id={qid}",
                    "source": "google_drive", "sheet_hint": None,
                    "note": "File must be shared as 'Anyone with the link can view'.",
                    "filename": f"drive_file_{qid[:8]}",
                }

        # ── Dropbox ──
        # ?dl=0 renders a preview page; ?dl=1 downloads the file
        if "dropbox.com" in host:
            q = parse_qs(p.query)
            q["dl"] = ["1"]
            q.pop("raw", None)
            direct = p._replace(query=urlencode(q, doseq=True)).geturl()
            return {"url": direct, "source": "dropbox", "sheet_hint": None,
                    "note": None, "filename": self._name_from_path(p.path) or "dropbox_file"}

        # ── OneDrive / SharePoint ──
        if "1drv.ms" in host or "onedrive.live.com" in host or "sharepoint.com" in host:
            sep = "&" if p.query else "?"
            return {"url": url + sep + "download=1", "source": "onedrive", "sheet_hint": None,
                    "note": "Link must allow anonymous view access.",
                    "filename": self._name_from_path(p.path) or "onedrive_file"}

        # ── GitHub blob page -> raw ──
        if "github.com" in host and "/blob/" in p.path:
            raw = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/", 1)
            return {"url": raw, "source": "github", "sheet_hint": None, "note": None,
                    "filename": self._name_from_path(p.path) or "github_file"}

        # ── Anything else: use as-is (S3, raw links, public buckets) ──
        return {"url": url, "source": "direct", "sheet_hint": None, "note": None,
                "filename": self._name_from_path(p.path) or "remote_file"}

    def _name_from_path(self, path: str) -> str:
        name = (path or "").rstrip("/").split("/")[-1]
        return name if "." in name else ""

    # ── FETCH ─────────────────────────────────────────────────────────────────

    def fetch(self, url: str) -> dict:
        """Resolve the URL, download it, and return raw bytes plus metadata."""
        try:
            import requests
        except ImportError:
            return {"success": False, "error": "The 'requests' library is not installed on the server.",
                    "hint": "Add requests>=2.31.0 to requirements.txt and redeploy."}

        try:
            info = self.normalise(url)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        try:
            r = requests.get(info["url"], timeout=TIMEOUT, allow_redirects=True,
                             headers={"User-Agent": UA}, stream=True)
        except requests.exceptions.Timeout:
            return {"success": False, "error": f"The server took longer than {TIMEOUT}s to respond.",
                    "source": info["source"]}
        except requests.exceptions.SSLError:
            return {"success": False, "error": "The site's SSL certificate could not be verified.",
                    "source": info["source"]}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Could not reach that URL: {e}",
                    "source": info["source"]}

        if r.status_code == 404:
            return {"success": False, "error": "That link returned 404 — the file does not exist.",
                    "source": info["source"], "hint": info.get("note")}
        if r.status_code in (401, 403):
            return {"success": False,
                    "error": "Access denied. The file is private.",
                    "source": info["source"],
                    "hint": info.get("note") or "Share the file so anyone with the link can view it."}
        if r.status_code >= 400:
            return {"success": False, "error": f"The server returned HTTP {r.status_code}.",
                    "source": info["source"]}

        # Read with a hard size cap so a huge file cannot exhaust memory
        limit = MAX_MB * 1024 * 1024
        buf, total = io.BytesIO(), 0
        for chunk in r.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > limit:
                return {"success": False, "error": f"File exceeds the {MAX_MB}MB limit.",
                        "source": info["source"]}
            buf.write(chunk)
        content = buf.getvalue()

        if not content:
            return {"success": False, "error": "That URL returned an empty file.",
                    "source": info["source"]}

        ctype = (r.headers.get("Content-Type") or "").lower()

        # An HTML page means we got a login wall or a preview page, not the data
        if any(h in ctype for h in HTML_TYPES) and not self._looks_like_data(content):
            return {
                "success": False,
                "error": "That link returned a web page, not a data file.",
                "source": info["source"],
                "hint": (info.get("note")
                         or "Check the file is shared publicly, and that the link points at the file itself."),
                "content_type": ctype,
            }

        filename = self._filename(r, info, ctype)
        return {
            "success": True,
            "content": content,
            "filename": filename,
            "size_bytes": len(content),
            "source": info["source"],
            "resolved_url": info["url"],
            "content_type": ctype,
            "note": info.get("note"),
        }

    def _looks_like_data(self, content: bytes) -> bool:
        """A CSV served with the wrong Content-Type should still be accepted."""
        head = content[:2048].lstrip()
        if head[:1] in (b"{", b"["):          # JSON
            return True
        if head[:4] == b"PK\x03\x04":          # xlsx / zip
            return True
        if head[:5] == b"%PDF-":               # pdf
            return True
        if head[:4] == b"PAR1":                # parquet
            return True
        low = head[:200].lower()
        if low.startswith(b"<!doctype") or low.startswith(b"<html"):
            return False
        # Comma/tab separated first line with no angle brackets
        first = head.split(b"\n", 1)[0]
        return (b"," in first or b"\t" in first) and b"<" not in first

    def _filename(self, r, info, ctype: str) -> str:
        """Work out a filename with an extension the parser will recognise."""
        # 1. Content-Disposition header
        cd = r.headers.get("Content-Disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, re.I)
        if m:
            name = m.group(1).strip()
            if "." in name:
                return name

        # 2. The normaliser's guess
        name = info.get("filename") or "remote_file"
        if "." in name:
            return name

        # 3. Infer from Content-Type
        ext = (".csv"     if "csv" in ctype else
               ".json"    if "json" in ctype else
               ".xlsx"    if "spreadsheetml" in ctype or "excel" in ctype else
               ".xls"     if "ms-excel" in ctype else
               ".pdf"     if "pdf" in ctype else
               ".tsv"     if "tab-separated" in ctype else
               ".csv")
        return name + ext

    # ── FULL PIPELINE ─────────────────────────────────────────────────────────

    def ingest(self, url: str, sheet_name: str = None) -> dict:
        """Fetch a URL and parse it into records via the existing file parser."""
        got = self.fetch(url)
        if not got.get("success"):
            return got

        try:
            from app.services.file_parser_service import file_parser_service
        except Exception as e:
            return {"success": False, "error": f"File parser unavailable: {e}"}

        parsed = file_parser_service.parse(got["content"], got["filename"], sheet_name)
        if not parsed.get("success"):
            parsed.setdefault("hint",
                "The file downloaded but could not be parsed. Check it is a table, not a document.")
            parsed["source"] = got["source"]
            parsed["filename"] = got["filename"]
            return parsed

        parsed["source"] = got["source"]
        parsed["filename"] = got["filename"]
        parsed["size_bytes"] = got["size_bytes"]
        parsed["resolved_url"] = got["resolved_url"]
        if got.get("note"):
            parsed["note"] = got["note"]
        return parsed

    def inspect(self, url: str) -> dict:
        """Peek at a remote file: Excel sheets, PDF tables, preview rows."""
        got = self.fetch(url)
        if not got.get("success"):
            return got
        try:
            from app.services.file_parser_service import file_parser_service
        except Exception as e:
            return {"success": False, "error": f"File parser unavailable: {e}"}

        result = file_parser_service.inspect(got["content"], got["filename"])
        result["source"] = got["source"]
        result["filename"] = got["filename"]
        result["size_bytes"] = got["size_bytes"]
        return result


url_ingest_service = URLIngestService()
