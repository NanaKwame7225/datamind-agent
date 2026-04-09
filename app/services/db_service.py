import logging, uuid
from typing import Optional
logger = logging.getLogger(__name__)
_connections: dict = {}

class DatabaseService:
    def connect_sqlite(self, db_path: str) -> dict:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        conn.close()
        cid = str(uuid.uuid4())[:8]
        _connections[cid] = {"type": "sqlite", "path": db_path}
        return {"connection_id": cid, "tables": tables, "db_type": "sqlite"}

    async def get_schema(self, connection_id: str) -> dict:
        return {}

    async def execute_query(self, connection_id: str, sql: str) -> list:
        conn = _connections.get(connection_id)
        if not conn:
            raise ValueError(f"Connection '{connection_id}' not found")
        if conn["type"] == "sqlite":
            import sqlite3
            c = sqlite3.connect(conn["path"])
            c.row_factory = sqlite3.Row
            rows = [dict(r) for r in c.cursor().execute(sql).fetchall()]
            c.close()
            return rows
        return []

db_service = DatabaseService()
