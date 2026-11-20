"""SQLite key-value cache for review analysis and aggregate insights.

Keyed by review text hash, prompt version, and model name to ensure that
re-uploading identical reviews incurs zero external LLM API calls.
"""

from typing import Optional
from pathlib import Path
import sqlite3


class Cache:
    """Lightweight SQLite-backed key-value cache."""

    def __init__(self, db_path: str = "cache.db") -> None:
        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        if not self._is_memory:
            parent = Path(db_path).parent
            if parent and str(parent) != ".":
                parent.mkdir(parents=True, exist_ok=True)
            self._persistent_conn = None
        else:
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory:
            return self._persistent_conn
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        cache_key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
        finally:
            if not self._is_memory:
                conn.close()

    def get(self, key: str) -> Optional[str]:
        """Retrieve stored JSON string by key, or None if not cached."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM cache WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            if not self._is_memory:
                conn.close()

    def set(self, key: str, value: str) -> None:
        """Store key-value pair, overwriting on conflict."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO cache (cache_key, value)
                    VALUES (?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        value = excluded.value,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    (key, value),
                )
        finally:
            if not self._is_memory:
                conn.close()

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None

    def clear(self) -> None:
        """Clear all entries in cache."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM cache")
        finally:
            if not self._is_memory:
                conn.close()

    def close(self) -> None:
        """Close connection if persistent."""
        if self._persistent_conn:
            self._persistent_conn.close()
            self._persistent_conn = None
