"""SQLite-based file modification time tracker for incremental re-indexing."""
import logging
import sqlite3
import time
from pathlib import Path

from config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

DB_PATH = CHROMA_PERSIST_DIR.parent / "file_tracker.db"

_conn = None


def _get_conn() -> sqlite3.Connection:
    """Lazy-initialize the SQLite connection."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH))
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_mtime (
                file_path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                indexed_at REAL NOT NULL
            )
        """
        )
        _conn.commit()
    return _conn


def needs_reindex(file_path: str | Path) -> bool:
    """Check if a file needs re-indexing based on mtime."""
    file_path = Path(file_path)
    try:
        current_mtime = file_path.stat().st_mtime
    except OSError:
        return False  # file doesn't exist

    conn = _get_conn()
    row = conn.execute(
        "SELECT mtime FROM file_mtime WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()

    if row is None:
        return True  # never indexed
    return current_mtime > row[0]  # modified since last index


def mark_indexed(file_path: str | Path) -> None:
    """Record that a file has been indexed at its current mtime."""
    file_path = Path(file_path)
    try:
        current_mtime = file_path.stat().st_mtime
    except OSError:
        return

    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO file_mtime (file_path, mtime, indexed_at) VALUES (?, ?, ?)",
        (str(file_path), current_mtime, time.time()),
    )
    conn.commit()


def remove_file(file_path: str | Path) -> None:
    """Remove a file from the tracker (when it's been deleted)."""
    conn = _get_conn()
    conn.execute("DELETE FROM file_mtime WHERE file_path = ?", (str(file_path),))
    conn.commit()


def get_all_tracked() -> dict[str, float]:
    """Return all tracked files and their mtimes."""
    conn = _get_conn()
    rows = conn.execute("SELECT file_path, mtime FROM file_mtime").fetchall()
    return {row[0]: row[1] for row in rows}


def get_stats() -> dict:
    """Return tracker statistics."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM file_mtime").fetchone()[0]
    latest = conn.execute("SELECT MAX(indexed_at) FROM file_mtime").fetchone()[0]
    return {
        "tracked_files": total,
        "last_indexed_at": latest,
    }
