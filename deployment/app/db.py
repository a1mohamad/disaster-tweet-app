import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    tweet TEXT NOT NULL,
    keyword TEXT,
    final_text TEXT NOT NULL,
    probability REAL NOT NULL,
    label INTEGER NOT NULL,
    label_name TEXT NOT NULL,
    threshold REAL NOT NULL,
    warnings_json TEXT
);
"""


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_PREDICTIONS_TABLE)
    conn.commit()


def log_prediction(
    conn: sqlite3.Connection,
    *,
    tweet: str,
    keyword: Optional[str],
    final_text: str,
    probability: float,
    label: int,
    label_name: str,
    threshold: float,
    warnings: Optional[List[Dict[str, Any]]],
) -> None:
    warnings_json = json.dumps(warnings or [], ensure_ascii=True)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conn.execute(
        """
        INSERT INTO prediction_logs (
            created_at,
            tweet,
            keyword,
            final_text,
            probability,
            label,
            label_name,
            threshold,
            warnings_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            tweet,
            keyword,
            final_text,
            probability,
            label,
            label_name,
            threshold,
            warnings_json,
        ),
    )
    conn.commit()


def fetch_recent_predictions(
    conn: sqlite3.Connection, limit: int = 50
) -> List[Dict[str, Any]]:
    cursor = conn.execute(
        """
        SELECT
            id,
            created_at,
            tweet,
            keyword,
            final_text,
            probability,
            label,
            label_name,
            threshold,
            warnings_json
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        warnings = json.loads(row["warnings_json"]) if row["warnings_json"] else []
        results.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "tweet": row["tweet"],
                "keyword": row["keyword"],
                "final_text": row["final_text"],
                "probability": row["probability"],
                "label": row["label"],
                "label_name": row["label_name"],
                "threshold": row["threshold"],
                "warnings": warnings,
            }
        )
    return results
