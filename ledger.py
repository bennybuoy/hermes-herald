"""Profile-aware SQLite audit ledger for Hermes Herald dispatch edges.

The JSON state file remains the bounded live-recovery cache. This module is the
append/update audit surface: it stores no transport or provider credentials and
can be pointed at one shared local path by several origin profiles.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config as cfg

_SCHEMA_VERSION = 1
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "timeout", "error"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    path = cfg.get_ledger_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10.0)
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            connection.close()
            raise
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dispatches (
            edge_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            origin_profile TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            dispatch_type TEXT NOT NULL,
            delivery TEXT NOT NULL,
            message TEXT NOT NULL,
            message_preview TEXT NOT NULL,
            instructions TEXT NOT NULL DEFAULT '',
            trace_id TEXT NOT NULL DEFAULT '',
            parent_edge_id TEXT NOT NULL DEFAULT '',
            hop_count INTEGER NOT NULL DEFAULT 1,
            max_hops INTEGER,
            origin_session_id TEXT NOT NULL DEFAULT '',
            requested_model TEXT NOT NULL DEFAULT '',
            resolved_model TEXT NOT NULL DEFAULT '',
            model_resolution TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            dispatched_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT '',
            output_preview TEXT NOT NULL DEFAULT '',
            duration_seconds REAL,
            usage_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_dispatches_run_id
            ON dispatches(run_id);
        CREATE INDEX IF NOT EXISTS idx_dispatches_trace_id
            ON dispatches(trace_id);
        CREATE INDEX IF NOT EXISTS idx_dispatches_origin_target
            ON dispatches(origin_profile, target_profile);
        CREATE INDEX IF NOT EXISTS idx_dispatches_dispatched_at
            ON dispatches(dispatched_at DESC);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(_SCHEMA_VERSION),),
    )
    # Forward-compatible additive migration for databases created by an
    # earlier pre-release build.
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(dispatches)")
    }
    if "hop_count" not in columns:
        connection.execute(
            "ALTER TABLE dispatches ADD COLUMN hop_count INTEGER NOT NULL DEFAULT 1"
        )
    if "max_hops" not in columns:
        connection.execute("ALTER TABLE dispatches ADD COLUMN max_hops INTEGER")
    connection.commit()
    return connection


def preflight() -> None:
    """Create/validate the ledger before starting a remote side effect."""
    with _connect() as connection:
        connection.execute("SELECT 1 FROM dispatches LIMIT 1").fetchone()


def record_dispatch(
    *,
    edge_id: str,
    run_id: str,
    origin_profile: str,
    target_profile: str,
    dispatch_type: str,
    delivery: str,
    message: str,
    instructions: str = "",
    trace_id: str = "",
    parent_edge_id: str = "",
    hop_count: int = 1,
    max_hops: Optional[int] = None,
    origin_session_id: str = "",
    requested_model: str = "",
    resolved_model: str = "",
    model_resolution: str = "",
    status: str = "dispatched",
    output_preview: str = "",
    duration_seconds: Optional[float] = None,
    usage: Optional[dict] = None,
    dispatched_at: str = "",
    completed_at: str = "",
) -> None:
    """Insert one credential-free directed dispatch edge."""
    now = dispatched_at or _now()
    completed_at = completed_at or (now if status in _TERMINAL_STATUSES else "")
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO dispatches (
                edge_id, run_id, origin_profile, target_profile, dispatch_type,
                delivery, message, message_preview, instructions, trace_id,
                parent_edge_id, hop_count, max_hops, origin_session_id, requested_model,
                resolved_model, model_resolution, status, dispatched_at,
                updated_at, completed_at, output_preview, duration_seconds,
                usage_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id, run_id, origin_profile, target_profile, dispatch_type,
                delivery, message, message[:120], instructions or "", trace_id or "",
                parent_edge_id or "", hop_count, max_hops, origin_session_id or "",
                requested_model or "", resolved_model or "",
                model_resolution or "", status, now, now, completed_at,
                (output_preview or "")[:500], duration_seconds,
                json.dumps(usage or {}, ensure_ascii=False),
            ),
        )
        connection.commit()


def known_run_ids(origin_profile: str) -> set[str]:
    """Return run IDs already represented for one origin."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT DISTINCT run_id FROM dispatches WHERE origin_profile = ?",
            (origin_profile,),
        ).fetchall()
    return {str(row["run_id"]) for row in rows}


def update_dispatch(
    *,
    run_id: str,
    origin_profile: str,
    status: str,
    output_preview: str = "",
    duration_seconds: Optional[float] = None,
    usage: Optional[dict] = None,
    requested_model: str = "",
    resolved_model: str = "",
) -> None:
    """Update all matching local-origin rows for one target run."""
    assignments = ["status = ?", "updated_at = ?"]
    values: list[Any] = [status, _now()]
    if status in _TERMINAL_STATUSES:
        assignments.append("completed_at = ?")
        values.append(_now())
    if output_preview:
        assignments.append("output_preview = ?")
        values.append(output_preview[:500])
    if duration_seconds is not None:
        assignments.append("duration_seconds = ?")
        values.append(round(float(duration_seconds), 2))
    if usage:
        assignments.append("usage_json = ?")
        values.append(json.dumps(usage, ensure_ascii=False))
    if requested_model:
        assignments.append("requested_model = ?")
        values.append(requested_model)
    if resolved_model:
        assignments.append("resolved_model = ?")
        values.append(resolved_model)
    values.extend([run_id, origin_profile])
    with _connect() as connection:
        connection.execute(
            f"UPDATE dispatches SET {', '.join(assignments)} "
            "WHERE run_id = ? AND origin_profile = ?",
            values,
        )
        connection.commit()


def _row_to_dict(row: sqlite3.Row, include_messages: bool) -> dict:
    item = {
        "edge_id": row["edge_id"],
        "run_id": row["run_id"],
        "origin_profile": row["origin_profile"],
        "target_profile": row["target_profile"],
        "dispatch_type": row["dispatch_type"],
        "delivery": row["delivery"],
    }
    if include_messages:
        item["message"] = row["message"]
    item["message_preview"] = row["message_preview"]
    if include_messages:
        item["instructions"] = row["instructions"]
    item.update({
        "trace_id": row["trace_id"],
        "parent_edge_id": row["parent_edge_id"],
        "hop_count": row["hop_count"],
        "max_hops": row["max_hops"],
        "origin_session_id": row["origin_session_id"],
        "requested_model": row["requested_model"],
        "resolved_model": row["resolved_model"],
        "model_resolution": row["model_resolution"],
        "status": row["status"],
        "dispatched_at": row["dispatched_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "output_preview": row["output_preview"],
        "duration_seconds": row["duration_seconds"],
        "usage": json.loads(row["usage_json"] or "{}"),
    })
    return item


def list_dispatches(
    *,
    limit: int = 100,
    include_messages: bool = False,
    target_profile: str = "",
    origin_profile: str = "",
    dispatch_type: str = "",
    status: str = "",
    delivery: str = "",
    trace_id: str = "",
) -> list[dict]:
    """List newest dispatch edges with optional exact-match filters."""
    limit = max(1, min(int(limit), 500))
    clauses: list[str] = []
    values: list[Any] = []
    for column, value in (
        ("origin_profile", origin_profile),
        ("target_profile", target_profile),
        ("dispatch_type", dispatch_type),
        ("status", status),
        ("delivery", delivery),
        ("trace_id", trace_id),
    ):
        if value:
            clauses.append(f"{column} = ?")
            values.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    values.append(limit)
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM dispatches" + where +
            " ORDER BY dispatched_at DESC, edge_id DESC LIMIT ?",
            values,
        ).fetchall()
    return [_row_to_dict(row, include_messages) for row in rows]


def observed_edges() -> list[dict]:
    """Aggregate durable call counts into a directed observed graph."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT origin_profile, target_profile, COUNT(*) AS calls
            FROM dispatches
            GROUP BY origin_profile, target_profile
            ORDER BY origin_profile, target_profile
            """
        ).fetchall()
    return [
        {
            "origin_profile": row["origin_profile"],
            "target_profile": row["target_profile"],
            "calls": row["calls"],
        }
        for row in rows
    ]
