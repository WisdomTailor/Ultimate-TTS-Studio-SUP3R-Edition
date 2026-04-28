from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS output_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_json_path TEXT NOT NULL UNIQUE,
    project TEXT NOT NULL,
    preset TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    datetime_iso TEXT,
    engine TEXT,
    seed INTEGER,
    speaker TEXT,
    chunks INTEGER,
    duration_seconds REAL,
    transform TEXT,
    llm_enabled INTEGER NOT NULL DEFAULT 0,
    manual_audio_path TEXT,
    autosave_audio_path TEXT,
    autosave_meta_path TEXT,
    autosave_scripts_json TEXT NOT NULL DEFAULT '[]',
    legacy_copy INTEGER NOT NULL DEFAULT 0,
    can_reload INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_output_history_project ON output_history(project);
CREATE INDEX IF NOT EXISTS idx_output_history_preset ON output_history(preset);
CREATE INDEX IF NOT EXISTS idx_output_history_timestamp ON output_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_output_history_engine ON output_history(engine);
CREATE INDEX IF NOT EXISTS idx_output_history_seed ON output_history(seed);
CREATE INDEX IF NOT EXISTS idx_output_history_speaker ON output_history(speaker);
"""


@dataclass(slots=True)
class OutputHistoryRecord:
    """Normalized persisted history record for one autosave bundle."""

    job_json_path: str
    project: str
    timestamp: str
    preset: str = ""
    datetime_iso: str | None = None
    engine: str | None = None
    seed: int | None = None
    speaker: str | None = None
    chunks: int | None = None
    duration_seconds: float | None = None
    transform: str | None = None
    llm_enabled: bool = False
    manual_audio_path: str | None = None
    autosave_audio_path: str | None = None
    autosave_meta_path: str | None = None
    autosave_scripts: list[str] = field(default_factory=list)
    legacy_copy: bool = False
    can_reload: bool = False
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "OutputHistoryRecord":
        """Build a record from a SQLite row."""
        return cls(
            id=row["id"],
            job_json_path=row["job_json_path"],
            project=row["project"],
            preset=row["preset"],
            timestamp=row["timestamp"],
            datetime_iso=row["datetime_iso"],
            engine=row["engine"],
            seed=row["seed"],
            speaker=row["speaker"],
            chunks=row["chunks"],
            duration_seconds=row["duration_seconds"],
            transform=row["transform"],
            llm_enabled=bool(row["llm_enabled"]),
            manual_audio_path=row["manual_audio_path"],
            autosave_audio_path=row["autosave_audio_path"],
            autosave_meta_path=row["autosave_meta_path"],
            autosave_scripts=json.loads(row["autosave_scripts_json"] or "[]"),
            legacy_copy=bool(row["legacy_copy"]),
            can_reload=bool(row["can_reload"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def to_db_params(self) -> dict[str, Any]:
        """Return parameter mapping for insert or update statements."""
        return {
            "job_json_path": self.job_json_path,
            "project": self.project,
            "preset": self.preset,
            "timestamp": self.timestamp,
            "datetime_iso": self.datetime_iso,
            "engine": self.engine,
            "seed": self.seed,
            "speaker": self.speaker,
            "chunks": self.chunks,
            "duration_seconds": self.duration_seconds,
            "transform": self.transform,
            "llm_enabled": 1 if self.llm_enabled else 0,
            "manual_audio_path": self.manual_audio_path,
            "autosave_audio_path": self.autosave_audio_path,
            "autosave_meta_path": self.autosave_meta_path,
            "autosave_scripts_json": json.dumps(self.autosave_scripts, ensure_ascii=False),
            "legacy_copy": 1 if self.legacy_copy else 0,
            "can_reload": 1 if self.can_reload else 0,
        }


class OutputHistoryStore:
    """SQLite-backed store for persisted TTS output history."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        """Create the schema and indexes if they do not exist."""
        with self._connect() as connection:
            connection.executescript(SCHEMA_SQL)
            columns = {
                str(row["name"]) for row in connection.execute("PRAGMA table_info(output_history)")
            }
            if "duration_seconds" not in columns:
                connection.execute("ALTER TABLE output_history ADD COLUMN duration_seconds REAL")

    def upsert_record(self, record: OutputHistoryRecord) -> OutputHistoryRecord:
        """Insert or update a record keyed by canonical job bundle path."""
        params = record.to_db_params()
        sql = """
        INSERT INTO output_history (
            job_json_path,
            project,
            preset,
            timestamp,
            datetime_iso,
            engine,
            seed,
            speaker,
            chunks,
            duration_seconds,
            transform,
            llm_enabled,
            manual_audio_path,
            autosave_audio_path,
            autosave_meta_path,
            autosave_scripts_json,
            legacy_copy,
            can_reload
        ) VALUES (
            :job_json_path,
            :project,
            :preset,
            :timestamp,
            :datetime_iso,
            :engine,
            :seed,
            :speaker,
            :chunks,
            :duration_seconds,
            :transform,
            :llm_enabled,
            :manual_audio_path,
            :autosave_audio_path,
            :autosave_meta_path,
            :autosave_scripts_json,
            :legacy_copy,
            :can_reload
        )
        ON CONFLICT(job_json_path) DO UPDATE SET
            project = excluded.project,
            preset = excluded.preset,
            timestamp = excluded.timestamp,
            datetime_iso = excluded.datetime_iso,
            engine = excluded.engine,
            seed = excluded.seed,
            speaker = excluded.speaker,
            chunks = excluded.chunks,
            duration_seconds = excluded.duration_seconds,
            transform = excluded.transform,
            llm_enabled = excluded.llm_enabled,
            manual_audio_path = excluded.manual_audio_path,
            autosave_audio_path = excluded.autosave_audio_path,
            autosave_meta_path = excluded.autosave_meta_path,
            autosave_scripts_json = excluded.autosave_scripts_json,
            legacy_copy = excluded.legacy_copy,
            can_reload = excluded.can_reload,
            updated_at = CURRENT_TIMESTAMP
        """
        with self._connect() as connection:
            connection.execute(sql, params)
            row = connection.execute(
                "SELECT * FROM output_history WHERE job_json_path = ?",
                (record.job_json_path,),
            ).fetchone()

        if row is None:
            raise RuntimeError("Failed to load upserted output history record")
        return OutputHistoryRecord.from_row(row)

    def get_record(self, record_id: int) -> OutputHistoryRecord | None:
        """Return a record by integer primary key."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM output_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        return OutputHistoryRecord.from_row(row) if row is not None else None

    def get_record_by_job_path(self, job_json_path: str) -> OutputHistoryRecord | None:
        """Return a record by canonical job bundle path."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM output_history WHERE job_json_path = ?",
                (job_json_path,),
            ).fetchone()
        return OutputHistoryRecord.from_row(row) if row is not None else None

    def list_records(
        self,
        *,
        project: str | None = None,
        preset: str | None = None,
        seed: int | None = None,
        speaker: str | None = None,
        query: str | None = None,
        from_timestamp: str | None = None,
        to_timestamp: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[OutputHistoryRecord]:
        """List records ordered newest-first with lightweight filters."""
        clauses = ["1=1"]
        params: list[Any] = []

        if project:
            clauses.append("project = ?")
            params.append(project)
        if preset:
            clauses.append("preset = ?")
            params.append(preset)
        if seed is not None:
            clauses.append("seed = ?")
            params.append(seed)
        if speaker:
            clauses.append("speaker = ?")
            params.append(speaker)
        if from_timestamp:
            clauses.append("timestamp >= ?")
            params.append(from_timestamp)
        if to_timestamp:
            clauses.append("timestamp <= ?")
            params.append(to_timestamp)
        if query:
            like_value = f"%{query}%"
            clauses.append(
                "("
                + " OR ".join(
                    [
                        "project LIKE ?",
                        "preset LIKE ?",
                        "engine LIKE ?",
                        "speaker LIKE ?",
                        "transform LIKE ?",
                        "timestamp LIKE ?",
                    ]
                )
                + ")"
            )
            params.extend([like_value] * 6)

        params.extend([limit, offset])
        sql = (
            "SELECT * FROM output_history "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?"
        )

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [OutputHistoryRecord.from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection


__all__ = ["OutputHistoryRecord", "OutputHistoryStore", "SCHEMA_SQL"]
