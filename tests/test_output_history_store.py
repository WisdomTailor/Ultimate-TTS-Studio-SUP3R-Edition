from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from output_history_store import OutputHistoryRecord, OutputHistoryStore


class TestOutputHistoryStore:
    def test_initializes_schema_and_persists_record(self, tmp_path: Path) -> None:
        db_path = tmp_path / "outputs.db"
        store = OutputHistoryStore(db_path)

        record = store.upsert_record(
            OutputHistoryRecord(
                job_json_path="F:/TTS Output Files/app_state_outputs/default/jobs/run.job.json",
                project="default",
                preset="no_preset",
                timestamp="20260425_042128",
                datetime_iso="2026-04-25T04:21:28",
                engine="Fish Speech",
                seed=501928455,
                speaker="Confidence Narration",
                autosave_meta_path="F:/TTS Output Files/app_state_outputs/default/meta/run.json",
                autosave_audio_path="F:/TTS Output Files/app_state_outputs/default/audio/run.wav",
                autosave_scripts=[
                    "F:/TTS Output Files/app_state_outputs/default/scripts/run.txt",
                ],
                can_reload=True,
            )
        )

        assert record.id is not None
        assert db_path.exists()

    def test_upsert_uses_job_json_path_identity_not_nullable_seed(self, tmp_path: Path) -> None:
        store = OutputHistoryStore(tmp_path / "outputs.db")

        first = store.upsert_record(
            OutputHistoryRecord(
                job_json_path="F:/root/jobs/example.job.json",
                project="default",
                preset="no_preset",
                timestamp="20260425_042128",
                seed=None,
                engine="Fish Speech",
            )
        )
        second = store.upsert_record(
            OutputHistoryRecord(
                job_json_path="F:/root/jobs/example.job.json",
                project="default",
                preset="no_preset",
                timestamp="20260425_042128",
                seed=None,
                engine="Chatterbox",
            )
        )

        rows = store.list_records(limit=10)

        assert first.id == second.id
        assert len(rows) == 1
        assert rows[0].engine == "Chatterbox"

    def test_same_seed_can_exist_on_multiple_job_paths(self, tmp_path: Path) -> None:
        store = OutputHistoryStore(tmp_path / "outputs.db")

        store.upsert_record(
            OutputHistoryRecord(
                job_json_path="F:/root/jobs/one.job.json",
                project="alpha",
                preset="preset-a",
                timestamp="20260425_042128",
                seed=42,
            )
        )
        store.upsert_record(
            OutputHistoryRecord(
                job_json_path="F:/root/jobs/two.job.json",
                project="beta",
                preset="preset-b",
                timestamp="20260425_042129",
                seed=42,
            )
        )

        rows = store.list_records(seed=42, limit=10)

        assert len(rows) == 2
        assert {row.project for row in rows} == {"alpha", "beta"}

    def test_seed_is_indexed_but_not_unique(self, tmp_path: Path) -> None:
        db_path = tmp_path / "outputs.db"
        OutputHistoryStore(db_path)

        with sqlite3.connect(str(db_path)) as connection:
            index_names = {
                row[1]
                for row in connection.execute("PRAGMA index_list('output_history')").fetchall()
            }

        assert "idx_output_history_seed" in index_names
