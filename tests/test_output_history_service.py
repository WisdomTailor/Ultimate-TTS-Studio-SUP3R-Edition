from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from output_history_service import (
    build_record_from_meta,
    build_reload_payload,
    default_db_path_for_autosave_root,
    feature_storage_root_from_autosave_root,
    is_path_within_root,
    reindex_root,
    resolve_playback_path,
)
from output_history_store import OutputHistoryStore


def _write_fixture_bundle(tmp_path: Path) -> tuple[Path, Path]:
    autosave_root = tmp_path / "app_state_outputs"
    project_root = autosave_root / "default"
    audio_dir = project_root / "audio"
    meta_dir = project_root / "meta"
    scripts_dir = project_root / "scripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    run_base = "default_no_preset_20260425_042128"
    audio_path = audio_dir / f"{run_base}.wav"
    script_path = scripts_dir / f"{run_base}.txt"
    original_path = scripts_dir / f"{run_base}.original.txt"
    transformed_path = scripts_dir / f"{run_base}.transformed.txt"
    meta_path = meta_dir / f"{run_base}.json"

    audio_path.write_bytes(b"RIFFfixture")
    script_path.write_text("Current text", encoding="utf-8")
    original_path.write_text("Original text", encoding="utf-8")
    transformed_path.write_text("Transformed text", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "project": "default",
                "preset": "no_preset",
                "engine": "Fish Speech",
                "seed": 501928455,
                "speaker": "Confidence Narration",
                "chunks": 105,
                "audio_format": "wav",
                "llm_transform": {
                    "status": "deterministic normalisation",
                    "applied": False,
                },
                "paths": {
                    "audio": str(audio_path),
                    "script": str(script_path),
                    "script_original": str(original_path),
                    "script_transformed": str(transformed_path),
                    "meta": str(meta_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return autosave_root, meta_path


class TestOutputHistoryService:
    def test_feature_storage_root_and_default_db_path(self, tmp_path: Path) -> None:
        autosave_root = tmp_path / "app_state_outputs"

        assert feature_storage_root_from_autosave_root(autosave_root) == tmp_path.resolve()
        assert default_db_path_for_autosave_root(autosave_root) == tmp_path.resolve() / "outputs.db"

    def test_build_record_from_meta_creates_job_json_and_normalizes_paths(
        self, tmp_path: Path
    ) -> None:
        autosave_root, meta_path = _write_fixture_bundle(tmp_path)

        record = build_record_from_meta(meta_path)

        assert record.project == "default"
        assert record.preset == "no_preset"
        assert record.timestamp == "20260425_042128"
        assert record.job_json_path.endswith("default_no_preset_20260425_042128.job.json")
        assert "/jobs/" in record.job_json_path
        assert record.autosave_audio_path is not None
        assert record.autosave_meta_path == meta_path.resolve().as_posix()
        assert all("/" in path for path in record.autosave_scripts)
        assert is_path_within_root(record.job_json_path, autosave_root)

    def test_reindex_root_upserts_fixture_bundle(self, tmp_path: Path) -> None:
        autosave_root, _meta_path = _write_fixture_bundle(tmp_path)
        store = OutputHistoryStore(tmp_path / "outputs.db")

        records = reindex_root(autosave_root, store=store)

        assert len(records) == 1
        persisted = store.get_record_by_job_path(records[0].job_json_path)
        assert persisted is not None
        assert persisted.engine == "Fish Speech"

    def test_resolve_playback_path_rejects_outside_root(self, tmp_path: Path) -> None:
        autosave_root, meta_path = _write_fixture_bundle(tmp_path)
        record = build_record_from_meta(meta_path)
        record.autosave_audio_path = None
        record.manual_audio_path = (tmp_path / "outside.wav").resolve().as_posix()

        with pytest.raises(ValueError, match="configured autosave root"):
            resolve_playback_path(record, autosave_root)

    def test_build_reload_payload_uses_job_json_texts(self, tmp_path: Path) -> None:
        _autosave_root, meta_path = _write_fixture_bundle(tmp_path)
        record = build_record_from_meta(meta_path)

        payload = build_reload_payload(record)

        assert payload["project"] == "default"
        assert payload["preset"] == "no_preset"
        assert payload["engine"] == "Fish Speech"
        assert payload["script_text"] == "Transformed text"
        assert payload["original_text"] == "Original text"
