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
                "reload_snapshot": {
                    "schema_version": 1,
                    "active_engine": "Fish Speech",
                    "control_values": {
                        "tts_engine": "Fish Speech",
                        "audio_format": "wav",
                        "fish_ref_text": "Reference transcript",
                        "fish_temperature": 0.65,
                        "fish_top_p": 0.75,
                        "llm_transform_enabled": True,
                        "llm_provider": "LM Studio OpenAI Server",
                        "llm_base_url": "http://localhost:1234/v1",
                        "llm_model_id": "qwen/test-model",
                        "llm_mode": "Polish",
                        "llm_locale": "en-US",
                        "llm_style": "cinematic_audiobook",
                        "llm_max_tag_density": 0.25,
                        "llm_system_prompt": "Keep punctuation natural.",
                        "llm_timeout_seconds": 45,
                        "llm_temperature": 0.3,
                        "llm_top_p": 0.8,
                        "llm_max_tokens": 1400,
                        "llm_allow_local_fallback": True,
                        "gain_db": 1.5,
                        "enable_eq": True,
                        "eq_bass": 2.0,
                        "speaker_name": "Confidence Narration",
                        "voice_preset": "no_preset",
                        "autosave_enabled": True,
                        "autosave_project_name": "default",
                        "autosave_store_audio_copy": True,
                        "keep_legacy_output_copy": False,
                        "last_seed_state": 501928455,
                    },
                    "excluded_controls": [
                        {"name": "llm_api_key", "reason": "secret"},
                        {"name": "fish_ref_audio", "reason": "transient_file_input"},
                    ],
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
        assert record.speaker == "Confidence Narration"

    def test_build_record_from_meta_falls_back_to_voice_metadata_when_speaker_missing(
        self, tmp_path: Path
    ) -> None:
        _autosave_root, meta_path = _write_fixture_bundle(tmp_path)
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload.pop("speaker", None)
        payload["engine"] = "Kokoro TTS"
        payload["voice"] = "af_heart"
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        record = build_record_from_meta(meta_path)

        assert record.speaker == "af_heart"

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
        assert payload["voice_narrator"] == "Confidence Narration"
        assert payload["audio_format"] == "wav"
        assert payload["script_text"] == "Original text"
        assert payload["original_text"] == "Original text"
        assert payload["transformed_text"] == "Transformed text"
        assert payload["reload_snapshot"]["legacy_fallback"] is False
        assert payload["reload_snapshot"]["control_values"]["fish_temperature"] == 0.65
        assert payload["reload_snapshot"]["control_values"]["llm_model_id"] == "qwen/test-model"
        assert payload["reload_snapshot"]["control_values"]["gain_db"] == 1.5
        assert payload["reload_snapshot"]["control_values"]["last_seed_state"] == 501928455
        assert payload["reload_snapshot"]["excluded_controls"] == [
            {"name": "llm_api_key", "reason": "secret"},
            {"name": "fish_ref_audio", "reason": "transient_file_input"},
        ]

    def test_build_reload_payload_falls_back_for_legacy_records(self, tmp_path: Path) -> None:
        _autosave_root, meta_path = _write_fixture_bundle(tmp_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata.pop("reload_snapshot", None)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        record = build_record_from_meta(meta_path)

        payload = build_reload_payload(record)

        assert payload["legacy_reload"] is True
        assert payload["reload_snapshot"]["legacy_fallback"] is True
        assert payload["reload_snapshot"]["control_values"]["tts_engine"] == "Fish Speech"
        assert payload["reload_snapshot"]["control_values"]["audio_format"] == "wav"
        assert payload["reload_snapshot"]["control_values"]["speaker_name"] == (
            "Confidence Narration"
        )
        assert payload["reload_snapshot"]["control_values"]["last_seed_state"] == 501928455
