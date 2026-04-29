from __future__ import annotations

import json
import os
import sys
import wave
from datetime import datetime
from pathlib import Path

import pytest


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from output_history_service import (
    HISTORY_PREVIEW_CURRENT_SCRIPT,
    HISTORY_PREVIEW_METADATA_JSON,
    add_run_base_collision_suffix,
    allocate_collision_safe_run_base,
    bundle_exists,
    build_record_from_meta,
    build_reload_payload,
    default_db_path_for_autosave_root,
    feature_storage_root_from_autosave_root,
    import_legacy_outputs,
    is_path_within_root,
    read_history_preview,
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
                "duration_seconds": 12.5,
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


def _write_valid_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)


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
        assert record.duration_seconds == 12.5

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

    def test_build_record_from_meta_sets_duration_none_when_unavailable(
        self, tmp_path: Path
    ) -> None:
        _autosave_root, meta_path = _write_fixture_bundle(tmp_path)
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload.pop("duration_seconds", None)
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        record = build_record_from_meta(meta_path)

        assert record.duration_seconds is None

    def test_allocate_collision_safe_run_base_uses_zero_padded_suffixes(
        self, tmp_path: Path
    ) -> None:
        project_root = tmp_path / "app_state_outputs" / "default"
        for folder_name in ("audio", "meta", "scripts", "jobs"):
            (project_root / folder_name).mkdir(parents=True, exist_ok=True)

        preferred = "default_story_20260429_101530"
        assert allocate_collision_safe_run_base(project_root, preferred) == preferred

        (project_root / "meta" / f"{preferred}.json").write_text("{}", encoding="utf-8")
        assert bundle_exists(project_root, preferred) is True
        assert allocate_collision_safe_run_base(project_root, preferred) == (
            "default_story_01_20260429_101530"
        )

        (project_root / "jobs" / "default_story_01_20260429_101530.job.json").write_text(
            "{}",
            encoding="utf-8",
        )
        assert allocate_collision_safe_run_base(project_root, preferred) == (
            "default_story_02_20260429_101530"
        )

    def test_add_run_base_collision_suffix_preserves_terminal_timestamp(self) -> None:
        assert add_run_base_collision_suffix("default_story_20260429_101530", 1) == (
            "default_story_01_20260429_101530"
        )
        assert add_run_base_collision_suffix("default_story_20260429_101530", 12) == (
            "default_story_12_20260429_101530"
        )

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

    def test_read_history_preview_returns_current_script_and_metadata(self, tmp_path: Path) -> None:
        autosave_root, meta_path = _write_fixture_bundle(tmp_path)
        record = build_record_from_meta(meta_path)

        script_preview = read_history_preview(record, autosave_root, HISTORY_PREVIEW_CURRENT_SCRIPT)
        metadata_preview = read_history_preview(
            record,
            autosave_root,
            HISTORY_PREVIEW_METADATA_JSON,
        )

        assert script_preview["title"] == "Current Script"
        assert script_preview["content"] == "Current text"
        assert script_preview["path"].endswith("default_no_preset_20260425_042128.txt")
        assert metadata_preview["title"] == "Metadata JSON"
        assert '"engine": "Fish Speech"' in metadata_preview["content"]

    def test_import_legacy_outputs_imports_complete_trio(self, tmp_path: Path) -> None:
        legacy_root = tmp_path / "outputs"
        autosave_root = tmp_path / "app_state_outputs"
        store = OutputHistoryStore(tmp_path / "outputs.db")
        audio_path = legacy_root / "legacy_clip_20260425_042128.wav"
        text_path = legacy_root / "legacy_clip_20260425_042128.txt"
        json_path = legacy_root / "legacy_clip_20260425_042128.json"
        _write_valid_wav(audio_path)
        text_path.write_text("Recovered script from legacy output.", encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "project": "old_project",
                    "preset": "archived_voice",
                    "engine": "Fish Speech",
                    "speaker": "Legacy Narrator",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        summary = import_legacy_outputs(legacy_root, autosave_root, store=store)

        assert summary == {
            "imported": 1,
            "skipped": 0,
            "synthesized_meta": 0,
            "synthesized_script": 0,
            "errors": 0,
        }
        records = store.list_records(limit=10)
        assert len(records) == 1
        record = records[0]
        assert record.project == "default"
        assert record.preset == "archived_voice"
        assert record.timestamp == "20260425_042128"
        assert record.speaker == "Legacy Narrator"
        assert record.autosave_audio_path is not None
        assert record.autosave_meta_path is not None
        assert Path(record.autosave_audio_path).exists()
        assert record.autosave_scripts == [
            (autosave_root / "default" / "scripts" / "default_archived_voice_20260425_042128.txt")
            .resolve()
            .as_posix()
        ]
        imported_meta = json.loads(Path(record.autosave_meta_path).read_text(encoding="utf-8"))
        assert imported_meta["legacy_import"]["source_audio"] == audio_path.resolve().as_posix()
        assert (
            Path(imported_meta["paths"]["audio"]).name
            == "default_archived_voice_20260425_042128.wav"
        )
        assert Path(imported_meta["paths"]["script"]).read_text(encoding="utf-8") == (
            "Recovered script from legacy output."
        )

    def test_import_legacy_outputs_synthesizes_missing_metadata_and_script(
        self, tmp_path: Path
    ) -> None:
        legacy_root = tmp_path / "outputs"
        autosave_root = tmp_path / "app_state_outputs"
        store = OutputHistoryStore(tmp_path / "outputs.db")
        audio_path = legacy_root / "orphan_clip.wav"
        _write_valid_wav(audio_path)
        mtime = datetime(2026, 4, 26, 5, 6, 7).timestamp()
        os.utime(audio_path, (mtime, mtime))

        summary = import_legacy_outputs(legacy_root, autosave_root, store=store)

        assert summary == {
            "imported": 1,
            "skipped": 0,
            "synthesized_meta": 1,
            "synthesized_script": 1,
            "errors": 0,
        }
        records = store.list_records(limit=10)
        assert len(records) == 1
        record = records[0]
        assert record.timestamp == datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
        assert record.autosave_meta_path is not None
        imported_meta = json.loads(Path(record.autosave_meta_path).read_text(encoding="utf-8"))
        assert imported_meta["preset"] == "legacy_import"
        assert imported_meta["project"] == "default"
        assert imported_meta["engine"] == "Legacy Import"
        assert Path(imported_meta["paths"]["script"]).read_text(encoding="utf-8") == (
            "Recovered legacy output where source text was unavailable."
        )

    def test_import_legacy_outputs_is_idempotent_for_repeated_runs(self, tmp_path: Path) -> None:
        legacy_root = tmp_path / "outputs"
        autosave_root = tmp_path / "app_state_outputs"
        store = OutputHistoryStore(tmp_path / "outputs.db")
        audio_path = legacy_root / "repeat_clip_20260425_042128.wav"
        _write_valid_wav(audio_path)

        first_summary = import_legacy_outputs(legacy_root, autosave_root, store=store)
        records_after_first = store.list_records(limit=10)
        assert len(records_after_first) == 1
        first_record = records_after_first[0]
        assert first_record.autosave_meta_path is not None
        first_meta_contents = Path(first_record.autosave_meta_path).read_text(encoding="utf-8")

        second_summary = import_legacy_outputs(legacy_root, autosave_root, store=store)
        records_after_second = store.list_records(limit=10)

        assert first_summary["imported"] == 1
        assert second_summary == {
            "imported": 0,
            "skipped": 1,
            "synthesized_meta": 0,
            "synthesized_script": 0,
            "errors": 0,
        }
        assert len(records_after_second) == 1
        assert records_after_second[0].job_json_path == first_record.job_json_path
        assert (
            Path(first_record.autosave_meta_path).read_text(encoding="utf-8") == first_meta_contents
        )

    def test_import_legacy_outputs_avoids_same_second_bundle_overwrite(self, tmp_path: Path) -> None:
        legacy_root = tmp_path / "outputs"
        autosave_root = tmp_path / "app_state_outputs"
        store = OutputHistoryStore(tmp_path / "outputs.db")

        first_audio = legacy_root / "clip_a_20260425_042128.wav"
        second_audio = legacy_root / "clip_b_20260425_042128.wav"
        _write_valid_wav(first_audio)
        _write_valid_wav(second_audio)

        (legacy_root / "clip_a_20260425_042128.json").write_text(
            json.dumps({"preset": "archived_voice", "speaker": "Narrator A"}, indent=2),
            encoding="utf-8",
        )
        (legacy_root / "clip_b_20260425_042128.json").write_text(
            json.dumps({"preset": "archived_voice", "speaker": "Narrator B"}, indent=2),
            encoding="utf-8",
        )

        summary = import_legacy_outputs(legacy_root, autosave_root, store=store)

        assert summary == {
            "imported": 2,
            "skipped": 0,
            "synthesized_meta": 0,
            "synthesized_script": 2,
            "errors": 0,
        }
        records = store.list_records(limit=10)
        assert len(records) == 2
        run_bases = sorted(Path(record.job_json_path).stem.replace(".job", "") for record in records)
        assert run_bases == [
            "default_archived_voice_01_20260425_042128",
            "default_archived_voice_20260425_042128",
        ]
        speakers = sorted(record.speaker for record in records if record.speaker)
        assert speakers == ["Narrator A", "Narrator B"]
