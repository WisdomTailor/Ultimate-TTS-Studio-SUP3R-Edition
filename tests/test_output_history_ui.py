from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from output_history_service import (
    HISTORY_PREVIEW_CURRENT_SCRIPT,
    HISTORY_PREVIEW_METADATA_JSON,
    build_record_from_meta,
)
from output_history_store import OutputHistoryStore
from output_history_ui import (
    HISTORY_AUDIO_EMPTY_HTML,
    HISTORY_EMPTY_DETAIL_MESSAGE,
    HISTORY_NO_SELECTION_MESSAGE,
    HISTORY_PREVIEW_EMPTY_MESSAGE,
    build_history_panel_refresh_response,
    build_history_preview_response,
    build_history_table_select_response,
)


def _write_history_bundle(tmp_path: Path) -> tuple[Path, Path]:
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
                "duration_seconds": 12.5,
                "audio_format": "wav",
                "reload_snapshot": {
                    "schema_version": 1,
                    "active_engine": "Fish Speech",
                    "control_values": {
                        "tts_engine": "Fish Speech",
                        "audio_format": "wav",
                        "speaker_name": "Confidence Narration",
                        "autosave_project_name": "default",
                        "last_seed_state": 501928455,
                    },
                    "excluded_controls": [],
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


def _proxy_url(record_id: int | None) -> str | None:
    if record_id is None:
        return None
    return f"/api/history/audio/{record_id}"


class TestOutputHistoryUi:
    def test_refresh_response_applies_filters_and_preserves_no_selection_message(
        self, tmp_path: Path
    ) -> None:
        autosave_root, meta_path = _write_history_bundle(tmp_path)
        store = OutputHistoryStore(tmp_path / "outputs.db")
        record = store.upsert_record(build_record_from_meta(meta_path))

        rows, detail, audio_value, preview = build_history_panel_refresh_response(
            store=store,
            autosave_root=str(autosave_root),
            audio_proxy_url_builder=_proxy_url,
            query="Fish",
            project="default",
            preset="",
            seed="",
            speaker="",
            from_timestamp="",
            to_timestamp="",
            record_id="",
        )

        assert rows[0][0] == record.id
        assert detail == HISTORY_NO_SELECTION_MESSAGE
        assert audio_value == HISTORY_AUDIO_EMPTY_HTML
        assert preview == HISTORY_PREVIEW_EMPTY_MESSAGE

        empty_rows, empty_detail, empty_audio, empty_preview = build_history_panel_refresh_response(
            store=store,
            autosave_root=str(autosave_root),
            audio_proxy_url_builder=_proxy_url,
            query="missing",
            project="",
            preset="",
            seed="",
            speaker="",
            from_timestamp="",
            to_timestamp="",
            record_id="",
        )

        assert empty_rows[0][1] == "No history yet"
        assert empty_detail == HISTORY_EMPTY_DETAIL_MESSAGE
        assert empty_audio == HISTORY_AUDIO_EMPTY_HTML
        assert empty_preview == HISTORY_PREVIEW_EMPTY_MESSAGE

    def test_table_select_response_updates_record_id_detail_and_audio(self, tmp_path: Path) -> None:
        autosave_root, meta_path = _write_history_bundle(tmp_path)
        store = OutputHistoryStore(tmp_path / "outputs.db")
        record = store.upsert_record(build_record_from_meta(meta_path))
        rows = [
            [
                record.id,
                "default",
                "—",
                record.timestamp,
                "Fish Speech",
                "Narrator",
                "12.5s",
                "—",
                "Yes",
            ]
        ]

        selected_id, detail, audio_value, preview = build_history_table_select_response(
            table_rows=rows,
            selected_row_index=0,
            store=store,
            autosave_root=str(autosave_root),
            audio_proxy_url_builder=_proxy_url,
        )

        assert selected_id == str(record.id)
        assert f"### History Record {record.id}" in detail
        assert "<audio controls" in audio_value
        assert f"/api/history/audio/{record.id}" in audio_value
        assert f"History record {record.id} selected." in preview

    def test_preview_response_returns_script_and_metadata_markdown(self, tmp_path: Path) -> None:
        autosave_root, meta_path = _write_history_bundle(tmp_path)
        store = OutputHistoryStore(tmp_path / "outputs.db")
        record = store.upsert_record(build_record_from_meta(meta_path))

        script_preview = build_history_preview_response(
            record.id,
            HISTORY_PREVIEW_CURRENT_SCRIPT,
            store=store,
            autosave_root=str(autosave_root),
        )
        metadata_preview = build_history_preview_response(
            record.id,
            HISTORY_PREVIEW_METADATA_JSON,
            store=store,
            autosave_root=str(autosave_root),
        )

        assert "### Current Script" in script_preview
        assert "Current text" in script_preview
        assert "```text" in script_preview
        assert "### Metadata JSON" in metadata_preview
        assert '"engine": "Fish Speech"' in metadata_preview
        assert "```json" in metadata_preview
