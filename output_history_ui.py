from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from output_history_service import (
    HISTORY_PREVIEW_CURRENT_SCRIPT,
    build_reload_payload,
    read_history_preview,
    resolve_playback_path,
)
from output_history_store import OutputHistoryRecord, OutputHistoryStore


HISTORY_EMPTY_ROWS: list[list[object]] = [
    ["—", "No history yet", "—", "—", "—", "—", "—", "—", "—"]
]
HISTORY_NO_SELECTION_MESSAGE = (
    "No history record selected yet. Enter a numeric History Record ID after refreshing or reindexing."
)
HISTORY_EMPTY_DETAIL_MESSAGE = (
    "No indexed autosave history yet. Generate a clip with autosave enabled or click Reindex Autosaves to scan the current autosave root."
)
HISTORY_PREVIEW_EMPTY_MESSAGE = (
    "Select a history record, then use Preview Script or Preview Metadata to inspect the saved bundle content."
)


class HistoryListFilters(TypedDict):
    query: str | None
    project: str | None
    preset: str | None
    seed: int | None
    speaker: str | None
    from_timestamp: str | None
    to_timestamp: str | None


def coerce_history_record_id(raw_record_id: object) -> int | None:
    if raw_record_id in {None, ""}:
        return None
    try:
        record_id = int(str(raw_record_id).strip())
    except (TypeError, ValueError):
        return None
    return record_id if record_id > 0 else None


def format_history_preset(preset: str | None) -> str:
    normalized = str(preset or "").strip()
    if not normalized or normalized.lower() == "no_preset":
        return "—"
    return normalized


def format_history_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "—"
    try:
        total_seconds = float(duration_seconds)
    except (TypeError, ValueError):
        return "—"
    if total_seconds < 0:
        return "—"
    if total_seconds >= 60:
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    return f"{total_seconds:.1f}s"


def build_history_settings_lines(metadata_snapshot: dict[str, object] | None) -> list[str]:
    if not isinstance(metadata_snapshot, dict):
        return []

    settings_lines: list[str] = []
    audio_format = str(metadata_snapshot.get("audio_format") or "").strip()
    if audio_format:
        settings_lines.append(f"- Audio format: `{audio_format}`")

    for key, label in (
        ("voice", "Built-in voice"),
        ("voice_preset", "Engine voice preset"),
        ("speaker_profile", "Speaker profile"),
        ("language", "Language"),
    ):
        value = str(metadata_snapshot.get(key) or "").strip()
        if value:
            settings_lines.append(f"- {label}: `{value}`")

    return settings_lines


def normalize_history_filter_text(raw_value: object) -> str | None:
    normalized_value = str(raw_value or "").strip()
    return normalized_value or None


def normalize_history_seed_filter(raw_value: object) -> int | None:
    normalized_seed = normalize_history_filter_text(raw_value)
    if normalized_seed is None:
        return None
    try:
        return int(normalized_seed)
    except (TypeError, ValueError):
        return None


def build_history_list_filters(
    query: object,
    project: object,
    preset: object,
    seed: object,
    speaker: object,
    from_timestamp: object,
    to_timestamp: object,
) -> HistoryListFilters:
    return {
        "query": normalize_history_filter_text(query),
        "project": normalize_history_filter_text(project),
        "preset": normalize_history_filter_text(preset),
        "seed": normalize_history_seed_filter(seed),
        "speaker": normalize_history_filter_text(speaker),
        "from_timestamp": normalize_history_filter_text(from_timestamp),
        "to_timestamp": normalize_history_filter_text(to_timestamp),
    }


def format_history_rows(records: list[OutputHistoryRecord]) -> list[list[object]]:
    if not records:
        return HISTORY_EMPTY_ROWS

    rows: list[list[object]] = []
    for record in records:
        rows.append(
            [
                record.id,
                record.project,
                format_history_preset(record.preset),
                record.timestamp,
                record.engine or "—",
                record.speaker or "—",
                format_history_duration(record.duration_seconds),
                str(record.seed) if record.seed is not None else "—",
                "Yes" if record.can_reload else "No",
            ]
        )
    return rows


def select_history_record_id_from_rows(
    table_rows: list[list[object]] | None,
    selected_row_index: int | None,
) -> int | None:
    if table_rows is None or selected_row_index is None:
        return None
    if not 0 <= selected_row_index < len(table_rows):
        return None
    row = table_rows[selected_row_index]
    if not row:
        return None
    return coerce_history_record_id(row[0])


def build_history_preview_placeholder(record_id: object) -> str:
    parsed_id = coerce_history_record_id(record_id)
    if parsed_id is None:
        return HISTORY_PREVIEW_EMPTY_MESSAGE
    return (
        f"History record {parsed_id} selected. Use Preview Script or Preview Metadata to inspect saved bundle files."
    )


def format_history_preview_markdown(preview: dict[str, str]) -> str:
    content = preview.get("content") or ""
    if not content.strip():
        content = "(empty)"
    language = preview.get("language") or "text"
    title = preview.get("title") or "Preview"
    path = preview.get("path") or "—"
    return f"### {title}\n**Source:** `{path}`\n\n```{language}\n{content}\n```"


def build_history_detail_response(
    record_id: object,
    *,
    store: OutputHistoryStore,
    autosave_root: str,
    audio_proxy_url_builder: Callable[[int | None], str | None],
) -> tuple[str, str | None]:
    parsed_id = coerce_history_record_id(record_id)
    if parsed_id is None:
        return HISTORY_NO_SELECTION_MESSAGE, None

    record = store.get_record(parsed_id)
    if record is None:
        return f"❌ History record not found: {record_id}", None

    payload: dict[str, object] = {}
    payload_error: str | None = None
    try:
        payload = build_reload_payload(record)
    except Exception as error:
        payload_error = str(error)

    metadata_snapshot = payload.get("metadata_snapshot", {})
    reload_snapshot = payload.get("reload_snapshot", {})
    reload_control_values = (
        reload_snapshot.get("control_values", {}) if isinstance(reload_snapshot, dict) else {}
    )
    excluded_controls = (
        reload_snapshot.get("excluded_controls", []) if isinstance(reload_snapshot, dict) else []
    )
    excluded_reasons = {
        item.get("reason")
        for item in excluded_controls
        if isinstance(item, dict) and item.get("reason")
    }
    voice_narrator = payload.get("voice_narrator") or record.speaker
    audio_format = payload.get("audio_format")
    uses_legacy_reload = bool(payload.get("legacy_reload"))

    lines = [f"### History Record {record.id}"]
    lines.append(f"**Project:** {record.project}")
    lines.append(f"**Preset:** {format_history_preset(record.preset)}")
    lines.append(f"**Timestamp:** {record.timestamp}")
    lines.append(f"**Engine:** {record.engine or '—'}")
    lines.append(f"**Voice / Narrator:** {voice_narrator or '—'}")
    lines.append(f"**Audio Length:** {format_history_duration(record.duration_seconds)}")
    lines.append(f"**Seed:** {record.seed if record.seed is not None else '—'}")
    lines.append(f"**Audio Format:** {audio_format or '—'}")
    lines.append(f"**Chunks:** {record.chunks if record.chunks is not None else '—'}")
    lines.append(f"**Transform:** {record.transform or '—'}")
    lines.append(f"**LLM Transform Applied:** {'Yes' if record.llm_enabled else 'No'}")
    lines.append(f"**Reload Ready:** {'Yes' if record.can_reload else 'No'}")
    lines.append("")
    lines.append("**Reload Into Text Tab**")
    if uses_legacy_reload:
        lines.append(
            "- Restores: original source text when available, project name, engine, audio format, voice/narrator label, preset dropdown, and last seed."
        )
        lines.append(
            "- This older record predates the richer production snapshot, so engine-specific controls, narration transform settings, audio effects, and autosave toggle details may be incomplete."
        )
    else:
        lines.append(
            "- Restores: original source text, engine, audio format, saved controls for the production engine, narration transform settings except API key, audio effects, speaker label, preset, autosave options/project name, and last seed."
        )
        lines.append(
            "- Does not restore uploaded/reference audio temp files directly, but the active engine reference audio can be repopulated from the selected preset when that preset still points to a valid saved audio file. Emotion audio uploads, API keys, and unrelated tab state are not restored."
        )

    if excluded_reasons:
        lines.append("")
        lines.append("**Reload Limitations**")
        if "secret" in excluded_reasons:
            lines.append("- API keys are intentionally excluded from history snapshots for safety.")
        if "transient_file_input" in excluded_reasons:
            lines.append(
                "- Uploaded reference/emotion audio file inputs are not restored directly because their original temp paths may no longer exist. If the saved preset still has a valid reference audio file, reload can repopulate the active engine's standard reference-audio control from that preset."
            )

    settings_lines = build_history_settings_lines(metadata_snapshot if isinstance(metadata_snapshot, dict) else None)
    if settings_lines:
        lines.append("")
        lines.append("**Stored Settings Context**")
        lines.extend(settings_lines)

    if reload_control_values and not uses_legacy_reload:
        lines.append("")
        lines.append(
            f"**Reload Snapshot:** {len(reload_control_values)} sanitized control value(s) captured at generation time."
        )

    if payload_error:
        lines.append("")
        lines.append(f"⚠️ Stored reload metadata unavailable: {payload_error}")

    lines.append("")
    lines.append("**Paths**")
    lines.append(f"- Job JSON: `{record.job_json_path}`")
    lines.append(f"- Metadata: `{record.autosave_meta_path or '—'}`")
    lines.append(f"- Autosave audio: `{record.autosave_audio_path or '—'}`")
    lines.append(f"- Manual audio: `{record.manual_audio_path or '—'}`")
    if record.autosave_scripts:
        lines.append("- Scripts:")
        lines.extend(f"  - `{script_path}`" for script_path in record.autosave_scripts)

    audio_value = None
    try:
        resolve_playback_path(record, autosave_root)
        audio_value = audio_proxy_url_builder(record.id)
    except ValueError as error:
        lines.append("")
        lines.append(f"⚠️ Audio preview unavailable: {error}")

    return "\n".join(lines), audio_value


def build_history_panel_refresh_response(
    *,
    store: OutputHistoryStore,
    autosave_root: str,
    audio_proxy_url_builder: Callable[[int | None], str | None],
    query: object,
    project: object,
    preset: object,
    seed: object,
    speaker: object,
    from_timestamp: object,
    to_timestamp: object,
    record_id: object,
) -> tuple[list[list[object]], str, str | None, str]:
    history_filters = build_history_list_filters(
        query,
        project,
        preset,
        seed,
        speaker,
        from_timestamp,
        to_timestamp,
    )
    records = store.list_records(
        query=history_filters["query"],
        project=history_filters["project"],
        preset=history_filters["preset"],
        seed=history_filters["seed"],
        speaker=history_filters["speaker"],
        from_timestamp=history_filters["from_timestamp"],
        to_timestamp=history_filters["to_timestamp"],
        limit=50,
    )
    rows = format_history_rows(records)
    detail, audio_value = build_history_detail_response(
        record_id,
        store=store,
        autosave_root=autosave_root,
        audio_proxy_url_builder=audio_proxy_url_builder,
    )
    if not records and coerce_history_record_id(record_id) is None:
        detail = HISTORY_EMPTY_DETAIL_MESSAGE
    return rows, detail, audio_value, build_history_preview_placeholder(record_id)


def build_history_preview_response(
    record_id: object,
    preview_kind: str,
    *,
    store: OutputHistoryStore,
    autosave_root: str,
) -> str:
    parsed_id = coerce_history_record_id(record_id)
    if parsed_id is None:
        return HISTORY_PREVIEW_EMPTY_MESSAGE

    record = store.get_record(parsed_id)
    if record is None:
        return f"❌ History record not found: {record_id}"

    try:
        preview = read_history_preview(record, autosave_root, preview_kind)
    except Exception as error:
        return f"❌ Unable to preview history artifact: {error}"

    return format_history_preview_markdown(preview)


def build_history_table_select_response(
    *,
    table_rows: list[list[object]] | None,
    selected_row_index: int | None,
    store: OutputHistoryStore,
    autosave_root: str,
    audio_proxy_url_builder: Callable[[int | None], str | None],
) -> tuple[str, str, str | None, str]:
    record_id = select_history_record_id_from_rows(table_rows, selected_row_index)
    if record_id is None:
        detail, audio_value = build_history_detail_response(
            None,
            store=store,
            autosave_root=autosave_root,
            audio_proxy_url_builder=audio_proxy_url_builder,
        )
        return "", detail, audio_value, build_history_preview_placeholder(None)

    detail, audio_value = build_history_detail_response(
        record_id,
        store=store,
        autosave_root=autosave_root,
        audio_proxy_url_builder=audio_proxy_url_builder,
    )
    return str(record_id), detail, audio_value, build_history_preview_placeholder(record_id)


__all__ = [
    "HISTORY_EMPTY_DETAIL_MESSAGE",
    "HISTORY_EMPTY_ROWS",
    "HISTORY_NO_SELECTION_MESSAGE",
    "HISTORY_PREVIEW_EMPTY_MESSAGE",
    "HISTORY_PREVIEW_CURRENT_SCRIPT",
    "build_history_detail_response",
    "build_history_list_filters",
    "build_history_panel_refresh_response",
    "build_history_preview_placeholder",
    "build_history_preview_response",
    "build_history_settings_lines",
    "build_history_table_select_response",
    "coerce_history_record_id",
    "format_history_duration",
    "format_history_preset",
    "format_history_rows",
    "normalize_history_filter_text",
    "normalize_history_seed_filter",
    "select_history_record_id_from_rows",
]