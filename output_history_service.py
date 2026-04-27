from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from app.output_history_store import OutputHistoryRecord, OutputHistoryStore


logger = logging.getLogger(__name__)

TIMESTAMP_PATTERN = re.compile(r"(?P<timestamp>\d{8}_\d{6})$")
MISSING_HISTORY_PRESET_NAMES = {"", "no_preset"}
VOICE_NARRATOR_METADATA_KEYS = (
    "speaker",
    "speaker_profile",
    "voice_preset",
    "voice",
)


@dataclass(slots=True)
class OutputHistoryPaths:
    """Resolved bundle paths for one autosave history entry."""

    manual_audio_path: str | None
    autosave_audio_path: str | None
    autosave_meta_path: str
    autosave_scripts: list[str]
    job_json_path: str
    legacy_copy: bool


def normalize_path(value: str | Path | None) -> str | None:
    """Return a full normalized forward-slash path when a value is present."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve(strict=False).as_posix()


def feature_storage_root_from_autosave_root(autosave_root: str | Path) -> Path:
    """Return the feature storage root that should own the SQLite DB."""
    root = Path(autosave_root).expanduser().resolve(strict=False)
    return root.parent if root.name == "app_state_outputs" else root


def default_db_path_for_autosave_root(autosave_root: str | Path) -> Path:
    """Return the default SQLite path for a configured autosave root."""
    return feature_storage_root_from_autosave_root(autosave_root) / "outputs.db"


def is_path_within_root(candidate: str | Path | None, root: str | Path) -> bool:
    """Return whether a path resolves inside the allowed root."""
    if candidate is None:
        return False
    candidate_path = Path(candidate).expanduser().resolve(strict=False)
    root_path = Path(root).expanduser().resolve(strict=False)
    try:
        candidate_path.relative_to(root_path)
    except ValueError:
        return False
    return True


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def is_history_preset_missing(preset: str | None) -> bool:
    normalized = _clean_metadata_text(preset)
    if normalized is None:
        return True
    return normalized.lower() in MISSING_HISTORY_PRESET_NAMES


def resolve_voice_narrator(
    metadata: dict[str, Any] | None,
    *,
    fallback: str | None = None,
) -> str | None:
    if isinstance(metadata, dict):
        for key in VOICE_NARRATOR_METADATA_KEYS:
            resolved = _clean_metadata_text(metadata.get(key))
            if resolved:
                return resolved
    return _clean_metadata_text(fallback)


def _timestamp_to_iso(timestamp: str) -> str | None:
    try:
        parsed = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%S")


def _extract_timestamp(run_base: str) -> str:
    match = TIMESTAMP_PATTERN.search(run_base)
    if not match:
        raise ValueError(f"Could not extract timestamp from run base '{run_base}'")
    return match.group("timestamp")


def _infer_preset(run_base: str, project: str, timestamp: str) -> str:
    prefix = f"{project}_"
    suffix = f"_{timestamp}"
    if run_base.startswith(prefix) and run_base.endswith(suffix):
        middle = run_base[len(prefix) : -len(suffix)]
        return middle or ""
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object metadata in {path}")
    return data


def _read_text_if_exists(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _collect_scripts(paths_block: dict[str, Any], scripts_dir: Path, run_base: str) -> list[str]:
    ordered = [
        normalize_path(paths_block.get("script")),
        normalize_path(paths_block.get("script_original")),
        normalize_path(paths_block.get("script_transformed")),
    ]
    fallback_names = [
        scripts_dir / f"{run_base}.txt",
        scripts_dir / f"{run_base}.original.txt",
        scripts_dir / f"{run_base}.transformed.txt",
    ]
    ordered.extend(normalize_path(path) for path in fallback_names if path.exists())

    unique: list[str] = []
    for item in ordered:
        if item and item not in unique:
            unique.append(item)
    return unique


def _resolve_audio_paths(
    metadata: dict[str, Any],
    project_root: Path,
    autosave_root: Path,
    run_base: str,
) -> tuple[str | None, str | None, bool]:
    paths_block = metadata.get("paths", {}) if isinstance(metadata.get("paths"), dict) else {}
    metadata_audio = normalize_path(paths_block.get("audio"))

    bundle_audio_candidates = sorted(project_root.joinpath("audio").glob(f"{run_base}.*"))
    autosave_audio = normalize_path(bundle_audio_candidates[0]) if bundle_audio_candidates else None
    manual_audio = None

    if metadata_audio:
        if autosave_audio and metadata_audio == autosave_audio:
            manual_audio = None
        elif is_path_within_root(metadata_audio, autosave_root):
            autosave_audio = metadata_audio
        else:
            manual_audio = metadata_audio

    legacy_copy = bool(manual_audio and autosave_audio and manual_audio != autosave_audio)
    return manual_audio, autosave_audio, legacy_copy


def _build_job_payload(
    *,
    project: str,
    preset: str,
    timestamp: str,
    datetime_iso: str | None,
    metadata: dict[str, Any],
    paths: OutputHistoryPaths,
) -> dict[str, Any]:
    return {
        "project": project,
        "preset": preset,
        "timestamp": timestamp,
        "datetime_iso": datetime_iso,
        "engine": metadata.get("engine"),
        "seed": _safe_int(metadata.get("seed")),
        "speaker": metadata.get("speaker"),
        "audio_format": metadata.get("audio_format"),
        "paths": {
            "manual_audio": paths.manual_audio_path,
            "autosave_audio": paths.autosave_audio_path,
            "autosave_meta": paths.autosave_meta_path,
            "autosave_scripts": paths.autosave_scripts,
            "job_json": paths.job_json_path,
        },
        "texts": {
            "current": (
                _read_text_if_exists(paths.autosave_scripts[0]) if paths.autosave_scripts else None
            ),
            "original": (
                _read_text_if_exists(paths.autosave_scripts[1])
                if len(paths.autosave_scripts) > 1
                else None
            ),
            "transformed": (
                _read_text_if_exists(paths.autosave_scripts[2])
                if len(paths.autosave_scripts) > 2
                else None
            ),
        },
        "metadata_snapshot": metadata,
        "can_reload": True,
    }


def create_or_repair_job_json(meta_path: str | Path) -> str:
    """Ensure the canonical jobs/*.job.json exists and carries core reload data."""
    meta_path_obj = Path(meta_path).expanduser().resolve(strict=False)
    metadata = _read_json(meta_path_obj)
    run_base = meta_path_obj.stem
    timestamp = _extract_timestamp(run_base)
    project_root = meta_path_obj.parent.parent
    project = str(metadata.get("project") or project_root.name)
    preset = str(metadata.get("preset") or _infer_preset(run_base, project, timestamp) or "")
    datetime_iso = _timestamp_to_iso(timestamp)

    autosave_root = project_root.parent.resolve(strict=False)
    scripts = _collect_scripts(metadata.get("paths", {}), project_root / "scripts", run_base)
    manual_audio, autosave_audio, legacy_copy = _resolve_audio_paths(
        metadata,
        project_root,
        autosave_root,
        run_base,
    )

    jobs_dir = project_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_path = jobs_dir / f"{run_base}.job.json"

    resolved_paths = OutputHistoryPaths(
        manual_audio_path=manual_audio,
        autosave_audio_path=autosave_audio,
        autosave_meta_path=normalize_path(meta_path_obj) or "",
        autosave_scripts=scripts,
        job_json_path=normalize_path(job_path) or job_path.as_posix(),
        legacy_copy=legacy_copy,
    )
    desired = _build_job_payload(
        project=project,
        preset=preset,
        timestamp=timestamp,
        datetime_iso=datetime_iso,
        metadata=metadata,
        paths=resolved_paths,
    )

    existing: dict[str, Any] = {}
    if job_path.exists():
        try:
            raw_existing = json.loads(job_path.read_text(encoding="utf-8"))
            if isinstance(raw_existing, dict):
                existing = raw_existing
        except json.JSONDecodeError:
            logger.warning("Rebuilding corrupt job bundle file: %s", job_path)

    merged = {**existing, **desired}
    job_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalize_path(job_path) or job_path.as_posix()


def build_record_from_meta(meta_path: str | Path) -> OutputHistoryRecord:
    """Parse one metadata file into a normalized output history record."""
    meta_path_obj = Path(meta_path).expanduser().resolve(strict=False)
    metadata = _read_json(meta_path_obj)
    run_base = meta_path_obj.stem
    timestamp = _extract_timestamp(run_base)
    datetime_iso = _timestamp_to_iso(timestamp)
    project_root = meta_path_obj.parent.parent
    autosave_root = project_root.parent.resolve(strict=False)
    project = str(metadata.get("project") or project_root.name)
    preset = str(metadata.get("preset") or _infer_preset(run_base, project, timestamp) or "")
    speaker = resolve_voice_narrator(metadata)
    raw_llm_transform = metadata.get("llm_transform")
    llm_transform = raw_llm_transform if isinstance(raw_llm_transform, dict) else {}
    transform_status = str(llm_transform.get("status") or "") or None
    llm_enabled = bool(llm_transform.get("applied"))
    scripts = _collect_scripts(metadata.get("paths", {}), project_root / "scripts", run_base)
    manual_audio, autosave_audio, legacy_copy = _resolve_audio_paths(
        metadata,
        project_root,
        autosave_root,
        run_base,
    )
    job_json_path = create_or_repair_job_json(meta_path_obj)

    return OutputHistoryRecord(
        job_json_path=job_json_path,
        project=project,
        preset=preset,
        timestamp=timestamp,
        datetime_iso=datetime_iso,
        engine=str(metadata.get("engine") or "") or None,
        seed=_safe_int(metadata.get("seed")),
        speaker=speaker,
        chunks=_safe_int(metadata.get("chunks")),
        transform=transform_status,
        llm_enabled=llm_enabled,
        manual_audio_path=manual_audio,
        autosave_audio_path=autosave_audio,
        autosave_meta_path=normalize_path(meta_path_obj),
        autosave_scripts=scripts,
        legacy_copy=legacy_copy,
        can_reload=True,
    )


def upsert_meta_file(
    meta_path: str | Path,
    *,
    store: OutputHistoryStore | None = None,
    db_path: str | Path | None = None,
) -> OutputHistoryRecord:
    """Upsert one autosave bundle into the SQLite store."""
    record = build_record_from_meta(meta_path)
    active_store = store
    if active_store is None:
        resolved_db_path = db_path or default_db_path_for_autosave_root(Path(meta_path).parents[2])
        active_store = OutputHistoryStore(resolved_db_path)
    return active_store.upsert_record(record)


def reindex_root(
    autosave_root: str | Path,
    *,
    store: OutputHistoryStore | None = None,
    db_path: str | Path | None = None,
) -> list[OutputHistoryRecord]:
    """Scan an autosave root and upsert all discovered metadata bundles."""
    root = Path(autosave_root).expanduser().resolve(strict=False)
    active_store = store or OutputHistoryStore(db_path or default_db_path_for_autosave_root(root))
    records: list[OutputHistoryRecord] = []
    for meta_path in sorted(root.glob("*/meta/*.json")):
        records.append(active_store.upsert_record(build_record_from_meta(meta_path)))
    return records


def resolve_playback_path(record: OutputHistoryRecord, autosave_root: str | Path) -> str:
    """Return a validated playable file path for a history record."""
    for candidate in (record.autosave_audio_path, record.manual_audio_path):
        if candidate and is_path_within_root(candidate, autosave_root):
            return candidate
    raise ValueError("No playable audio path exists under the configured autosave root")


def build_reload_payload(record: OutputHistoryRecord) -> dict[str, Any]:
    """Return a Gradio-friendly reload payload using job JSON and current files."""
    job_path = Path(record.job_json_path)
    payload = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() else {}
    paths_block = payload.get("paths", {}) if isinstance(payload.get("paths"), dict) else {}
    texts_block = payload.get("texts", {}) if isinstance(payload.get("texts"), dict) else {}
    metadata_snapshot = (
        payload.get("metadata_snapshot", {})
        if isinstance(payload.get("metadata_snapshot"), dict)
        else {}
    )
    raw_reload_snapshot = (
        metadata_snapshot.get("reload_snapshot", {})
        if isinstance(metadata_snapshot.get("reload_snapshot"), dict)
        else {}
    )
    snapshot_control_values = (
        raw_reload_snapshot.get("control_values", {})
        if isinstance(raw_reload_snapshot.get("control_values"), dict)
        else {}
    )
    voice_narrator = resolve_voice_narrator(metadata_snapshot, fallback=record.speaker)

    transformed_text = texts_block.get("transformed") or _read_text_if_exists(
        (paths_block.get("autosave_scripts") or [None, None, None])[2]
        if len(paths_block.get("autosave_scripts") or []) > 2
        else None
    )
    original_text = texts_block.get("original") or _read_text_if_exists(
        (paths_block.get("autosave_scripts") or [None, None])[1]
        if len(paths_block.get("autosave_scripts") or []) > 1
        else None
    )
    current_text = texts_block.get("current") or _read_text_if_exists(
        (paths_block.get("autosave_scripts") or [None])[0]
        if paths_block.get("autosave_scripts")
        else None
    )

    fallback_control_values = {
        "autosave_project_name": record.project,
        "tts_engine": record.engine,
        "audio_format": _clean_metadata_text(metadata_snapshot.get("audio_format")),
        "speaker_name": voice_narrator,
        "voice_preset": "" if is_history_preset_missing(record.preset) else record.preset,
        "autosave_enabled": True,
        "last_seed_state": record.seed,
    }
    reload_snapshot = {
        "schema_version": raw_reload_snapshot.get("schema_version", 0),
        "active_engine": raw_reload_snapshot.get("active_engine") or record.engine,
        "control_values": {**fallback_control_values, **snapshot_control_values},
        "excluded_controls": (
            raw_reload_snapshot.get("excluded_controls", [])
            if isinstance(raw_reload_snapshot.get("excluded_controls"), list)
            else []
        ),
        "legacy_fallback": not bool(snapshot_control_values),
    }
    reload_text = original_text or current_text or transformed_text or ""

    return {
        "project": record.project,
        "preset": record.preset,
        "timestamp": record.timestamp,
        "engine": record.engine,
        "speaker": record.speaker,
        "voice_narrator": voice_narrator,
        "seed": record.seed,
        "audio_format": _clean_metadata_text(metadata_snapshot.get("audio_format")),
        "script_text": reload_text,
        "original_text": original_text or "",
        "transformed_text": transformed_text or current_text or "",
        "metadata_snapshot": metadata_snapshot,
        "reload_snapshot": reload_snapshot,
        "legacy_reload": reload_snapshot["legacy_fallback"],
        "job_json_path": record.job_json_path,
    }


__all__ = [
    "OutputHistoryPaths",
    "build_record_from_meta",
    "build_reload_payload",
    "create_or_repair_job_json",
    "default_db_path_for_autosave_root",
    "feature_storage_root_from_autosave_root",
    "is_history_preset_missing",
    "is_path_within_root",
    "normalize_path",
    "reindex_root",
    "resolve_voice_narrator",
    "resolve_playback_path",
    "upsert_meta_file",
]
