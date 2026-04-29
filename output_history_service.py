from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import wave
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
RUN_BASE_SAFE_PATTERN = re.compile(r"[^\w\-.]+")
MISSING_HISTORY_PRESET_NAMES = {"", "no_preset"}
LEGACY_AUDIO_EXTENSIONS = {".wav", ".mp3"}
LEGACY_IMPORT_PROJECT = "default"
LEGACY_IMPORT_PRESET = "legacy_import"
LEGACY_IMPORT_ENGINE = "Legacy Import"
LEGACY_IMPORT_SCRIPT_PLACEHOLDER = "Recovered legacy output where source text was unavailable."
HISTORY_PREVIEW_CURRENT_SCRIPT = "current_script"
HISTORY_PREVIEW_METADATA_JSON = "metadata_json"
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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
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


def _sanitize_run_base_label(value: str | None, *, fallback: str) -> str:
    cleaned = RUN_BASE_SAFE_PATTERN.sub("_", str(value or "").strip()).strip("._")
    return cleaned[:80] or fallback


def _split_run_base_timestamp(run_base: str) -> tuple[str, str]:
    timestamp = _extract_timestamp(run_base)
    suffix = f"_{timestamp}"
    if not run_base.endswith(suffix):
        raise ValueError(f"Run base '{run_base}' does not end with expected timestamp suffix")
    prefix = run_base[: -len(suffix)].rstrip("_")
    return prefix, timestamp


def add_run_base_collision_suffix(run_base: str, collision_index: int) -> str:
    """Return a deterministic collision-safe run base preserving the terminal timestamp."""
    if collision_index < 1:
        return run_base

    prefix, timestamp = _split_run_base_timestamp(run_base)
    return f"{prefix}_{collision_index:02d}_{timestamp}" if prefix else f"{collision_index:02d}_{timestamp}"


def bundle_exists(project_root: str | Path, run_base: str) -> bool:
    project_root_path = Path(project_root).expanduser().resolve(strict=False)
    return any(
        [
            any(project_root_path.joinpath("audio").glob(f"{run_base}.*")),
            (project_root_path / "meta" / f"{run_base}.json").exists(),
            (project_root_path / "scripts" / f"{run_base}.txt").exists(),
            (project_root_path / "scripts" / f"{run_base}.original.txt").exists(),
            (project_root_path / "scripts" / f"{run_base}.transformed.txt").exists(),
            (project_root_path / "jobs" / f"{run_base}.job.json").exists(),
        ]
    )


def allocate_collision_safe_run_base(project_root: str | Path, preferred_run_base: str) -> str:
    """Allocate a run base that does not collide with existing canonical bundle artifacts."""
    project_root_path = Path(project_root).expanduser().resolve(strict=False)
    if not bundle_exists(project_root_path, preferred_run_base):
        return preferred_run_base

    collision_index = 1
    while True:
        candidate = add_run_base_collision_suffix(preferred_run_base, collision_index)
        if not bundle_exists(project_root_path, candidate):
            return candidate
        collision_index += 1


def _derive_timestamp_from_audio_path(audio_path: Path) -> str:
    match = TIMESTAMP_PATTERN.search(audio_path.stem)
    if match:
        return match.group("timestamp")
    return datetime.fromtimestamp(audio_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")


def _iter_legacy_audio_files(legacy_root: Path) -> list[Path]:
    if not legacy_root.exists() or not legacy_root.is_dir():
        return []
    return sorted(
        path
        for path in legacy_root.iterdir()
        if path.is_file() and path.suffix.lower() in LEGACY_AUDIO_EXTENSIONS
    )


def _load_optional_legacy_metadata(json_path: Path) -> tuple[dict[str, Any], bool]:
    if not json_path.exists() or not json_path.is_file():
        return {}, True

    try:
        raw_payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Falling back to synthesized metadata for %s: %s", json_path, error)
        return {}, True

    if not isinstance(raw_payload, dict):
        logger.warning("Falling back to synthesized metadata for %s: expected an object", json_path)
        return {}, True
    return raw_payload, False


def _load_optional_legacy_script(script_path: Path) -> tuple[str, bool]:
    if not script_path.exists() or not script_path.is_file():
        return LEGACY_IMPORT_SCRIPT_PLACEHOLDER, True

    try:
        text = script_path.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning("Falling back to synthesized script for %s: %s", script_path, error)
        return LEGACY_IMPORT_SCRIPT_PLACEHOLDER, True

    normalized = text.strip()
    if not normalized:
        return LEGACY_IMPORT_SCRIPT_PLACEHOLDER, True
    return text, False


def _build_import_run_base(preset: str, timestamp: str, *, suffix: str | None = None) -> str:
    preset_label = _sanitize_run_base_label(preset, fallback=LEGACY_IMPORT_PRESET)
    suffix_label = _sanitize_run_base_label(suffix, fallback="import") if suffix else ""
    middle = f"{preset_label}_{suffix_label}" if suffix_label else preset_label
    return f"{LEGACY_IMPORT_PROJECT}_{middle}_{timestamp}"


def _allocate_import_run_base(project_root: Path, preset: str, timestamp: str) -> str:
    return allocate_collision_safe_run_base(project_root, _build_import_run_base(preset, timestamp))


def _index_existing_legacy_imports(autosave_root: Path) -> dict[str, Path]:
    existing: dict[str, Path] = {}
    for meta_path in sorted(autosave_root.glob("*/meta/*.json")):
        try:
            metadata = _read_json(meta_path)
        except Exception:
            continue

        legacy_import = metadata.get("legacy_import")
        if not isinstance(legacy_import, dict):
            continue

        source_audio = normalize_path(legacy_import.get("source_audio"))
        if source_audio:
            existing[source_audio] = meta_path
    return existing


def _build_legacy_import_metadata(
    *,
    run_base: str,
    timestamp: str,
    canonical_audio_path: Path,
    canonical_script_path: Path,
    canonical_meta_path: Path,
    legacy_audio_path: Path,
    legacy_json_path: Path,
    legacy_script_path: Path,
    legacy_metadata: dict[str, Any],
    script_text: str,
) -> dict[str, Any]:
    metadata = dict(legacy_metadata)
    preset = _clean_metadata_text(metadata.get("preset")) or LEGACY_IMPORT_PRESET
    engine = _clean_metadata_text(metadata.get("engine")) or LEGACY_IMPORT_ENGINE
    speaker = resolve_voice_narrator(metadata)
    timestamp_iso = _timestamp_to_iso(timestamp) or datetime.fromtimestamp(
        legacy_audio_path.stat().st_mtime
    ).isoformat(timespec="seconds")

    metadata.update(
        {
            "project": LEGACY_IMPORT_PROJECT,
            "preset": preset,
            "engine": engine,
            "timestamp": timestamp_iso,
            "audio_format": legacy_audio_path.suffix.lstrip(".").lower(),
            "paths": {
                "audio": canonical_audio_path.resolve(strict=False).as_posix(),
                "script": canonical_script_path.resolve(strict=False).as_posix(),
                "meta": canonical_meta_path.resolve(strict=False).as_posix(),
            },
            "legacy_import": {
                "source_audio": normalize_path(legacy_audio_path),
                "source_meta": (
                    normalize_path(legacy_json_path) if legacy_json_path.exists() else None
                ),
                "source_script": (
                    normalize_path(legacy_script_path) if legacy_script_path.exists() else None
                ),
                "source_stem": legacy_audio_path.stem,
                "imported_run_base": run_base,
                "imported_at": datetime.now().isoformat(timespec="seconds"),
            },
        }
    )
    if speaker:
        metadata["speaker"] = speaker

    duration_seconds = _probe_wav_duration_seconds(normalize_path(canonical_audio_path))
    if duration_seconds is not None:
        metadata["duration_seconds"] = duration_seconds

    if script_text:
        metadata.setdefault(
            "text_versions",
            {
                "original": {"chars": len(script_text)},
                "transformed": {"chars": len(script_text)},
            },
        )
    return metadata


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


def _probe_wav_duration_seconds(audio_path: str | None) -> float | None:
    if not audio_path:
        return None
    path_obj = Path(audio_path)
    if path_obj.suffix.lower() != ".wav" or not path_obj.exists() or not path_obj.is_file():
        return None

    try:
        with wave.open(str(path_obj), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate <= 0:
                return None
            return frame_count / frame_rate
    except (OSError, wave.Error) as error:
        logger.debug("Unable to read WAV duration from %s: %s", path_obj, error)
        return None


def _extract_duration_seconds(
    metadata: dict[str, Any],
    *,
    autosave_audio_path: str | None,
    manual_audio_path: str | None,
) -> float | None:
    duration_seconds = _safe_float(metadata.get("duration_seconds"))
    if duration_seconds is not None and duration_seconds >= 0:
        return duration_seconds

    total_duration_seconds = _safe_float(metadata.get("total_duration_seconds"))
    if total_duration_seconds is not None and total_duration_seconds >= 0:
        return total_duration_seconds

    duration_minutes = _safe_float(metadata.get("duration_minutes"))
    if duration_minutes is not None and duration_minutes >= 0:
        return duration_minutes * 60.0

    duration = _safe_float(metadata.get("duration"))
    if duration is not None and duration >= 0:
        return duration

    for candidate in (autosave_audio_path, manual_audio_path):
        probed = _probe_wav_duration_seconds(candidate)
        if probed is not None:
            return probed

    return None


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
    duration_seconds = _extract_duration_seconds(
        metadata,
        autosave_audio_path=autosave_audio,
        manual_audio_path=manual_audio,
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
        duration_seconds=duration_seconds,
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


def import_legacy_outputs(
    legacy_root: str | Path,
    autosave_root: str | Path,
    *,
    store: OutputHistoryStore | None = None,
    db_path: str | Path | None = None,
) -> dict[str, int]:
    """Import flat legacy outputs into canonical autosave bundles under the default project."""
    legacy_root_path = Path(legacy_root).expanduser().resolve(strict=False)
    autosave_root_path = Path(autosave_root).expanduser().resolve(strict=False)
    project_root = autosave_root_path / LEGACY_IMPORT_PROJECT
    for folder_name in ("audio", "meta", "scripts"):
        (project_root / folder_name).mkdir(parents=True, exist_ok=True)

    active_store = store or OutputHistoryStore(
        db_path or default_db_path_for_autosave_root(autosave_root_path)
    )
    existing_imports = _index_existing_legacy_imports(autosave_root_path)
    summary = {
        "imported": 0,
        "skipped": 0,
        "synthesized_meta": 0,
        "synthesized_script": 0,
        "errors": 0,
    }

    for audio_path in _iter_legacy_audio_files(legacy_root_path):
        normalized_source_audio = normalize_path(audio_path)
        if not normalized_source_audio:
            summary["errors"] += 1
            continue

        existing_meta_path = existing_imports.get(normalized_source_audio)
        if existing_meta_path and existing_meta_path.exists():
            upsert_meta_file(existing_meta_path, store=active_store)
            summary["skipped"] += 1
            continue

        try:
            legacy_json_path = audio_path.with_suffix(".json")
            legacy_script_path = audio_path.with_suffix(".txt")
            legacy_metadata, synthesized_meta = _load_optional_legacy_metadata(legacy_json_path)
            script_text, synthesized_script = _load_optional_legacy_script(legacy_script_path)

            timestamp = _derive_timestamp_from_audio_path(audio_path)
            preset = _clean_metadata_text(legacy_metadata.get("preset")) or LEGACY_IMPORT_PRESET
            run_base = _allocate_import_run_base(project_root, preset, timestamp)

            canonical_audio_path = project_root / "audio" / f"{run_base}{audio_path.suffix.lower()}"
            canonical_script_path = project_root / "scripts" / f"{run_base}.txt"
            canonical_meta_path = project_root / "meta" / f"{run_base}.json"

            shutil.copy2(audio_path, canonical_audio_path)
            canonical_script_path.write_text(script_text, encoding="utf-8")

            metadata = _build_legacy_import_metadata(
                run_base=run_base,
                timestamp=timestamp,
                canonical_audio_path=canonical_audio_path,
                canonical_script_path=canonical_script_path,
                canonical_meta_path=canonical_meta_path,
                legacy_audio_path=audio_path,
                legacy_json_path=legacy_json_path,
                legacy_script_path=legacy_script_path,
                legacy_metadata=legacy_metadata,
                script_text=script_text,
            )
            canonical_meta_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            upsert_meta_file(canonical_meta_path, store=active_store)
            existing_imports[normalized_source_audio] = canonical_meta_path
            summary["imported"] += 1
            if synthesized_meta:
                summary["synthesized_meta"] += 1
            if synthesized_script:
                summary["synthesized_script"] += 1
        except Exception as error:
            summary["errors"] += 1
            logger.warning("Failed to import legacy output %s: %s", audio_path, error)

    return summary


def resolve_playback_path(record: OutputHistoryRecord, autosave_root: str | Path) -> str:
    """Return a validated playable file path for a history record."""
    for candidate in (record.autosave_audio_path, record.manual_audio_path):
        if candidate and is_path_within_root(candidate, autosave_root):
            return candidate
    raise ValueError("No playable audio path exists under the configured autosave root")


def _resolve_validated_history_preview_path(
    candidate: str | Path | None,
    autosave_root: str | Path,
    *,
    label: str,
) -> Path:
    normalized = normalize_path(candidate)
    if not normalized:
        raise ValueError(f"No {label} is available for this history record")
    if not is_path_within_root(normalized, autosave_root):
        raise ValueError(f"{label.capitalize()} preview is only available under the autosave root")

    path = Path(normalized)
    if not path.exists() or not path.is_file():
        raise ValueError(f"{label.capitalize()} file is missing: {normalized}")
    return path


def read_history_preview(record: OutputHistoryRecord, autosave_root: str | Path, preview_kind: str) -> dict[str, str]:
    """Return validated preview content for a history record artifact."""
    normalized_kind = str(preview_kind or "").strip().lower()

    if normalized_kind == HISTORY_PREVIEW_CURRENT_SCRIPT:
        script_path = _resolve_validated_history_preview_path(
            record.autosave_scripts[0] if record.autosave_scripts else None,
            autosave_root,
            label="current script",
        )
        return {
            "kind": HISTORY_PREVIEW_CURRENT_SCRIPT,
            "title": "Current Script",
            "path": script_path.resolve(strict=False).as_posix(),
            "content": script_path.read_text(encoding="utf-8"),
            "language": "text",
        }

    if normalized_kind == HISTORY_PREVIEW_METADATA_JSON:
        meta_path = _resolve_validated_history_preview_path(
            record.autosave_meta_path,
            autosave_root,
            label="metadata json",
        )
        return {
            "kind": HISTORY_PREVIEW_METADATA_JSON,
            "title": "Metadata JSON",
            "path": meta_path.resolve(strict=False).as_posix(),
            "content": meta_path.read_text(encoding="utf-8"),
            "language": "json",
        }

    raise ValueError(f"Unsupported history preview kind: {preview_kind}")


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
    "bundle_exists",
    "create_or_repair_job_json",
    "default_db_path_for_autosave_root",
    "feature_storage_root_from_autosave_root",
    "HISTORY_PREVIEW_CURRENT_SCRIPT",
    "HISTORY_PREVIEW_METADATA_JSON",
    "import_legacy_outputs",
    "is_history_preset_missing",
    "is_path_within_root",
    "normalize_path",
    "read_history_preview",
    "reindex_root",
    "resolve_voice_narrator",
    "resolve_playback_path",
    "upsert_meta_file",
    "add_run_base_collision_suffix",
    "allocate_collision_safe_run_base",
]
