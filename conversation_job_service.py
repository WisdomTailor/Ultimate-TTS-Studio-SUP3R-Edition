"""Headless conversation job execution for JobManager workers."""

from __future__ import annotations

import os
from typing import Any

from conversation_logic import format_conversation_info


def generate_conversation_job(request_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute a queued conversation synthesis job and return a serializable result."""
    from launch import (
        _build_conversation_history_metadata,
        _validate_required_project_name,
        autosave_generation_artifacts,
        generate_conversation_audio_indextts2,
        generate_conversation_audio_kitten,
        generate_conversation_audio_kokoro,
        generate_conversation_audio_simple,
        write_generation_sidecar_metadata,
    )

    script_text = str(request_dict.get("text", "") or "")
    if not script_text.strip():
        raise ValueError("ERROR: No conversation script provided")

    selected_engine = str(request_dict.get("engine", "Kokoro TTS") or "Kokoro TTS")
    audio_format = str(request_dict.get("audio_format", "wav") or "wav")
    engine_params = dict(request_dict.get("engine_params") or {})

    project_name = str(engine_params.get("project_name", "") or "")
    resolved_project, project_error = _validate_required_project_name(project_name)
    if project_error:
        raise ValueError(project_error)

    voice_samples = list(engine_params.get("voice_samples") or [])
    ref_texts = [str(value or "") for value in list(engine_params.get("ref_texts") or [])]
    kokoro_voices = [str(value or "") for value in list(engine_params.get("kokoro_voices") or [])]
    kitten_voices = [str(value or "") for value in list(engine_params.get("kitten_voices") or [])]
    emotion_modes = list(engine_params.get("emotion_modes") or [])
    emotion_audios = list(engine_params.get("emotion_audios") or [])
    emotion_descriptions = list(engine_params.get("emotion_descriptions") or [])
    emotion_vectors = list(engine_params.get("emotion_vectors") or [])
    hydration_warnings = [
        str(value) for value in list(engine_params.get("hydration_warnings") or []) if str(value)
    ]
    preflight_warnings = [
        str(value) for value in list(engine_params.get("preflight_warnings") or []) if str(value)
    ]

    pause_duration = float(engine_params.get("pause_duration") or 0.8)
    transition_pause = float(engine_params.get("transition_pause") or 0.3)
    autosave_enabled = bool(engine_params.get("autosave_enabled", True))
    autosave_store_audio_copy = bool(engine_params.get("autosave_store_audio_copy", True))
    keep_legacy_output_copy = bool(engine_params.get("keep_legacy_output_copy", True))

    if selected_engine == "Kokoro TTS":
        result = generate_conversation_audio_kokoro(
            script_text,
            kokoro_voices,
            selected_engine=selected_engine,
            conversation_pause_duration=pause_duration,
            speaker_transition_pause=transition_pause,
            effects_settings=None,
            audio_format=audio_format,
            project_name=resolved_project,
        )
    elif selected_engine == "KittenTTS":
        result = generate_conversation_audio_kitten(
            script_text,
            kitten_voices,
            selected_engine=selected_engine,
            conversation_pause_duration=pause_duration,
            speaker_transition_pause=transition_pause,
            effects_settings=None,
            audio_format=audio_format,
            project_name=resolved_project,
        )
    elif selected_engine == "IndexTTS2":
        result = generate_conversation_audio_indextts2(
            script_text,
            voice_samples,
            emotion_modes,
            emotion_audios,
            emotion_descriptions,
            emotion_vectors,
            selected_engine=selected_engine,
            conversation_pause_duration=pause_duration,
            speaker_transition_pause=transition_pause,
            effects_settings=None,
            audio_format=audio_format,
            project_name=resolved_project,
        )
    else:
        result = generate_conversation_audio_simple(
            script_text,
            voice_samples,
            ref_texts=ref_texts,
            selected_engine=selected_engine,
            conversation_pause_duration=pause_duration,
            speaker_transition_pause=transition_pause,
            effects_settings=None,
            audio_format=audio_format,
            project_name=resolved_project,
        )

    if result[0] is None:
        raise RuntimeError(str(result[1]))

    audio_data, summary = result
    summary_dict = summary if isinstance(summary, dict) else {}
    conversation_metadata = _build_conversation_history_metadata(
        script_text=script_text,
        summary=summary_dict,
        selected_engine=selected_engine,
        project_name=resolved_project,
        audio_format=audio_format,
        pause_duration=pause_duration,
        transition_pause=transition_pause,
        voice_samples=voice_samples,
        ref_texts=ref_texts,
        kokoro_voices=kokoro_voices,
        kitten_voices=kitten_voices,
        emotion_modes=emotion_modes,
        emotion_descriptions=emotion_descriptions,
        emotion_vectors=emotion_vectors,
    )

    saved_audio_path = str(summary_dict.get("saved_audio_path") or "").strip()
    if saved_audio_path:
        rich_meta_path, rich_script_path = write_generation_sidecar_metadata(
            saved_audio_path,
            conversation_metadata,
            script_text,
            original_text=script_text,
            transformed_text=script_text,
        )
        summary_dict["metadata_file"] = rich_meta_path
        summary_dict["script_file"] = rich_script_path

    history_status_lines: list[str] = []
    autosave_paths = None
    if autosave_enabled:
        autosave_paths, autosave_error = autosave_generation_artifacts(
            audio_data,
            script_text,
            audio_format,
            resolved_project,
            conversation_metadata.get("speaker") or "conversation",
            conversation_metadata,
            source_audio_path=saved_audio_path,
            store_audio_copy=autosave_store_audio_copy,
            original_text_input=script_text,
            transformed_text_input=script_text,
        )
        if autosave_error:
            history_status_lines.append(f"Autosave failed: {autosave_error}")
        elif autosave_paths:
            history_status_lines.append("History autosave captured for this job.")

    output_path = saved_audio_path
    if autosave_paths and autosave_paths.get("audio_path"):
        output_path = str(autosave_paths["audio_path"])

    if autosave_enabled and saved_audio_path and autosave_paths and not keep_legacy_output_copy:
        saved_audio_abs = os.path.abspath(saved_audio_path)
        autosave_audio_abs = os.path.abspath(autosave_paths.get("audio_path", ""))
        if saved_audio_abs != autosave_audio_abs and os.path.exists(saved_audio_abs):
            try:
                os.remove(saved_audio_abs)
                history_status_lines.append(f"Legacy output removed: {saved_audio_abs}")
            except Exception as cleanup_error:
                history_status_lines.append(f"Legacy output cleanup failed: {cleanup_error}")

    summary_text = format_conversation_info(summary_dict)
    resume_info = str(summary_dict.get("resume_info", "") or "").strip()
    if resume_info:
        summary_text = f"INFO: {resume_info}\n\n{summary_text}"
    if hydration_warnings:
        summary_text = "\n".join(
            [
                *(f"WARNING: {warning}" for warning in hydration_warnings),
                "",
                summary_text,
            ]
        )
    if preflight_warnings:
        summary_text = "\n".join(
            [
                *(f"WARNING: {warning}" for warning in preflight_warnings),
                "",
                summary_text,
            ]
        )
    if history_status_lines:
        summary_text = summary_text + "\n\n" + "\n".join(history_status_lines)

    return {
        "job_type": "conversation",
        "status": "completed",
        "engine": selected_engine,
        "project": resolved_project,
        "output_path": output_path,
        "saved_audio_path": saved_audio_path,
        "summary_text": summary_text,
        "saved_file": str(summary_dict.get("saved_file") or ""),
        "metadata_file": str(summary_dict.get("metadata_file") or ""),
        "script_file": str(summary_dict.get("script_file") or ""),
    }
