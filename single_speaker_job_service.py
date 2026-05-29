"""Background worker helpers for queued single-speaker synthesis jobs."""

from __future__ import annotations

from typing import Any


def _extract_output_path(status_text: str) -> str:
    for line in str(status_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("Autosave audio: "):
            return stripped.split(": ", 1)[1].strip()
    return ""


def generate_single_speaker_job(request_dict: dict[str, Any]) -> dict[str, Any]:
    """Execute a queued single-speaker job via the main launch wrapper."""
    from launch import generate_unified_tts_wrapped

    engine_params = dict(request_dict.get("engine_params") or {})
    wrapped_args = list(engine_params.get("wrapped_args") or [])
    if not wrapped_args:
        raise ValueError("Queued single-speaker job is missing wrapped arguments")

    generation_output, status_text, seed_label, used_seed = generate_unified_tts_wrapped(*wrapped_args)
    if generation_output is None:
        raise RuntimeError(str(status_text or "Single-speaker generation failed"))

    result_payload: dict[str, Any] = {
        "job_type": "single_speaker",
        "status": str(status_text or "SUCCESS"),
        "output_path": _extract_output_path(status_text),
        "seed_label": str(seed_label or ""),
    }
    if used_seed is not None:
        result_payload["used_seed"] = used_seed
    if isinstance(generation_output, tuple) and len(generation_output) == 2:
        sample_rate, _audio = generation_output
        result_payload["sample_rate"] = sample_rate
        result_payload["audio_format"] = request_dict.get("audio_format", "wav")
    return result_payload