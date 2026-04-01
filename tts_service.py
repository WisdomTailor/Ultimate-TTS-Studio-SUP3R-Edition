"""TTS service layer with zero Gradio imports.

Provides a pure-Python contract for MCP tools and other headless consumers.
Engine handlers are imported lazily to avoid loading model dependencies at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import inspect
from pathlib import Path
from typing import Any, Optional

import numpy as np

from engine_registry import ENGINE_EXPRESSIVENESS, ENGINE_METADATA_CONTROL_MAP


KOKORO_CHOICES: dict[str, str] = {
    "US Heart": "af_heart",
    "US Bella": "af_bella",
    "US Nicole": "af_nicole",
    "US Aoede": "af_aoede",
    "US Kore": "af_kore",
    "US Sarah": "af_sarah",
    "US Nova": "af_nova",
    "US Sky": "af_sky",
    "US Alloy": "af_alloy",
    "US Jessica": "af_jessica",
    "US River": "af_river",
    "US Michael": "am_michael",
    "US Fenrir": "am_fenrir",
    "US Puck": "am_puck",
    "US Echo": "am_echo",
    "US Eric": "am_eric",
    "US Liam": "am_liam",
    "US Onyx": "am_onyx",
    "US Santa": "am_santa",
    "US Adam": "am_adam",
    "UK Emma": "bf_emma",
    "UK Isabella": "bf_isabella",
    "UK Alice": "bf_alice",
    "UK Lily": "bf_lily",
    "UK George": "bm_george",
    "UK Fable": "bm_fable",
    "UK Lewis": "bm_lewis",
    "UK Daniel": "bm_daniel",
    "PF Dora": "pf_dora",
    "PM Alex": "pm_alex",
    "PM Santa": "pm_santa",
    "IT Sara": "if_sara",
    "IT Nicola": "im_nicola",
}

KITTEN_VOICES: list[str] = [
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
]

QWEN_SPEAKERS: list[str] = [
    "Aiden",
    "Dylan",
    "Eric",
    "Ono_anna",
    "Ryan",
    "Serena",
    "Sohee",
    "Uncle_fu",
    "Vivian",
]


ENGINE_HANDLERS: dict[str, tuple[str, str]] = {
    "Chatterbox Turbo": ("chatterbox_turbo_handler", "generate_chatterbox_turbo_tts"),
    "Higgs Audio": ("higgs_audio_handler", "generate_higgs_audio_tts"),
    "IndexTTS2": ("indextts2_handler", "generate_indextts2_tts"),
    "KittenTTS": ("kitten_tts_handler", "generate_kitten_tts"),
    "Qwen Voice Design": ("qwen_tts_handler", "generate_qwen_voice_design_tts"),
    "Qwen Voice Clone": ("qwen_tts_handler", "generate_qwen_voice_clone_tts"),
    "Qwen Custom Voice": ("qwen_tts_handler", "generate_qwen_custom_voice_tts"),
    "VoxCPM": ("voxcpm_handler", "generate_voxcpm_tts"),
}

ENGINE_METHOD_HANDLERS: dict[str, tuple[str, str, str]] = {
    "F5-TTS": ("f5_tts_handler", "get_f5_tts_handler", "generate_speech"),
}

ENGINE_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "F5-TTS": {
        "reference_audio": "ref_audio_path",
        "reference_text": "ref_text",
        "cross_fade": "cross_fade_duration",
    },
    "Qwen Voice Clone": {
        "reference_audio": "ref_audio",
        "reference_text": "ref_text",
        "xvector_only": "use_xvector_only",
    },
    "Qwen Custom Voice": {
        "speaker_profile": "speaker",
        "style_instruct": "instruct",
    },
}

REFERENCE_AUDIO_ENGINES: set[str] = {
    "ChatterboxTTS",
    "Chatterbox Multilingual",
    "Chatterbox Turbo",
    "Fish Speech",
    "IndexTTS",
    "IndexTTS2",
    "F5-TTS",
    "Higgs Audio",
    "VoxCPM",
    "Qwen Voice Clone",
}

SERVICE_AVAILABLE_ENGINES: set[str] = set(ENGINE_HANDLERS) | set(ENGINE_METHOD_HANDLERS)


@dataclass
class TtsRequest:
    """Parameters for a TTS generation request."""

    text: str
    engine: str
    audio_format: str = "wav"
    engine_params: dict[str, Any] = field(default_factory=dict)
    effects: dict[str, Any] = field(default_factory=dict)


@dataclass
class TtsResult:
    """Result of a TTS generation."""

    audio: Optional[tuple[int, np.ndarray]] = None
    status: str = ""
    output_path: Optional[str] = None


def list_engines() -> list[dict[str, Any]]:
    """List all available TTS engines with capability metadata.

    Returns:
        List of dicts with engine names, capabilities, and service-layer availability.
    """
    return [get_engine_info(engine_name) for engine_name in ENGINE_EXPRESSIVENESS]


def get_engine_info(engine_name: str) -> dict[str, Any]:
    """Get detailed info for a specific engine.

    Args:
        engine_name: Engine identifier.

    Returns:
        Dict with capabilities, parameter schema, and service-layer status.

    Raises:
        ValueError: If engine_name is not recognized.
    """
    if engine_name not in ENGINE_EXPRESSIVENESS:
        raise ValueError(f"Unknown engine: {engine_name}")

    capabilities = dict(ENGINE_EXPRESSIVENESS[engine_name])
    supported_cues = [name for name, enabled in capabilities.items() if enabled]
    controls = ENGINE_METADATA_CONTROL_MAP.get(engine_name, [])
    parameter_schema = [
        {
            "ui_key": ui_key,
            "parameter": _service_parameter_name(engine_name, parameter_name),
        }
        for ui_key, parameter_name in controls
    ]
    service_available = engine_name in SERVICE_AVAILABLE_ENGINES

    return {
        "name": engine_name,
        "display_name": engine_name,
        "capabilities": capabilities,
        "supported_cues": supported_cues,
        "supports_expressiveness": bool(supported_cues),
        "voice_mode": _voice_mode_for_engine(engine_name),
        "parameter_schema": parameter_schema,
        "service_layer_available": service_available,
        "service_layer_status": "available" if service_available else "not_yet_extracted",
    }


def list_voices(engine_name: str | None = None) -> list[dict[str, Any]]:
    """List available voices, optionally filtered by engine.

    Args:
        engine_name: If provided, list voices only for this engine.

    Returns:
        List of voice dicts with id, name, engine, and type fields.

    Raises:
        ValueError: If engine_name is not recognized.
    """
    if engine_name is not None and engine_name not in ENGINE_EXPRESSIVENESS:
        raise ValueError(f"Unknown engine: {engine_name}")

    voices: list[dict[str, Any]] = []

    if engine_name in (None, "Kokoro TTS"):
        voices.extend(_kokoro_voices())
        voices.extend(_custom_kokoro_voices())

    if engine_name in (None, "KittenTTS"):
        voices.extend(
            {
                "id": voice_id,
                "name": voice_id,
                "engine": "KittenTTS",
                "type": "built_in",
            }
            for voice_id in KITTEN_VOICES
        )

    if engine_name in (None, "Qwen Custom Voice"):
        voices.extend(
            {
                "id": speaker,
                "name": speaker,
                "engine": "Qwen Custom Voice",
                "type": "speaker_profile",
            }
            for speaker in QWEN_SPEAKERS
        )

    if engine_name is not None and not voices:
        indicator = _voice_indicator(engine_name)
        return [indicator] if indicator is not None else []

    return voices


def get_output_dir(kind: str = "outputs") -> Path:
    """Get the output directory path.

    Args:
        kind: Either "outputs" or "audiobooks".

    Returns:
        Path to the output directory, created if needed.

    Raises:
        ValueError: If kind is not supported.
    """
    if kind not in {"outputs", "audiobooks"}:
        raise ValueError(f"Unsupported output directory kind: {kind}")

    output_dir = Path.cwd() / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def list_outputs(
    output_dir: str | Path | None = None,
    extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg"),
) -> list[dict[str, Any]]:
    """List generated audio files in an output directory.

    Args:
        output_dir: Directory to scan. Defaults to get_output_dir("outputs").
        extensions: Audio file extensions to include.

    Returns:
        List of dicts with path, filename, size_bytes, and modified_at fields.
    """
    directory = Path(output_dir) if output_dir is not None else get_output_dir("outputs")
    if not directory.exists():
        return []

    allowed_extensions = {suffix.lower() for suffix in extensions}
    files = [
        file_path
        for file_path in directory.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions
    ]
    files.sort(key=lambda file_path: file_path.stat().st_mtime, reverse=True)

    return [
        {
            "path": str(file_path.resolve()),
            "filename": file_path.name,
            "size_bytes": file_path.stat().st_size,
            "modified_at": file_path.stat().st_mtime,
        }
        for file_path in files
    ]


def generate_tts(request: TtsRequest) -> TtsResult:
    """Dispatch TTS generation to an extracted engine handler.

    Args:
        request: TTS generation request with engine and parameters.

    Returns:
        TtsResult with audio data and status.
    """
    if request.engine not in ENGINE_EXPRESSIVENESS:
        return TtsResult(status=f"Unknown engine '{request.engine}'")

    if request.engine not in SERVICE_AVAILABLE_ENGINES:
        return TtsResult(status=f"Engine '{request.engine}' not yet available via service layer")

    if not request.text.strip():
        return TtsResult(status="No text provided for synthesis")

    try:
        callable_obj = _load_engine_callable(request.engine)
        kwargs = _filter_kwargs(callable_obj, _request_kwargs(request))
        payload, status = callable_obj(**kwargs)
        return _coerce_result(payload, status)
    except ImportError as error:
        return TtsResult(status=f"Engine '{request.engine}' unavailable: {error}")
    except Exception as error:
        return TtsResult(status=f"Engine '{request.engine}' failed: {error}")


def _coerce_result(payload: Any, status: str) -> TtsResult:
    if _is_audio_tuple(payload):
        sample_rate, audio_data = payload
        return TtsResult(audio=(int(sample_rate), np.asarray(audio_data)), status=status)

    if isinstance(payload, (str, Path)):
        return TtsResult(status=status, output_path=str(payload))

    return TtsResult(status=status)


def _custom_kokoro_voices() -> list[dict[str, Any]]:
    voices_dir = Path.cwd() / "custom_voices"
    if not voices_dir.exists():
        return []

    return [
        {
            "id": f"custom_{voice_path.stem}",
            "name": f"Custom: {voice_path.stem}",
            "engine": "Kokoro TTS",
            "type": "custom",
        }
        for voice_path in sorted(voices_dir.glob("*.pt"), key=lambda path: path.name.lower())
        if voice_path.is_file()
    ]


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs

    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }


def _import_module(module_name: str) -> Any:
    return import_module(module_name)


def _is_audio_tuple(payload: Any) -> bool:
    if not isinstance(payload, tuple) or len(payload) != 2:
        return False

    sample_rate, audio_data = payload
    return isinstance(sample_rate, (int, np.integer)) and hasattr(audio_data, "__array__")


def _kokoro_voices() -> list[dict[str, Any]]:
    return [
        {
            "id": voice_id,
            "name": voice_name,
            "engine": "Kokoro TTS",
            "type": "built_in",
        }
        for voice_name, voice_id in KOKORO_CHOICES.items()
    ]


def _load_engine_callable(engine_name: str) -> Any:
    if engine_name in ENGINE_HANDLERS:
        module_name, function_name = ENGINE_HANDLERS[engine_name]
        module = _import_module(module_name)
        return getattr(module, function_name)

    module_name, getter_name, method_name = ENGINE_METHOD_HANDLERS[engine_name]
    module = _import_module(module_name)
    handler = getattr(module, getter_name)()
    return getattr(handler, method_name)


def _request_kwargs(request: TtsRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "text": request.text,
        "audio_format": request.audio_format,
        "effects_settings": request.effects or None,
        "skip_file_saving": request.engine_params.get("skip_file_saving", True),
    }
    kwargs.update(request.engine_params)

    for source_name, target_name in ENGINE_PARAM_ALIASES.get(request.engine, {}).items():
        if source_name in kwargs and target_name not in kwargs:
            kwargs[target_name] = kwargs.pop(source_name)

    return kwargs


def _service_parameter_name(engine_name: str, parameter_name: str) -> str:
    return ENGINE_PARAM_ALIASES.get(engine_name, {}).get(parameter_name, parameter_name)


def _voice_indicator(engine_name: str) -> dict[str, Any] | None:
    if engine_name in REFERENCE_AUDIO_ENGINES:
        return {
            "id": "reference_audio",
            "name": "Reference audio provided at request time",
            "engine": engine_name,
            "type": "reference_audio",
        }

    if engine_name == "Qwen Voice Design":
        return {
            "id": "voice_description",
            "name": "Voice defined by text description at request time",
            "engine": engine_name,
            "type": "voice_description",
        }

    if engine_name == "VibeVoice":
        return {
            "id": "multi_speaker_assignment",
            "name": "Speaker voices supplied at request time",
            "engine": engine_name,
            "type": "multi_speaker",
        }

    return None


def _voice_mode_for_engine(engine_name: str) -> str:
    if engine_name == "Kokoro TTS":
        return "named_voices"
    if engine_name == "KittenTTS":
        return "named_voices"
    if engine_name == "Qwen Custom Voice":
        return "speaker_profiles"
    if engine_name == "Qwen Voice Design":
        return "voice_description"
    if engine_name == "VibeVoice":
        return "multi_speaker"
    if engine_name in REFERENCE_AUDIO_ENGINES:
        return "reference_audio"
    return "none"


__all__ = [
    "ENGINE_HANDLERS",
    "ENGINE_METHOD_HANDLERS",
    "KOKORO_CHOICES",
    "QWEN_SPEAKERS",
    "TtsRequest",
    "TtsResult",
    "generate_tts",
    "get_engine_info",
    "get_output_dir",
    "list_engines",
    "list_outputs",
    "list_voices",
]