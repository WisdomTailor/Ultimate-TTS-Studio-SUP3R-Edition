from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from engine_registry import (  # noqa: E402
    ENGINE_EXPRESSIVENESS,
    ENGINE_METADATA_CONTROL_MAP,
    _DEFAULT_ENGINE_EXPRESSIVENESS,
    strip_unsupported_cues,
)


def test_engine_expressiveness_contains_all_expected_engines() -> None:
    expected_engines = {
        "ChatterboxTTS",
        "Chatterbox Multilingual",
        "Chatterbox Turbo",
        "Kokoro TTS",
        "Fish Speech",
        "IndexTTS",
        "IndexTTS2",
        "F5-TTS",
        "Higgs Audio",
        "VoxCPM",
        "KittenTTS",
        "Qwen Voice Design",
        "Qwen Voice Clone",
        "Qwen Custom Voice",
        "VibeVoice",
    }

    assert set(ENGINE_EXPRESSIVENESS) == expected_engines


def test_unknown_engine_falls_back_to_default_capabilities() -> None:
    stripped = strip_unsupported_cues("[whispers] HELLO", "UnknownEngine")

    assert stripped == "Hello"
    assert _DEFAULT_ENGINE_EXPRESSIVENESS["bracket_cues"] is False
    assert _DEFAULT_ENGINE_EXPRESSIVENESS["emotion_vectors"] is False


def test_strip_unsupported_cues_removes_brackets_when_not_supported() -> None:
    assert strip_unsupported_cues("[whispers] Hello there", "ChatterboxTTS") == "Hello there"


def test_strip_unsupported_cues_preserves_brackets_when_supported() -> None:
    assert strip_unsupported_cues("[whispers] Hello there", "IndexTTS2") == "[whispers] Hello there"


def test_engine_metadata_control_map_has_known_entries() -> None:
    assert ENGINE_METADATA_CONTROL_MAP["ChatterboxTTS"][0] == (
        "chatterbox_exaggeration",
        "exaggeration",
    )
    assert ("indextts2_emotion_mode", "emotion_mode") in ENGINE_METADATA_CONTROL_MAP["IndexTTS2"]
    assert ("qwen_language", "language") in ENGINE_METADATA_CONTROL_MAP["Qwen Voice Clone"]
