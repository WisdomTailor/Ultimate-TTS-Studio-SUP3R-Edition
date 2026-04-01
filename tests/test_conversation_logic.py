from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from conversation_logic import (  # noqa: E402
    create_default_speaker_settings,
    format_conversation_info,
    get_speaker_names_from_script,
    parse_conversation_script,
    parse_to_narration_script,
)
from narration_script import NarrationScript  # noqa: E402


def test_parse_conversation_script_parses_two_speakers() -> None:
    conversation, error = parse_conversation_script("Alice: Hello\nBob: Hi")

    assert error is None
    assert conversation == [
        {"speaker": "Alice", "text": "Hello"},
        {"speaker": "Bob", "text": "Hi"},
    ]


def test_parse_conversation_script_handles_continuation_lines() -> None:
    conversation, error = parse_conversation_script("Alice: Hello\nthere\nBob: Hi")

    assert error is None
    assert conversation == [
        {"speaker": "Alice", "text": "Hello there"},
        {"speaker": "Bob", "text": "Hi"},
    ]


def test_parse_conversation_script_empty_input_returns_empty_list() -> None:
    conversation, error = parse_conversation_script("")

    assert error is None
    assert conversation == []


def test_parse_conversation_script_invalid_input_returns_error() -> None:
    conversation, error = parse_conversation_script(None)  # type: ignore[arg-type]

    assert conversation == []
    assert error is not None
    assert "Error parsing conversation" in error


def test_get_speaker_names_from_script_returns_sorted_unique_names() -> None:
    speakers = get_speaker_names_from_script("Bob: Hi\nAlice: Hello\nBob: Again")

    assert speakers == ["Alice", "Bob"]


def test_create_default_speaker_settings_creates_entries_for_all_speakers() -> None:
    settings = create_default_speaker_settings(["Alice", "Bob"])

    assert set(settings) == {"Alice", "Bob"}
    assert settings["Alice"]["tts_engine"] == "chatterbox"
    assert settings["Bob"]["kokoro_voice"] == "af_heart"


def test_format_conversation_info_formats_summary_dict() -> None:
    summary = {
        "saved_file": "outputs/test.wav",
        "engine_used": "ChatterboxTTS",
        "total_lines": 2,
        "unique_speakers": 2,
        "total_duration": 3.25,
        "speakers": ["Alice", "Bob"],
        "conversation_info": [
            {"speaker": "Alice", "text": "Hello", "duration": 1.5},
            {"speaker": "Bob", "text": "Hi", "duration": 1.75},
        ],
    }

    formatted = format_conversation_info(summary)

    assert "Conversation Generated Successfully" in formatted
    assert "outputs/test.wav" in formatted
    assert '1. Alice: "Hello" (1.5s)' in formatted


def test_format_conversation_info_returns_string_input_unchanged() -> None:
    assert format_conversation_info("ready") == "ready"


def test_parse_to_narration_script_returns_script_for_valid_input() -> None:
    script, error = parse_to_narration_script("Alice: Hello\nBob: Hi")

    assert error is None
    assert isinstance(script, NarrationScript)
    assert script.to_conversation_list() == [
        {"speaker": "Alice", "text": "Hello"},
        {"speaker": "Bob", "text": "Hi"},
    ]


def test_parse_to_narration_script_returns_error_for_invalid_script() -> None:
    script, error = parse_to_narration_script("")

    assert script is None
    assert error == "No valid conversation found in script"


def test_parse_to_narration_script_attaches_metadata() -> None:
    script, error = parse_to_narration_script(
        "Alice: Hello",
        metadata={"source_format": "speaker_colon", "scene": "intro"},
    )

    assert error is None
    assert script is not None
    assert script.metadata == {"source_format": "speaker_colon", "scene": "intro"}