from __future__ import annotations

import inspect
import sys
from pathlib import Path

import conversation_logic as conversation_logic_module


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from conversation_logic import (  # noqa: E402
    CONVERSATION_FORMATTER_SYSTEM_PROMPT,
    PerLineTransformSettings,
    _extract_json_from_llm_response,
    apply_per_line_transform,
    create_default_speaker_settings,
    format_conversation_with_llm,
    format_conversation_info,
    get_speaker_names_from_script,
    parse_conversation_script,
    parse_to_narration_script,
)
from narration_script import NarrationLine, NarrationScript, SemanticCue  # noqa: E402
from pronunciation import PronunciationOverride, ProtectedTerm  # noqa: E402


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


class TestAIConversationFormatter:
    """Tests for LLM-powered conversation formatting."""

    def test_extract_json_from_direct_json(self) -> None:
        """Direct JSON string parses correctly."""
        payload = '{"version": "1.0", "lines": [{"speaker": "Narrator", "text": "Hello", "line_type": "narration", "cues": [], "confidence": 1.0, "ambiguous": false}], "metadata": {}}'

        parsed = _extract_json_from_llm_response(payload)

        assert parsed["version"] == "1.0"
        assert parsed["lines"][0]["speaker"] == "Narrator"

    def test_extract_json_strips_markdown_fences(self) -> None:
        """JSON wrapped in ```json ... ``` fences is extracted."""
        payload = """```json
{"version": "1.0", "lines": [{"speaker": "Narrator", "text": "Hello", "line_type": "narration", "cues": [], "confidence": 1.0, "ambiguous": false}], "metadata": {}}
```"""

        parsed = _extract_json_from_llm_response(payload)

        assert parsed["lines"][0]["text"] == "Hello"

    def test_extract_json_finds_embedded_json(self) -> None:
        """JSON embedded in explanation text is found."""
        payload = (
            "Here is the structured result:\n"
            '{"version": "1.0", "lines": [{"speaker": "Alice", "text": "Hi", '
            '"line_type": "dialogue", "cues": [], "confidence": 0.8, "ambiguous": false}], '
            '"metadata": {}}\nThanks.'
        )

        parsed = _extract_json_from_llm_response(payload)

        assert parsed["lines"][0]["speaker"] == "Alice"

    def test_extract_json_raises_on_invalid(self) -> None:
        """Non-JSON input raises ValueError."""
        try:
            _extract_json_from_llm_response("not json")
        except ValueError as error:
            assert "JSON" in str(error)
        else:
            raise AssertionError("Expected ValueError for invalid JSON input")

    def test_format_conversation_with_llm_returns_error_on_empty_text(self) -> None:
        """Empty text input returns error tuple."""
        script, error = format_conversation_with_llm(
            text="   ",
            base_url="http://localhost:1234/v1",
            api_key="",
            model_id="test-model",
        )

        assert script is None
        assert error == "Conversation text is required"

    def test_conversation_formatter_system_prompt_exists(self) -> None:
        """System prompt constant is defined and non-empty."""
        assert isinstance(CONVERSATION_FORMATTER_SYSTEM_PROMPT, str)
        assert CONVERSATION_FORMATTER_SYSTEM_PROMPT.strip()
        assert "Return ONLY a valid JSON object" in CONVERSATION_FORMATTER_SYSTEM_PROMPT

    def test_format_conversation_with_llm_signature(self) -> None:
        """Function has expected signature with all required params."""
        signature = inspect.signature(format_conversation_with_llm)
        params = signature.parameters

        assert list(params) == [
            "text",
            "base_url",
            "api_key",
            "model_id",
            "timeout_seconds",
            "extra_headers",
            "auth_style",
        ]
        assert params["timeout_seconds"].default == 120
        assert params["extra_headers"].default is None
        assert params["auth_style"].default == "bearer"


class TestApplyPerLineTransform:
    def test_apply_per_line_transform_with_mock_llm(self, monkeypatch) -> None:
        script = NarrationScript(
            lines=[
                NarrationLine(speaker="Alice", text="hello there", line_type="dialogue"),
                NarrationLine(speaker="Bob", text="general kenobi", line_type="dialogue"),
            ],
            metadata={"scene": "intro"},
        )

        def mock_transform(**kwargs) -> tuple[str, str]:
            return kwargs["source_text"].upper(), "mock transform applied"

        monkeypatch.setattr(
            conversation_logic_module,
            "apply_llm_narration_transform",
            mock_transform,
        )

        transformed_script, status_messages = apply_per_line_transform(
            script,
            enabled=True,
            provider_name="test-provider",
            base_url="http://localhost:1234/v1",
            api_key="",
            model_id="test-model",
        )

        assert [line.text for line in transformed_script.lines] == ["HELLO THERE", "GENERAL KENOBI"]
        assert transformed_script.metadata == {"scene": "intro"}
        assert len(status_messages) == 2
        assert "mock transform applied" in status_messages[0]

    def test_apply_per_line_transform_uses_per_speaker_overrides(self, monkeypatch) -> None:
        script = NarrationScript(
            lines=[
                NarrationLine(speaker="Alice", text="hello", line_type="dialogue"),
                NarrationLine(speaker="Bob", text="hi", line_type="dialogue"),
            ]
        )
        received_settings: list[tuple[str, str, str]] = []

        def mock_transform(**kwargs) -> tuple[str, str]:
            received_settings.append((kwargs["mode"], kwargs["style"], kwargs["locale"]))
            return kwargs["source_text"], "settings captured"

        monkeypatch.setattr(
            conversation_logic_module,
            "apply_llm_narration_transform",
            mock_transform,
        )

        transformed_script, _ = apply_per_line_transform(
            script,
            provider_name="test-provider",
            base_url="http://localhost:1234/v1",
            model_id="test-model",
            mode="minimal",
            style="conversational",
            locale="en-US",
            speaker_settings={
                "Alice": PerLineTransformSettings(
                    mode="vivid",
                    style="formal",
                    locale="en-GB",
                ),
                "Bob": PerLineTransformSettings(style="casual"),
            },
        )

        assert [line.text for line in transformed_script.lines] == ["hello", "hi"]
        assert received_settings == [
            ("vivid", "formal", "en-GB"),
            ("minimal", "casual", "en-US"),
        ]

    def test_apply_per_line_transform_integrates_pronunciation_pipeline(self, monkeypatch) -> None:
        script = NarrationScript(
            lines=[
                NarrationLine(
                    speaker="Narrator",
                    text="Hermione met Nguyen.",
                    line_type="narration",
                )
            ]
        )

        def mock_transform(**kwargs) -> tuple[str, str]:
            return kwargs["source_text"].replace("met", "greeted"), "pronunciation test"

        monkeypatch.setattr(
            conversation_logic_module,
            "apply_llm_narration_transform",
            mock_transform,
        )

        transformed_script, _ = apply_per_line_transform(
            script,
            provider_name="test-provider",
            base_url="http://localhost:1234/v1",
            model_id="test-model",
            protected_terms=[ProtectedTerm(term="Hermione"), ProtectedTerm(term="Nguyen")],
            pronunciation_overrides=[
                PronunciationOverride(word="Hermione", phonetic="Her-MY-oh-nee"),
                PronunciationOverride(word="Nguyen", phonetic="Win"),
            ],
        )

        assert transformed_script.lines[0].text == "Her-MY-oh-nee greeted Win."

    def test_apply_per_line_transform_reports_progress_per_line(self, monkeypatch) -> None:
        script = NarrationScript(
            lines=[
                NarrationLine(speaker="Alice", text="one", line_type="dialogue"),
                NarrationLine(speaker="Bob", text="two", line_type="dialogue"),
            ]
        )
        progress_events: list[tuple[int, int, str]] = []

        def mock_transform(**kwargs) -> tuple[str, str]:
            return kwargs["source_text"], "progress test"

        monkeypatch.setattr(
            conversation_logic_module,
            "apply_llm_narration_transform",
            mock_transform,
        )

        apply_per_line_transform(
            script,
            provider_name="test-provider",
            base_url="http://localhost:1234/v1",
            model_id="test-model",
            progress_callback=lambda current, total, speaker: progress_events.append(
                (current, total, speaker)
            ),
        )

        assert progress_events == [(1, 2, "Alice"), (2, 2, "Bob")]

    def test_apply_per_line_transform_returns_empty_script_for_empty_input(
        self, monkeypatch
    ) -> None:
        empty_script = NarrationScript.model_construct(
            version="1.0",
            lines=[],
            metadata={"scene": "empty"},
        )

        def unexpected_transform(**kwargs) -> tuple[str, str]:
            raise AssertionError("Transform should not be called for an empty script")

        monkeypatch.setattr(
            conversation_logic_module,
            "apply_llm_narration_transform",
            unexpected_transform,
        )

        transformed_script, status_messages = apply_per_line_transform(empty_script)

        assert transformed_script.lines == []
        assert transformed_script.metadata == {"scene": "empty"}
        assert status_messages == []

    def test_apply_per_line_transform_disabled_uses_deterministic_and_pronunciation(self) -> None:
        script = NarrationScript(
            lines=[
                NarrationLine(
                    speaker="Narrator",
                    text="Call 555-123-4567, Nguyen.",
                    line_type="narration",
                )
            ]
        )

        transformed_script, status_messages = apply_per_line_transform(
            script,
            enabled=False,
            pronunciation_overrides=[PronunciationOverride(word="Nguyen", phonetic="Win")],
        )

        assert (
            transformed_script.lines[0].text
            == "Call five five five, one two three, four five six seven, Win."
        )
        assert "LLM disabled" in status_messages[0]

    def test_apply_per_line_transform_preserves_non_text_line_fields(self, monkeypatch) -> None:
        script = NarrationScript(
            lines=[
                NarrationLine(
                    speaker="Narrator",
                    text="Pause here.",
                    line_type="narration",
                    cues=[SemanticCue.PAUSE, SemanticCue.EMPHASIS],
                    confidence=0.75,
                    ambiguous=True,
                )
            ],
            metadata={"scene": "dramatic"},
        )

        def mock_transform(**kwargs) -> tuple[str, str]:
            return kwargs["source_text"] + " Again.", "field preservation test"

        monkeypatch.setattr(
            conversation_logic_module,
            "apply_llm_narration_transform",
            mock_transform,
        )

        transformed_script, _ = apply_per_line_transform(
            script,
            provider_name="test-provider",
            base_url="http://localhost:1234/v1",
            model_id="test-model",
        )
        transformed_line = transformed_script.lines[0]

        assert transformed_line.text == "Pause here. Again."
        assert transformed_line.speaker == "Narrator"
        assert transformed_line.line_type == "narration"
        assert transformed_line.cues == [SemanticCue.PAUSE, SemanticCue.EMPHASIS]
        assert transformed_line.confidence == 0.75
        assert transformed_line.ambiguous is True
