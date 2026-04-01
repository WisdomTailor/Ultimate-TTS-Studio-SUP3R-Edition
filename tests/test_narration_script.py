from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError


APP_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent / "golden_scripts"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from narration_script import (  # noqa: E402
    NarrationLine,
    NarrationScript,
    SemanticCue,
    migrate_v1_to_v2,
)


def build_sample_script() -> NarrationScript:
    return NarrationScript(
        lines=[
            NarrationLine(
                speaker="Narrator",
                text="A storm gathered over the harbor.",
                line_type="narration",
            ),
            NarrationLine(
                speaker="Alice",
                text="We need to leave now.",
                line_type="dialogue",
                cues=[SemanticCue.EMPHASIS],
                confidence=0.9,
            ),
            NarrationLine(
                speaker="Stage",
                text="Footsteps fade into the tunnel.",
                line_type="stage_direction",
                ambiguous=True,
            ),
            NarrationLine(
                speaker="Bob",
                text="I'm right behind you.",
                line_type="dialogue",
                cues=[SemanticCue.PAUSE],
            ),
        ],
        metadata={"source_format": "mixed"},
    )


class TestNarrationLineConstruction:
    def test_valid_narration_line(self) -> None:
        line = NarrationLine(
            speaker="Narrator",
            text="The hall fell silent.",
            line_type="narration",
            cues=[SemanticCue.WHISPER],
            confidence=0.75,
            ambiguous=False,
        )

        assert line.speaker == "Narrator"
        assert line.text == "The hall fell silent."
        assert line.line_type == "narration"
        assert line.cues == [SemanticCue.WHISPER]
        assert line.confidence == pytest.approx(0.75)
        assert line.ambiguous is False

    def test_valid_narration_script(self) -> None:
        script = build_sample_script()

        assert script.version == "1.0"
        assert len(script.lines) == 4
        assert script.metadata == {"source_format": "mixed"}


class TestNarrationValidation:
    @pytest.mark.parametrize("speaker", ["", "   ", "\t"])
    def test_empty_speaker_raises(self, speaker: str) -> None:
        with pytest.raises(ValidationError):
            NarrationLine(speaker=speaker, text="Hello", line_type="dialogue")

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_text_raises(self, text: str) -> None:
        with pytest.raises(ValidationError):
            NarrationLine(speaker="Alice", text=text, line_type="dialogue")

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_out_of_range_raises(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            NarrationLine(
                speaker="Alice",
                text="Hello",
                line_type="dialogue",
                confidence=confidence,
            )

    def test_empty_lines_list_raises(self) -> None:
        with pytest.raises(ValidationError):
            NarrationScript(lines=[])


class TestSemanticCue:
    def test_all_valid_enum_values_parse(self) -> None:
        line = NarrationLine(
            speaker="Actor",
            text="Take a breath.",
            line_type="dialogue",
            cues=["whisper", "pause", "emphasis", "emotional_beat"],
        )

        assert line.cues == [
            SemanticCue.WHISPER,
            SemanticCue.PAUSE,
            SemanticCue.EMPHASIS,
            SemanticCue.EMOTIONAL_BEAT,
        ]

    def test_invalid_enum_value_raises(self) -> None:
        with pytest.raises(ValidationError):
            NarrationLine(
                speaker="Actor",
                text="Take a breath.",
                line_type="dialogue",
                cues=["invalid_cue"],
            )


class TestSerialization:
    def test_dict_roundtrip_preserves_equality(self) -> None:
        script = build_sample_script()

        restored = NarrationScript.from_dict(script.to_dict())

        assert restored == script

    def test_json_roundtrip_preserves_equality(self) -> None:
        script = build_sample_script()

        restored = NarrationScript.from_json(script.to_json())

        assert restored == script


class TestLegacyConversion:
    def test_to_conversation_list_returns_legacy_shape(self) -> None:
        script = build_sample_script()

        items = script.to_conversation_list()

        assert items == [
            {"speaker": "Narrator", "text": "A storm gathered over the harbor."},
            {"speaker": "Alice", "text": "We need to leave now."},
            {"speaker": "Stage", "text": "Footsteps fade into the tunnel."},
            {"speaker": "Bob", "text": "I'm right behind you."},
        ]

    def test_from_conversation_list_roundtrips(self) -> None:
        items = [
            {"speaker": "Host", "text": "Welcome back."},
            {"speaker": "Guest", "text": "Thanks for having me."},
        ]

        script = NarrationScript.from_conversation_list(
            items,
            metadata={"source_format": "speaker_colon"},
        )

        assert script.to_conversation_list() == items
        assert [line.line_type for line in script.lines] == ["dialogue", "dialogue"]
        assert script.metadata == {"source_format": "speaker_colon"}


class TestGoldenScriptParsing:
    @pytest.mark.parametrize(
        "fixture_name",
        ["01_two_speaker_interview.json", "03_narrator_plus_dialogue.json"],
    )
    def test_expected_output_parses(self, fixture_name: str) -> None:
        fixture_path = GOLDEN_DIR / fixture_name
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))

        parsed = NarrationScript.from_dict(payload["expected_output"])

        assert parsed.version == "1.0"
        assert len(parsed.lines) >= 1
        assert parsed.to_dict() == payload["expected_output"]


class TestUtilityProperties:
    def test_speakers_returns_sorted_unique_values(self) -> None:
        script = build_sample_script()

        assert script.speakers == ["Alice", "Bob", "Narrator", "Stage"]

    def test_speaker_line_counts_returns_counts_by_speaker(self) -> None:
        script = build_sample_script()

        assert script.speaker_line_counts == {
            "Narrator": 1,
            "Alice": 1,
            "Stage": 1,
            "Bob": 1,
        }

    def test_dialogue_lines_filters_only_dialogue(self) -> None:
        script = build_sample_script()

        assert [line.speaker for line in script.dialogue_lines] == ["Alice", "Bob"]
        assert all(line.line_type == "dialogue" for line in script.dialogue_lines)


class TestMigrationStub:
    def test_migration_stub_returns_input_unchanged(self) -> None:
        payload = {"version": "1.0", "lines": [{"speaker": "A", "text": "Hi"}]}

        assert migrate_v1_to_v2(payload) is payload