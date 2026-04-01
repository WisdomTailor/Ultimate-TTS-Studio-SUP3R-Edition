import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SemanticCue(str, Enum):
    """Semantic annotations that downstream narration engines may interpret."""

    WHISPER = "whisper"
    PAUSE = "pause"
    EMPHASIS = "emphasis"
    EMOTIONAL_BEAT = "emotional_beat"


class NarrationLine(BaseModel):
    """One structured narration or dialogue line.

    Attributes:
        speaker: Display name for the speaker or narrator.
        text: Spoken or narrated content for the line.
        line_type: Semantic category for the line.
        cues: Optional semantic cues associated with the line.
        confidence: Parser confidence score in the range [0.0, 1.0].
        ambiguous: Whether speaker attribution is ambiguous.
    """

    speaker: str
    text: str
    line_type: Literal["dialogue", "narration", "stage_direction"]
    cues: list[SemanticCue] = Field(default_factory=list)
    confidence: float = 1.0
    ambiguous: bool = False

    @field_validator("speaker", "text")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Validate that speaker and text are not empty or whitespace-only."""
        if not value.strip():
            raise ValueError("Value must be a non-empty, non-whitespace string")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        """Validate confidence falls within the supported range."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return value


class NarrationScript(BaseModel):
    """Structured narration script with versioning and serialization helpers."""

    version: str = "1.0"
    lines: list[NarrationLine] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            dict[str, Any]: Script data suitable for JSON serialization.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NarrationScript":
        """Parse a dictionary into a narration script.

        Args:
            data: Dictionary representation of a narration script.

        Returns:
            NarrationScript: Parsed narration script model.
        """
        return cls.model_validate(data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the script to a JSON string.

        Args:
            indent: Indentation level for the generated JSON.

        Returns:
            str: JSON representation of the narration script.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "NarrationScript":
        """Parse a JSON string into a narration script.

        Args:
            json_str: Serialized narration script JSON.

        Returns:
            NarrationScript: Parsed narration script model.
        """
        return cls.from_dict(json.loads(json_str))

    def to_conversation_list(self) -> list[dict[str, str]]:
        """Convert the script to the legacy speaker/text list format.

        Returns:
            list[dict[str, str]]: Legacy conversation items with speaker and text keys.
        """
        return [{"speaker": line.speaker, "text": line.text} for line in self.lines]

    @classmethod
    def from_conversation_list(
        cls,
        items: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> "NarrationScript":
        """Build a narration script from the legacy conversation format.

        Args:
            items: Legacy items containing speaker and text keys.
            metadata: Optional metadata to attach to the resulting script.

        Returns:
            NarrationScript: Parsed narration script with dialogue line defaults.
        """
        lines = [
            NarrationLine(
                speaker=item["speaker"],
                text=item["text"],
                line_type="dialogue",
            )
            for item in items
        ]
        return cls(lines=lines, metadata=metadata or {})

    @property
    def speakers(self) -> list[str]:
        """Return the sorted unique list of speaker names."""
        return sorted({line.speaker for line in self.lines})

    @property
    def speaker_line_counts(self) -> dict[str, int]:
        """Return a mapping of speaker name to number of lines."""
        counts: dict[str, int] = {}
        for line in self.lines:
            counts[line.speaker] = counts.get(line.speaker, 0) + 1
        return counts

    @property
    def dialogue_lines(self) -> list[NarrationLine]:
        """Return only dialogue lines from the script."""
        return [line for line in self.lines if line.line_type == "dialogue"]


def migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Return unchanged v1 data until a v2 narration schema is defined.

    This stub exists so future schema migrations can be introduced without changing
    current callers or backfilling a migration entry point later.

    Args:
        data: Existing version 1 narration payload.

    Returns:
        dict[str, Any]: The unmodified input payload.
    """
    return data
