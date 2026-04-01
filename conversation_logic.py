"""Conversation parsing and NarrationScript bridging helpers.

This module contains pure conversation utilities that can be imported independently
from the Gradio UI layer.
"""

from __future__ import annotations

from typing import Any

from narration_script import NarrationScript


def parse_conversation_script(script_text: str) -> tuple[list[dict[str, str]], str | None]:
    """Parse conversation script text in Speaker: Text format."""
    try:
        lines = script_text.strip().split("\n")
        conversation: list[dict[str, str]] = []
        current_speaker: str | None = None
        current_text = ""

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if ":" in line and not line.startswith(" "):
                if current_speaker and current_text:
                    conversation.append({"speaker": current_speaker, "text": current_text.strip()})

                parts = line.split(":", 1)
                if len(parts) == 2:
                    current_speaker = parts[0].strip()
                    current_text = parts[1].strip()
                else:
                    current_text += " " + line
            else:
                current_text += " " + line

        if current_speaker and current_text:
            conversation.append({"speaker": current_speaker, "text": current_text.strip()})

        return conversation, None
    except Exception as error:
        return [], f"Error parsing conversation: {str(error)}"


def get_speaker_names_from_script(script_text: str) -> list[str]:
    """Return sorted unique speaker names from a parsed conversation script."""
    conversation, error = parse_conversation_script(script_text)
    if error:
        return []

    return sorted({item["speaker"] for item in conversation})


def create_default_speaker_settings(speakers: list[str]) -> dict[str, dict[str, Any]]:
    """Create default per-speaker synthesis settings for conversation mode."""
    default_settings: dict[str, dict[str, Any]] = {}

    for speaker in speakers:
        default_settings[speaker] = {
            "ref_audio": "",
            "tts_engine": "chatterbox",
            "exaggeration": 0.5,
            "temperature": 0.8,
            "cfg_weight": 0.5,
            "kokoro_voice": "af_heart",
            "kokoro_speed": 1.0,
            "fish_ref_text": "",
            "fish_temperature": 0.8,
            "fish_top_p": 0.8,
            "fish_repetition_penalty": 1.1,
            "fish_max_tokens": 1024,
            "fish_seed": None,
        }

    return default_settings


def format_conversation_info(summary: dict[str, Any] | str) -> str:
    """Format a conversation-generation summary for display."""
    if isinstance(summary, str):
        return summary

    saved_file = summary.get("saved_file", "conversation_audio")
    engine_used = summary.get("engine_used", "Unknown")

    info_text = f"""🎭 **Conversation Generated Successfully!**

💾 **File Saved:** {saved_file}
🎵 **Engine Used:** {engine_used}

📊 **Summary:**
• Total Lines: {summary['total_lines']} | Speakers: {summary['unique_speakers']} | Duration: {summary['total_duration']:.1f}s
• Speakers: {', '.join(summary['speakers'])}

📝 **Line Breakdown:**"""

    for index, line_info in enumerate(summary["conversation_info"], 1):
        speaker = line_info["speaker"]
        text_preview = line_info["text"]
        duration = line_info["duration"]
        info_text += f'\n{index:2d}. {speaker}: "{text_preview}" ({duration:.1f}s)'

    info_text += "\n\n✅ **Status:** Conversation audio saved to outputs folder!"
    return info_text.strip()


def parse_to_narration_script(
    script_text: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[NarrationScript | None, str | None]:
    """Parse conversation script text into a NarrationScript model.

    Args:
        script_text: Raw text in Speaker: Text format.
        metadata: Optional metadata dict to attach.

    Returns:
        Tuple of (NarrationScript, None) on success or (None, error_message) on failure.
    """
    conversation, error = parse_conversation_script(script_text)
    if error:
        return None, error
    if not conversation:
        return None, "No valid conversation found in script"

    try:
        script = NarrationScript.from_conversation_list(
            conversation,
            metadata=metadata or {"source_format": "speaker_colon"},
        )
        return script, None
    except Exception as error:
        return None, f"Error creating NarrationScript: {str(error)}"


__all__ = [
    "parse_conversation_script",
    "get_speaker_names_from_script",
    "create_default_speaker_settings",
    "format_conversation_info",
    "parse_to_narration_script",
]