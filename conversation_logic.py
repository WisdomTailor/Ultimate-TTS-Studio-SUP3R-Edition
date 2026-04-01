"""Conversation parsing and NarrationScript bridging helpers.

This module contains pure conversation utilities that can be imported independently
from the Gradio UI layer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from narration_script import NarrationScript
from narration_transform import call_openai_compatible_chat


logger = logging.getLogger(__name__)


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


CONVERSATION_FORMATTER_SYSTEM_PROMPT = """You are a conversation structuring assistant. Your task is to analyze text and identify distinct speakers, attributing each line of dialogue or narration to the correct speaker.

Rules:
1. Return ONLY a valid JSON object matching the NarrationScript schema.
2. Every line must have: "speaker" (string), "text" (string), "line_type" (one of "dialogue", "narration", "stage_direction").
3. If a speaker name is explicitly given (e.g., "Alice:", "Speaker 1:"), use that exact name.
4. If speakers are not explicitly named, infer reasonable names from context (e.g., "Interviewer", "Guest", "Narrator").
5. For narration or description text not spoken by a character, use "Narrator" as the speaker and "narration" as line_type.
6. For stage directions or action descriptions, use "Stage Direction" as speaker and "stage_direction" as line_type.
7. Preserve the original text content exactly — do not paraphrase, summarize, or add words.
8. Set "confidence" to 1.0 for clear attributions, 0.5-0.9 for uncertain ones.
9. Set "ambiguous" to true when speaker attribution is genuinely uncertain.
10. The "cues" array should be empty (cues are applied separately).

Output schema:
{
  "version": "1.0",
  "lines": [
    {"speaker": "...", "text": "...", "line_type": "dialogue|narration|stage_direction", "cues": [], "confidence": 1.0, "ambiguous": false}
  ],
  "metadata": {"source_format": "ai_attributed", "model": "..."}
}

Return ONLY the JSON object. No markdown fences, no explanation text."""


def _extract_json_from_llm_response(response: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response, handling common formatting issues.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    if not isinstance(response, str) or not response.strip():
        raise ValueError("LLM response is empty or not a string")

    cleaned_response = response.strip()

    def _parse_json(candidate: str) -> dict[str, Any]:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON must be an object")
        return parsed

    try:
        return _parse_json(cleaned_response)
    except (json.JSONDecodeError, ValueError):
        pass

    if cleaned_response.startswith("```"):
        fenced_lines = cleaned_response.splitlines()
        if len(fenced_lines) >= 3 and fenced_lines[-1].strip() == "```":
            fence_body = "\n".join(fenced_lines[1:-1]).strip()
            try:
                return _parse_json(fence_body)
            except (json.JSONDecodeError, ValueError):
                pass

    json_start = cleaned_response.find("{")
    json_end = cleaned_response.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        embedded_json = cleaned_response[json_start : json_end + 1]
        try:
            return _parse_json(embedded_json)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Failed to parse JSON object from LLM response: {error}") from error

    raise ValueError("No valid JSON object found in LLM response")


def format_conversation_with_llm(
    text: str,
    base_url: str,
    api_key: str,
    model_id: str,
    timeout_seconds: int = 120,
    extra_headers: dict[str, str] | None = None,
    auth_style: str = "bearer",
) -> tuple[NarrationScript | None, str | None]:
    """Use an LLM to parse free-form text into a structured NarrationScript.

    Takes arbitrary text (prose, dialogue, screenplay format, etc.) and sends it
    to an LLM endpoint for speaker detection and attribution. Returns a validated
    NarrationScript model.

    Args:
        text: The raw text to structure into a conversation script.
        base_url: LLM API base URL (OpenAI-compatible).
        api_key: API key for authentication.
        model_id: Model identifier to use.
        timeout_seconds: Request timeout.
        extra_headers: Additional HTTP headers for the request.
        auth_style: Authentication style ("bearer" or "api-key").

    Returns:
        Tuple of (NarrationScript, None) on success, or (None, error_message) on failure.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "Conversation text is required"
    if not isinstance(base_url, str) or not base_url.strip():
        return None, "LLM base URL is required"
    if not isinstance(model_id, str) or not model_id.strip():
        return None, "LLM model ID is required"

    try:
        raw_response = call_openai_compatible_chat(
            base_url=base_url.strip(),
            api_key=api_key,
            model_id=model_id.strip(),
            system_prompt=CONVERSATION_FORMATTER_SYSTEM_PROMPT,
            user_prompt=text,
            timeout_seconds=timeout_seconds,
            temperature=0.1,
            max_tokens=4096,
            extra_headers=extra_headers,
            auth_style=auth_style,
        )
        response_data = _extract_json_from_llm_response(raw_response)
        metadata = response_data.get("metadata")
        normalized_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        normalized_metadata.update({"source_format": "ai_attributed", "model": model_id.strip()})
        response_data["metadata"] = normalized_metadata

        script = NarrationScript.from_dict(response_data)
        return script, None
    except Exception as error:
        logger.exception("Failed to format conversation with LLM")
        return None, f"Error formatting conversation with LLM: {error}"


__all__ = [
    *__all__,
    "CONVERSATION_FORMATTER_SYSTEM_PROMPT",
    "_extract_json_from_llm_response",
    "format_conversation_with_llm",
]
