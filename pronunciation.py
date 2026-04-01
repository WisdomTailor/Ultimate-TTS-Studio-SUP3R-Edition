"""Pronunciation masking and substitution pipeline utilities."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path


_PLACEHOLDER_TEMPLATE = "«PROT_{index}»"
_LEXICON_VERSION = "1.0"


@dataclass(slots=True, frozen=True)
class ProtectedTerm:
    """A term that should be preserved through LLM transformation."""

    term: str
    case_sensitive: bool = True


@dataclass(slots=True, frozen=True)
class PronunciationOverride:
    """A word-to-phonetic mapping for TTS pronunciation."""

    word: str
    phonetic: str
    case_sensitive: bool = False


def _next_placeholder(text: str, used_placeholders: set[str], start_index: int) -> tuple[str, int]:
    next_index = start_index

    while True:
        placeholder = _PLACEHOLDER_TEMPLATE.format(index=next_index)
        next_index += 1
        if placeholder in used_placeholders:
            continue
        if placeholder in text:
            continue
        return placeholder, next_index


def _sorted_protected_terms(protected_terms: list[ProtectedTerm]) -> list[ProtectedTerm]:
    return sorted(
        [protected_term for protected_term in protected_terms if protected_term.term],
        key=lambda protected_term: len(protected_term.term),
        reverse=True,
    )


def mask_protected_terms(
    text: str,
    protected_terms: list[ProtectedTerm],
) -> tuple[str, dict[str, str]]:
    """Replace protected terms with placeholders.

    Args:
        text: Source text to mask.
        protected_terms: Terms that should be preserved through LLM transforms.

    Returns:
        Tuple of masked text and a placeholder-to-original-text mapping.
    """

    sorted_terms = _sorted_protected_terms(protected_terms)
    if not text or not sorted_terms:
        return text, {}

    cursor = 0
    placeholder_index = 0
    masked_fragments: list[str] = []
    placeholder_map: dict[str, str] = {}
    used_placeholders: set[str] = set()

    while cursor < len(text):
        matched_text: str | None = None

        for protected_term in sorted_terms:
            candidate = text[cursor : cursor + len(protected_term.term)]
            if len(candidate) != len(protected_term.term):
                continue

            if protected_term.case_sensitive:
                is_match = candidate == protected_term.term
            else:
                is_match = candidate.casefold() == protected_term.term.casefold()

            if is_match:
                matched_text = candidate
                break

        if matched_text is None:
            masked_fragments.append(text[cursor])
            cursor += 1
            continue

        placeholder, placeholder_index = _next_placeholder(
            text=text,
            used_placeholders=used_placeholders,
            start_index=placeholder_index,
        )
        used_placeholders.add(placeholder)
        placeholder_map[placeholder] = matched_text
        masked_fragments.append(placeholder)
        cursor += len(matched_text)

    return "".join(masked_fragments), placeholder_map


def unmask_protected_terms(
    text: str,
    placeholder_map: dict[str, str],
) -> str:
    """Restore placeholders back to original terms.

    Args:
        text: Text containing placeholders.
        placeholder_map: Mapping from placeholder to original text.

    Returns:
        Unmasked text.
    """

    restored_text = text
    for placeholder in sorted(placeholder_map, key=len, reverse=True):
        restored_text = restored_text.replace(placeholder, placeholder_map[placeholder])
    return restored_text


def _sorted_overrides(overrides: list[PronunciationOverride]) -> list[PronunciationOverride]:
    return sorted(
        [override for override in overrides if override.word],
        key=lambda override: len(override.word),
        reverse=True,
    )


def apply_pronunciation_overrides(
    text: str,
    overrides: list[PronunciationOverride],
) -> str:
    """Replace words with their phonetic spellings before TTS.

    Args:
        text: Source text.
        overrides: Whole-word pronunciation override rules.

    Returns:
        Text with phonetic replacements applied.
    """

    if not text or not overrides:
        return text

    adjusted_text = text
    for override in _sorted_overrides(overrides):
        flags = 0 if override.case_sensitive else re.IGNORECASE
        pattern = re.compile(rf"\b{re.escape(override.word)}\b", flags=flags)
        adjusted_text = pattern.sub(override.phonetic, adjusted_text)
    return adjusted_text


def pronunciation_pipeline(
    text: str,
    protected_terms: list[ProtectedTerm] | None = None,
    overrides: list[PronunciationOverride] | None = None,
    llm_transform_fn: Callable[[str], str] | None = None,
) -> str:
    """Full pipeline: mask, transform, unmask, then apply pronunciation overrides.

    Args:
        text: Source text.
        protected_terms: Terms to preserve through the optional LLM stage.
        overrides: Pronunciation substitutions to apply before synthesis.
        llm_transform_fn: Optional transform function for narration rewriting.

    Returns:
        Processed text ready for synthesis.
    """

    effective_protected_terms = protected_terms or []
    effective_overrides = overrides or []

    masked_text, placeholder_map = mask_protected_terms(text, effective_protected_terms)
    transformed_text = (
        llm_transform_fn(masked_text) if llm_transform_fn is not None else masked_text
    )
    unmasked_text = unmask_protected_terms(transformed_text, placeholder_map)
    return apply_pronunciation_overrides(unmasked_text, effective_overrides)


def _load_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected '{field_name}' to be a boolean.")


def _load_string(value: object, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected '{field_name}' to be a string.")


def load_lexicon(path: Path) -> tuple[list[ProtectedTerm], list[PronunciationOverride]]:
    """Load protected terms and pronunciation overrides from a JSON file.

    Args:
        path: Lexicon JSON file path.

    Returns:
        Tuple of protected terms and pronunciation overrides.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON payload is invalid.
    """

    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, dict):
        raise ValueError("Lexicon JSON must be an object.")

    version = payload.get("version")
    if version != _LEXICON_VERSION:
        raise ValueError(f"Unsupported lexicon version: {version!r}")

    protected_terms_payload = payload.get("protected_terms", [])
    overrides_payload = payload.get("pronunciation_overrides", [])

    if not isinstance(protected_terms_payload, list):
        raise ValueError("'protected_terms' must be a list.")
    if not isinstance(overrides_payload, list):
        raise ValueError("'pronunciation_overrides' must be a list.")

    protected_terms = [
        ProtectedTerm(
            term=_load_string(item.get("term"), field_name="term"),
            case_sensitive=_load_bool(
                item.get("case_sensitive", True),
                field_name="case_sensitive",
            ),
        )
        for item in protected_terms_payload
        if isinstance(item, dict)
    ]

    if len(protected_terms) != len(protected_terms_payload):
        raise ValueError("Each protected term entry must be an object.")

    overrides = [
        PronunciationOverride(
            word=_load_string(item.get("word"), field_name="word"),
            phonetic=_load_string(item.get("phonetic"), field_name="phonetic"),
            case_sensitive=_load_bool(
                item.get("case_sensitive", False),
                field_name="case_sensitive",
            ),
        )
        for item in overrides_payload
        if isinstance(item, dict)
    ]

    if len(overrides) != len(overrides_payload):
        raise ValueError("Each pronunciation override entry must be an object.")

    return protected_terms, overrides


def save_lexicon(
    path: Path,
    protected_terms: list[ProtectedTerm],
    overrides: list[PronunciationOverride],
) -> None:
    """Save protected terms and pronunciation overrides to a JSON file.

    Args:
        path: Destination JSON path.
        protected_terms: Protected terms to persist.
        overrides: Pronunciation overrides to persist.
    """

    payload = {
        "version": _LEXICON_VERSION,
        "protected_terms": [asdict(protected_term) for protected_term in protected_terms],
        "pronunciation_overrides": [asdict(override) for override in overrides],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.write("\n")
