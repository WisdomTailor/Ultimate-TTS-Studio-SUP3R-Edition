from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pronunciation import (  # noqa: E402
    PronunciationOverride,
    ProtectedTerm,
    apply_pronunciation_overrides,
    load_lexicon,
    mask_protected_terms,
    pronunciation_pipeline,
    save_lexicon,
    unmask_protected_terms,
)


class TestMaskProtectedTerms:
    def test_round_trip_preserves_original_text(self) -> None:
        text = "Hermione met Nguyen in Diagon Alley."
        protected_terms = [
            ProtectedTerm(term="Hermione"),
            ProtectedTerm(term="Diagon Alley"),
        ]

        masked_text, placeholder_map = mask_protected_terms(text, protected_terms)

        assert masked_text != text
        assert len(placeholder_map) == 2
        assert unmask_protected_terms(masked_text, placeholder_map) == text

    def test_case_sensitive_and_case_insensitive_matching(self) -> None:
        text = "Hermione met hermione."
        protected_terms = [
            ProtectedTerm(term="Hermione", case_sensitive=True),
            ProtectedTerm(term="hermione", case_sensitive=False),
        ]

        masked_text, placeholder_map = mask_protected_terms(text, protected_terms)

        assert masked_text.count("«PROT_") == 2
        assert sorted(placeholder_map.values()) == ["Hermione", "hermione"]

    def test_multiple_terms_and_longest_match_wins(self) -> None:
        text = "I left New York City for New York today."
        protected_terms = [
            ProtectedTerm(term="New York"),
            ProtectedTerm(term="New York City"),
        ]

        masked_text, placeholder_map = mask_protected_terms(text, protected_terms)

        assert len(placeholder_map) == 2
        assert "New York City" in placeholder_map.values()
        assert "New York" in placeholder_map.values()
        assert unmask_protected_terms(masked_text, placeholder_map) == text

    def test_placeholder_tokens_do_not_collide_with_real_text(self) -> None:
        text = "Keep «PROT_0» literal and protect Wizard."

        masked_text, placeholder_map = mask_protected_terms(
            text,
            [ProtectedTerm(term="Wizard")],
        )

        assert "«PROT_0»" in masked_text
        assert list(placeholder_map) == ["«PROT_1»"]
        assert unmask_protected_terms(masked_text, placeholder_map) == text

    def test_empty_inputs_return_empty_mapping(self) -> None:
        assert mask_protected_terms("", [ProtectedTerm(term="Hermione")]) == ("", {})
        assert mask_protected_terms("Text", []) == ("Text", {})


class TestApplyPronunciationOverrides:
    def test_case_sensitive_and_case_insensitive_overrides(self) -> None:
        text = "Hermione met hermione."
        overrides = [
            PronunciationOverride(word="Hermione", phonetic="Her-MY-oh-nee", case_sensitive=True),
            PronunciationOverride(word="hermione", phonetic="her-MY-oh-nee", case_sensitive=False),
        ]

        adjusted_text = apply_pronunciation_overrides(text, overrides)

        assert adjusted_text == "Her-MY-oh-nee met her-MY-oh-nee."

    def test_whole_word_matching_does_not_replace_inside_other_words(self) -> None:
        text = "The theater is next to the other building."
        overrides = [PronunciationOverride(word="the", phonetic="thee")]

        adjusted_text = apply_pronunciation_overrides(text, overrides)

        assert adjusted_text == "thee theater is next to thee other building."

    def test_empty_override_inputs_leave_text_unchanged(self) -> None:
        assert apply_pronunciation_overrides("Hello", []) == "Hello"
        assert apply_pronunciation_overrides("", [PronunciationOverride("the", "thee")]) == ""


class TestPronunciationPipeline:
    def test_pipeline_with_mock_llm_transform(self) -> None:
        def mock_llm_transform(text: str) -> str:
            return text.replace("walked", "strolled").replace("quietly", "softly")

        result = pronunciation_pipeline(
            text="Hermione walked quietly with Nguyen.",
            protected_terms=[ProtectedTerm(term="Hermione"), ProtectedTerm(term="Nguyen")],
            overrides=[
                PronunciationOverride(word="Hermione", phonetic="Her-MY-oh-nee"),
                PronunciationOverride(word="Nguyen", phonetic="Win"),
            ],
            llm_transform_fn=mock_llm_transform,
        )

        assert result == "Her-MY-oh-nee strolled softly with Win."

    def test_pipeline_skips_llm_when_no_transform_is_supplied(self) -> None:
        result = pronunciation_pipeline(
            text="Hermione stayed put.",
            protected_terms=[ProtectedTerm(term="Hermione")],
            overrides=[PronunciationOverride(word="Hermione", phonetic="Her-MY-oh-nee")],
        )

        assert result == "Her-MY-oh-nee stayed put."


class TestLexiconPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        lexicon_path = tmp_path / "app_state" / "lexicon.json"
        protected_terms = [ProtectedTerm(term="Hermione", case_sensitive=False)]
        overrides = [
            PronunciationOverride(word="Nguyen", phonetic="Win"),
            PronunciationOverride(word="Hermione", phonetic="Her-MY-oh-nee", case_sensitive=True),
        ]

        save_lexicon(lexicon_path, protected_terms, overrides)
        loaded_protected_terms, loaded_overrides = load_lexicon(lexicon_path)

        assert loaded_protected_terms == protected_terms
        assert loaded_overrides == overrides
        saved_payload = json.loads(lexicon_path.read_text(encoding="utf-8"))
        assert saved_payload["version"] == "1.0"

    def test_load_lexicon_rejects_invalid_version(self, tmp_path: Path) -> None:
        lexicon_path = tmp_path / "lexicon.json"
        lexicon_path.write_text(
            json.dumps(
                {
                    "version": "2.0",
                    "protected_terms": [],
                    "pronunciation_overrides": [],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Unsupported lexicon version"):
            load_lexicon(lexicon_path)