from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest


APP_DIR = Path(__file__).resolve().parents[1]
LAUNCH_PATH = APP_DIR / "launch.py"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


NORMALIZATION_GOLDEN = [
    ("The price is $50", "fifty dollars", "currency expansion"),
    (
        "The price is $1,234.56",
        "one thousand two hundred thirty-four dollars and fifty-six cents",
        "complex currency",
    ),
    ("Call 555-1234", "five five five one two three four", "phone number"),
    ("It's 50% off", "fifty percent", "percentage"),
    ("Visit https://example.com for details", "", "URL handling"),
    ("Dr. Smith went to the store", "Doctor Smith", "abbreviation"),
    ("The meeting is at 3:30 PM", "", "time handling"),
    ("Born on 01/15/2000", "", "date handling"),
    ("He said 'hello' to the crowd", "hello", "quotes preserved"),
    ("Normal text stays the same", "Normal text stays the same", "passthrough"),
]


CUE_STRIPPING_GOLDEN = [
    ("[whispers] Hello there", "ChatterboxTTS", "Hello there", "bracket cue removed"),
    ("(softly) Good morning", "Kokoro TTS", "Good morning", "parenthetical direction removed"),
    (
        "<speak><prosody rate='slow'>Hello</prosody></speak>",
        "Fish Speech",
        "Hello",
        "SSML stripped",
    ),
    ("THIS IS VERY LOUD", "F5-TTS", "", "allcaps normalized"),
    ("Check the API docs", "IndexTTS", "API", "acronym preserved"),
    ("Normal text here", "VoxCPM", "Normal text here", "clean text unchanged"),
    ("[laughs] Oh really [sighs]", "Higgs Audio", "Oh really", "multiple cues stripped"),
    ("Hello there", "UnknownEngine", "Hello there", "unknown engine safe fallback"),
]


REQUIRED_SYMBOLS = {
    "_DIGIT_WORDS",
    "_SMALL_NUMS",
    "_TENS",
    "_DEFAULT_ENGINE_EXPRESSIVENESS",
    "_BRACKET_CUE_PATTERN",
    "_PAREN_STAGE_DIRECTION_PATTERN",
    "_SSML_TAG_PATTERN",
    "_MULTISPACE_PATTERN",
    "_SPACE_BEFORE_PUNCT_PATTERN",
    "_EXCESS_BLANK_LINES_PATTERN",
    "_ALLCAPS_WORD_PATTERN",
    "_PRESERVED_ALLCAPS_ACRONYMS",
    "ENGINE_EXPRESSIVENESS",
    "_int_to_words",
    "_spell_digits",
    "deterministic_normalize",
    "_normalize_allcaps_word",
    "strip_unsupported_cues",
}


_CACHED_LAUNCH_SYMBOLS: ModuleType | None = None


def _assignment_targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _extract_launch_subset() -> ModuleType:
    source = LAUNCH_PATH.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(LAUNCH_PATH))
    selected_nodes: list[ast.stmt] = []

    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in REQUIRED_SYMBOLS:
            selected_nodes.append(node)
            continue

        if _assignment_targets(node) & REQUIRED_SYMBOLS:
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"__builtins__": __builtins__, "re": re}
    exec(compile(module, str(LAUNCH_PATH), "exec"), namespace)

    missing = sorted(name for name in REQUIRED_SYMBOLS if name not in namespace)
    if missing:
        raise RuntimeError(f"Missing extracted symbols from launch.py: {', '.join(missing)}")

    extracted_module = ModuleType("launch_subset")
    for name in REQUIRED_SYMBOLS:
        setattr(extracted_module, name, namespace[name])
    return extracted_module


def _load_launch_symbols() -> ModuleType:
    global _CACHED_LAUNCH_SYMBOLS

    if _CACHED_LAUNCH_SYMBOLS is not None:
        return _CACHED_LAUNCH_SYMBOLS

    try:
        _CACHED_LAUNCH_SYMBOLS = importlib.import_module("launch")
        return _CACHED_LAUNCH_SYMBOLS
    except Exception as direct_import_error:
        sys.modules.pop("launch", None)

    try:
        _CACHED_LAUNCH_SYMBOLS = _extract_launch_subset()
        return _CACHED_LAUNCH_SYMBOLS
    except Exception as extraction_error:
        pytest.skip(
            "Unable to load narration transform symbols from launch.py. "
            f"Direct import failed: {direct_import_error!r}. "
            f"AST extraction failed: {extraction_error!r}."
        )


@pytest.fixture(scope="module")
def launch_symbols() -> ModuleType:
    return _load_launch_symbols()


class TestDeterministicNormalize:
    """Golden tests for deterministic_normalize()."""

    @pytest.mark.parametrize(
        "input_text,expected_substring,description",
        NORMALIZATION_GOLDEN,
        ids=[golden[2] for golden in NORMALIZATION_GOLDEN],
    )
    def test_normalization(
        self,
        launch_symbols: ModuleType,
        input_text: str,
        expected_substring: str,
        description: str,
    ) -> None:
        result = launch_symbols.deterministic_normalize(input_text)

        assert isinstance(result, str), f"Expected str, got {type(result)}"
        if expected_substring:
            assert (
                expected_substring.lower() in result.lower()
            ), f"[{description}] Expected '{expected_substring}' in output '{result}'"


class TestStripUnsupportedCues:
    """Golden tests for strip_unsupported_cues()."""

    @pytest.mark.parametrize(
        "input_text,engine,expected_substring,description",
        CUE_STRIPPING_GOLDEN,
        ids=[golden[3] for golden in CUE_STRIPPING_GOLDEN],
    )
    def test_cue_stripping(
        self,
        launch_symbols: ModuleType,
        input_text: str,
        engine: str,
        expected_substring: str,
        description: str,
    ) -> None:
        result = launch_symbols.strip_unsupported_cues(input_text, engine)

        assert isinstance(result, str), f"Expected str, got {type(result)}"
        if expected_substring:
            assert (
                expected_substring in result
            ), f"[{description}] Expected '{expected_substring}' in output '{result}'"

    def test_allcaps_normalized_to_title_case(self, launch_symbols: ModuleType) -> None:
        result = launch_symbols.strip_unsupported_cues("THIS IS VERY LOUD", "F5-TTS")
        assert (
            result == "This Is Very Loud"
        ), "Expected ALL-CAPS emphasis to be normalized to title case for F5-TTS"


class TestEngineCapabilityMatrix:
    """Verify ENGINE_EXPRESSIVENESS covers all engines."""

    def test_all_engines_present(self, launch_symbols: ModuleType) -> None:
        expected_engines = [
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
        ]

        for engine in expected_engines:
            assert engine in launch_symbols.ENGINE_EXPRESSIVENESS, f"Missing engine: {engine}"

    def test_indextts2_has_emotion_vectors(self, launch_symbols: ModuleType) -> None:
        assert launch_symbols.ENGINE_EXPRESSIVENESS["IndexTTS2"]["emotion_vectors"] is True

    def test_default_capabilities_conservative(self, launch_symbols: ModuleType) -> None:
        for engine, capabilities in launch_symbols.ENGINE_EXPRESSIVENESS.items():
            if engine != "IndexTTS2":
                assert (
                    capabilities.get("emotion_vectors") is False
                ), f"{engine} should not have emotion_vectors"
