"""Pure narration transform and LLM provider utilities.

This module holds deterministic normalization, provider configuration, model discovery,
OpenAI-compatible chat calls, and narration transform orchestration. It intentionally avoids any
Gradio imports so the logic can be tested and reused independently from the UI layer.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from engine_registry import (
    ENGINE_EXPRESSIVENESS,
    _ALLCAPS_WORD_PATTERN,
    _BRACKET_CUE_PATTERN,
    _DEFAULT_ENGINE_EXPRESSIVENESS,
    _EXCESS_BLANK_LINES_PATTERN,
    _MULTISPACE_PATTERN,
    _PAREN_STAGE_DIRECTION_PATTERN,
    _PRESERVED_ALLCAPS_ACRONYMS,
    _SPACE_BEFORE_PUNCT_PATTERN,
    _SSML_TAG_PATTERN,
    _normalize_allcaps_word,
    strip_unsupported_cues,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformChunk:
    text: str
    separator_before: str = ""


DEFAULT_LLM_NARRATION_SYSTEM_PROMPT = """You are a TTS script preparation specialist. Your sole task is to transform raw text into clean, speakable narration for ElevenLabs v3 TTS.

OUTPUT RULES — apply unconditionally:
- Return plain narration text only. No SSML, no markdown, no code fences.
- Do not include explanations, meta-commentary, or framing phrases.
- Preserve speaker attributions (e.g., "Alice:" or "NARRATOR:") exactly as written.
- Respect the LOCALE provided: use the appropriate spoken forms, idioms, and spelling.
- Never emit SSML break tags (<break/>). Use punctuation for pacing.

MODE CONTRACTS:
Minimal — Only resolve what deterministic rules cannot: ambiguous abbreviations, context-dependent
expressions, initialisms that need spelt-out or spoken form based on surrounding context. Do NOT
rephrase, restructure, reorder, or add any styling, emphasis, or audio tags.

Polish — Everything Minimal does, plus: split overly long run-on sentences, smooth genuinely
awkward phrasing, remove markdown artifacts (**, __, ##, bullet dashes), convert parenthetical
asides to natural spoken equivalents. Do NOT add emotional cues, audio tags, or ALL-CAPS emphasis.

Vivid — Everything Polish does, plus: add sparse [emotional] audio cues honoring MAX_TAG_DENSITY
(tags per 100 words), use em-dashes (\u2014) for pivots or interruptions, ellipses (\u2026) for
dramatic pauses, occasional ALL-CAPS for spoken stress. Calibrate tone and rhythm to the STYLE.
"""

VOICE_CASTING_SYSTEM_PROMPT = """You are a professional voice casting director for audio productions. Given a list of character names from a script, generate a concise voice profile for each character.

For each character, provide:
- A brief character description (age estimate, personality, speaking style)
- A 3-5 word voice profile descriptor (e.g., "Warm, authoritative baritone" or "Bright, energetic soprano")

Format your response as a simple list:
CHARACTER_NAME:
    Description: [brief description]
    Voice Profile: [3-5 word descriptor]

Be creative but realistic. Infer character traits from their names and typical narrative conventions. Keep descriptions concise - a few sentences maximum per character."""

CONTENT_TYPE_PRESETS: dict[str, dict[str, str | None]] = {
    "General (Default)": {
        "system_prompt": None,
        "description": "General-purpose TTS cleanup that works across narration, dialogue, and mixed text.",
    },
    "Audiobook Narration": {
        "system_prompt": """You are a long-form audiobook narration specialist preparing text for natural spoken delivery.

Keep the author's meaning, sequence, and scene structure intact. Preserve chapter headings, part headings, and section titles as spoken anchors. If the input already uses speaker tags such as [narrator], [speaker], or Name:, keep them consistent and do not rename them.

Priorities:
- Separate narration from direct dialogue when the distinction is explicit in the source.
- Preserve narrator passages as narrative prose and spoken lines as dialogue.
- Expand abbreviations, initials, numbers, dates, currencies, symbols, and shorthand into locale-appropriate spoken language.
- Convert markdown, ornamental separators, footnote markers, and other visual-only artifacts into clean speech or remove them if they have no spoken value.
- Smooth punctuation so the text reads aloud clearly, but do not summarize, modernize, or rewrite the author's voice.
- When mixed paragraphs contain both action and quoted speech, split only where the boundary is unambiguous.

Guardrails:
- Respect MODE exactly: Minimal stays conservative, Polish may smooth awkward phrasing, and Vivid may add sparse expressive cues within MAX_TAG_DENSITY.
- Respect STYLE and LOCALE as downstream steering signals rather than overriding them.
- Do not invent new plot detail, internal thoughts, or character intent.
- Keep existing speaker tags and narrator tags structurally valid and aligned with the original content.
""",
        "description": "For long-form books and stories: preserve headings, narration flow, and clean spoken rendering.",
    },
    "Natural Dialogue": {
        "system_prompt": """You are a dialogue naturalization specialist preparing text to sound like believable spoken conversation.

Your job is to make dialogue feel authentic for mature adults without changing the facts, intent, or speaker identity. Preserve existing speaker labels and keep narration distinct from speech.

Priorities:
- Prefer natural contractions where they fit the character and context.
- Break stiff or overly formal sentences into shorter spoken units when needed.
- Use light conversational bridges such as well, you know, honestly, right, or I mean only when they improve flow.
- Never use like as a filler.
- Keep the tone adult, grounded, and contemporary rather than exaggerated, adolescent, or slang-heavy.
- Preserve interruptions, hesitations, and back-and-forth rhythm when they are already implied by the text.

Guardrails:
- Respect MODE, STYLE, LOCALE, and MAX_TAG_DENSITY supplied by the user prompt.
- Do not turn narration into dialogue unless the source clearly supports it.
- Do not add verbal clutter to every line; authenticity beats quantity.
- Keep speaker attribution stable and do not merge separate voices.
""",
        "description": "For character-heavy dialogue: more natural contractions, cadence, and conversational flow.",
    },
    "Instructional / Tutorial": {
        "system_prompt": """You are an instructional script editor preparing educational and technical material for a single teacher-style narrator.

Transform dry directions into clear spoken guidance while preserving technical accuracy. Favor a calm mentor tone that sounds supportive, competent, and easy to follow.

Priorities:
- Rewrite detached or third-person instructions into direct second-person guidance when meaning stays intact.
- Keep steps, prerequisites, warnings, and outcomes explicit and easy to follow aloud.
- Expand technical acronyms letter-by-letter when they are ordinarily spoken that way, such as U S B, G P U, A P I, or H T M L.
- Preserve product names, commands, file names, and code identifiers accurately.
- Write measurements, units, symbols, and numeric ranges in spoken form appropriate to the locale.
- Use encouraging transitions that help a listener follow a procedure without sounding salesy or overexcited.

Guardrails:
- Respect MODE, STYLE, and LOCALE supplied separately.
- Do not omit safety warnings, ordering, or technical constraints.
- Keep the script suitable for one narrator unless the source already contains explicit speaker labels.
- Avoid unnecessary dramatization; clarity comes first.
""",
        "description": "For tutorials and explainers: second-person mentor tone with spoken acronyms and measurements.",
    },
    "Script Cleanup": {
        "system_prompt": """You are a script cleanup specialist performing the lightest useful pass before TTS.

Assume the input may already be tagged or structurally formatted for narration. Your main goal is to preserve the script faithfully while repairing obvious formatting defects that would hurt speech synthesis.

Priorities:
- Verify speaker tags, narrator tags, and label formatting stay consistent from start to finish.
- Repair obvious unclosed or mismatched tags when the intended pairing is clear.
- Remove markdown artifacts, stray bullets, duplicate emphasis markers, and copy-paste debris.
- Normalize whitespace and punctuation only where needed for clean speech.
- Preserve wording, ordering, and speaker assignment as closely as possible.

Guardrails:
- Do not paraphrase, embellish, summarize, or reinterpret content unless a tiny edit is required to fix malformed structure.
- Keep MODE effects restrained when they would otherwise alter the script's structure.
- Respect STYLE and LOCALE, but do not let them override fidelity.
- Never drop valid content just because it looks unusual.
""",
        "description": "For already-structured scripts: faithful cleanup, tag repair, and markdown removal with minimal rewriting.",
    },
}

DEFAULT_CONTENT_TYPE_PRESET = "General (Default)"


def get_content_type_preset_names() -> list[str]:
    return list(CONTENT_TYPE_PRESETS.keys())


def get_content_type_system_prompt(preset_name: str) -> str:
    preset = CONTENT_TYPE_PRESETS.get(preset_name)
    system_prompt = preset.get("system_prompt") if preset else None
    if isinstance(system_prompt, str) and system_prompt.strip():
        return system_prompt.strip()
    return DEFAULT_LLM_NARRATION_SYSTEM_PROMPT


_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_SMALL_NUMS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]

_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(number: int) -> str:
    if number < 20:
        return _SMALL_NUMS[number]
    if number < 100:
        tens, remainder = divmod(number, 10)
        return _TENS[tens] if remainder == 0 else f"{_TENS[tens]}-{_SMALL_NUMS[remainder]}"
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        return (
            f"{_SMALL_NUMS[hundreds]} hundred"
            if remainder == 0
            else f"{_SMALL_NUMS[hundreds]} hundred {_int_to_words(remainder)}"
        )
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1000)
        return (
            f"{_int_to_words(thousands)} thousand"
            if remainder == 0
            else f"{_int_to_words(thousands)} thousand {_int_to_words(remainder)}"
        )
    millions, remainder = divmod(number, 1_000_000)
    return (
        f"{_int_to_words(millions)} million"
        if remainder == 0
        else f"{_int_to_words(millions)} million {_int_to_words(remainder)}"
    )


def _spell_digits(text: str) -> str:
    return " ".join(_DIGIT_WORDS.get(char, char) for char in text)


def deterministic_normalize(source_text: str) -> str:
    """Apply deterministic narration cleanup before any LLM styling pass.

    Args:
        source_text: Raw source text to normalize.

    Returns:
        The normalized text with deterministic replacements applied.
    """
    if not isinstance(source_text, str):
        return ""

    text = source_text

    abbreviation_map = {
        r"\bDr\.(?=\s|$)": "Doctor",
        r"\bMr\.(?=\s|$)": "Mister",
        r"\bMrs\.(?=\s|$)": "Missus",
        r"\bMs\.(?=\s|$)": "Miss",
        r"\bAve\.(?=\s|$)": "Avenue",
        r"\bSt\.(?=\s|$)": "Street",
    }
    for pattern, replacement in abbreviation_map.items():
        text = re.sub(pattern, replacement, text)

    def short_phone_replacer(match: re.Match[str]) -> str:
        parts = [group for group in match.groups() if group]
        return " ".join(_spell_digits(part) for part in parts)

    def full_phone_replacer(match: re.Match[str]) -> str:
        g1, g2, g3 = match.groups()
        return f"{_spell_digits(g1)}, {_spell_digits(g2)}, {_spell_digits(g3)}"

    text = re.sub(r"\b(\d{3})-(\d{3})-(\d{4})\b", full_phone_replacer, text)
    text = re.sub(r"\b(\d{3})-(\d{4})\b", short_phone_replacer, text)

    def currency_replacer(match: re.Match[str]) -> str:
        symbol, number = match.groups()
        currency_name = {"$": "dollars", "£": "pounds", "€": "euros", "¥": "yen"}.get(
            symbol, "currency"
        )
        clean = number.replace(",", "")
        if "." in clean:
            whole, cents = clean.split(".", 1)
            whole_words = _int_to_words(int(whole)) if whole.isdigit() else whole
            cents_words = _int_to_words(int(cents)) if cents.isdigit() else cents
            return f"{whole_words} {currency_name} and {cents_words} cents"
        return f"{_int_to_words(int(clean)) if clean.isdigit() else clean} {currency_name}"

    text = re.sub(r"([$£€¥])(\d+(?:,\d{3})*(?:\.\d{2})?)", currency_replacer, text)

    def percent_replacer(match: re.Match[str]) -> str:
        digits = match.group(1)
        return f"{_int_to_words(int(digits)) if digits.isdigit() else digits} percent"

    text = re.sub(r"\b(\d+)%", percent_replacer, text)

    def date_replacer(match: re.Match[str]) -> str:
        year, month, day = match.groups()
        month_names = {
            "01": "January",
            "02": "February",
            "03": "March",
            "04": "April",
            "05": "May",
            "06": "June",
            "07": "July",
            "08": "August",
            "09": "September",
            "10": "October",
            "11": "November",
            "12": "December",
        }
        day_int = int(day)
        day_words = _int_to_words(day_int)
        year_words = _int_to_words(int(year))
        return f"{month_names.get(month, month)} {day_words}, {year_words}"

    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", date_replacer, text)

    def time_replacer(match: re.Match[str]) -> str:
        hh, mm = match.groups()
        hour = int(hh)
        minute = int(mm)
        suffix = "AM"
        if hour == 0:
            hour = 12
        elif hour == 12:
            suffix = "PM"
        elif hour > 12:
            hour -= 12
            suffix = "PM"
        minute_words = _int_to_words(minute) if minute > 0 else ""
        return f"{_int_to_words(hour)} {minute_words} {suffix}".replace("  ", " ").strip()

    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", time_replacer, text)

    def url_replacer(match: re.Match[str]) -> str:
        value = match.group(0)
        spoken = value.replace("https://", "").replace("http://", "")
        spoken = spoken.replace(".", " dot ").replace("/", " slash ")
        return re.sub(r"\s+", " ", spoken).strip()

    text = re.sub(
        r"\bhttps?://[^\s]+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?", url_replacer, text
    )

    return text


def _apply_local_narration_transform(source_text: str, mode: str) -> tuple[str, str]:
    """Apply local non-LLM narration transform with mode-aware status reporting."""
    result = deterministic_normalize(source_text)
    if mode == "Vivid":
        status = (
            "Narration transform: deterministic normalization applied; "
            "Vivid expressive styling requires an LLM — connect a provider for full effect"
        )
    else:
        status = "Narration transform: deterministic normalization applied"
    return result, status


LLM_PROVIDER_CONFIGS = {
    "Custom OpenAI-compatible": {
        "base_url": "http://localhost:8000/v1",
        "default_model": "",
        "env_var": "OPENAI_API_KEY",
        "requires_api_key": False,
        "kind": "custom",
        "auth_style": "bearer",
        "headers": {},
    },
    "GitHub Models (OpenAI-compatible)": {
        "base_url": "https://models.github.ai/inference",
        "catalog_url": "https://models.github.ai/catalog/models",
        "default_model": "openai/gpt-4.1-mini",
        "env_var": "GITHUB_MODELS_TOKEN",
        "requires_api_key": True,
        "kind": "cloud",
        "auth_style": "bearer",
        "headers": {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    },
    "Google Gemini API (OpenAI-compatible)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
        "env_var": "GOOGLE_API_KEY",
        "requires_api_key": True,
        "kind": "cloud",
        "auth_style": "bearer",
        "headers": {},
    },
    "Hugging Face Inference API": {
        "base_url": "https://api-inference.huggingface.co/v1",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "env_var": "HF_TOKEN",
        "requires_api_key": True,
        "kind": "cloud",
        "auth_style": "bearer",
        "headers": {},
    },
    "Hugging Face Inference Endpoints": {
        "base_url": "https://your-endpoint.endpoints.huggingface.cloud/v1",
        "default_model": "tgi",
        "env_var": "HF_TOKEN",
        "requires_api_key": True,
        "kind": "cloud",
        "auth_style": "bearer",
        "headers": {},
    },
    "LM Studio OpenAI Server": {
        "base_url": "http://localhost:1234/v1",
        "default_model": "qwen/qwen3-30b-a3b-instruct-2507",
        "env_var": "OPENAI_API_KEY",
        "requires_api_key": False,
        "kind": "local",
        "auth_style": "bearer",
        "headers": {},
    },
    "Microsoft Foundry (OpenAI-compatible)": {
        "base_url": "https://eastus2.api.cognitive.microsoft.com",
        "default_model": "gpt-4o-mini",
        "env_var": "AZURE_AI_API_KEY",
        "requires_api_key": True,
        "kind": "cloud",
        "auth_style": "api-key",
        "headers": {},
    },
    "Ollama (OpenAI-compatible)": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen3:30b-a3b",
        "env_var": "OPENAI_API_KEY",
        "requires_api_key": False,
        "kind": "local",
        "auth_style": "bearer",
        "headers": {},
    },
    "vLLM OpenAI Server": {
        "base_url": "http://localhost:8000/v1",
        "default_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "env_var": "OPENAI_API_KEY",
        "requires_api_key": False,
        "kind": "local",
        "auth_style": "bearer",
        "headers": {},
    },
}

LLM_PROVIDER_MODEL_SUGGESTIONS = {
    "Custom OpenAI-compatible": [],
    "GitHub Models (OpenAI-compatible)": [
        "openai/gpt-4o-mini",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
    "Google Gemini API (OpenAI-compatible)": [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash-lite",
    ],
    "Hugging Face Inference API": [
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "google/gemma-2-9b-it",
    ],
    "Hugging Face Inference Endpoints": [
        "tgi",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ],
    "LM Studio OpenAI Server": [
        "qwen/qwen3-30b-a3b-instruct-2507",
        "lmstudio-community/qwen3-30b-a3b",
        "lmstudio-community/llama-3.1-8b",
    ],
    "Microsoft Foundry (OpenAI-compatible)": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ],
    "Ollama (OpenAI-compatible)": [
        "qwen3:30b-a3b",
        "qwen3:8b",
        "llama3.1:8b",
        "mistral:7b",
        "gemma2:9b",
    ],
    "vLLM OpenAI Server": [
        "Qwen/Qwen3-30B-A3B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
}

_DEFAULT_PROVIDER_CONFIG = {
    "base_url": "http://localhost:8000/v1",
    "default_model": "",
    "env_var": "OPENAI_API_KEY",
    "requires_api_key": False,
    "kind": "custom",
    "auth_style": "bearer",
    "headers": {},
}

_MODE_BEHAVIORAL_REMINDERS: dict[str, str] = {
    "Minimal": "fix only undetermined abbreviations and context-dependent expansions; do not rephrase or add styling",
    "Polish": "normalize + smooth phrasing and remove markdown artifacts; no emotional cues or audio tags",
    "Vivid": "normalize + polish + add sparse emotional cues and expressive punctuation per STYLE",
}

_STYLE_DESCRIPTIONS: dict[str, str] = {
    "neutral": "plain, clear delivery with no emotional coloring",
    "dramatic": "heightened tension, deliberate pacing, weight on key moments",
    "conversational": "warm, natural, relaxed — like speaking to a friend",
    "documentary": "measured, authoritative, informative tone",
    "children": "bright, friendly, slightly animated and encouraging",
}

LLM_OUTCOME_PRESETS: dict[str, dict[str, float | int]] = {
    "Conservative": {"temperature": 0.3, "top_p": 0.8, "max_tokens": 2048},
    "Balanced": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
    "Creative": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096},
}

DEFAULT_LLM_OUTCOME_PRESET = "Balanced"


def _get_provider_config(provider_name: str) -> dict:
    return LLM_PROVIDER_CONFIGS.get(provider_name, _DEFAULT_PROVIDER_CONFIG)


def get_llm_provider_env_var(provider_name: str) -> str:
    return _get_provider_config(provider_name)["env_var"]


def get_llm_shell_key_setup_hint(provider_name: str) -> str:
    cfg = _get_provider_config(provider_name)
    env_name = cfg["env_var"]
    hint = (
        f"Set key in shell (placeholder only):\n"
        f'PowerShell: $env:{env_name} = "<PASTE_KEY_HERE>"\n'
        f"CMD: set {env_name}=<PASTE_KEY_HERE>\n"
        f'Bash: export {env_name}="<PASTE_KEY_HERE>"'
    )
    if provider_name == "Microsoft Foundry (OpenAI-compatible)":
        hint += "\nOr use: Azure Portal → AI Foundry → Project → Keys"
    elif provider_name == "GitHub Models (OpenAI-compatible)":
        hint += "\nGenerate at: github.com/settings/tokens (Fine-grained or Classic)"
    elif provider_name == "Hugging Face Inference API":
        hint += "\nGenerate at: huggingface.co/settings/tokens"
    return hint


def resolve_llm_api_key(provider_name: str, api_key: str) -> tuple[str, str]:
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip(), "ui"

    env_name = get_llm_provider_env_var(provider_name)
    env_value = os.environ.get(env_name, "")
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip(), "env"

    return "", "missing"


def _resolve_api_key_internal(provider_name: str, api_key: str) -> tuple[str, str]:
    return resolve_llm_api_key(provider_name, api_key)


def fetch_provider_models(
    provider_name: str,
    base_url: str,
    api_key: str = "",
    timeout_sec: float = 5.0,
) -> tuple[list[str], str]:
    """Fetch model IDs from provider model discovery endpoints."""
    cfg = _get_provider_config(provider_name)
    clean_base = (base_url or "").strip().rstrip("/")
    catalog_url = str(cfg.get("catalog_url") or "").strip()
    if not clean_base:
        return [], "❌ Base URL is required to fetch models."

    if catalog_url:
        models_url = catalog_url
    elif "/openai" in clean_base:
        base_without_openai = clean_base.rsplit("/openai", 1)[0]
        if cfg.get("auth_style") == "api-key":
            models_url = base_without_openai + "/models?api-version=2024-10-21"
        else:
            models_url = base_without_openai + "/models"
    elif cfg.get("auth_style") == "api-key":
        models_url = clean_base + "/models?api-version=2024-10-21"
    else:
        models_url = clean_base + "/models"

    headers = dict(cfg.get("headers", {}))
    resolved_key, _key_source = _resolve_api_key_internal(provider_name, api_key)
    if resolved_key:
        if cfg.get("auth_style") == "api-key":
            headers["api-key"] = resolved_key
        else:
            headers["Authorization"] = f"Bearer {resolved_key}"
    elif cfg.get("requires_api_key"):
        env_var = get_llm_provider_env_var(provider_name)
        return [], f"⚠ API key required. Set {env_var} or enter it in the API Key field."

    if "generativelanguage.googleapis.com" in models_url and resolved_key:
        separator = "&" if "?" in models_url else "?"
        encoded_key = urllib.parse.quote(resolved_key, safe="")
        models_url = models_url + separator + "key=" + encoded_key
        headers.pop("Authorization", None)

    try:
        req = urllib.request.Request(url=models_url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if isinstance(payload, dict):
            items = payload.get("data", [])
            if not items:
                items = payload.get("models", [])
        else:
            items = []

        models: list[str] = []
        seen: set[str] = set()
        for item in items:
            if isinstance(item, dict):
                model_id = str(item.get("id") or "")
                if not model_id and item.get("name"):
                    model_id = str(item["name"])
                    if model_id.startswith("models/"):
                        model_id = model_id[7:]
                if model_id not in seen:
                    seen.add(model_id)
                    models.append(model_id)

        if models:
            return models, f"✅ Found {len(models)} model(s)"
        return [], "⚠ API responded but no models were listed. Load a model first."
    except urllib.error.HTTPError as error:
        return [], f"❌ HTTP {error.code}: {error.reason}"
    except (urllib.error.URLError, TimeoutError):
        return [], f"❌ Cannot reach {provider_name} API at {clean_base}"
    except Exception as error:
        return [], f"❌ Error fetching models: {error}"


def try_start_lm_studio() -> str:
    """Attempt to launch LM Studio if it is not already running."""
    try:
        req = urllib.request.Request(url="http://localhost:1234/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.status in (200, 204):
                return "✅ LM Studio already running"
    except Exception:
        pass

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LM Studio" / "LM Studio.exe",
        Path("C:/Program Files/LM Studio/LM Studio.exe"),
        Path("C:/Program Files (x86)/LM Studio/LM Studio.exe"),
        Path("/Applications/LM Studio.app/Contents/MacOS/LM Studio"),
    ]

    launched = False
    for candidate in candidates:
        if candidate.exists():
            try:
                subprocess.Popen([str(candidate)], cwd=str(candidate.parent))
                launched = True
                break
            except Exception:
                continue

    if not launched:
        return "❌ LM Studio not found. Install it or start it manually."

    deadline = time.time() + 25.0
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url="http://localhost:1234/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status in (200, 204):
                    return "✅ LM Studio started successfully"
        except Exception:
            pass
        time.sleep(1.5)

    return (
        "⚠ LM Studio launched but the API is not ready yet. "
        "Enable the local server in the Developer tab."
    )


def strip_think_tags(text: str) -> str:
    """Remove Qwen3-style <think>...</think> reasoning blocks from LLM output."""
    if not isinstance(text, str):
        return ""

    token_pattern = re.compile(r"</?think>", flags=re.IGNORECASE)
    depth = 0
    cursor = 0
    visible_parts: list[str] = []

    for match in token_pattern.finditer(text):
        if depth == 0 and match.start() > cursor:
            visible_parts.append(text[cursor : match.start()])

        if match.group(0).startswith("</"):
            depth = max(0, depth - 1)
        else:
            depth += 1
        cursor = match.end()

    if depth == 0 and cursor < len(text):
        visible_parts.append(text[cursor:])

    return "".join(visible_parts).strip()


def _clean_llm_transform_output(text: str, engine: str | None = None) -> str:
    text = strip_think_tags(text)
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(
        r"<\s*(speak|break|phoneme|prosody|lexicon|voice)\b[^>]*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"</\s*(speak|phoneme|prosody|lexicon|voice)\s*>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if engine:
        cleaned = strip_unsupported_cues(cleaned, engine)
    return cleaned.strip()


def _resolve_transform_engine_name(engine_name: str = "", engine: str = "") -> str:
    return (engine_name or engine or "").strip()


def _build_llm_system_prompt(
    system_prompt: str,
    engine_name: str = "",
    engine: str = "",
) -> str:
    effective_system_prompt = (system_prompt or "").strip() or DEFAULT_LLM_NARRATION_SYSTEM_PROMPT
    resolved_engine_name = _resolve_transform_engine_name(engine_name=engine_name, engine=engine)
    if not resolved_engine_name:
        return effective_system_prompt

    from engine_script_profiles import get_engine_prompt_addendum

    engine_addendum = get_engine_prompt_addendum(resolved_engine_name).strip()
    if not engine_addendum:
        return effective_system_prompt
    return f"{effective_system_prompt}\n\n## ENGINE-SPECIFIC OPTIMIZATION\n{engine_addendum}"


def _resolve_transform_max_chunk_chars(
    max_chunk_chars: int,
    engine_name: str = "",
    engine: str = "",
) -> int:
    resolved_engine_name = _resolve_transform_engine_name(engine_name=engine_name, engine=engine)
    if not resolved_engine_name or max_chunk_chars <= 0:
        return max_chunk_chars

    from engine_script_profiles import get_engine_max_chunk_chars

    engine_max_chunk_chars = get_engine_max_chunk_chars(resolved_engine_name)
    if engine_max_chunk_chars <= 0:
        return max_chunk_chars
    return min(max_chunk_chars, engine_max_chunk_chars)


def _build_llm_transform_user_prompt(
    source_text: str,
    mode: str,
    locale: str,
    style: str,
    max_tag_density: float,
) -> str:
    mode_reminder = _MODE_BEHAVIORAL_REMINDERS.get(
        mode, "apply appropriate narration normalization"
    )
    style_key = (style or "neutral").lower()
    style_desc = _STYLE_DESCRIPTIONS.get(style_key, style or "neutral")
    lines = [
        f"MODE: {mode} — {mode_reminder}",
        f"LOCALE: {locale}",
        f"STYLE: {style} — {style_desc}",
    ]
    if mode == "Vivid":
        lines.append(f"MAX_TAG_DENSITY: {max_tag_density} audio tags per 100 words")
    lines.append("")
    lines.append("Return ONLY the transformed text. No commentary.")
    lines.append("")
    lines.append("TEXT:")
    lines.append(source_text)
    return "\n".join(lines)


def _split_text_by_words(text: str, max_chunk_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if max_chunk_chars <= 0 or len(normalized) <= max_chunk_chars:
        return [normalized]

    chunks: list[str] = []
    current = ""
    for word in normalized.split():
        if len(word) > max_chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(word), max_chunk_chars):
                chunks.append(word[start : start + max_chunk_chars])
            continue

        if not current:
            current = word
            continue

        candidate = f"{current} {word}"
        if len(candidate) <= max_chunk_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)
    return chunks


def _split_paragraph_for_transform(paragraph: str, max_chunk_chars: int) -> list[str]:
    cleaned_paragraph = paragraph.strip()
    if not cleaned_paragraph:
        return []
    if max_chunk_chars <= 0 or len(cleaned_paragraph) <= max_chunk_chars:
        return [cleaned_paragraph]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_paragraph)
        if sentence.strip()
    ]
    if len(sentences) <= 1:
        return _split_text_by_words(cleaned_paragraph, max_chunk_chars)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence_parts = (
            _split_text_by_words(sentence, max_chunk_chars)
            if len(sentence) > max_chunk_chars
            else [sentence]
        )
        for part in sentence_parts:
            if not current:
                current = part
                continue

            candidate = f"{current} {part}"
            if len(candidate) <= max_chunk_chars:
                current = candidate
            else:
                chunks.append(current)
                current = part

    if current:
        chunks.append(current)
    return chunks


def chunk_text_for_transform(source_text: str, max_chunk_chars: int = 3000) -> list[TransformChunk]:
    """Split narration text into bounded chunks for LLM transforms.

    Args:
        source_text: Deterministically normalized narration text.
        max_chunk_chars: Soft maximum number of characters per chunk.

    Returns:
        Ordered chunks with separator metadata. Reassemble by concatenating each
        chunk's ``separator_before`` and ``text`` in sequence.
    """
    if not isinstance(source_text, str) or not source_text.strip():
        return []

    if max_chunk_chars <= 0:
        return [TransformChunk(text=source_text.strip())]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", source_text.strip())
        if paragraph.strip()
    ]
    if not paragraphs:
        return []

    units: list[TransformChunk] = []
    for paragraph in paragraphs:
        paragraph_segments = _split_paragraph_for_transform(paragraph, max_chunk_chars)
        for segment_index, segment in enumerate(paragraph_segments):
            if not units:
                separator_before = ""
            elif segment_index == 0:
                separator_before = "\n\n"
            else:
                separator_before = " "
            units.append(TransformChunk(text=segment, separator_before=separator_before))

    packed_chunks: list[TransformChunk] = []
    current_text = ""
    current_separator = ""
    for unit in units:
        if not current_text:
            current_text = unit.text
            current_separator = unit.separator_before
            continue

        candidate = f"{current_text}{unit.separator_before}{unit.text}"
        if len(candidate) <= max_chunk_chars:
            current_text = candidate
        else:
            packed_chunks.append(
                TransformChunk(text=current_text, separator_before=current_separator)
            )
            current_text = unit.text
            current_separator = unit.separator_before

    if current_text:
        packed_chunks.append(TransformChunk(text=current_text, separator_before=current_separator))

    return packed_chunks


def _run_llm_transform_chunk(
    chunk_text: str,
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str,
    mode: str,
    locale: str,
    style: str,
    max_tag_density: float,
    timeout_seconds: int,
    temperature: float = 0.25,
    top_p: float = 0.85,
    repeat_penalty: float = 1.08,
    max_tokens: int = 1024,
    extra_headers: dict[str, str] | None = None,
    auth_style: str = "bearer",
    engine: str = "",
) -> str:
    user_prompt = _build_llm_transform_user_prompt(
        source_text=chunk_text,
        mode=mode,
        locale=locale,
        style=style,
        max_tag_density=max_tag_density,
    )
    raw_output = call_openai_compatible_chat(
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        extra_headers=extra_headers,
        auth_style=auth_style,
    )
    return _clean_llm_transform_output(raw_output, engine=engine)


def call_openai_compatible_chat(
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int = 60,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repeat_penalty: float = 0.0,
    max_tokens: int = 1024,
    extra_headers: dict[str, str] | None = None,
    auth_style: str = "bearer",
) -> str:
    if not base_url or not model_id:
        raise ValueError("Base URL and model ID are required")

    if auth_style == "api-key":
        endpoint = (
            base_url.rstrip("/")
            + f"/openai/deployments/{model_id}/chat/completions?api-version=2024-10-21"
        )
    else:
        endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if float(repeat_penalty) > 0:
        payload["repetition_penalty"] = float(repeat_penalty)

    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if isinstance(api_key, str) and api_key.strip():
        if auth_style == "api-key":
            headers["api-key"] = api_key.strip()
        else:
            headers["Authorization"] = f"Bearer {api_key.strip()}"

    request_body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=request_body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=max(5, int(timeout_seconds))) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)

    choices = data.get("choices") or []
    if not choices:
        raise ValueError("No choices returned by LLM endpoint")

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty")
    return content.strip()


def _normalize_voice_casting_speaker_names(speaker_names: list[str]) -> list[str]:
    normalized_names: list[str] = []
    seen_names: set[str] = set()

    for speaker_name in speaker_names:
        clean_name = str(speaker_name or "").strip()
        if not clean_name:
            continue

        folded_name = clean_name.casefold()
        if folded_name in seen_names:
            continue

        seen_names.add(folded_name)
        normalized_names.append(clean_name)

    return normalized_names


def _normalize_voice_casting_block(block_lines: list[str]) -> tuple[str, str]:
    description_parts: list[str] = []
    voice_profile_parts: list[str] = []
    extra_lines: list[str] = []
    current_field = ""

    for block_line in block_lines:
        stripped_line = block_line.strip()
        if not stripped_line:
            current_field = ""
            continue

        description_match = re.match(r"^Description:\s*(.+)$", stripped_line, re.IGNORECASE)
        if description_match:
            description_parts = [description_match.group(1).strip()]
            current_field = "description"
            continue

        voice_profile_match = re.match(
            r"^Voice\s*Profile:\s*(.+)$",
            stripped_line,
            re.IGNORECASE,
        )
        if voice_profile_match:
            voice_profile_parts = [voice_profile_match.group(1).strip()]
            current_field = "voice_profile"
            continue

        if current_field == "description":
            description_parts.append(stripped_line)
        elif current_field == "voice_profile":
            voice_profile_parts.append(stripped_line)
        else:
            extra_lines.append(stripped_line)

    description_text = " ".join(description_parts).strip()
    voice_profile_text = " ".join(voice_profile_parts).strip()

    if not description_text and extra_lines:
        description_text = " ".join(extra_lines).strip()
    if extra_lines and voice_profile_text:
        description_text = " ".join(
            part for part in [description_text, *extra_lines] if part
        ).strip()

    description_text = re.sub(r"\s+", " ", description_text).strip()
    voice_profile_text = re.sub(r"\s+", " ", voice_profile_text).strip()

    return description_text, voice_profile_text


def _format_voice_casting_result(raw_text: str, speaker_names: list[str]) -> str:
    clean_names = _normalize_voice_casting_speaker_names(speaker_names)
    if not clean_names:
        return ""

    heading_pattern = re.compile(
        r"^\s*(?:[-*]\s*|\d+\.\s*)?(?P<name>"
        + "|".join(re.escape(name) for name in sorted(clean_names, key=len, reverse=True))
        + r")\s*:?\s*$",
        re.IGNORECASE,
    )
    canonical_names = {name.casefold(): name for name in clean_names}
    sections: dict[str, list[str]] = {name: [] for name in clean_names}
    current_name: str | None = None

    for raw_line in str(raw_text or "").splitlines():
        header_match = heading_pattern.match(raw_line.rstrip())
        if header_match:
            current_name = canonical_names[header_match.group("name").casefold()]
            continue

        if current_name is not None:
            sections[current_name].append(raw_line)

    formatted_sections: list[str] = []
    for speaker_name in clean_names:
        description_text, voice_profile_text = _normalize_voice_casting_block(
            sections.get(speaker_name, [])
        )
        if not description_text and not voice_profile_text:
            description_text = "No description returned."
        if not voice_profile_text:
            voice_profile_text = "Not provided"

        formatted_sections.append(
            "\n".join(
                [
                    f"{speaker_name}:",
                    f"  Description: {description_text}",
                    f"  Voice Profile: {voice_profile_text}",
                ]
            )
        )

    return "\n\n".join(formatted_sections)


def generate_voice_casting(
    speaker_names: list[str],
    base_url: str,
    api_key: str,
    model_id: str,
    timeout_seconds: int,
    extra_headers: dict[str, str],
    auth_style: str,
) -> tuple[str, str]:
    clean_speaker_names = _normalize_voice_casting_speaker_names(speaker_names)
    if not clean_speaker_names:
        return "", "At least one speaker name is required."

    clean_base_url = str(base_url or "").strip()
    if not clean_base_url:
        return "", "Base URL is required."

    clean_model_id = str(model_id or "").strip()
    if not clean_model_id:
        return "", "Model ID is required."

    user_prompt = "\n".join(
        [
            "Generate voice casting guidance for the following speakers:",
            *(f"- {speaker_name}" for speaker_name in clean_speaker_names),
            "",
            "Return every speaker using the exact format requested in the system prompt.",
        ]
    )

    try:
        raw_output = call_openai_compatible_chat(
            base_url=clean_base_url,
            api_key=str(api_key or "").strip(),
            model_id=clean_model_id,
            system_prompt=VOICE_CASTING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            timeout_seconds=int(timeout_seconds or 60),
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            extra_headers=extra_headers,
            auth_style=auth_style,
        )
        raw_output = strip_think_tags(raw_output)
        return _format_voice_casting_result(raw_output, clean_speaker_names), ""
    except urllib.error.HTTPError as error:
        return "", f"LLM HTTP error: {error.code} {error.reason}"
    except urllib.error.URLError as error:
        return "", f"LLM URL error: {error.reason}"
    except Exception as error:
        return "", str(error)


def test_llm_connection(
    provider_name: str,
    base_url: str,
    api_key: str,
    model_id: str,
    timeout_seconds: int,
) -> str:
    if not model_id or not str(model_id).strip():
        return "❌ Enter a model ID before testing connection."

    cfg = _get_provider_config(provider_name)
    resolved_api_key, key_source = resolve_llm_api_key(provider_name, api_key)
    if cfg["requires_api_key"] and not resolved_api_key:
        return f"❌ API key required for {provider_name}.\n" + get_llm_shell_key_setup_hint(
            provider_name
        )

    try:
        response_text = call_openai_compatible_chat(
            base_url=base_url,
            api_key=resolved_api_key,
            model_id=model_id,
            system_prompt="Return exactly: OK",
            user_prompt="Reply with OK only.",
            timeout_seconds=timeout_seconds,
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
            extra_headers=cfg["headers"],
            auth_style=cfg["auth_style"],
        )
        return (
            f"✅ LLM connection successful\n"
            f"Provider: {provider_name}\n"
            f"URL: {base_url}\n"
            f"Model: {model_id}\n"
            f"API key source: {key_source}\n"
            f"Sample response: {response_text[:120]}"
        )
    except urllib.error.HTTPError as error:
        hint = get_llm_shell_key_setup_hint(provider_name)
        return f"❌ LLM HTTP error: {error.code} {error.reason}\n{hint}"
    except urllib.error.URLError as error:
        return f"❌ LLM URL error: {error.reason}"
    except Exception as error:
        hint = get_llm_shell_key_setup_hint(provider_name)
        return f"❌ LLM connection failed: {error}\n{hint}"


def apply_llm_narration_transform(
    source_text: str,
    enabled: bool,
    provider_name: str,
    base_url: str,
    api_key: str,
    model_id: str,
    mode: str,
    locale: str,
    style: str,
    max_tag_density: float,
    system_prompt: str,
    timeout_seconds: int,
    temperature: float = 0.25,
    top_p: float = 0.85,
    repeat_penalty: float = 1.08,
    max_tokens: int = 1024,
    allow_local_fallback: bool = True,
    engine_name: str = "",
    max_chunk_chars: int = 3000,
    engine: str = "",
) -> tuple[str, str]:
    if not isinstance(source_text, str) or not source_text.strip():
        return source_text, "Narration transform: skipped (empty text)"

    deterministic_text = deterministic_normalize(source_text)
    resolved_engine_name = _resolve_transform_engine_name(engine_name=engine_name, engine=engine)

    if not enabled:
        return (
            deterministic_text,
            "Narration transform: deterministic normalization applied; LLM disabled",
        )

    if not model_id or not str(model_id).strip():
        return (
            deterministic_text,
            "Narration transform: deterministic normalization applied; "
            "LLM skipped (no model ID provided)",
        )

    cfg = _get_provider_config(provider_name)
    resolved_api_key, key_source = resolve_llm_api_key(provider_name, api_key)
    if cfg["requires_api_key"] and not resolved_api_key:
        return deterministic_text, (
            "Narration transform: deterministic normalization applied; "
            f"LLM skipped ({provider_name} API key missing)\n"
            + get_llm_shell_key_setup_hint(provider_name)
        )

    effective_system_prompt = _build_llm_system_prompt(
        system_prompt=system_prompt,
        engine_name=resolved_engine_name,
    )
    effective_max_chunk_chars = _resolve_transform_max_chunk_chars(
        max_chunk_chars=max_chunk_chars,
        engine_name=resolved_engine_name,
    )
    chunk_specs = chunk_text_for_transform(
        deterministic_text,
        max_chunk_chars=effective_max_chunk_chars,
    )
    if not chunk_specs:
        chunk_specs = [TransformChunk(text=deterministic_text)]

    if len(deterministic_text) > 3000 and len(chunk_specs) > 1:
        logger.warning(
            "LLM narration transform input is %s chars; chunking into %s requests "
            "with max_chunk_chars=%s",
            len(deterministic_text),
            len(chunk_specs),
            effective_max_chunk_chars,
        )

    try:
        if len(chunk_specs) == 1:
            cleaned = _run_llm_transform_chunk(
                deterministic_text,
                base_url=base_url,
                api_key=resolved_api_key,
                model_id=model_id,
                system_prompt=effective_system_prompt,
                mode=mode,
                locale=locale,
                style=style,
                max_tag_density=max_tag_density,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens,
                extra_headers=cfg["headers"],
                auth_style=cfg["auth_style"],
                engine=resolved_engine_name,
            )
        else:
            transformed_parts: list[str] = []
            for chunk_spec in chunk_specs:
                cleaned_chunk = _run_llm_transform_chunk(
                    chunk_spec.text,
                    base_url=base_url,
                    api_key=resolved_api_key,
                    model_id=model_id,
                    system_prompt=effective_system_prompt,
                    mode=mode,
                    locale=locale,
                    style=style,
                    max_tag_density=max_tag_density,
                    timeout_seconds=timeout_seconds,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    max_tokens=max_tokens,
                    extra_headers=cfg["headers"],
                    auth_style=cfg["auth_style"],
                    engine=resolved_engine_name,
                )
                if not cleaned_chunk:
                    if allow_local_fallback:
                        return (
                            deterministic_text,
                            "Narration transform: deterministic normalization applied; "
                            "LLM returned empty output for a chunk, returned deterministic result",
                        )
                    return (
                        source_text,
                        "Narration transform: deterministic normalization applied; "
                        "LLM returned empty output for a chunk, using original text",
                    )

                if transformed_parts:
                    transformed_parts.append(chunk_spec.separator_before)
                transformed_parts.append(cleaned_chunk.strip())

            cleaned = "".join(transformed_parts).strip()

        if not cleaned:
            if allow_local_fallback:
                return (
                    deterministic_text,
                    "Narration transform: deterministic normalization applied; "
                    "LLM returned empty output, returned deterministic result",
                )
            return (
                source_text,
                "Narration transform: deterministic normalization applied; "
                "LLM returned empty output, using original text",
            )
        return cleaned, (
            "Narration transform: deterministic normalization applied; "
            "LLM styling applied\n"
            f"Provider: {provider_name}\n"
            f"Model: {model_id}\n"
            f"API key source: {key_source}\n"
            f"Mode: {mode}" + (f"\nChunks: {len(chunk_specs)}" if len(chunk_specs) > 1 else "")
        )
    except Exception as error:
        if allow_local_fallback:
            return (
                deterministic_text,
                "Narration transform: deterministic normalization applied; "
                f"LLM failed ({error}), returned deterministic result",
            )
        return (
            source_text,
            "Narration transform: deterministic normalization applied; "
            f"LLM failed ({error}), using original text",
        )


def apply_llm_transform_to_textbox(
    source_text: str,
    provider_name: str,
    base_url: str,
    api_key: str,
    model_id: str,
    mode: str,
    locale: str,
    style: str,
    max_tag_density: float,
    system_prompt: str,
    timeout_seconds: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    allow_local_fallback: bool,
    engine_name: str = "",
    engine: str = "",
) -> tuple[str, str]:
    transformed_text, status = apply_llm_narration_transform(
        source_text=source_text,
        enabled=True,
        provider_name=provider_name,
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        mode=mode,
        locale=locale,
        style=style,
        max_tag_density=max_tag_density,
        system_prompt=system_prompt,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        allow_local_fallback=allow_local_fallback,
        engine_name=engine_name,
        engine=engine,
    )
    if isinstance(status, str) and status.startswith("LLM transform:"):
        status = status.replace("LLM transform:", "Narration transform:", 1)
    return transformed_text, status


def format_provenance(status_text: str) -> str:
    normalized_status = (status_text or "").lower()
    if "deterministic" in normalized_status or "fallback" in normalized_status:
        return "⚠️ **Fallback result** — LLM was unavailable; deterministic rules applied."
    return "✨ **AI-transformed** — Review changes before accepting."
