"""Pure narration transform and LLM provider utilities.

This module holds deterministic normalization, provider configuration, model discovery,
OpenAI-compatible chat calls, and narration transform orchestration. It intentionally avoids any
Gradio imports so the logic can be tested and reused independently from the UI layer.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


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
        r"\bDr\.\b": "Doctor",
        r"\bMr\.\b": "Mister",
        r"\bMrs\.\b": "Missus",
        r"\bMs\.\b": "Miss",
        r"\bAve\.\b": "Avenue",
        r"\bSt\.\b": "Street",
    }
    for pattern, replacement in abbreviation_map.items():
        text = re.sub(pattern, replacement, text)

    def phone_replacer(match: re.Match[str]) -> str:
        g1, g2, g3 = match.groups()
        return f"{_spell_digits(g1)}, {_spell_digits(g2)}, {_spell_digits(g3)}"

    text = re.sub(r"\b(\d{3})-(\d{3})-(\d{4})\b", phone_replacer, text)

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

    text = re.sub(r"\b(\d+)%\b", percent_replacer, text)

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
        "base_url": "https://models.github.ai/v1",
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
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
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

_DEFAULT_ENGINE_EXPRESSIVENESS = {
    "bracket_cues": False,
    "ssml": False,
    "allcaps_emphasis": False,
    "ellipsis_pause": True,
    "emotion_vectors": False,
}

_BRACKET_CUE_PATTERN = re.compile(r"\[(?=[^\]\n]*[A-Za-z])[^\]\n]{1,80}\]")
_PAREN_STAGE_DIRECTION_PATTERN = re.compile(r"\((?=[^\)\n]*[A-Za-z])[^\)\n]{1,80}\)")
_SSML_TAG_PATTERN = re.compile(r"</?\s*[A-Za-z][\w:-]*(?:\s+[^<>]*)?\s*/?>")
_MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:!?])")
_EXCESS_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
_ALLCAPS_WORD_PATTERN = re.compile(r"\b[A-Z][A-Z'-]{2,}\b")
_PRESERVED_ALLCAPS_ACRONYMS = {
    "AI",
    "API",
    "ASCII",
    "CIA",
    "CLI",
    "CPU",
    "CSS",
    "CSV",
    "DVD",
    "EU",
    "FAQ",
    "FBI",
    "GPU",
    "HTML",
    "HTTP",
    "HTTPS",
    "IDE",
    "JSON",
    "LLM",
    "ML",
    "NASA",
    "NATO",
    "NLP",
    "OCR",
    "PDF",
    "RAM",
    "SDK",
    "SQL",
    "SSH",
    "TCP",
    "TTS",
    "UI",
    "UN",
    "URI",
    "URL",
    "USA",
    "USB",
    "UX",
    "UDP",
    "WAV",
    "XML",
    "YAML",
}

ENGINE_EXPRESSIVENESS: dict[str, dict[str, bool]] = {
    "ChatterboxTTS": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "Chatterbox Multilingual": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "Chatterbox Turbo": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "Kokoro TTS": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "Fish Speech": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "IndexTTS": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "IndexTTS2": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": True,
    },
    "F5-TTS": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "Higgs Audio": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "VoxCPM": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "KittenTTS": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "Qwen Voice Design": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "Qwen Voice Clone": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "Qwen Custom Voice": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
    "VibeVoice": {
        "bracket_cues": False,
        "ssml": False,
        "allcaps_emphasis": False,
        "ellipsis_pause": True,
        "emotion_vectors": False,
    },
}


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
    if not clean_base:
        return [], "❌ Base URL is required to fetch models."

    if "/openai" in clean_base:
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


def _clean_llm_transform_output(text: str, engine: str | None = None) -> str:
    if not isinstance(text, str):
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


def _normalize_allcaps_word(match: re.Match[str]) -> str:
    word = match.group(0)
    if word in _PRESERVED_ALLCAPS_ACRONYMS:
        return word
    return word.title()


def strip_unsupported_cues(text: str, engine: str) -> str:
    """Strip expressive markup that the selected engine will read literally."""
    if not isinstance(text, str):
        return ""

    capabilities = ENGINE_EXPRESSIVENESS.get(engine, _DEFAULT_ENGINE_EXPRESSIVENESS)
    cleaned = text

    if not capabilities.get("bracket_cues", False):
        cleaned = _BRACKET_CUE_PATTERN.sub("", cleaned)
        cleaned = _PAREN_STAGE_DIRECTION_PATTERN.sub("", cleaned)
        cleaned = _MULTISPACE_PATTERN.sub(" ", cleaned)
        cleaned = _SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", cleaned)

    if not capabilities.get("ssml", False):
        cleaned = _SSML_TAG_PATTERN.sub("", cleaned)

    if not capabilities.get("allcaps_emphasis", False):
        cleaned = _ALLCAPS_WORD_PATTERN.sub(_normalize_allcaps_word, cleaned)

    cleaned = _MULTISPACE_PATTERN.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", cleaned)
    cleaned = _EXCESS_BLANK_LINES_PATTERN.sub("\n\n", cleaned)
    return cleaned.strip()


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


def call_openai_compatible_chat(
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int = 60,
    temperature: float = 0.2,
    top_p: float = 0.9,
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
    temperature: float,
    top_p: float,
    max_tokens: int,
    allow_local_fallback: bool = True,
    engine: str = "",
) -> tuple[str, str]:
    if not isinstance(source_text, str) or not source_text.strip():
        return source_text, "Narration transform: skipped (empty text)"

    deterministic_text = deterministic_normalize(source_text)

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

    effective_system_prompt = (system_prompt or "").strip() or DEFAULT_LLM_NARRATION_SYSTEM_PROMPT
    user_prompt = _build_llm_transform_user_prompt(
        source_text=deterministic_text,
        mode=mode,
        locale=locale,
        style=style,
        max_tag_density=max_tag_density,
    )

    try:
        raw_output = call_openai_compatible_chat(
            base_url=base_url,
            api_key=resolved_api_key,
            model_id=model_id,
            system_prompt=effective_system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_headers=cfg["headers"],
            auth_style=cfg["auth_style"],
        )
        cleaned = _clean_llm_transform_output(raw_output, engine=engine)
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
            f"Mode: {mode}"
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