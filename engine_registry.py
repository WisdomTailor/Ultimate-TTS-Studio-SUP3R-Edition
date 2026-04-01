"""Engine capability registry and cue stripping for TTS engines.

Central source of truth for engine capabilities, supported features,
and metadata control mappings.
"""

from __future__ import annotations

import re


_DEFAULT_ENGINE_EXPRESSIVENESS: dict[str, bool] = {
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
        "bracket_cues": True,
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

ENGINE_METADATA_CONTROL_MAP: dict[str, list[tuple[str, str]]] = {
    "ChatterboxTTS": [
        ("chatterbox_exaggeration", "exaggeration"),
        ("chatterbox_temperature", "temperature"),
        ("chatterbox_cfg_weight", "cfg_weight"),
        ("chatterbox_chunk_size", "chunk_size"),
    ],
    "Chatterbox Multilingual": [
        ("chatterbox_mtl_language", "language"),
        ("chatterbox_mtl_exaggeration", "exaggeration"),
        ("chatterbox_mtl_temperature", "temperature"),
        ("chatterbox_mtl_cfg_weight", "cfg_weight"),
        ("chatterbox_mtl_repetition_penalty", "repetition_penalty"),
        ("chatterbox_mtl_min_p", "min_p"),
        ("chatterbox_mtl_top_p", "top_p"),
        ("chatterbox_mtl_chunk_size", "chunk_size"),
    ],
    "Chatterbox Turbo": [
        ("chatterbox_turbo_exaggeration", "exaggeration"),
        ("chatterbox_turbo_temperature", "temperature"),
        ("chatterbox_turbo_cfg_weight", "cfg_weight"),
        ("chatterbox_turbo_repetition_penalty", "repetition_penalty"),
        ("chatterbox_turbo_min_p", "min_p"),
        ("chatterbox_turbo_top_p", "top_p"),
        ("chatterbox_turbo_chunk_size", "chunk_size"),
    ],
    "Kokoro TTS": [
        ("kokoro_voice", "voice"),
        ("kokoro_speed", "speed"),
    ],
    "Fish Speech": [
        ("fish_temperature", "temperature"),
        ("fish_top_p", "top_p"),
        ("fish_repetition_penalty", "repetition_penalty"),
        ("fish_max_tokens", "max_tokens"),
    ],
    "IndexTTS": [
        ("indextts_temperature", "temperature"),
    ],
    "IndexTTS2": [
        ("indextts2_emotion_mode", "emotion_mode"),
        ("indextts2_emo_alpha", "emo_alpha"),
        ("indextts2_happy", "happy"),
        ("indextts2_angry", "angry"),
        ("indextts2_sad", "sad"),
        ("indextts2_afraid", "afraid"),
        ("indextts2_disgusted", "disgusted"),
        ("indextts2_melancholic", "melancholic"),
        ("indextts2_surprised", "surprised"),
        ("indextts2_calm", "calm"),
        ("indextts2_temperature", "temperature"),
        ("indextts2_top_p", "top_p"),
        ("indextts2_top_k", "top_k"),
        ("indextts2_repetition_penalty", "repetition_penalty"),
        ("indextts2_max_mel_tokens", "max_mel_tokens"),
        ("indextts2_use_random", "use_random"),
    ],
    "F5-TTS": [
        ("f5_speed", "speed"),
        ("f5_cross_fade", "cross_fade"),
        ("f5_remove_silence", "remove_silence"),
    ],
    "Higgs Audio": [
        ("higgs_voice_preset", "voice_preset"),
        ("higgs_system_prompt", "system_prompt"),
        ("higgs_temperature", "temperature"),
        ("higgs_top_p", "top_p"),
        ("higgs_top_k", "top_k"),
        ("higgs_max_tokens", "max_tokens"),
        ("higgs_ras_win_len", "ras_win_len"),
        ("higgs_ras_win_max_num_repeat", "ras_win_max_num_repeat"),
    ],
    "KittenTTS": [
        ("kitten_voice", "voice"),
    ],
    "VoxCPM": [
        ("voxcpm_cfg_value", "cfg_value"),
        ("voxcpm_inference_timesteps", "inference_timesteps"),
        ("voxcpm_normalize", "normalize"),
        ("voxcpm_denoise", "denoise"),
        ("voxcpm_retry_badcase", "retry_badcase"),
        ("voxcpm_retry_badcase_max_times", "retry_badcase_max_times"),
        (
            "voxcpm_retry_badcase_ratio_threshold",
            "retry_badcase_ratio_threshold",
        ),
    ],
    "Qwen Voice Design": [
        ("qwen_language", "language"),
        ("qwen_voice_description", "voice_description"),
    ],
    "Qwen Voice Clone": [
        ("qwen_language", "language"),
        ("qwen_ref_text", "ref_text"),
        ("qwen_xvector_only", "xvector_only"),
        ("qwen_clone_model_size", "model_size"),
        ("qwen_chunk_size", "chunk_size"),
        ("qwen_chunk_gap", "chunk_gap"),
    ],
    "Qwen Custom Voice": [
        ("qwen_speaker", "speaker_profile"),
        ("qwen_language", "language"),
        ("qwen_style_instruct", "style_instruct"),
        ("qwen_custom_model_size", "model_size"),
    ],
}


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


__all__ = [
    "_DEFAULT_ENGINE_EXPRESSIVENESS",
    "ENGINE_EXPRESSIVENESS",
    "strip_unsupported_cues",
    "ENGINE_METADATA_CONTROL_MAP",
]
