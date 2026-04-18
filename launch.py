# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalIterable=false, reportPossiblyUnboundVariable=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportAssignmentType=false, reportGeneralTypeIssues=false
# NOTE: Before editing this file, read Docs/launch-py-index.md.
# app/launch.py is a large coupled Gradio/runtime module; use the index first
# to find the correct section, handlers, and event bindings.

import random
import copy
import numpy as np
import torch
import gradio as gr
import os
import subprocess
import sys
import warnings
import re
import json
import io
import logging
import hashlib
import inspect
import importlib
import importlib.util as importlib_util
import importlib.metadata as importlib_metadata
import unicodedata
import urllib.request
import urllib.error
from typing import Any, Optional
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)

# Suppress redirect warning on Windows/MacOS
warnings.filterwarnings("ignore", message="Redirects are currently not supported")
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

# ===== COMPREHENSIVE WARNING SUPPRESSION =====
# Suppress all warnings to clean up console output
warnings.filterwarnings("ignore")

# Suppress specific warning categories
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set environment variables to suppress various library warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings

# Suppress torch distributed warnings
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

# Redirect stderr temporarily to suppress specific warnings during imports
import contextlib
from io import StringIO

AudioSegment = None


# Custom context manager to suppress specific warning patterns
@contextlib.contextmanager
def suppress_specific_warnings():
    """Context manager to suppress specific warning patterns"""
    old_stderr = sys.stderr
    sys.stderr = captured_stderr = StringIO()
    try:
        yield
    finally:
        # Filter out specific warning patterns and only show important errors
        captured_output = captured_stderr.getvalue()
        filtered_lines = []

        # Patterns to completely suppress
        suppress_patterns = [
            "Setting ds_accelerator to cuda",
            "cannot open input file",
            "LINK : fatal error LNK1181",
            "Redirects are currently not supported",
            "DeepSpeed info:",
            "Config parameter mp_size is deprecated",
            "quantize_bits =",
            "Removing weight norm",
            "bigvgan weights restored",
            "Text normalization dependencies not available",
            "No module named 'pynini'",
            "Using fallback normalizer",
            "test.c",
        ]

        for line in captured_output.split("\n"):
            should_suppress = False
            for pattern in suppress_patterns:
                if pattern in line:
                    should_suppress = True
                    break

            if not should_suppress and line.strip():
                filtered_lines.append(line)

        # Only print non-suppressed lines
        if filtered_lines:
            sys.stderr = old_stderr
            for line in filtered_lines:
                print(line, file=sys.stderr)

        sys.stderr = old_stderr


# ===== END WARNING SUPPRESSION =====

# Add current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add indextts module path for imports
indextts_path = os.path.join(current_dir, "indextts")
if indextts_path not in sys.path:
    sys.path.insert(0, indextts_path)

# Use warning suppression context for imports that generate warnings
with suppress_specific_warnings():
    from scipy.io import wavfile
    from scipy import signal
    import tempfile
    import shutil
    import glob
    from tqdm import tqdm
    from scipy.io.wavfile import write

try:
    with suppress_specific_warnings():
        from scipy.signal import filtfilt
    AUDIO_PROCESSING_AVAILABLE = True
except Exception as error:
    filtfilt = None
    AUDIO_PROCESSING_AVAILABLE = False
    print(f"⚠️ Advanced audio processing unavailable. Error: {error}")

try:
    with suppress_specific_warnings():
        import soundfile as sf
except Exception as error:
    sf = None
    print(f"⚠️ soundfile not available. Falling back where possible. Error: {error}")

try:
    with suppress_specific_warnings():
        from pydub import AudioSegment
except Exception:
    AudioSegment = None

# Chatterbox imports
try:
    with suppress_specific_warnings():
        from chatterbox.src.chatterbox.tts import ChatterboxTTS
        from chatterbox.src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    CHATTERBOX_AVAILABLE = True
    CHATTERBOX_MULTILINGUAL_AVAILABLE = True
except Exception as error:
    CHATTERBOX_AVAILABLE = False
    CHATTERBOX_MULTILINGUAL_AVAILABLE = False
    ChatterboxTTS = None
    ChatterboxMultilingualTTS = None
    SUPPORTED_LANGUAGES = []
    print(f"⚠️ ChatterboxTTS not available. Some features will be disabled. Error: {error}")

# Kokoro imports
try:
    with suppress_specific_warnings():
        from kokoro import KModel, KPipeline
    KOKORO_AVAILABLE = True
except Exception as error:
    KOKORO_AVAILABLE = False
    KModel = None
    KPipeline = None
    print(f"⚠️ Kokoro TTS not available. Some features will be disabled. Error: {error}")

# Fish Speech imports
try:
    with suppress_specific_warnings():
        import queue
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
        from fish_speech.utils.file import audio_to_bytes
    FISH_SPEECH_AVAILABLE = True
except Exception as error:
    FISH_SPEECH_AVAILABLE = False
    TTSInferenceEngine = None
    load_decoder_model = None
    launch_thread_safe_queue = None
    ServeTTSRequest = None
    ServeReferenceAudio = None
    audio_to_bytes = lambda *args, **kwargs: b""
    print(f"⚠️ Fish Speech not available. Some features will be disabled. Error: {error}")

# F5-TTS imports
try:
    with suppress_specific_warnings():
        from f5_tts_handler import get_f5_tts_handler
    F5_TTS_AVAILABLE = True
    print("✅ F5-TTS handler loaded")
except Exception as error:
    F5_TTS_AVAILABLE = False
    get_f5_tts_handler = lambda *args, **kwargs: None
    print(f"⚠️ F5-TTS not available. Some features will be disabled. Error: {error}")


# Higgs Audio imports
def _get_higgs_audio_handler_placeholder():
    """Placeholder for get_higgs_audio_handler when handler is unavailable"""
    return None


def _tts_unavailable(*args, **kwargs):
    """Common placeholder for unavailable TTS generation functions."""
    return None, "❌ TTS handler not available"


def _init_unavailable(*args, **kwargs):
    """Common placeholder for unavailable model init functions."""
    return False, "❌ TTS handler not available"


def _unload_unavailable(*args, **kwargs):
    """Common placeholder for unavailable model unload functions."""
    return "❌ TTS handler not available"


def _transcribe_unavailable(*args, **kwargs):
    """Common placeholder for unavailable transcription functions."""
    return ""


# Provide safe defaults so static analysis does not treat these as unbound.
generate_kitten_tts = _tts_unavailable
init_kitten_tts = _init_unavailable
unload_kitten_tts = _unload_unavailable
KITTEN_VOICES = []

generate_vibevoice_podcast = _tts_unavailable
init_vibevoice = _init_unavailable
unload_vibevoice = _unload_unavailable
get_vibevoice_status = _transcribe_unavailable
get_vibevoice_voices = lambda *args, **kwargs: []
scan_vibevoice_models = lambda *args, **kwargs: []
download_vibevoice_model = _init_unavailable

generate_chatterbox_turbo_tts = _tts_unavailable
init_chatterbox_turbo = _init_unavailable
unload_chatterbox_turbo = _unload_unavailable
get_chatterbox_turbo_status = _transcribe_unavailable

init_qwen_tts = _init_unavailable
unload_qwen_tts = _unload_unavailable
get_qwen_tts_status = _transcribe_unavailable
generate_qwen_voice_design_tts = _tts_unavailable
generate_qwen_voice_clone_tts = _tts_unavailable
generate_qwen_custom_voice_tts = _tts_unavailable
transcribe_qwen_audio = _transcribe_unavailable

generate_indextts2_tts = _tts_unavailable
init_indextts2 = _init_unavailable
unload_indextts2 = _unload_unavailable
get_indextts2_status = _transcribe_unavailable
check_indextts2_models = lambda *args, **kwargs: False
download_indextts2_models = _init_unavailable
EMOTION_PRESETS = {}

generate_higgs_audio_tts = _tts_unavailable

transcribe_voxcpm_audio = _transcribe_unavailable


try:
    with suppress_specific_warnings():
        from higgs_audio_handler import generate_higgs_audio_tts, get_higgs_audio_handler
    HIGGS_AUDIO_AVAILABLE = True
    print("✅ Higgs Audio handler loaded")
except Exception as error:
    HIGGS_AUDIO_AVAILABLE = False
    get_higgs_audio_handler = _get_higgs_audio_handler_placeholder
    print(f"⚠️ Higgs Audio not available. Some features will be disabled. Error: {error}")

# KittenTTS imports
try:
    with suppress_specific_warnings():
        from kitten_tts_handler import (
            generate_kitten_tts,
            get_kitten_tts_handler,
            init_kitten_tts,
            unload_kitten_tts,
            KITTEN_VOICES,
        )
    KITTEN_TTS_AVAILABLE = True
    print("✅ KittenTTS handler loaded")
except Exception as error:
    KITTEN_TTS_AVAILABLE = False
    print(f"⚠️ KittenTTS not available. Some features will be disabled. Error: {error}")

# VibeVoice imports
try:
    with suppress_specific_warnings():
        from vibevoice_handler import (
            get_vibevoice_handler,
            generate_vibevoice_podcast,
            init_vibevoice,
            unload_vibevoice,
            get_vibevoice_status,
            get_vibevoice_voices,
            scan_vibevoice_models,
            download_vibevoice_model,
        )
    VIBEVOICE_AVAILABLE = True
    print("✅ VibeVoice handler loaded")
except Exception as error:
    VIBEVOICE_AVAILABLE = False
    print(f"⚠️ VibeVoice not available. Some features will be disabled. Error: {error}")


# VoxCPM imports
def _unload_voxcpm_placeholder():
    """Placeholder for unload_voxcpm when handler is unavailable"""
    return "⚠️ VoxCPM not available"


try:
    with suppress_specific_warnings():
        from voxcpm_handler import (
            get_voxcpm_handler,
            generate_voxcpm_tts,
            init_voxcpm,
            unload_voxcpm,
            get_voxcpm_status,
            transcribe_voxcpm_audio,
        )
    VOXCPM_AVAILABLE = True
    print("✅ VoxCPM handler loaded")
except Exception as error:
    VOXCPM_AVAILABLE = False
    unload_voxcpm = _unload_voxcpm_placeholder
    print(f"⚠️ VoxCPM not available. Some features will be disabled. Error: {error}")

# Chatterbox Turbo imports
try:
    with suppress_specific_warnings():
        from chatterbox_turbo_handler import (
            get_chatterbox_turbo_handler,
            generate_chatterbox_turbo_tts,
            init_chatterbox_turbo,
            unload_chatterbox_turbo,
            get_chatterbox_turbo_status,
            CHATTERBOX_TURBO_AVAILABLE as _TURBO_AVAILABLE,
        )
    CHATTERBOX_TURBO_AVAILABLE = _TURBO_AVAILABLE
    print("✅ Chatterbox Turbo handler loaded")
except Exception as error:
    CHATTERBOX_TURBO_AVAILABLE = False
    print(f"⚠️ Chatterbox Turbo not available. Some features will be disabled. Error: {error}")

# Qwen TTS imports
try:
    with suppress_specific_warnings():
        from qwen_tts_handler import (
            get_qwen_tts_handler,
            init_qwen_tts,
            unload_qwen_tts,
            get_qwen_tts_status,
            generate_qwen_voice_design_tts,
            generate_qwen_voice_clone_tts,
            generate_qwen_custom_voice_tts,
            transcribe_qwen_audio,
            QWEN_TTS_AVAILABLE as _QWEN_AVAILABLE,
            QWEN_TTS_MODELS,
            QWEN_SPEAKERS,
            QWEN_LANGUAGES,
        )
    QWEN_TTS_AVAILABLE = _QWEN_AVAILABLE
    print("✅ Qwen TTS handler loaded")
except Exception as error:
    QWEN_TTS_AVAILABLE = False
    QWEN_TTS_MODELS = {}
    QWEN_SPEAKERS = []
    QWEN_LANGUAGES = []
    print(f"⚠️ Qwen TTS not available. Some features will be disabled. Error: {error}")

try:
    from ebook_converter import (
        analyze_ebook,
        convert_ebook_to_text_chunks,
        get_supported_formats,
    )

    EBOOK_CONVERTER_AVAILABLE = True
except Exception as error:
    EBOOK_CONVERTER_AVAILABLE = False

    def analyze_ebook(file_path: str) -> dict[str, Any]:
        raise RuntimeError("eBook converter not available")

    def convert_ebook_to_text_chunks(
        file_path: str, max_chunk_length: int = 500
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise RuntimeError("eBook converter not available")

    def get_supported_formats() -> dict[str, Any]:
        return {}

    print(f"⚠️ eBook converter not available. Error: {error}")

from narration_transform import (
    CONTENT_TYPE_PRESETS,
    DEFAULT_LLM_NARRATION_SYSTEM_PROMPT,
    DEFAULT_CONTENT_TYPE_PRESET,
    LLM_PROVIDER_CONFIGS,
    LLM_PROVIDER_MODEL_SUGGESTIONS,
    LLM_OUTCOME_PRESETS,
    DEFAULT_LLM_OUTCOME_PRESET,
    _MODE_BEHAVIORAL_REMINDERS,
    _STYLE_DESCRIPTIONS,
    _DEFAULT_PROVIDER_CONFIG,
    _get_provider_config,
    deterministic_normalize,
    strip_unsupported_cues,
    _build_llm_transform_user_prompt,
    _clean_llm_transform_output,
    _apply_local_narration_transform,
    get_content_type_preset_names,
    get_content_type_system_prompt,
    get_llm_provider_env_var,
    get_llm_shell_key_setup_hint,
    fetch_provider_models,
    try_start_lm_studio,
    call_openai_compatible_chat,
    test_llm_connection,
    apply_llm_narration_transform,
    apply_llm_transform_to_textbox,
    generate_voice_casting,
    format_provenance,
)
from assistant_service import DEFAULT_ASSISTANT_SYSTEM_PROMPT
from pronunciation import (
    PronunciationOverride,
    ProtectedTerm,
    load_lexicon,
    save_lexicon,
)
from conversation_logic import (
    create_default_speaker_settings,
    detect_suspect_speaker_names,
    format_conversation_with_llm,
    format_conversation_info,
    get_speaker_names_from_script,
    parse_conversation_script,
)
from engine_registry import (
    ENGINE_EXPRESSIVENESS,
    ENGINE_METADATA_CONTROL_MAP,
    _DEFAULT_ENGINE_EXPRESSIVENESS,
)


# ===== VOXCPM MODEL MANAGEMENT =====
def init_voxcpm():
    """Initialize VoxCPM - placeholder implementation"""
    try:
        from voxcpm_handler import init_voxcpm as handler_init_voxcpm

        return handler_init_voxcpm()
    except ImportError:
        return False


INDEXTTS_MODELS_AVAILABLE = False
INDEXTTS_AVAILABLE = False
INDEXTTS2_AVAILABLE = False
EBOOK_CONVERTER_AVAILABLE = bool(globals().get("EBOOK_CONVERTER_AVAILABLE", False))
IndexTTS = None


def _has_indextts_package() -> bool:
    try:
        if importlib_util.find_spec("indextts.infer") is not None:
            return True
    except Exception:
        return False

    return False


def load_indextts_class() -> tuple[Any | None, str | None]:
    """Load IndexTTS class lazily and return import errors without crashing startup."""
    try:
        module = importlib.import_module("indextts.infer")
        _IndexTTS = getattr(module, "IndexTTS", None)
        if _IndexTTS is None:
            return None, "IndexTTS class not found in indextts.infer"

        return _IndexTTS, None
    except Exception as error:
        return None, str(error)


def download_indextts_models_auto() -> bool:
    """Attempt to download IndexTTS models using the local downloader utility."""
    try:
        downloader_path = Path(__file__).resolve().parent / "tools" / "download_indextts_models.py"
        if not downloader_path.exists():
            print(f"❌ IndexTTS downloader not found: {downloader_path}")
            return False

        spec = importlib_util.spec_from_file_location(
            "download_indextts_models", str(downloader_path)
        )
        if spec is None or spec.loader is None:
            print("❌ Failed to load IndexTTS downloader module spec")
            return False

        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        downloader = getattr(module, "download_indextts_models", None)
        if downloader is None:
            print("❌ download_indextts_models function not found")
            return False

        return bool(downloader())
    except Exception as error:
        print(f"❌ IndexTTS auto-download failed: {error}")
        return False


def detect_language(text: str) -> str:
    """Lightweight fallback language detection for multilingual Chatterbox routing."""
    if not isinstance(text, str) or not text.strip():
        return "en"

    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"

    return "en"


def init_higgs_audio() -> tuple[bool, str]:
    """Compatibility wrapper for model manager handlers."""
    if not HIGGS_AUDIO_AVAILABLE:
        return False, "❌ Higgs Audio not available"

    try:
        handler = get_higgs_audio_handler()
        if handler is None:
            return False, "❌ Higgs Audio handler unavailable"
        success = bool(handler.initialize_engine())
        return (
            (True, "✅ Higgs Audio loaded successfully")
            if success
            else (False, "❌ Failed to initialize Higgs Audio")
        )
    except Exception as error:
        return False, f"❌ Error loading Higgs Audio: {error}"


def unload_higgs_audio() -> str:
    """Compatibility wrapper for model manager handlers."""
    try:
        handler = get_higgs_audio_handler()
        if handler is None:
            return "⚠️ Higgs Audio handler unavailable"
        if getattr(handler, "engine", None) is not None:
            handler.engine = None

        import gc

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return "✅ Higgs Audio unloaded"
    except Exception as error:
        return f"⚠️ Error unloading Higgs Audio: {error}"


def init_voxcpm_model() -> tuple[bool, str]:
    result = init_voxcpm()
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), str(result[1])
    if isinstance(result, bool):
        return result, (
            "✅ VoxCPM models loaded successfully" if result else "❌ VoxCPM not available"
        )
    return False, "❌ VoxCPM initialization failed"


def unload_voxcpm_model() -> str:
    return unload_voxcpm()


def init_kitten_tts_model() -> tuple[bool, str]:
    return init_kitten_tts()


def unload_kitten_tts_model() -> str:
    return unload_kitten_tts()


def init_indextts2_model() -> tuple[bool, str]:
    return init_indextts2()


def unload_indextts2_model() -> str:
    return unload_indextts2()


def init_qwen_tts_model(model_type: str = "Base", model_size: str = "1.7B") -> tuple[bool, str]:
    return init_qwen_tts(model_type, model_size)


def unload_qwen_tts_model(
    model_type: Optional[str] = None, model_size: Optional[str] = None
) -> str:
    return unload_qwen_tts(model_type or "", model_size or "")


def init_vibevoice_model(
    model_path: str = "models/VibeVoice-1.5B", use_flash_attention: bool = False
) -> tuple[bool, str]:
    return init_vibevoice(model_path, use_flash_attention=use_flash_attention)


def unload_vibevoice_model() -> str:
    return unload_vibevoice()


INDEXTTS_AVAILABLE = _has_indextts_package()


def generate_conversation_audio_simple(
    conversation_script,
    voice_samples,  # List of voice sample file paths
    ref_texts=None,  # List of reference texts for each speaker (for Qwen TTS)
    selected_engine="chatterbox",
    conversation_pause_duration=0.8,
    speaker_transition_pause=0.3,
    effects_settings=None,
    audio_format="wav",
):
    """Generate a complete conversation with multiple voices - Simplified version."""
    try:
        print("🎭 Starting conversation generation...")

        # Parse the conversation script
        conversation, parse_error = parse_conversation_script(conversation_script)
        if parse_error:
            return None, f"❌ Script parsing error: {parse_error}"

        if not conversation:
            return None, "❌ No valid conversation found in script"

        print(f"📝 Parsed {len(conversation)} conversation lines")

        # Get unique speakers and map them to voice samples
        speakers = get_speaker_names_from_script(conversation_script)
        print(f"🎤 Found speakers: {speakers}")

        # Initialize ref_texts if not provided
        if ref_texts is None:
            ref_texts = [None] * 5

        # Map speakers to voice samples, reference texts, and generate consistent seeds
        speaker_voice_map = {}
        speaker_ref_text_map = {}
        speaker_seed_map = {}
        for i, speaker in enumerate(speakers):
            if i < len(voice_samples) and voice_samples[i] is not None:
                speaker_voice_map[speaker] = voice_samples[i]
                print(f"🎤 {speaker} -> {voice_samples[i]}")
            else:
                speaker_voice_map[speaker] = None
                print(f"� {speaker} -> No voice sample")

            # Map reference text for this speaker
            ref_text_value = ref_texts[i] if i < len(ref_texts) else None
            if isinstance(ref_text_value, str) and ref_text_value.strip():
                speaker_ref_text_map[speaker] = ref_text_value.strip()
                print(f"📝 {speaker} -> ref_text: {ref_text_value[:30]}...")
            else:
                speaker_ref_text_map[speaker] = None
                print(f"📝 {speaker} -> No reference text")

            # Generate a consistent seed for each speaker
            speaker_seed_map[speaker] = np.random.randint(0, 2147483647)
            print(f"🎲 {speaker} -> seed: {speaker_seed_map[speaker]}")

        conversation_audio_chunks = []
        conversation_info = []
        sample_rate = 22050

        # Generate audio for each conversation line
        for i, line in enumerate(conversation):
            speaker = line["speaker"]
            text = line["text"]

            print(f'🗣️ Generating line {i+1}/{len(conversation)}: {speaker} - "{text[:30]}..."')

            ref_audio = speaker_voice_map.get(speaker)

            # Generate audio based on selected engine
            try:
                if selected_engine == "chatterbox" or selected_engine == "ChatterboxTTS":
                    result = generate_chatterbox_tts(
                        text,
                        ref_audio or "",
                        0.5,  # exaggeration
                        0.8,  # temperature
                        0,  # seed
                        0.5,  # cfg_weight
                        300,  # chunk_size
                        effects_settings,
                        audio_format,
                        skip_file_saving=True,
                    )
                elif selected_engine == "Chatterbox Multilingual":
                    print(f"🌍 Using Chatterbox Multilingual for {speaker}")
                    lang = detect_language(text)
                    print(f"🌍 Multilingual detected language: {lang}")
                    result = generate_chatterbox_multilingual_tts(
                        text,
                        lang,  # language_id - auto-detected
                        ref_audio or "",
                        0.5,  # exaggeration
                        0.8,  # temperature
                        0,  # seed
                        0.5,  # cfg_weight
                        2.0,  # repetition_penalty
                        0.05,  # min_p
                        1.0,  # top_p
                        300,  # chunk_size
                        effects_settings,
                        audio_format,
                        skip_file_saving=True,
                    )
                elif selected_engine == "Chatterbox Turbo":
                    print(f"🚀 Using Chatterbox Turbo for {speaker}")
                    result = generate_chatterbox_turbo_tts(
                        text,
                        ref_audio or "",
                        0.5,  # exaggeration
                        0.8,  # temperature
                        0.5,  # cfg_weight
                        1.2,  # repetition_penalty
                        0.05,  # min_p
                        1.0,  # top_p
                        0,  # seed
                        300,  # chunk_size
                        effects_settings,
                        audio_format,
                        skip_file_saving=True,
                    )
                elif selected_engine == "kokoro" or selected_engine == "Kokoro TTS":
                    print(f"🗣️ Using Kokoro TTS for speaker '{speaker}'")
                    result = generate_kokoro_conversation_tts(
                        text, speaker, speakers, effects_settings, audio_format
                    )
                elif selected_engine == "Fish Speech":
                    print(f"🐟 Using Fish Speech for {speaker}")
                    # Simplified Fish Speech call
                    result = generate_fish_speech_simple(
                        text, ref_audio, effects_settings, audio_format
                    )
                elif selected_engine == "IndexTTS":
                    print(f"🎯 Using IndexTTS for {speaker}")
                    result = generate_indextts_tts(
                        text,
                        ref_audio,
                        0.8,  # temperature
                        None,  # seed
                        effects_settings,
                        audio_format,
                        skip_file_saving=True,
                    )
                elif selected_engine == "IndexTTS2":
                    print(f"🎯 Using IndexTTS2 for {speaker}")
                    result = generate_indextts2_tts(
                        text,
                        ref_audio,
                        "audio_reference",  # emotion_mode
                        None,  # emotion_audio
                        None,  # emotion_vectors
                        "",  # emotion_description
                        0.8,  # temperature
                        0.9,  # top_p
                        50,  # top_k
                        1.1,  # repetition_penalty
                        1500,  # max_mel_tokens
                        None,  # seed
                        True,  # use_random
                        1.0,  # emo_alpha
                        effects_settings,
                        audio_format,
                        skip_file_saving=True,
                    )
                elif selected_engine == "F5-TTS":
                    print(f"🎵 Using F5-TTS for {speaker}")
                    result = generate_f5_tts(
                        text,
                        ref_audio,
                        None,  # ref_text
                        1.0,  # speed
                        0.15,  # cross_fade
                        False,  # remove_silence
                        None,  # seed
                        effects_settings,
                        audio_format,
                        skip_file_saving=True,
                    )
                elif selected_engine == "Higgs Audio":
                    print(f"🎙️ Using Higgs Audio for {speaker}")
                    result = generate_higgs_audio_tts(
                        text,
                        ref_audio,
                        "",  # ref_text
                        "EMPTY",  # voice_preset
                        "",  # system_prompt
                        1.0,  # temperature
                        0.95,  # top_p
                        50,  # top_k
                        1024,  # max_tokens
                        7,  # ras_win_len
                        2,  # ras_win_max_num_repeat
                        100,  # chunk_length
                        effects_settings,
                        audio_format,
                        skip_file_saving=True,
                    )
                elif selected_engine == "KittenTTS":
                    print(f"🐱 Using KittenTTS for {speaker}")
                    # For conversation mode, assign different voices to different speakers
                    available_voices = [
                        "expr-voice-2-f",
                        "expr-voice-2-m",
                        "expr-voice-3-f",
                        "expr-voice-3-m",
                        "expr-voice-4-f",
                        "expr-voice-4-m",
                        "expr-voice-5-f",
                        "expr-voice-5-m",
                    ]
                    speaker_index = speakers.index(speaker) if speaker in speakers else 0
                    kitten_voice = available_voices[speaker_index % len(available_voices)]
                    print(f"🎤 Assigned voice '{kitten_voice}' to speaker '{speaker}'")
                    result = generate_kitten_tts(
                        text, kitten_voice, effects_settings, audio_format, skip_file_saving=True
                    )
                elif selected_engine == "VoxCPM":
                    print(f"🎤 Using VoxCPM for {speaker}")

                    # Auto-transcribe reference audio if provided
                    ref_text = None
                    if ref_audio and VOXCPM_AVAILABLE:
                        try:
                            print(f"🎤 Auto-transcribing reference audio for {speaker}...")
                            ref_text = transcribe_voxcpm_audio(ref_audio)
                            if ref_text:
                                print(f"📝 Transcribed: {ref_text[:50]}...")
                            else:
                                print("⚠️ No transcription result, using default voice")
                        except Exception as e:
                            print(f"⚠️ Transcription failed for {speaker}: {e}")
                            ref_text = None

                    # Use consistent seed for this speaker
                    speaker_seed = speaker_seed_map.get(speaker, None)
                    print(f"🎲 Using consistent seed {speaker_seed} for {speaker}")

                    result = generate_voxcpm_unified_tts(
                        text,
                        ref_audio,
                        ref_text,
                        2.0,  # cfg_value
                        10,  # inference_timesteps
                        True,  # normalize
                        True,  # denoise
                        True,  # retry_badcase
                        3,  # retry_badcase_max_times
                        6.0,  # retry_badcase_ratio_threshold
                        speaker_seed,  # Use consistent seed per speaker
                        effects_settings,
                        audio_format,
                    )
                elif selected_engine == "Qwen Voice Clone":
                    print(f"🎙️ Using Qwen Voice Clone for {speaker}")

                    # Get pre-provided reference text for this speaker (from UI transcribe button)
                    ref_text = speaker_ref_text_map.get(speaker)

                    # If no pre-provided ref_text, try auto-transcribe as fallback
                    if not ref_text and ref_audio and QWEN_TTS_AVAILABLE:
                        try:
                            print(f"🎤 Auto-transcribing reference audio for {speaker}...")
                            ref_text = transcribe_qwen_audio(ref_audio)
                            if ref_text:
                                print(f"📝 Transcribed: {ref_text[:50]}...")
                            else:
                                print("⚠️ No transcription result, using x-vector only mode")
                        except Exception as e:
                            print(f"⚠️ Transcription failed for {speaker}: {e}")
                            ref_text = None
                    elif ref_text:
                        print(f"📝 Using pre-provided ref_text for {speaker}: {ref_text[:50]}...")

                    # Use consistent seed for this speaker
                    speaker_seed = speaker_seed_map.get(speaker, None)
                    print(f"🎲 Using consistent seed {speaker_seed} for {speaker}")

                    # Get the currently loaded model size from MODEL_STATUS
                    qwen_model_size = "0.6B"  # Default to smaller model
                    if MODEL_STATUS.get("qwen_tts", {}).get("loaded"):
                        qwen_model_size = MODEL_STATUS["qwen_tts"].get("model_size", "0.6B")
                    print(f"🎭 Using Qwen model size: {qwen_model_size}")

                    result = generate_qwen_voice_clone_tts(
                        text,
                        ref_audio,
                        ref_text or "",
                        "Auto",  # language
                        ref_text is None or ref_text == "",  # use_xvector_only
                        qwen_model_size,  # Use currently loaded model size
                        200,  # max_chunk_chars
                        0.0,  # chunk_gap
                        speaker_seed,  # seed
                        effects_settings,
                        audio_format,
                        skip_file_saving=True,
                    )
                else:
                    return None, f"❌ Unsupported TTS engine: {selected_engine}"

                if result[0] is None:
                    return None, f"❌ Error generating audio for {speaker}: {result[1]}"

                audio_data, info_text = result
                if audio_data is None:
                    return None, f"❌ No audio generated for {speaker}"

                # Extract audio array from tuple
                if isinstance(audio_data, tuple):
                    sample_rate, line_audio = audio_data
                else:
                    return None, f"❌ Invalid audio format for {speaker}"

                conversation_audio_chunks.append(line_audio)
                conversation_info.append(
                    {
                        "speaker": speaker,
                        "text": text[:50] + ("..." if len(text) > 50 else ""),
                        "duration": len(line_audio) / sample_rate,
                        "samples": len(line_audio),
                    }
                )

                print(f"✅ Generated {len(line_audio)} samples for {speaker}")

            except Exception as gen_error:
                import traceback

                traceback.print_exc()
                return None, f"❌ Error generating audio for {speaker}: {str(gen_error)}"

        # Combine all audio with proper timing
        print("🎵 Combining conversation audio with proper timing...")

        # Calculate pause durations in samples
        conversation_pause_samples = int(sample_rate * conversation_pause_duration)
        transition_pause_samples = int(sample_rate * speaker_transition_pause)

        # Handle negative pauses (overlapping audio)
        if conversation_pause_samples < 0 or transition_pause_samples < 0:
            print("🔄 Using overlapping audio mode for negative pauses...")

            # For negative pauses, we'll need to overlap the audio chunks
            final_conversation_audio = None
            current_position = 0

            for i, (audio_chunk, info) in enumerate(
                zip(conversation_audio_chunks, conversation_info)
            ):
                current_speaker = info["speaker"]

                if final_conversation_audio is None:
                    # First chunk - initialize the final audio
                    final_conversation_audio = audio_chunk.copy()
                    current_position = len(audio_chunk)
                else:
                    # Determine pause/overlap based on speaker change
                    if i < len(conversation_audio_chunks):
                        prev_speaker = conversation_info[i - 1]["speaker"]

                        if current_speaker != prev_speaker:
                            pause_samples = conversation_pause_samples
                        else:
                            pause_samples = transition_pause_samples

                        # Calculate where to place this chunk
                        start_position = current_position + pause_samples

                        if pause_samples < 0:
                            # Negative pause means overlap
                            overlap_samples = abs(pause_samples)
                            start_position = max(0, current_position - overlap_samples)

                        # Extend final audio if needed
                        end_position = start_position + len(audio_chunk)
                        if end_position > len(final_conversation_audio):
                            extension = np.zeros(end_position - len(final_conversation_audio))
                            final_conversation_audio = np.concatenate(
                                [final_conversation_audio, extension]
                            )

                        # Mix overlapping audio (average to prevent clipping)
                        if pause_samples < 0:
                            # For overlap region, mix the audio
                            overlap_end = min(start_position + len(audio_chunk), current_position)
                            if overlap_end > start_position:
                                overlap_length = overlap_end - start_position
                                final_conversation_audio[start_position:overlap_end] = (
                                    final_conversation_audio[start_position:overlap_end] * 0.5
                                    + audio_chunk[:overlap_length] * 0.5
                                )
                                # Add the non-overlapping part
                                if overlap_length < len(audio_chunk):
                                    final_conversation_audio[overlap_end:end_position] = (
                                        audio_chunk[overlap_length:]
                                    )
                            else:
                                final_conversation_audio[start_position:end_position] = audio_chunk
                        else:
                            # Normal placement with positive pause
                            final_conversation_audio[start_position:end_position] = audio_chunk

                        current_position = end_position
        else:
            # Original code for positive pauses only
            final_audio_parts = []

            for i, (audio_chunk, info) in enumerate(
                zip(conversation_audio_chunks, conversation_info)
            ):
                current_speaker = info["speaker"]

                # Add audio chunk
                final_audio_parts.append(audio_chunk)

                # Add pause after each line (except the last one)
                if i < len(conversation_audio_chunks) - 1:
                    next_speaker = conversation_info[i + 1]["speaker"]

                    # Different pause duration based on speaker change
                    if current_speaker != next_speaker:
                        # Speaker transition - longer pause
                        pause_samples = conversation_pause_samples
                    else:
                        # Same speaker continuing - shorter pause
                        pause_samples = transition_pause_samples

                    pause_audio = np.zeros(pause_samples)
                    final_audio_parts.append(pause_audio)

            # Concatenate all parts
            final_conversation_audio = np.concatenate(final_audio_parts)

        # Save the conversation audio to outputs folder
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"conversation_{selected_engine.lower().replace(' ', '_')}_{timestamp}"
            filepath, filename = save_audio_with_format(
                final_conversation_audio, sample_rate, audio_format, output_folder, filename_base
            )
            print(f"💾 Conversation saved as: {filename}")
        except Exception as save_error:
            print(f"Warning: Could not save conversation file: {save_error}")
            filename = "conversation_audio"

        # Create conversation summary
        total_duration = len(final_conversation_audio) / sample_rate
        unique_speakers = len(set([info["speaker"] for info in conversation_info]))
        meta_path, script_path = write_generation_sidecar_metadata(
            filepath if "filepath" in locals() else "",
            {
                "mode": "conversation",
                "engine": selected_engine,
                "total_lines": len(conversation),
                "unique_speakers": unique_speakers,
                "total_duration_seconds": float(total_duration),
            },
            conversation_script,
            original_text=conversation_script,
            transformed_text=conversation_script,
        )

        summary = {
            "total_lines": len(conversation),
            "unique_speakers": unique_speakers,
            "total_duration": total_duration,
            "speakers": list(set([info["speaker"] for info in conversation_info])),
            "conversation_info": conversation_info,
            "engine_used": selected_engine,
            "saved_file": filename,
            "metadata_file": meta_path,
            "script_file": script_path,
        }

        print(
            f"✅ Conversation generated: {len(conversation)} lines, {unique_speakers} speakers, {total_duration:.1f}s"
        )

        return (sample_rate, final_conversation_audio), summary

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"❌ Conversation generation error: {str(e)}"


def generate_conversation_audio_kokoro(
    conversation_script,
    kokoro_voices,  # List of selected Kokoro voices for each speaker
    selected_engine="Kokoro TTS",
    conversation_pause_duration=0.8,
    speaker_transition_pause=0.3,
    effects_settings=None,
    audio_format="wav",
):
    """Generate a complete conversation with Kokoro TTS using selected voices for each speaker."""
    try:
        print("🎭 Starting Kokoro conversation generation...")

        # Parse the conversation script
        conversation, parse_error = parse_conversation_script(conversation_script)
        if parse_error:
            return None, f"❌ Script parsing error: {parse_error}"

        if not conversation:
            return None, "❌ No valid conversation found in script"

        print(f"📝 Parsed {len(conversation)} conversation lines")

        # Get unique speakers and map them to selected Kokoro voices
        speakers = get_speaker_names_from_script(conversation_script)
        print(f"🎤 Found speakers: {speakers}")

        # Map speakers to selected Kokoro voices
        speaker_voice_map = {}
        for i, speaker in enumerate(speakers):
            if i < len(kokoro_voices) and kokoro_voices[i] is not None:
                speaker_voice_map[speaker] = kokoro_voices[i]
                print(f"🗣️ {speaker} -> {kokoro_voices[i]}")
            else:
                # Fallback to default voices if not enough selections
                default_voices = [
                    "af_heart",
                    "am_adam",
                    "bf_emma",
                    "bm_lewis",
                    "af_sarah",
                    "am_michael",
                ]
                fallback_voice = default_voices[i % len(default_voices)]
                speaker_voice_map[speaker] = fallback_voice
                print(f"🗣️ {speaker} -> {fallback_voice} (fallback)")

        conversation_audio_chunks = []
        conversation_info = []
        sample_rate = 22050

        # Generate audio for each conversation line
        for i, line in enumerate(conversation):
            speaker = line["speaker"]
            text = line["text"]

            print(f'🗣️ Generating line {i+1}/{len(conversation)}: {speaker} - "{text[:30]}..."')

            selected_voice = speaker_voice_map.get(speaker)

            # Generate audio using Kokoro TTS with selected voice
            try:
                result = generate_kokoro_tts(
                    text,
                    selected_voice,
                    1,  # speed
                    effects_settings,
                    audio_format,
                    skip_file_saving=True,
                )

                if result[0] is None:
                    return None, f"❌ Error generating audio for {speaker}: {result[1]}"

                audio_data, info_text = result
                if audio_data is None:
                    return None, f"❌ No audio generated for {speaker}"

                # Extract audio array from tuple
                if isinstance(audio_data, tuple):
                    sample_rate, line_audio = audio_data
                else:
                    return None, f"❌ Invalid audio format for {speaker}"

                conversation_audio_chunks.append(line_audio)
                conversation_info.append(
                    {
                        "speaker": speaker,
                        "text": text[:50] + ("..." if len(text) > 50 else ""),
                        "duration": len(line_audio) / sample_rate,
                        "samples": len(line_audio),
                        "voice": selected_voice,
                    }
                )

                print(
                    f"✅ Generated {len(line_audio)} samples for {speaker} using voice {selected_voice}"
                )

            except Exception as gen_error:
                import traceback

                traceback.print_exc()
                return None, f"❌ Error generating audio for {speaker}: {str(gen_error)}"

        # Combine all audio with proper timing
        print("🎵 Combining conversation audio with proper timing...")

        # Calculate pause durations in samples
        conversation_pause_samples = int(sample_rate * conversation_pause_duration)
        transition_pause_samples = int(sample_rate * speaker_transition_pause)

        # Handle negative pauses (overlapping audio)
        if conversation_pause_samples < 0 or transition_pause_samples < 0:
            print("🔄 Using overlapping audio mode for negative pauses...")

            # For negative pauses, we'll need to overlap the audio chunks
            final_conversation_audio = None
            current_position = 0

            for i, (audio_chunk, info) in enumerate(
                zip(conversation_audio_chunks, conversation_info)
            ):
                current_speaker = info["speaker"]

                if final_conversation_audio is None:
                    # First chunk - initialize the final audio
                    final_conversation_audio = audio_chunk.copy()
                    current_position = len(audio_chunk)
                else:
                    # Determine pause/overlap based on speaker change
                    if i < len(conversation_audio_chunks):
                        prev_speaker = conversation_info[i - 1]["speaker"]

                        if current_speaker != prev_speaker:
                            pause_samples = conversation_pause_samples
                        else:
                            pause_samples = transition_pause_samples

                        # Calculate where to place this chunk
                        start_position = current_position + pause_samples

                        if pause_samples < 0:
                            # Negative pause means overlap
                            overlap_samples = abs(pause_samples)
                            start_position = max(0, current_position - overlap_samples)

                        # Extend final audio if needed
                        end_position = start_position + len(audio_chunk)
                        if end_position > len(final_conversation_audio):
                            extension = np.zeros(end_position - len(final_conversation_audio))
                            final_conversation_audio = np.concatenate(
                                [final_conversation_audio, extension]
                            )

                        # Mix overlapping audio (average to prevent clipping)
                        if pause_samples < 0:
                            # For overlap region, mix the audio
                            overlap_end = min(start_position + len(audio_chunk), current_position)
                            if overlap_end > start_position:
                                overlap_length = overlap_end - start_position
                                final_conversation_audio[start_position:overlap_end] = (
                                    final_conversation_audio[start_position:overlap_end] * 0.5
                                    + audio_chunk[:overlap_length] * 0.5
                                )
                                # Add the non-overlapping part
                                if overlap_length < len(audio_chunk):
                                    final_conversation_audio[overlap_end:end_position] = (
                                        audio_chunk[overlap_length:]
                                    )
                            else:
                                final_conversation_audio[start_position:end_position] = audio_chunk
                        else:
                            # Normal placement with positive pause
                            final_conversation_audio[start_position:end_position] = audio_chunk

                        current_position = end_position
        else:
            # Original code for positive pauses only
            final_audio_parts = []

            for i, (audio_chunk, info) in enumerate(
                zip(conversation_audio_chunks, conversation_info)
            ):
                current_speaker = info["speaker"]

                # Add audio chunk
                final_audio_parts.append(audio_chunk)

                # Add pause after each line (except the last one)
                if i < len(conversation_audio_chunks) - 1:
                    next_speaker = conversation_info[i + 1]["speaker"]

                    # Different pause duration based on speaker change
                    if current_speaker != next_speaker:
                        # Speaker transition - longer pause
                        pause_samples = conversation_pause_samples
                    else:
                        # Same speaker continuing - shorter pause
                        pause_samples = transition_pause_samples

                    pause_audio = np.zeros(pause_samples)
                    final_audio_parts.append(pause_audio)

            # Concatenate all parts
            final_conversation_audio = np.concatenate(final_audio_parts)

        # Save the conversation audio to outputs folder
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"conversation_kokoro_{timestamp}"
            filepath, filename = save_audio_with_format(
                final_conversation_audio, sample_rate, audio_format, output_folder, filename_base
            )
            print(f"💾 Conversation saved as: {filename}")
        except Exception as save_error:
            print(f"Warning: Could not save conversation file: {save_error}")
            filename = "conversation_kokoro_audio"

        # Create conversation summary
        total_duration = len(final_conversation_audio) / sample_rate
        unique_speakers = len(set([info["speaker"] for info in conversation_info]))
        meta_path, script_path = write_generation_sidecar_metadata(
            filepath if "filepath" in locals() else "",
            {
                "mode": "conversation",
                "engine": selected_engine,
                "total_lines": len(conversation),
                "unique_speakers": unique_speakers,
                "total_duration_seconds": float(total_duration),
            },
            conversation_script,
            original_text=conversation_script,
            transformed_text=conversation_script,
        )

        summary = {
            "total_lines": len(conversation),
            "unique_speakers": unique_speakers,
            "total_duration": total_duration,
            "speakers": list(set([info["speaker"] for info in conversation_info])),
            "conversation_info": conversation_info,
            "engine_used": selected_engine,
            "saved_file": filename,
            "metadata_file": meta_path,
            "script_file": script_path,
        }

        print(
            f"✅ Kokoro conversation generated: {len(conversation)} lines, {unique_speakers} speakers, {total_duration:.1f}s"
        )

        return (sample_rate, final_conversation_audio), summary

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"❌ Kokoro conversation generation error: {str(e)}"


def generate_kokoro_conversation_tts(
    text, speaker, speakers_list, effects_settings=None, audio_format="wav"
):
    """Generate TTS audio using Kokoro TTS with speaker-specific voice assignment for conversation mode."""
    if not KOKORO_AVAILABLE:
        return None, "❌ Kokoro TTS not available - check installation"

    if not MODEL_STATUS["kokoro"]["loaded"] or not KOKORO_PIPELINES:
        return None, "❌ Kokoro TTS not loaded - please load the model first"

    try:
        # Voice assignment logic for conversation mode
        available_voices = ["af_heart", "am_adam", "bf_emma", "bm_lewis", "af_sarah", "am_michael"]
        speaker_index = speakers_list.index(speaker) if speaker in speakers_list else 0
        assigned_voice = available_voices[speaker_index % len(available_voices)]

        print(f"🗣️ Generating Kokoro TTS for speaker '{speaker}' using voice '{assigned_voice}'")

        # Generate using the assigned voice
        result = generate_kokoro_tts(
            text,
            assigned_voice,
            1,  # speed
            effects_settings,
            audio_format,
            skip_file_saving=True,
        )

        return result

    except Exception as e:
        return None, f"❌ Kokoro conversation error: {str(e)}"


def generate_conversation_audio_kitten(
    conversation_script,
    kitten_voices,  # List of selected KittenTTS voices for each speaker
    selected_engine="KittenTTS",
    conversation_pause_duration=0.8,
    speaker_transition_pause=0.3,
    effects_settings=None,
    audio_format="wav",
):
    """Generate a complete conversation with KittenTTS using selected voices for each speaker."""
    try:
        print("🐱 Starting KittenTTS conversation generation...")

        # Parse the conversation script
        conversation, parse_error = parse_conversation_script(conversation_script)
        if parse_error:
            return None, f"❌ Script parsing error: {parse_error}"

        if not conversation:
            return None, "❌ No valid conversation found in script"

        print(f"📝 Parsed {len(conversation)} conversation lines")

        # Get unique speakers and map them to selected KittenTTS voices
        speakers = get_speaker_names_from_script(conversation_script)
        print(f"🎤 Found speakers: {speakers}")

        # Map speakers to selected KittenTTS voices
        speaker_voice_map = {}
        for i, speaker in enumerate(speakers):
            if i < len(kitten_voices) and kitten_voices[i] is not None:
                speaker_voice_map[speaker] = kitten_voices[i]
                print(f"🐱 {speaker} -> {kitten_voices[i]}")
            else:
                # Fallback to default voices if not enough selections
                available_voices = [
                    "expr-voice-2-f",
                    "expr-voice-2-m",
                    "expr-voice-3-f",
                    "expr-voice-3-m",
                    "expr-voice-4-f",
                    "expr-voice-4-m",
                    "expr-voice-5-f",
                    "expr-voice-5-m",
                ]
                fallback_voice = available_voices[i % len(available_voices)]
                speaker_voice_map[speaker] = fallback_voice
                print(f"🐱 {speaker} -> {fallback_voice} (fallback)")

        conversation_audio_chunks = []
        conversation_info = []
        sample_rate = 22050

        # Generate audio for each conversation line
        for i, line in enumerate(conversation):
            speaker = line["speaker"]
            text = line["text"]

            print(f'🐱 Generating line {i+1}/{len(conversation)}: {speaker} - "{text[:30]}..."')

            kitten_voice = speaker_voice_map.get(speaker, "expr-voice-2-f")

            # Generate audio using KittenTTS
            try:
                result = generate_kitten_tts(
                    text, kitten_voice, effects_settings, audio_format, skip_file_saving=True
                )

                if result[0] is None:
                    return None, f"❌ Error generating audio for {speaker}: {result[1]}"

                audio_data, info_text = result
                if audio_data is None:
                    return None, f"❌ No audio generated for {speaker}"

                # Extract audio array from tuple
                if isinstance(audio_data, tuple):
                    sample_rate, line_audio = audio_data
                else:
                    return None, f"❌ Invalid audio format for {speaker}"

                conversation_audio_chunks.append(line_audio)
                conversation_info.append(
                    {
                        "speaker": speaker,
                        "text": text[:50] + ("..." if len(text) > 50 else ""),
                        "duration": len(line_audio) / sample_rate,
                        "samples": len(line_audio),
                        "voice": kitten_voice,
                    }
                )

                print(f"✅ Generated {len(line_audio)} samples for {speaker} using {kitten_voice}")

            except Exception as gen_error:
                import traceback

                traceback.print_exc()
                return None, f"❌ Error generating audio for {speaker}: {str(gen_error)}"

        # Combine all audio with proper timing (same logic as other conversation functions)
        print("🎵 Combining conversation audio with proper timing...")

        # Calculate pause durations in samples
        conversation_pause_samples = int(sample_rate * conversation_pause_duration)
        transition_pause_samples = int(sample_rate * speaker_transition_pause)

        # Combine audio parts
        final_audio_parts = []

        for i, (audio_chunk, info) in enumerate(zip(conversation_audio_chunks, conversation_info)):
            current_speaker = info["speaker"]

            # Add audio chunk
            final_audio_parts.append(audio_chunk)

            # Add pause after each line (except the last one)
            if i < len(conversation_audio_chunks) - 1:
                next_speaker = conversation_info[i + 1]["speaker"]

                # Different pause duration based on speaker change
                if current_speaker != next_speaker:
                    # Speaker transition - longer pause
                    pause_samples = conversation_pause_samples
                else:
                    # Same speaker continuing - shorter pause
                    pause_samples = transition_pause_samples

                if pause_samples > 0:
                    pause_audio = np.zeros(pause_samples)
                    final_audio_parts.append(pause_audio)

        # Concatenate all parts
        final_conversation_audio = np.concatenate(final_audio_parts)

        # Save the conversation audio to outputs folder
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"conversation_kitten_tts_{timestamp}"
            filepath, filename = save_audio_with_format(
                final_conversation_audio, sample_rate, audio_format, output_folder, filename_base
            )
            print(f"💾 KittenTTS conversation saved as: {filename}")
        except Exception as save_error:
            print(f"Warning: Could not save conversation file: {save_error}")
            filename = "kitten_conversation_audio"

        # Create conversation summary
        total_duration = len(final_conversation_audio) / sample_rate
        unique_speakers = len(set([info["speaker"] for info in conversation_info]))
        meta_path, script_path = write_generation_sidecar_metadata(
            filepath if "filepath" in locals() else "",
            {
                "mode": "conversation",
                "engine": selected_engine,
                "total_lines": len(conversation),
                "unique_speakers": unique_speakers,
                "total_duration_seconds": float(total_duration),
            },
            conversation_script,
            original_text=conversation_script,
            transformed_text=conversation_script,
        )

        summary = {
            "total_lines": len(conversation),
            "unique_speakers": unique_speakers,
            "total_duration": total_duration,
            "speakers": list(set([info["speaker"] for info in conversation_info])),
            "conversation_info": conversation_info,
            "engine_used": selected_engine,
            "saved_file": filename,
            "metadata_file": meta_path,
            "script_file": script_path,
        }

        print(
            f"✅ KittenTTS conversation generated: {len(conversation)} lines, {unique_speakers} speakers, {total_duration:.1f}s"
        )

        return (sample_rate, final_conversation_audio), summary

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"❌ KittenTTS conversation error: {str(e)}"


def generate_conversation_audio_indextts2(
    conversation_script,
    voice_samples,  # List of voice sample files for each speaker
    emotion_modes,  # List of emotion modes for each speaker
    emotion_audios,  # List of emotion audio files for each speaker
    emotion_descriptions,  # List of emotion descriptions for each speaker
    emotion_vectors,  # List of emotion vector dicts for each speaker
    selected_engine="IndexTTS2",
    conversation_pause_duration=0.8,
    speaker_transition_pause=0.3,
    effects_settings=None,
    audio_format="wav",
):
    """Generate a complete conversation with IndexTTS2 using emotion controls for each speaker."""
    try:
        print("🎯 Starting IndexTTS2 conversation generation...")

        # Parse the conversation script
        conversation, parse_error = parse_conversation_script(conversation_script)
        if parse_error:
            return None, f"❌ Script parsing error: {parse_error}"

        if not conversation:
            return None, "❌ No valid conversation found in script"

        print(f"📝 Parsed {len(conversation)} conversation lines")

        # Get unique speakers and map them to voice samples and emotion settings
        speakers = get_speaker_names_from_script(conversation_script)
        print(f"🎤 Found speakers: {speakers}")

        # Map speakers to voice samples and emotion settings
        speaker_voice_map = {}
        speaker_emotion_map = {}

        for i, speaker in enumerate(speakers):
            # Voice sample mapping
            if i < len(voice_samples) and voice_samples[i] is not None:
                speaker_voice_map[speaker] = voice_samples[i]
                print(f"🎤 {speaker} -> {voice_samples[i]}")
            else:
                speaker_voice_map[speaker] = None
                print(f"🎤 {speaker} -> No voice sample")

            # Emotion settings mapping
            emotion_settings = {
                "mode": emotion_modes[i] if i < len(emotion_modes) else "audio_reference",
                "audio": emotion_audios[i] if i < len(emotion_audios) else None,
                "description": emotion_descriptions[i] if i < len(emotion_descriptions) else "",
                "vectors": emotion_vectors[i] if i < len(emotion_vectors) else {},
            }
            speaker_emotion_map[speaker] = emotion_settings
            print(f"🎭 {speaker} emotion mode: {emotion_settings['mode']}")

        conversation_audio_chunks = []
        conversation_info = []
        sample_rate = 22050

        # Generate audio for each conversation line
        for i, line in enumerate(conversation):
            speaker = line["speaker"]
            text = line["text"]

            print(f'🎯 Generating line {i+1}/{len(conversation)}: {speaker} - "{text[:30]}..."')

            ref_audio = speaker_voice_map.get(speaker)
            emotion_settings = speaker_emotion_map.get(speaker, {})

            if not ref_audio:
                print(f"⚠️ No voice sample for {speaker}, skipping line")
                continue

            # Generate audio using IndexTTS2 with emotion controls
            # Use conservative parameters to avoid tensor dimension issues
            try:
                # Adjust max_mel_tokens based on text length to prevent tensor issues
                text_length = len(text)
                if text_length > 300:
                    max_mel_tokens = 800  # Smaller for long text
                elif text_length > 150:
                    max_mel_tokens = 1000  # Medium for medium text
                else:
                    max_mel_tokens = 1200  # Larger for short text

                result = generate_indextts2_tts(
                    text,
                    ref_audio,
                    emotion_settings.get("mode", "audio_reference"),
                    emotion_settings.get("audio"),
                    emotion_settings.get("vectors"),
                    emotion_settings.get("description", ""),
                    0.8,  # temperature
                    0.9,  # top_p
                    50,  # top_k
                    1.1,  # repetition_penalty
                    max_mel_tokens,  # Dynamic max_mel_tokens
                    None,  # seed
                    True,  # use_random
                    1.0,  # emo_alpha
                    effects_settings,
                    audio_format,
                    skip_file_saving=True,
                )

                if result[0] is None:
                    print(f"❌ Failed to generate audio for {speaker}: {result[1]}")

                    # Try with even more conservative settings as fallback
                    if "tensor" in result[1].lower() or "dimension" in result[1].lower():
                        print(f"   🔄 Attempting fallback with minimal parameters...")
                        try:
                            fallback_result = generate_indextts2_tts(
                                (
                                    text[:100] + "..." if len(text) > 100 else text
                                ),  # Truncate if too long
                                ref_audio,
                                "audio_reference",  # Use simplest emotion mode
                                None,  # No emotion audio
                                {},  # No emotion vectors
                                "",  # No emotion description
                                0.7,  # Lower temperature
                                0.8,  # Lower top_p
                                30,  # Lower top_k
                                1.0,  # No repetition penalty
                                300,  # Very small max_mel_tokens
                                None,  # seed
                                False,  # No random sampling
                                0.5,  # Lower emo_alpha
                                None,  # No effects
                                audio_format,
                                skip_file_saving=True,
                            )

                            if fallback_result[0] is not None:
                                print(f"   ✅ Fallback successful for {speaker}")
                                result = fallback_result
                            else:
                                print(f"   ❌ Fallback also failed for {speaker}")
                                continue
                        except Exception as fallback_error:
                            print(f"   ❌ Fallback error for {speaker}: {fallback_error}")
                            continue
                    else:
                        continue

                # Extract audio data
                if isinstance(result[0], tuple):
                    current_sample_rate, audio_data = result[0]
                else:
                    current_sample_rate = 22050  # Default IndexTTS2 sample rate
                    audio_data = result[0]

                if sample_rate is None:
                    sample_rate = current_sample_rate
                elif sample_rate != current_sample_rate:
                    # Resample if needed
                    import librosa

                    audio_data = librosa.resample(
                        audio_data, orig_sr=current_sample_rate, target_sr=sample_rate
                    )

                conversation_audio_chunks.append(audio_data)
                conversation_info.append(
                    {
                        "speaker": speaker,
                        "text": text,
                        "duration": len(audio_data) / sample_rate,
                        "emotion_mode": emotion_settings.get("mode", "audio_reference"),
                    }
                )

                print(f"✅ Generated {len(audio_data)} samples for {speaker}")

            except Exception as e:
                print(f"❌ Error generating audio for {speaker}: {e}")
                continue

        if not conversation_audio_chunks:
            return None, "❌ No audio generated for any speakers"

        # Combine all audio chunks with appropriate pauses
        print("🔗 Combining audio chunks...")
        final_conversation_audio = []

        for i, audio_chunk in enumerate(conversation_audio_chunks):
            # Add the current audio chunk
            final_conversation_audio.extend(audio_chunk)

            # Add pause after each line (except the last one)
            if i < len(conversation_audio_chunks) - 1:
                current_speaker = conversation_info[i]["speaker"]
                next_speaker = conversation_info[i + 1]["speaker"]

                # Different pause duration based on speaker change
                if current_speaker != next_speaker:
                    pause_duration = conversation_pause_duration
                else:
                    pause_duration = speaker_transition_pause

                # Handle negative pause (overlap)
                if pause_duration < 0:
                    # Overlap: remove samples from the end of current chunk
                    overlap_samples = int(abs(pause_duration) * sample_rate)
                    if overlap_samples < len(final_conversation_audio):
                        final_conversation_audio = final_conversation_audio[:-overlap_samples]
                else:
                    # Add silence
                    pause_samples = int(pause_duration * sample_rate)
                    silence = np.zeros(pause_samples)
                    final_conversation_audio.extend(silence)

        # Convert to numpy array
        final_conversation_audio = np.array(final_conversation_audio, dtype=np.float32)

        # Save the conversation audio to outputs folder
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"conversation_{selected_engine.lower().replace(' ', '_')}_{timestamp}"
            filepath, filename = save_audio_with_format(
                final_conversation_audio, sample_rate, audio_format, output_folder, filename_base
            )
            print(f"💾 Conversation saved as: {filename}")
        except Exception as save_error:
            print(f"Warning: Could not save conversation file: {save_error}")
            filename = "conversation_audio"

        # Create conversation summary
        total_duration = len(final_conversation_audio) / sample_rate
        unique_speakers = len(set([info["speaker"] for info in conversation_info]))
        meta_path, script_path = write_generation_sidecar_metadata(
            filepath if "filepath" in locals() else "",
            {
                "mode": "conversation",
                "engine": selected_engine,
                "total_lines": len(conversation),
                "unique_speakers": unique_speakers,
                "total_duration_seconds": float(total_duration),
                "emotion_controls_used": True,
            },
            conversation_script,
            original_text=conversation_script,
            transformed_text=conversation_script,
        )

        summary = {
            "total_lines": len(conversation),
            "unique_speakers": unique_speakers,
            "total_duration": total_duration,
            "speakers": list(set([info["speaker"] for info in conversation_info])),
            "conversation_info": conversation_info,
            "engine_used": selected_engine,
            "emotion_controls_used": True,
            "saved_file": filename,
            "metadata_file": meta_path,
            "script_file": script_path,
        }

        print(
            f"✅ IndexTTS2 conversation generated: {len(conversation)} lines, {unique_speakers} speakers, {total_duration:.1f}s"
        )

        return (sample_rate, final_conversation_audio), summary

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"❌ IndexTTS2 conversation error: {str(e)}"


def generate_fish_speech_simple(text, ref_audio=None, effects_settings=None, audio_format="wav"):
    """Simplified Fish Speech generation for conversation mode."""
    if not FISH_SPEECH_AVAILABLE:
        return None, "❌ Fish Speech not available"

    if not MODEL_STATUS["fish_speech"]["loaded"] or FISH_SPEECH_ENGINE is None:
        return None, "❌ Fish Speech not loaded"

    try:
        cleaned_text = _sanitize_fish_speech_text(text)
        if not cleaned_text:
            return None, "❌ Fish Speech input is empty after text cleanup"

        print(f"🐟 Fish Speech generating: {cleaned_text[:50]}...")

        # Prepare reference audio if provided
        references = []
        if ref_audio and os.path.exists(ref_audio):
            print(f"🎤 Using reference audio: {ref_audio}")
            ref_audio_bytes = audio_to_bytes(ref_audio)
            references.append(ServeReferenceAudio(audio=ref_audio_bytes, text=""))  # type: ignore

        # Generate consistent seed for voice consistency if no reference
        seed = None
        if not references:
            import time

            seed = int(time.time()) % 1000000
            print(f"🐟 Using seed {seed} for voice consistency")

        # Create simple TTS request
        request = ServeTTSRequest(
            text=cleaned_text,
            references=references,
            reference_id=None,
            format="wav",
            max_new_tokens=768,
            chunk_length=180,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
            streaming=False,
            use_memory_cache="off",
            seed=seed,  # Use consistent seed
            normalize=False,
        )

        print("🐟 Calling Fish Speech inference...")

        # Generate audio
        results = list(FISH_SPEECH_ENGINE.inference(request))

        # Find the final result
        final_result = None
        for result in results:
            if result.code == "final":
                final_result = result
                break
            elif result.code == "error":
                return None, f"❌ Fish Speech error: {str(result.error)}"

        if final_result is None or final_result.error is not None:
            error_msg = str(final_result.error) if final_result else "No audio generated"
            return None, f"❌ Fish Speech error: {error_msg}"

        # Extract audio data
        sample_rate, audio_data = final_result.audio

        # Convert to float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Simple normalization
        peak = np.max(np.abs(audio_data))
        if peak > 1.0:
            audio_data = audio_data / peak

        print(f"✅ Fish Speech generated: {len(audio_data)} samples")

        return (sample_rate, audio_data), "✅ Generated with Fish Speech"

    except Exception as e:
        import traceback

        traceback.print_exc()
        message = str(e)
        if "device-side assert" in message or "index out of bounds" in message:
            return None, (
                "❌ Fish Speech CUDA index error. Text was auto-cleaned but GPU context is now unstable. "
                "Please unload/reload Fish Speech (or restart app), then retry with plain narration text "
                "without style tags like (calm)/(break)."
            )
        return None, f"❌ Fish Speech error: {str(e)}"


def _sanitize_fish_speech_text(text_input: str) -> str:
    if not isinstance(text_input, str):
        return ""

    text = unicodedata.normalize("NFKC", text_input)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"\((?:\s*[A-Za-z][A-Za-z _\-]{0,40}\s*)\)", " ", text)
    text = re.sub(r"\[(?:\s*[A-Za-z][A-Za-z _\-]{0,40}\s*)\]", " ", text)
    text = re.sub(r"\{(?:\s*[A-Za-z][A-Za-z _\-]{0,40}\s*)\}", " ", text)

    text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===== HELPER FUNCTIONS =====
def save_audio_with_format(
    audio_data, sample_rate, output_format="wav", output_folder=None, filename_base=None
):
    """
    Save audio data in the specified format (WAV or MP3).

    Args:
        audio_data: numpy array of audio samples
        sample_rate: sample rate of the audio
        output_format: "wav" or "mp3"
        output_folder: folder to save the file (default: global output_folder)
        filename_base: base filename without extension (default: auto-generated)

    Returns:
        tuple: (filepath, filename) of the saved file
    """
    if output_folder is None:
        output_folder = get_runtime_output_dir("outputs")

    if filename_base is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"tts_output_{timestamp}"

    # Ensure output format is lowercase
    output_format = output_format.lower()

    # Ensure the folder exists
    os.makedirs(output_folder, exist_ok=True)

    if output_format == "wav":
        # Save directly as WAV using soundfile for better quality
        filename = f"{filename_base}.wav"
        filepath = os.path.join(output_folder, filename)

        try:
            import soundfile as sf

            # Check if audio_data is already in the right range
            if np.max(np.abs(audio_data)) <= 1.0:
                # Audio is already normalized, save directly
                sf.write(filepath, audio_data, sample_rate)
            else:
                # Audio needs normalization
                normalized_audio = audio_data / np.max(np.abs(audio_data))
                sf.write(filepath, normalized_audio, sample_rate)
        except ImportError:
            # Fallback to scipy if soundfile not available
            if np.max(np.abs(audio_data)) <= 1.0:
                # Audio is normalized, scale to int16 range
                write(filepath, sample_rate, (audio_data * 32767).astype(np.int16))
            else:
                # Audio needs normalization first
                normalized_audio = audio_data / np.max(np.abs(audio_data))
                write(filepath, sample_rate, (normalized_audio * 32767).astype(np.int16))

        return filepath, filename

    elif output_format == "mp3":
        # Convert to MP3 using pydub with high quality settings
        try:
            import tempfile
            import soundfile as sf

            # First save as temporary high-quality WAV using soundfile
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()

            # Normalize audio if needed and save as WAV
            if np.max(np.abs(audio_data)) <= 1.0:
                # Audio is already normalized
                sf.write(temp_wav.name, audio_data, sample_rate)
            else:
                # Audio needs normalization
                normalized_audio = audio_data / np.max(np.abs(audio_data))
                sf.write(temp_wav.name, normalized_audio, sample_rate)

            # Convert to MP3 with high quality settings
            audio_segment = AudioSegment.from_wav(temp_wav.name)
            filename = f"{filename_base}.mp3"
            filepath = os.path.join(output_folder, filename)

            # Export with high quality settings
            audio_segment.export(
                filepath,
                format="mp3",
                bitrate="320k",  # Higher bitrate for better quality
                parameters=["-q:a", "0"],  # Highest quality setting for ffmpeg
            )

            # Clean up temporary file
            os.unlink(temp_wav.name)

            return filepath, filename

        except Exception as e:
            print(f"Error converting to MP3: {e}")
            # Fallback to WAV with proper normalization
            print("Falling back to WAV format...")
            filename = f"{filename_base}.wav"
            filepath = os.path.join(output_folder, filename)

            if np.max(np.abs(audio_data)) <= 1.0:
                write(filepath, sample_rate, (audio_data * 32767).astype(np.int16))
            else:
                normalized_audio = audio_data / np.max(np.abs(audio_data))
                write(filepath, sample_rate, (normalized_audio * 32767).astype(np.int16))

            return filepath, filename

    else:
        raise ValueError(f"Unsupported audio format: {output_format}. Supported formats: wav, mp3")


# ===== GLOBAL CONFIGURATION =====
with suppress_specific_warnings():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# Cache configuration for Kokoro
cache_base = os.path.abspath(os.path.join(os.getcwd(), "cache"))
os.environ["HF_HOME"] = os.path.abspath(os.path.join(cache_base, "HF_HOME"))
os.environ["TORCH_HOME"] = os.path.abspath(os.path.join(cache_base, "TORCH_HOME"))
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Disable warnings
warnings.filterwarnings("ignore")
if hasattr(torch.nn.utils, "parametrizations") and hasattr(
    torch.nn.utils.parametrizations, "weight_norm"
):
    torch.nn.utils.parametrize = torch.nn.utils.parametrizations.weight_norm

# ===== DIRECTORY SETUP =====
APP_STATE_DIR = os.path.join(os.getcwd(), "app_state")
APP_STATE_PRESETS_FILE = os.path.join(APP_STATE_DIR, "presets.json")
APP_STATE_BUNDLES_FILE = os.path.join(APP_STATE_DIR, "bundles.json")
APP_STATE_SETTINGS_FILE = os.path.join(APP_STATE_DIR, "settings.json")
APP_STATE_SPEAKER_PROFILES_FILE = Path(APP_STATE_DIR) / "speaker_profiles.json"
APP_STATE_VOICES_DIR = os.path.join(APP_STATE_DIR, "voices")
APP_STATE_OUTPUTS_DIR = os.path.join(APP_STATE_DIR, "outputs")

# Legacy preset file (migrated one-way to app_state/presets.json)
PRESETS_FILE = "voice_presets.json"

output_folder = os.path.join(os.getcwd(), "outputs")
custom_voices_folder = os.path.join(os.getcwd(), "custom_voices")
audiobooks_folder = os.path.join(os.getcwd(), "audiobooks")

DEFAULT_AUTOSAVE_SETTINGS = {
    "filename_template": "{project}_{preset}_{timestamp}",
    "output_storage_mode": "project",
    "output_storage_path": "",
    "narration_llm_provider": "LM Studio OpenAI Server",
    "narration_llm_preset": "Balanced",
    "narration_llm_base_url": "",
    "narration_llm_model_id": "",
    "narration_llm_content_type": DEFAULT_CONTENT_TYPE_PRESET,
    "narration_llm_system_prompt": "",
    "assistant_llm_provider": "LM Studio OpenAI Server",
    "assistant_llm_preset": "Balanced",
    "assistant_llm_base_url": "",
    "assistant_llm_model_id": "",
    "assistant_llm_system_prompt": "",
}

LEGACY_LLM_SETTINGS_KEY_MAP = {
    "llm_provider": "narration_llm_provider",
    "llm_preset": "narration_llm_preset",
    "llm_base_url": "narration_llm_base_url",
    "llm_model_id": "narration_llm_model_id",
    "llm_system_prompt": "narration_llm_system_prompt",
}

LEGACY_DEFAULT_FILENAME_TEMPLATE = "{project}_{speaker}_{timestamp}"
PREVIOUS_DEFAULT_FILENAME_TEMPLATE = "{preset}_{project}_{timestamp}"
PRESET_ONLY_FILENAME_TEMPLATE = "{preset}_{timestamp}"


def ensure_app_state_dirs():
    os.makedirs(APP_STATE_DIR, exist_ok=True)
    os.makedirs(APP_STATE_VOICES_DIR, exist_ok=True)
    os.makedirs(APP_STATE_OUTPUTS_DIR, exist_ok=True)
    if not os.path.exists(APP_STATE_SETTINGS_FILE):
        with open(APP_STATE_SETTINGS_FILE, "w", encoding="utf-8") as file:
            json.dump(DEFAULT_AUTOSAVE_SETTINGS, file, indent=2, ensure_ascii=False)


def load_app_state_settings() -> dict:
    ensure_app_state_dirs()
    try:
        with open(APP_STATE_SETTINGS_FILE, "r", encoding="utf-8") as file:
            settings = json.load(file)
            if isinstance(settings, dict):
                merged = {**DEFAULT_AUTOSAVE_SETTINGS, **settings}
                needs_persist = False

                if "llm_provider" in settings and "narration_llm_provider" not in settings:
                    for legacy_key, namespaced_key in LEGACY_LLM_SETTINGS_KEY_MAP.items():
                        if legacy_key in settings:
                            merged[namespaced_key] = settings[legacy_key]
                        merged.pop(legacy_key, None)
                    needs_persist = True

                if merged.get("filename_template") in {
                    LEGACY_DEFAULT_FILENAME_TEMPLATE,
                    PREVIOUS_DEFAULT_FILENAME_TEMPLATE,
                    PRESET_ONLY_FILENAME_TEMPLATE,
                }:
                    merged["filename_template"] = DEFAULT_AUTOSAVE_SETTINGS["filename_template"]
                    needs_persist = True

                if needs_persist:
                    try:
                        with open(APP_STATE_SETTINGS_FILE, "w", encoding="utf-8") as settings_file:
                            json.dump(merged, settings_file, indent=2, ensure_ascii=False)
                    except Exception as write_error:
                        print(f"⚠️ Failed to persist settings migration: {write_error}")
                return merged
    except Exception as error:
        print(f"⚠️ Failed to load app_state settings: {error}")
    return dict(DEFAULT_AUTOSAVE_SETTINGS)


def resolve_output_storage_settings(settings: dict | None = None):
    if settings is None:
        settings = load_app_state_settings()

    mode = str(settings.get("output_storage_mode", "project") or "project").strip().lower()
    if mode not in {"project", "custom"}:
        mode = "project"

    raw_path = str(settings.get("output_storage_path", "") or "").strip()
    custom_base = os.path.abspath(os.path.expanduser(raw_path)) if raw_path else ""

    if mode == "custom" and custom_base:
        try:
            os.makedirs(custom_base, exist_ok=True)
        except Exception as error:
            print(f"⚠️ Could not access custom output storage '{custom_base}': {error}")
            mode = "project"
            custom_base = ""

    return mode, custom_base


def get_runtime_output_dir(kind: str, settings: dict | None = None) -> str:
    mode, custom_base = resolve_output_storage_settings(settings)

    if kind == "outputs":
        target = (
            os.path.join(custom_base, "outputs")
            if mode == "custom" and custom_base
            else os.path.join(os.getcwd(), "outputs")
        )
    elif kind == "audiobooks":
        target = (
            os.path.join(custom_base, "audiobooks")
            if mode == "custom" and custom_base
            else os.path.join(os.getcwd(), "audiobooks")
        )
    elif kind == "autosave":
        target = (
            os.path.join(custom_base, "app_state_outputs")
            if mode == "custom" and custom_base
            else APP_STATE_OUTPUTS_DIR
        )
    else:
        target = os.path.join(os.getcwd(), "outputs")

    os.makedirs(target, exist_ok=True)
    return target


def refresh_runtime_storage_paths(settings: dict | None = None):
    global output_folder, audiobooks_folder
    output_folder = get_runtime_output_dir("outputs", settings)
    audiobooks_folder = get_runtime_output_dir("audiobooks", settings)
    return output_folder, audiobooks_folder


def save_app_state_settings(updates: dict) -> dict:
    ensure_app_state_dirs()
    current = load_app_state_settings()
    current.update(updates or {})
    with open(APP_STATE_SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(current, file, indent=2, ensure_ascii=False)
    return current


def normalize_llm_outcome_preset(preset_name: str | None) -> str:
    candidate = str(preset_name or "").strip()
    if candidate not in LLM_OUTCOME_PRESETS:
        return DEFAULT_LLM_OUTCOME_PRESET
    return candidate


def normalize_llm_content_type(content_type_name: str | None) -> str:
    candidate = str(content_type_name or "").strip()
    if candidate not in get_content_type_preset_names():
        return DEFAULT_CONTENT_TYPE_PRESET
    return candidate


def get_llm_outcome_preset_values(preset_name: str | None) -> tuple[float, float, int]:
    normalized_preset = normalize_llm_outcome_preset(preset_name)
    params = LLM_OUTCOME_PRESETS[normalized_preset]
    return float(params["temperature"]), float(params["top_p"]), int(params["max_tokens"])


def _get_llm_settings_key(namespace: str, setting_name: str) -> str:
    return f"{namespace}_llm_{setting_name}"


def _get_default_llm_setting(namespace: str, setting_name: str) -> str:
    return str(DEFAULT_AUTOSAVE_SETTINGS[_get_llm_settings_key(namespace, setting_name)])


def _get_initial_namespaced_llm_settings(
    namespace: str,
    default_system_prompt: str,
    settings: dict | None = None,
) -> dict:
    if settings is None:
        settings = load_app_state_settings()

    provider_key = _get_llm_settings_key(namespace, "provider")
    preset_key = _get_llm_settings_key(namespace, "preset")
    base_url_key = _get_llm_settings_key(namespace, "base_url")
    model_id_key = _get_llm_settings_key(namespace, "model_id")
    system_prompt_key = _get_llm_settings_key(namespace, "system_prompt")

    default_provider = _get_default_llm_setting(namespace, "provider")
    provider_name = str(settings.get(provider_key, default_provider) or default_provider)
    if provider_name not in LLM_PROVIDER_CONFIGS:
        provider_name = default_provider

    preset_name = normalize_llm_outcome_preset(settings.get(preset_key))
    temperature, top_p, max_tokens = get_llm_outcome_preset_values(preset_name)

    provider_config = _get_provider_config(provider_name)
    base_url = str(settings.get(base_url_key, "") or "").strip() or provider_config["base_url"]
    # Secret hygiene: API keys are session-only and resolved via env vars at runtime.
    api_key = ""
    model_id = str(settings.get(model_id_key, "") or "").strip() or provider_config["default_model"]
    system_prompt = str(settings.get(system_prompt_key, "") or "")
    if not system_prompt:
        system_prompt = default_system_prompt

    model_choices = list(LLM_PROVIDER_MODEL_SUGGESTIONS.get(provider_name, []))
    if provider_config["default_model"] and provider_config["default_model"] not in model_choices:
        model_choices.insert(0, provider_config["default_model"])
    if model_id and model_id not in model_choices:
        model_choices.insert(0, model_id)

    return {
        "provider": provider_name,
        "preset": preset_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "base_url": base_url,
        "api_key": api_key,
        "model_id": model_id,
        "model_choices": model_choices,
        "system_prompt": system_prompt,
    }


def get_initial_llm_panel_settings(settings: dict | None = None) -> dict:
    if settings is None:
        settings = load_app_state_settings()

    content_type = normalize_llm_content_type(
        settings.get(_get_llm_settings_key("narration", "content_type"))
    )
    llm_settings = _get_initial_namespaced_llm_settings(
        namespace="narration",
        default_system_prompt=get_content_type_system_prompt(content_type),
        settings=settings,
    )
    llm_settings["content_type"] = content_type
    return llm_settings


def get_initial_assistant_llm_settings(settings: dict | None = None) -> dict:
    return _get_initial_namespaced_llm_settings(
        namespace="assistant",
        default_system_prompt="",
        settings=settings,
    )


def _save_namespaced_llm_settings(
    namespace: str,
    provider_name: str,
    base_url: str,
    model_id: str,
    system_prompt: str,
    preset_name: str | None = None,
    default_system_prompt: str = "",
    content_type_name: str | None = None,
) -> None:
    normalized_provider = str(provider_name or "").strip()
    default_provider = _get_default_llm_setting(namespace, "provider")
    if normalized_provider not in LLM_PROVIDER_CONFIGS:
        normalized_provider = default_provider

    normalized_preset = normalize_llm_outcome_preset(preset_name)
    normalized_content_type = None
    effective_default_system_prompt = default_system_prompt
    if namespace == "narration":
        normalized_content_type = normalize_llm_content_type(content_type_name)
        effective_default_system_prompt = get_content_type_system_prompt(normalized_content_type)

    # Secret hygiene: API keys are session-only and resolved via env vars at runtime.
    updates = {
        _get_llm_settings_key(namespace, "provider"): normalized_provider,
        _get_llm_settings_key(namespace, "preset"): normalized_preset,
        _get_llm_settings_key(namespace, "base_url"): str(base_url or "").strip(),
        _get_llm_settings_key(namespace, "model_id"): str(model_id or "").strip(),
        _get_llm_settings_key(namespace, "system_prompt"): (
            ""
            if str(system_prompt or "") == effective_default_system_prompt
            else str(system_prompt or "")
        ),
    }
    if namespace == "narration" and normalized_content_type is not None:
        updates[_get_llm_settings_key(namespace, "content_type")] = normalized_content_type

    save_app_state_settings(updates)


def save_llm_panel_settings(
    provider_name: str,
    base_url: str,
    model_id: str,
    api_key: str,
    content_type_name: str,
    system_prompt: str,
    preset_name: str | None = None,
) -> None:
    try:
        _save_namespaced_llm_settings(
            namespace="narration",
            provider_name=provider_name,
            base_url=base_url,
            model_id=model_id,
            system_prompt=system_prompt,
            preset_name=preset_name,
            default_system_prompt=DEFAULT_LLM_NARRATION_SYSTEM_PROMPT,
            content_type_name=content_type_name,
        )
    except Exception as error:
        print(f"⚠️ Failed to save LLM settings: {error}")


def save_assistant_llm_settings(
    provider_name: str,
    base_url: str,
    model_id: str,
    api_key: str,
    system_prompt: str,
    preset_name: str | None = None,
) -> None:
    try:
        _save_namespaced_llm_settings(
            namespace="assistant",
            provider_name=provider_name,
            base_url=base_url,
            model_id=model_id,
            system_prompt=system_prompt,
            preset_name=preset_name,
            default_system_prompt="",
        )
    except Exception as error:
        print(f"⚠️ Failed to save assistant LLM settings: {error}")


def save_output_storage_settings(mode_label: str, custom_path: str):
    mode = "custom" if str(mode_label or "").lower().startswith("custom") else "project"
    normalized_path = str(custom_path or "").strip()

    if mode == "custom" and not normalized_path:
        return "❌ Custom output mode requires a base path"

    try:
        updates = {
            "output_storage_mode": mode,
            "output_storage_path": normalized_path if mode == "custom" else "",
        }
        settings = save_app_state_settings(updates)
        active_outputs, active_audiobooks = refresh_runtime_storage_paths(settings)
        active_autosave = get_runtime_output_dir("autosave", settings)

        mode_text = "Custom Path" if mode == "custom" else "Project Folders"
        return (
            f"✅ Output storage updated\n"
            f"Mode: {mode_text}\n"
            f"Outputs: {os.path.abspath(active_outputs)}\n"
            f"Audiobooks: {os.path.abspath(active_audiobooks)}\n"
            f"Autosave: {os.path.abspath(active_autosave)}"
        )
    except Exception as error:
        return f"❌ Failed to save output storage settings: {error}"


def choose_custom_output_storage_path(mode_label: str, current_path: str):
    if not str(mode_label or "").lower().startswith("custom"):
        return gr.update(value=current_path), "ℹ️ Output storage mode set to project folders"

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()

        initial_dir = (
            current_path if isinstance(current_path, str) and current_path.strip() else os.getcwd()
        )
        selected_path = filedialog.askdirectory(initialdir=initial_dir, mustexist=False)
        root.destroy()

        if selected_path:
            return (
                gr.update(value=selected_path),
                f"📁 Selected custom output path:\n{os.path.abspath(selected_path)}\nClick 'Save Output Storage' to apply",
            )

        return gr.update(value=current_path), "ℹ️ Custom output path selection canceled"
    except Exception as error:
        return gr.update(value=current_path), f"⚠️ Could not open folder picker: {error}"


def open_active_output_folder():
    try:
        folder_path = os.path.abspath(get_runtime_output_dir("outputs"))
        os.makedirs(folder_path, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(folder_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder_path])
        else:
            subprocess.Popen(["xdg-open", folder_path])

        return f"📂 Opened output folder:\n{folder_path}"
    except Exception as error:
        return f"❌ Failed to open output folder: {error}"


def open_active_autosave_folder():
    try:
        folder_path = os.path.abspath(get_runtime_output_dir("autosave"))
        os.makedirs(folder_path, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(folder_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder_path])
        else:
            subprocess.Popen(["xdg-open", folder_path])

        return f"📂 Opened autosave folder:\n{folder_path}"
    except Exception as error:
        return f"❌ Failed to open autosave folder: {error}"


def _list_audio_files(folder_path: str):
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    collected = set()
    try:
        for name in os.listdir(folder_path):
            abs_path = os.path.join(folder_path, name)
            if os.path.isfile(abs_path) and os.path.splitext(name)[1].lower() in audio_extensions:
                collected.add(os.path.abspath(abs_path))
    except Exception:
        pass
    return collected


def _detect_new_audio_file(before_files, after_files):
    new_files = list((after_files or set()) - (before_files or set()))
    if not new_files:
        return None
    try:
        return max(new_files, key=lambda candidate: os.path.getmtime(candidate))
    except Exception:
        return new_files[-1]


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def write_generation_sidecar_metadata(
    audio_path: str,
    metadata: Optional[dict[str, Any]] = None,
    script_text: Optional[str] = None,
    original_text: Optional[str] = None,
    transformed_text: Optional[str] = None,
):
    if not audio_path:
        return None, None

    abs_audio_path = os.path.abspath(audio_path)
    base, _ = os.path.splitext(abs_audio_path)
    meta_path = f"{base}.json"
    script_path = None

    original_script_path = None
    transformed_script_path = None

    base_script_text = script_text if isinstance(script_text, str) else ""
    original_script_text = original_text if isinstance(original_text, str) else base_script_text
    transformed_script_text = (
        transformed_text if isinstance(transformed_text, str) else base_script_text
    )

    include_text_versions = bool(original_script_text or transformed_script_text)

    text_versions_payload = None
    if include_text_versions:
        text_versions_payload = {
            "original": {
                "sha256": hashlib.sha256((original_script_text or "").encode("utf-8")).hexdigest(),
                "chars": len(original_script_text or ""),
            },
            "transformed": {
                "sha256": hashlib.sha256(
                    (transformed_script_text or "").encode("utf-8")
                ).hexdigest(),
                "chars": len(transformed_script_text or ""),
            },
        }

    safe_metadata = _json_safe(metadata or {})
    if not isinstance(safe_metadata, dict):
        safe_metadata = {}

    payload = {
        "timestamp": datetime.now().isoformat(),
        "audio_path": abs_audio_path,
        **safe_metadata,
    }

    if text_versions_payload:
        payload["text_versions"] = text_versions_payload

    if include_text_versions:
        script_path = f"{base}.txt"
        original_script_path = f"{base}.original.txt"
        transformed_script_path = f"{base}.transformed.txt"

        with open(script_path, "w", encoding="utf-8") as file:
            file.write(transformed_script_text)

        with open(original_script_path, "w", encoding="utf-8") as file:
            file.write(original_script_text)

        with open(transformed_script_path, "w", encoding="utf-8") as file:
            file.write(transformed_script_text)

        payload["paths"] = {
            "script": os.path.abspath(script_path),
            "script_original": os.path.abspath(original_script_path),
            "script_transformed": os.path.abspath(transformed_script_path),
            "meta": os.path.abspath(meta_path),
        }

    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    if not include_text_versions and isinstance(script_text, str) and script_text.strip():
        script_path = f"{base}.txt"
        with open(script_path, "w", encoding="utf-8") as file:
            file.write(script_text)

    return meta_path, script_path


# Create necessary folders
ensure_app_state_dirs()
refresh_runtime_storage_paths()
os.makedirs(custom_voices_folder, exist_ok=True)

# ===== MODEL INITIALIZATION =====
CHATTERBOX_MODEL = None
CHATTERBOX_MULTILINGUAL_MODEL = None
KOKORO_PIPELINES = {}
FISH_SPEECH_ENGINE = None
FISH_SPEECH_LLAMA_QUEUE = None
INDEXTTS_MODEL = None
loaded_voices = {}

# Model loading status
MODEL_STATUS = {
    "chatterbox": {"loaded": False, "loading": False},
    "chatterbox_multilingual": {"loaded": False, "loading": False},
    "chatterbox_turbo": {"loaded": False, "loading": False},
    "kokoro": {"loaded": False, "loading": False},
    "vibevoice": {"loaded": False, "loading": False},
    "fish_speech": {"loaded": False, "loading": False},
    "indextts": {"loaded": False, "loading": False},
    "indextts2": {"loaded": False, "loading": False},
    "f5_tts": {"loaded": False, "loading": False, "models": {}},
    "higgs_audio": {"loaded": False, "loading": False},
    "kitten_tts": {"loaded": False, "loading": False},
    "qwen_tts": {"loaded": False, "loading": False},
    "voxcpm": {"loaded": False, "loading": False},
}


def init_chatterbox():
    """Initialize ChatterboxTTS model."""
    global CHATTERBOX_MODEL, MODEL_STATUS
    if not CHATTERBOX_AVAILABLE:
        return False, "❌ ChatterboxTTS not available - check installation"

    if MODEL_STATUS["chatterbox"]["loaded"]:
        return True, "✅ ChatterboxTTS already loaded"

    if MODEL_STATUS["chatterbox"]["loading"]:
        return False, "⏳ ChatterboxTTS is currently loading..."

    try:
        MODEL_STATUS["chatterbox"]["loading"] = True
        print("🔄 Loading ChatterboxTTS...")
        with suppress_specific_warnings():
            CHATTERBOX_MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        MODEL_STATUS["chatterbox"]["loaded"] = True
        MODEL_STATUS["chatterbox"]["loading"] = False
        print("✅ ChatterboxTTS loaded successfully")
        return True, "✅ ChatterboxTTS loaded successfully"
    except Exception as e:
        MODEL_STATUS["chatterbox"]["loading"] = False
        error_msg = f"❌ Failed to load ChatterboxTTS: {e}"
        print(error_msg)
        return False, error_msg


def unload_chatterbox():
    """Unload ChatterboxTTS model to free memory."""
    global CHATTERBOX_MODEL, MODEL_STATUS
    try:
        if CHATTERBOX_MODEL is not None:
            del CHATTERBOX_MODEL
            CHATTERBOX_MODEL = None

        # Force garbage collection
        import gc

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        MODEL_STATUS["chatterbox"]["loaded"] = False
        print("✅ ChatterboxTTS unloaded successfully")
        return "✅ ChatterboxTTS unloaded - memory freed"
    except Exception as e:
        error_msg = f"❌ Error unloading ChatterboxTTS: {e}"
        print(error_msg)
        return error_msg


def init_chatterbox_multilingual():
    """Initialize ChatterboxMultilingualTTS model."""
    global CHATTERBOX_MULTILINGUAL_MODEL, MODEL_STATUS
    if not CHATTERBOX_MULTILINGUAL_AVAILABLE:
        return False, "❌ ChatterboxMultilingualTTS not available - check installation"

    if MODEL_STATUS["chatterbox_multilingual"]["loaded"]:
        return True, "✅ ChatterboxMultilingualTTS already loaded"

    if MODEL_STATUS["chatterbox_multilingual"]["loading"]:
        return False, "⏳ ChatterboxMultilingualTTS is currently loading..."

    try:
        MODEL_STATUS["chatterbox_multilingual"]["loading"] = True
        print("🔄 Loading ChatterboxMultilingualTTS...")
        with suppress_specific_warnings():
            CHATTERBOX_MULTILINGUAL_MODEL = ChatterboxMultilingualTTS.from_pretrained(
                torch.device(DEVICE)
            )
        MODEL_STATUS["chatterbox_multilingual"]["loaded"] = True
        MODEL_STATUS["chatterbox_multilingual"]["loading"] = False
        print("✅ ChatterboxMultilingualTTS loaded successfully")
        return True, "✅ ChatterboxMultilingualTTS loaded successfully"
    except Exception as e:
        MODEL_STATUS["chatterbox_multilingual"]["loading"] = False
        error_msg = f"❌ Failed to load ChatterboxMultilingualTTS: {e}"
        print(error_msg)
        return False, error_msg


def unload_chatterbox_multilingual():
    """Unload ChatterboxMultilingualTTS model to free memory."""
    global CHATTERBOX_MULTILINGUAL_MODEL, MODEL_STATUS
    try:
        if CHATTERBOX_MULTILINGUAL_MODEL is not None:
            del CHATTERBOX_MULTILINGUAL_MODEL
            CHATTERBOX_MULTILINGUAL_MODEL = None

        # Force garbage collection
        import gc

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        MODEL_STATUS["chatterbox_multilingual"]["loaded"] = False
        print("✅ ChatterboxMultilingualTTS unloaded successfully")
        return "✅ ChatterboxMultilingualTTS unloaded - memory freed"
    except Exception as e:
        error_msg = f"❌ Error unloading ChatterboxMultilingualTTS: {e}"
        print(error_msg)
        return error_msg


def init_chatterbox_turbo_model():
    """Initialize Chatterbox Turbo model."""
    global MODEL_STATUS
    if not CHATTERBOX_TURBO_AVAILABLE:
        return False, "❌ Chatterbox Turbo not available - check installation"

    if MODEL_STATUS["chatterbox_turbo"]["loaded"]:
        return True, "✅ Chatterbox Turbo already loaded"

    if MODEL_STATUS["chatterbox_turbo"]["loading"]:
        return False, "⏳ Chatterbox Turbo is currently loading..."

    try:
        MODEL_STATUS["chatterbox_turbo"]["loading"] = True
        print("🔄 Loading Chatterbox Turbo...")
        success, message = init_chatterbox_turbo()
        if success:
            MODEL_STATUS["chatterbox_turbo"]["loaded"] = True
            MODEL_STATUS["chatterbox_turbo"]["loading"] = False
            print("✅ Chatterbox Turbo loaded successfully")
            return True, "✅ Chatterbox Turbo loaded successfully"
        else:
            MODEL_STATUS["chatterbox_turbo"]["loading"] = False
            return False, message
    except Exception as e:
        MODEL_STATUS["chatterbox_turbo"]["loading"] = False
        error_msg = f"❌ Failed to load Chatterbox Turbo: {e}"
        print(error_msg)
        return False, error_msg


def unload_chatterbox_turbo_model():
    """Unload Chatterbox Turbo model to free memory."""
    global MODEL_STATUS
    try:
        message = unload_chatterbox_turbo()
        MODEL_STATUS["chatterbox_turbo"]["loaded"] = False
        print("✅ Chatterbox Turbo unloaded successfully")
        return message
    except Exception as e:
        error_msg = f"❌ Error unloading Chatterbox Turbo: {e}"
        print(error_msg)
        return error_msg


def init_kokoro():
    """Initialize Kokoro TTS models and pipelines."""
    global KOKORO_PIPELINES, MODEL_STATUS
    if not KOKORO_AVAILABLE:
        return False, "❌ Kokoro TTS not available - check installation"

    if MODEL_STATUS["kokoro"]["loaded"]:
        return True, "✅ Kokoro TTS already loaded"

    if MODEL_STATUS["kokoro"]["loading"]:
        return False, "⏳ Kokoro TTS is currently loading..."

    try:
        MODEL_STATUS["kokoro"]["loading"] = True
        print("🔄 Loading Kokoro TTS...")

        # Check if first run
        if not os.path.exists(os.path.join(cache_base, "HF_HOME/hub/models--hexgrad--Kokoro-82M")):
            print("Downloading/Loading Kokoro models...")
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)

        # Load pipelines only (no need for separate KModel)
        with suppress_specific_warnings():
            KOKORO_PIPELINES = {
                lang_code: KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code=lang_code)
                for lang_code in "abpi"
            }

        # Configure lexicons
        KOKORO_PIPELINES["a"].g2p.lexicon.golds["kokoro"] = "kˈOkəɹO"
        KOKORO_PIPELINES["b"].g2p.lexicon.golds["kokoro"] = "kˈQkəɹQ"

        try:
            if hasattr(KOKORO_PIPELINES["i"].g2p, "lexicon"):
                KOKORO_PIPELINES["i"].g2p.lexicon.golds["kokoro"] = "kˈkɔro"
        except Exception as e:
            print(f"Warning: Could not set Italian pronunciation: {e}")

        # Re-enable offline mode
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        MODEL_STATUS["kokoro"]["loaded"] = True
        MODEL_STATUS["kokoro"]["loading"] = False
        print("✅ Kokoro TTS loaded successfully")
        return True, "✅ Kokoro TTS loaded successfully"

    except Exception as e:
        MODEL_STATUS["kokoro"]["loading"] = False
        error_msg = f"❌ Failed to load Kokoro TTS: {e}"
        print(error_msg)
        return False, error_msg


def unload_kokoro():
    """Unload Kokoro TTS models to free memory."""
    global KOKORO_PIPELINES, loaded_voices, MODEL_STATUS
    try:
        # Clear pipelines
        for pipeline in KOKORO_PIPELINES.values():
            del pipeline
        KOKORO_PIPELINES.clear()

        # Clear loaded voices
        loaded_voices.clear()

        # Force garbage collection
        import gc

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        MODEL_STATUS["kokoro"]["loaded"] = False
        print("✅ Kokoro TTS unloaded successfully")
        return "✅ Kokoro TTS unloaded - memory freed"
    except Exception as e:
        error_msg = f"❌ Error unloading Kokoro TTS: {e}"
        print(error_msg)
        return error_msg


def init_fish_speech():
    """Initialize Fish Speech TTS engine."""
    global FISH_SPEECH_ENGINE, FISH_SPEECH_LLAMA_QUEUE, MODEL_STATUS
    if not FISH_SPEECH_AVAILABLE:
        return False, "❌ Fish Speech not available - check installation"

    if MODEL_STATUS["fish_speech"]["loaded"]:
        return True, "✅ Fish Speech already loaded"

    if MODEL_STATUS["fish_speech"]["loading"]:
        return False, "⏳ Fish Speech is currently loading..."

    try:
        MODEL_STATUS["fish_speech"]["loading"] = True
        print("🔄 Loading Fish Speech...")

        # Check for model checkpoints
        checkpoint_path = "checkpoints/openaudio-s1-mini"
        if not os.path.exists(checkpoint_path):
            MODEL_STATUS["fish_speech"]["loading"] = False
            error_msg = "❌ Fish Speech checkpoints not found. Please download them first:\nhf download cocktailpeanut/oa --local-dir ./checkpoints/openaudio-s1-mini"
            print(error_msg)
            return False, error_msg

        # Initialize LLAMA queue for text2semantic processing
        precision = torch.half if DEVICE == "cuda" else torch.bfloat16
        with suppress_specific_warnings():
            FISH_SPEECH_LLAMA_QUEUE = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=DEVICE,
                precision=precision,
                compile=False,  # Can be enabled for faster inference
            )

            # Load decoder model
            decoder_model = load_decoder_model(
                config_name="modded_dac_vq",
                checkpoint_path=os.path.join(checkpoint_path, "codec.pth"),
                device=DEVICE,
            )

            # Initialize TTS inference engine
            FISH_SPEECH_ENGINE = TTSInferenceEngine(
                llama_queue=FISH_SPEECH_LLAMA_QUEUE,
                decoder_model=decoder_model,
                precision=precision,
                compile=False,
            )

        MODEL_STATUS["fish_speech"]["loaded"] = True
        MODEL_STATUS["fish_speech"]["loading"] = False
        print("✅ Fish Speech loaded successfully")
        return True, "✅ Fish Speech loaded successfully"

    except Exception as e:
        MODEL_STATUS["fish_speech"]["loading"] = False
        error_msg = f"❌ Failed to load Fish Speech: {e}"
        print(error_msg)
        return False, error_msg


def unload_fish_speech():
    """Unload Fish Speech TTS engine to free memory."""
    global FISH_SPEECH_ENGINE, FISH_SPEECH_LLAMA_QUEUE, MODEL_STATUS
    try:
        if FISH_SPEECH_ENGINE is not None:
            del FISH_SPEECH_ENGINE
            FISH_SPEECH_ENGINE = None

        if FISH_SPEECH_LLAMA_QUEUE is not None:
            del FISH_SPEECH_LLAMA_QUEUE
            FISH_SPEECH_LLAMA_QUEUE = None

        # Force garbage collection
        import gc

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        MODEL_STATUS["fish_speech"]["loaded"] = False
        print("✅ Fish Speech unloaded successfully")
        return "✅ Fish Speech unloaded - memory freed"
    except Exception as e:
        error_msg = f"❌ Error unloading Fish Speech: {e}"
        print(error_msg)
        return error_msg


def init_indextts():
    """Initialize IndexTTS model."""
    global INDEXTTS_MODEL, MODEL_STATUS, INDEXTTS_MODELS_AVAILABLE
    if not INDEXTTS_AVAILABLE:
        return False, "❌ IndexTTS not available - check installation"

    if MODEL_STATUS["indextts"]["loaded"]:
        return True, "✅ IndexTTS already loaded"

    if MODEL_STATUS["indextts"]["loading"]:
        return False, "⏳ IndexTTS is currently loading..."

    try:
        MODEL_STATUS["indextts"]["loading"] = True
        print("🔄 Loading IndexTTS...")

        # Lazy import IndexTTS class only when initializing model
        indextts_class, import_error = load_indextts_class()
        if indextts_class is None:
            MODEL_STATUS["indextts"]["loading"] = False
            error_msg = f"❌ IndexTTS import failed: {import_error}"
            print(error_msg)
            return False, error_msg

        # Check if models are available, try to download if not
        if not INDEXTTS_MODELS_AVAILABLE:
            print("🎯 IndexTTS models not found - attempting download...")
            if download_indextts_models_auto():
                INDEXTTS_MODELS_AVAILABLE = True
                print("✅ IndexTTS models downloaded successfully")
            else:
                MODEL_STATUS["indextts"]["loading"] = False
                error_msg = "❌ IndexTTS models not available and download failed.\nRun: python tools/download_indextts_models.py"
                print(error_msg)
                return False, error_msg

        # Check for model checkpoints
        checkpoint_path = "indextts/checkpoints"
        config_path = os.path.join(checkpoint_path, "config.yaml")

        if not os.path.exists(config_path):
            MODEL_STATUS["indextts"]["loading"] = False
            error_msg = "❌ IndexTTS config not found after download attempt."
            print(error_msg)
            return False, error_msg

        # Initialize IndexTTS model
        with suppress_specific_warnings():
            INDEXTTS_MODEL = indextts_class(
                cfg_path=config_path,
                model_dir=checkpoint_path,
                is_fp16=DEVICE == "cuda",
                device=DEVICE,
                use_cuda_kernel=False,  # Disable to avoid compilation issues
            )

        MODEL_STATUS["indextts"]["loaded"] = True
        MODEL_STATUS["indextts"]["loading"] = False
        print("✅ IndexTTS loaded successfully")
        return True, "✅ IndexTTS loaded successfully"

    except Exception as e:
        MODEL_STATUS["indextts"]["loading"] = False
        error_msg = f"❌ Failed to load IndexTTS: {e}"
        print(error_msg)
        return False, error_msg


def unload_indextts():
    """Unload IndexTTS model to free memory."""
    global INDEXTTS_MODEL, MODEL_STATUS
    try:
        if INDEXTTS_MODEL is not None:
            del INDEXTTS_MODEL
            INDEXTTS_MODEL = None

        # Force garbage collection
        import gc

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        MODEL_STATUS["indextts"]["loaded"] = False
        print("✅ IndexTTS unloaded successfully")
        return "✅ IndexTTS unloaded - memory freed"
    except Exception as e:
        error_msg = f"❌ Error unloading IndexTTS: {e}"
        print(error_msg)
        return error_msg


def clear_gradio_temp_files():
    """Clear Gradio temporary files from the temp directory."""
    try:
        import tempfile
        import os

        deleted_count = 0
        deleted_size = 0

        # Function to calculate directory size and count files
        def get_directory_size(directory):
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                        file_count += 1
                    except (OSError, IOError):
                        pass
            return total_size, file_count

        # Check for Pinokio environment - look for cache/GRADIO_TEMP_DIR
        # Try multiple possible locations for Pinokio cache
        possible_pinokio_paths = [
            os.path.join(os.getcwd(), "cache", "GRADIO_TEMP_DIR"),  # Current directory
            os.path.join(
                os.path.dirname(os.getcwd()), "cache", "GRADIO_TEMP_DIR"
            ),  # Parent directory (likely for app/ subdirectory)
            os.path.join(
                os.path.dirname(os.path.dirname(os.getcwd())), "cache", "GRADIO_TEMP_DIR"
            ),  # Grandparent directory
        ]

        # Also try using the workspace path if we can detect it
        current_path = os.getcwd()
        if "Ultimate-TTS-Studio-SUP3R-Edition-Pinokio.git" in current_path:
            git_root = (
                current_path.split("Ultimate-TTS-Studio-SUP3R-Edition-Pinokio.git")[0]
                + "Ultimate-TTS-Studio-SUP3R-Edition-Pinokio.git"
            )
            possible_pinokio_paths.append(os.path.join(git_root, "cache", "GRADIO_TEMP_DIR"))

        # Check Pinokio cache directories silently
        pinokio_cache_found = False
        for pinokio_cache_dir in possible_pinokio_paths:
            if os.path.exists(pinokio_cache_dir):
                pinokio_cache_found = True
                try:
                    size, count = get_directory_size(pinokio_cache_dir)
                    shutil.rmtree(pinokio_cache_dir)
                    deleted_size += size
                    deleted_count += count
                except (OSError, IOError, PermissionError) as e:
                    pass  # Silent failure
                break  # Only delete the first one found

        # Check standard Windows temp location: AppData\Local\Temp\gradio
        if os.name == "nt":  # Windows
            user_temp = os.path.expanduser(r"~\AppData\Local\Temp\gradio")
        else:  # Linux/Mac
            user_temp = os.path.join(tempfile.gettempdir(), "gradio")

        if os.path.exists(user_temp):
            try:
                size, count = get_directory_size(user_temp)
                shutil.rmtree(user_temp)
                deleted_size += size
                deleted_count += count
            except (OSError, IOError, PermissionError) as e:
                pass  # Silent failure

        # Also check default temp directory for any gradio patterns
        temp_dir = tempfile.gettempdir()
        gradio_temp_patterns = [
            os.path.join(temp_dir, "gradio*"),
            os.path.join(temp_dir, "*gradio*"),
        ]

        for pattern in gradio_temp_patterns:
            for path in glob.glob(pattern):
                if os.path.exists(path):
                    try:
                        if os.path.isfile(path):
                            size = os.path.getsize(path)
                            os.remove(path)
                            deleted_count += 1
                            deleted_size += size
                        elif os.path.isdir(path):
                            size, count = get_directory_size(path)
                            shutil.rmtree(path)
                            deleted_size += size
                            deleted_count += count
                    except (OSError, IOError, PermissionError) as e:
                        continue  # Silent failure

        # Also check for common temp audio files in current directory
        current_dir_patterns = [
            "tmp*.wav",
            "tmp*.mp3",
            "tmp*.flac",
            "gradio_*.wav",
            "gradio_*.mp3",
            "gradio_*.flac",
            "temp_*.wav",
            "temp_*.mp3",
            "temp_*.flac",
        ]

        for pattern in current_dir_patterns:
            for filepath in glob.glob(pattern):
                try:
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_count += 1
                    deleted_size += size
                except (OSError, IOError, PermissionError):
                    continue

        # Format size for display
        if deleted_size > 1024**3:  # GB
            size_str = f"{deleted_size / (1024**3):.2f} GB"
        elif deleted_size > 1024**2:  # MB
            size_str = f"{deleted_size / (1024**2):.2f} MB"
        elif deleted_size > 1024:  # KB
            size_str = f"{deleted_size / 1024:.2f} KB"
        else:
            size_str = f"{deleted_size} bytes"

        if deleted_count > 0:
            return f"✅ Successfully deleted {deleted_count} temporary files ({size_str} freed)"
        else:
            return "ℹ️ No Gradio temporary files found to delete"

    except Exception as e:
        return f"❌ Error clearing temp files: {str(e)}"


def get_model_status():
    """Get current status of all models."""
    status_text = "📊 **Model Status:**\n\n"

    # ChatterboxTTS status
    if CHATTERBOX_AVAILABLE:
        if MODEL_STATUS["chatterbox"]["loading"]:
            status_text += "🎤 **ChatterboxTTS:** ⏳ Loading...\n"
        elif MODEL_STATUS["chatterbox"]["loaded"]:
            status_text += "🎤 **ChatterboxTTS:** ✅ Loaded\n"
        else:
            status_text += "🎤 **ChatterboxTTS:** ⭕ Not loaded\n"
    else:
        status_text += "🎤 **ChatterboxTTS:** ❌ Not available\n"

    # Kokoro TTS status
    if KOKORO_AVAILABLE:
        if MODEL_STATUS["kokoro"]["loading"]:
            status_text += "🗣️ **Kokoro TTS:** ⏳ Loading...\n"
        elif MODEL_STATUS["kokoro"]["loaded"]:
            status_text += "🗣️ **Kokoro TTS:** ✅ Loaded\n"
        else:
            status_text += "🗣️ **Kokoro TTS:** ⭕ Not loaded\n"
    else:
        status_text += "🗣️ **Kokoro TTS:** ❌ Not available\n"

    # Fish Speech status
    if FISH_SPEECH_AVAILABLE:
        if MODEL_STATUS["fish_speech"]["loading"]:
            status_text += "🐟 **Fish Speech:** ⏳ Loading...\n"
        elif MODEL_STATUS["fish_speech"]["loaded"]:
            status_text += "🐟 **Fish Speech:** ✅ Loaded\n"
        else:
            status_text += "🐟 **Fish Speech:** ⭕ Not loaded\n"
    else:
        status_text += "🐟 **Fish Speech:** ❌ Not available\n"

    # IndexTTS status
    if INDEXTTS_AVAILABLE:
        if MODEL_STATUS["indextts"]["loading"]:
            status_text += "🎯 **IndexTTS:** ⏳ Loading...\n"
        elif MODEL_STATUS["indextts"]["loaded"]:
            status_text += "🎯 **IndexTTS:** ✅ Loaded\n"
        else:
            if INDEXTTS_MODELS_AVAILABLE:
                status_text += "🎯 **IndexTTS:** ⭕ Not loaded (Models ready)\n"
            else:
                status_text += "🎯 **IndexTTS:** ⭕ Not loaded (Models will auto-download)\n"
    else:
        status_text += "🎯 **IndexTTS:** ❌ Not available\n"

    # F5-TTS status
    if F5_TTS_AVAILABLE:
        if MODEL_STATUS["f5_tts"]["loading"]:
            status_text += "🎵 **F5-TTS:** ⏳ Loading...\n"
        elif MODEL_STATUS["f5_tts"]["loaded"]:
            handler = get_f5_tts_handler()
            model_info = handler.get_model_info()
            status_text += f"🎵 **F5-TTS:** ✅ Loaded ({model_info['model']})\n"
        else:
            status_text += "🎵 **F5-TTS:** ⭕ Not loaded\n"
    else:
        status_text += "🎵 **F5-TTS:** ❌ Not available\n"

    # Higgs Audio status
    if HIGGS_AUDIO_AVAILABLE:
        if MODEL_STATUS["higgs_audio"]["loading"]:
            status_text += "🎙️ **Higgs Audio:** ⏳ Loading...\n"
        elif MODEL_STATUS["higgs_audio"]["loaded"]:
            status_text += "🎙️ **Higgs Audio:** ✅ Loaded\n"
        else:
            status_text += "🎙️ **Higgs Audio:** ⭕ Not loaded\n"
    else:
        status_text += "🎙️ **Higgs Audio:** ❌ Not available\n"

    # VoxCPM status
    if VOXCPM_AVAILABLE:
        if MODEL_STATUS.get("voxcpm", {}).get("loading", False):
            status_text += "🎤 **VoxCPM:** ⏳ Loading...\n"
        elif MODEL_STATUS.get("voxcpm", {}).get("loaded", False):
            status_text += "🎤 **VoxCPM:** ✅ Loaded\n"
        else:
            status_text += "🎤 **VoxCPM:** ⭕ Not loaded\n"
    else:
        status_text += "🎤 **VoxCPM:** ❌ Not available\n"

    # KittenTTS status
    if KITTEN_TTS_AVAILABLE:
        if MODEL_STATUS["kitten_tts"]["loading"]:
            status_text += "🐱 **KittenTTS:** ⏳ Loading...\n"
        elif MODEL_STATUS["kitten_tts"]["loaded"]:
            status_text += "🐱 **KittenTTS:** ✅ Loaded\n"
        else:
            status_text += "🐱 **KittenTTS:** ⭕ Not loaded\n"
    else:
        status_text += "🐱 **KittenTTS:** ❌ Not available\n"

    return status_text


# Don't initialize models at startup - they will be loaded on demand
print("🚀 TTS models ready for on-demand loading...")

# ===== KOKORO VOICE DEFINITIONS =====
KOKORO_CHOICES = {
    "🇺🇸 🚺 Heart ❤️": "af_heart",
    "🇺🇸 🚺 Bella 🔥": "af_bella",
    "🇺🇸 🚺 Nicole 🎧": "af_nicole",
    "🇺🇸 🚺 Aoede": "af_aoede",
    "🇺🇸 🚺 Kore": "af_kore",
    "🇺🇸 🚺 Sarah": "af_sarah",
    "🇺🇸 🚺 Nova": "af_nova",
    "🇺🇸 🚺 Sky": "af_sky",
    "🇺🇸 🚺 Alloy": "af_alloy",
    "🇺🇸 🚺 Jessica": "af_jessica",
    "🇺🇸 🚺 River": "af_river",
    "🇺🇸 🚹 Michael": "am_michael",
    "🇺🇸 🚹 Fenrir": "am_fenrir",
    "🇺🇸 🚹 Puck": "am_puck",
    "🇺🇸 🚹 Echo": "am_echo",
    "🇺🇸 🚹 Eric": "am_eric",
    "🇺🇸 🚹 Liam": "am_liam",
    "🇺🇸 🚹 Onyx": "am_onyx",
    "🇺🇸 🚹 Santa": "am_santa",
    "🇺🇸 🚹 Adam": "am_adam",
    "🇬🇧 🚺 Emma": "bf_emma",
    "🇬🇧 🚺 Isabella": "bf_isabella",
    "🇬🇧 🚺 Alice": "bf_alice",
    "🇬🇧 🚺 Lily": "bf_lily",
    "🇬🇧 🚹 George": "bm_george",
    "🇬🇧 🚹 Fable": "bm_fable",
    "🇬🇧 🚹 Lewis": "bm_lewis",
    "🇬🇧 🚹 Daniel": "bm_daniel",
    "PF 🚺 Dora": "pf_dora",
    "PM 🚹 Alex": "pm_alex",
    "PM 🚹 Santa": "pm_santa",
    "🇮🇹 🚺 Sara": "if_sara",
    "🇮🇹 🚹 Nicola": "im_nicola",
}


# ===== SHARED UTILITY FUNCTIONS =====
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_text_into_chunks(text: str, max_chunk_length: int = 300) -> list[str]:
    """Split text into chunks that respect sentence boundaries."""
    if len(text) <= max_chunk_length:
        return [text]

    sentences = re.split(r"[.!?]+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 2 > max_chunk_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if len(sentence) > max_chunk_length:
                    parts = re.split(r"[,;]+", sentence)
                    for part in parts:
                        part = part.strip()
                        if len(current_chunk) + len(part) + 2 > max_chunk_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += (", " if current_chunk else "") + part
                else:
                    current_chunk = sentence
        else:
            current_chunk += (". " if current_chunk else "") + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ===== AUDIO EFFECTS FUNCTIONS =====
def apply_reverb(audio, sr, room_size=0.3, damping=0.5, wet_level=0.3):
    """Apply reverb effect to audio."""
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio

    try:
        reverb_audio = audio.copy()
        delays = [0.01, 0.02, 0.03, 0.05, 0.08]
        gains = [0.6, 0.4, 0.3, 0.2, 0.15]

        for delay_time, gain in zip(delays, gains):
            delay_samples = int(sr * delay_time)
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain * (1 - damping)
                reverb_audio += delayed * wet_level

        return np.clip(reverb_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Reverb error: {e}")
        return audio


def apply_echo(audio, sr, delay=0.3, decay=0.5):
    """Apply echo effect to audio."""
    try:
        delay_samples = int(sr * delay)
        if delay_samples < len(audio):
            echo_audio = audio.copy()
            echo_audio[delay_samples:] += audio[:-delay_samples] * decay
            return np.clip(echo_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Echo error: {e}")
    return audio


def apply_pitch_shift(audio, sr, semitones):
    """Apply simple pitch shift."""
    try:
        if semitones == 0:
            return audio

        factor = 2 ** (semitones / 12.0)
        indices = np.arange(0, len(audio), factor)
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return audio


def apply_gain(audio, gain_db):
    """Apply gain/volume adjustment in dB."""
    try:
        if gain_db == 0:
            return audio

        # Convert dB to linear scale
        gain_linear = 10 ** (gain_db / 20.0)
        gained_audio = audio * gain_linear

        # Prevent clipping
        return np.clip(gained_audio, -1.0, 1.0)
    except Exception as e:
        print(f"Gain error: {e}")
        return audio


def apply_eq_filter(audio, sr, freq, gain_db, q_factor=1.0, filter_type="peak"):
    """Apply EQ filter at specific frequency."""
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio

    try:
        if gain_db == 0:
            return audio

        # Normalize frequency
        nyquist = sr / 2
        norm_freq = freq / nyquist

        # Clamp frequency to valid range
        norm_freq = np.clip(norm_freq, 0.01, 0.99)

        if filter_type == "lowpass":
            # Low-pass filter for bass boost
            coeffs = signal.butter(2, norm_freq, btype="low")
            if coeffs is None:
                return audio
            b, a = coeffs[0], coeffs[1]
        elif filter_type == "highpass":
            # High-pass filter for treble boost
            coeffs = signal.butter(2, norm_freq, btype="high")
            if coeffs is None:
                return audio
            b, a = coeffs[0], coeffs[1]
        elif filter_type == "peak":
            # Peaking EQ filter (more complex, simplified version)
            # This is a basic implementation - for production use, consider more sophisticated EQ
            coeffs = signal.butter(
                2, [max(0.01, norm_freq - 0.1), min(0.99, norm_freq + 0.1)], btype="band"
            )
            if coeffs is None:
                return audio
            b, a = coeffs[0], coeffs[1]

        # Apply filter
        filtered = filtfilt(b, a, audio)

        # Mix with original based on gain
        gain_linear = 10 ** (gain_db / 20.0)
        if gain_db > 0:
            # Boost: blend filtered signal
            mix_ratio = min(gain_linear - 1, 1.0)
            result = audio + filtered * mix_ratio
        else:
            # Cut: reduce filtered signal
            mix_ratio = 1 - (1 / gain_linear)
            result = audio - filtered * mix_ratio

        return np.clip(result, -1.0, 1.0)
    except Exception as e:
        print(f"EQ filter error: {e}")
        return audio


def apply_three_band_eq(audio, sr, bass_gain=0, mid_gain=0, treble_gain=0):
    """Apply 3-band EQ (bass, mid, treble)."""
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio

    try:
        result = audio.copy()

        # Bass: ~80-250 Hz
        if bass_gain != 0:
            result = apply_eq_filter(result, sr, 150, bass_gain, filter_type="lowpass")

        # Mid: ~250-4000 Hz
        if mid_gain != 0:
            result = apply_eq_filter(result, sr, 1000, mid_gain, filter_type="peak")

        # Treble: ~4000+ Hz
        if treble_gain != 0:
            result = apply_eq_filter(result, sr, 6000, treble_gain, filter_type="highpass")

        return result
    except Exception as e:
        print(f"3-band EQ error: {e}")
        return audio


def apply_audio_effects(audio, sr, effects_settings):
    """Apply selected audio effects to the generated audio."""
    if not effects_settings:
        return audio

    processed_audio = audio.copy()

    # Apply EQ first (before other effects)
    if effects_settings.get("enable_eq", False):
        processed_audio = apply_three_band_eq(
            processed_audio,
            sr,
            bass_gain=effects_settings.get("eq_bass", 0),
            mid_gain=effects_settings.get("eq_mid", 0),
            treble_gain=effects_settings.get("eq_treble", 0),
        )

    # Apply gain/volume adjustment
    if effects_settings.get("gain_db", 0) != 0:
        processed_audio = apply_gain(processed_audio, gain_db=effects_settings.get("gain_db", 0))

    if effects_settings.get("enable_reverb", False):
        processed_audio = apply_reverb(
            processed_audio,
            sr,
            room_size=effects_settings.get("reverb_room", 0.3),
            damping=effects_settings.get("reverb_damping", 0.5),
            wet_level=effects_settings.get("reverb_wet", 0.3),
        )

    if effects_settings.get("enable_echo", False):
        processed_audio = apply_echo(
            processed_audio,
            sr,
            delay=effects_settings.get("echo_delay", 0.3),
            decay=effects_settings.get("echo_decay", 0.5),
        )

    if effects_settings.get("enable_pitch", False):
        processed_audio = apply_pitch_shift(
            processed_audio, sr, semitones=effects_settings.get("pitch_semitones", 0)
        )

    return processed_audio


# ===== CHATTERBOX TTS FUNCTIONS =====
def generate_chatterbox_tts(
    text_input: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    chunk_size_input: int,
    effects_settings=None,
    audio_format: str = "wav",
    skip_file_saving: bool = False,
):
    """Generate TTS audio using ChatterboxTTS."""
    if not CHATTERBOX_AVAILABLE:
        return None, "❌ ChatterboxTTS not available - check installation"

    if not MODEL_STATUS["chatterbox"]["loaded"] or CHATTERBOX_MODEL is None:
        return None, "❌ ChatterboxTTS not loaded - please load the model first"

    try:
        if seed_num_input != 0:
            set_seed(int(seed_num_input))

        # Split text into chunks
        text_chunks = split_text_into_chunks(text_input, max_chunk_length=chunk_size_input)
        audio_chunks = []

        # Generate audio chunks with progress information
        print(f"🎙️ Generating ChatterboxTTS audio for {len(text_chunks)} chunk(s)...")
        if len(text_chunks) == 1:
            print("📊 Progress information will appear below during generation...")

        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                print(f"📝 Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")

            # Only suppress specific warnings, not all output (to allow tqdm progress bars)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                wav = CHATTERBOX_MODEL.generate(
                    chunk,
                    audio_prompt_path=audio_prompt_path_input,
                    exaggeration=exaggeration_input,
                    temperature=temperature_input,
                    cfg_weight=cfgw_input,
                )
                audio_chunks.append(wav.squeeze(0).numpy())

            if len(text_chunks) > 1:
                print(f"✅ Chunk {i+1}/{len(text_chunks)} completed")

        # Concatenate chunks
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            silence_samples = int(CHATTERBOX_MODEL.sr * 0.05)
            silence = np.zeros(silence_samples)

            concatenated_chunks = []
            for i, chunk in enumerate(audio_chunks):
                concatenated_chunks.append(chunk)
                if i < len(audio_chunks) - 1:
                    concatenated_chunks.append(silence)

            final_audio = np.concatenate(concatenated_chunks)

        # Apply effects
        if effects_settings:
            final_audio = apply_audio_effects(final_audio, CHATTERBOX_MODEL.sr, effects_settings)

        # Save audio file in specified format (skip if requested, e.g., for audiobook chunks)
        if skip_file_saving:
            status_message = "✅ Generated with ChatterboxTTS"
        else:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"chatterbox_output_{timestamp}"
                filepath, filename = save_audio_with_format(
                    final_audio, CHATTERBOX_MODEL.sr, audio_format, output_folder, filename_base
                )
                status_message = f"✅ Generated with ChatterboxTTS - Saved as: {filename}"
            except Exception as e:
                print(f"Warning: Could not save audio file: {e}")
                status_message = "✅ Generated with ChatterboxTTS (file saving failed)"

        return (CHATTERBOX_MODEL.sr, final_audio), status_message

    except Exception as e:
        return None, f"❌ ChatterboxTTS error: {str(e)}"


def generate_chatterbox_multilingual_tts(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str,
    exaggeration_input: float,
    temperature_input: float,
    seed_num_input: int,
    cfgw_input: float,
    repetition_penalty_input: float,
    min_p_input: float,
    top_p_input: float,
    chunk_size_input: int,
    effects_settings=None,
    audio_format: str = "wav",
    skip_file_saving: bool = False,
):
    """Generate TTS audio using ChatterboxMultilingualTTS."""
    if not CHATTERBOX_MULTILINGUAL_AVAILABLE:
        return None, "❌ ChatterboxMultilingualTTS not available - check installation"

    if (
        not MODEL_STATUS["chatterbox_multilingual"]["loaded"]
        or CHATTERBOX_MULTILINGUAL_MODEL is None
    ):
        return None, "❌ ChatterboxMultilingualTTS not loaded - please load the model first"

    try:
        print(
            f"🌍 Multilingual TTS - Language: {language_id}, Ref Audio: {audio_prompt_path_input}"
        )
        if seed_num_input != 0:
            set_seed(int(seed_num_input))

        # Split text into chunks
        text_chunks = split_text_into_chunks(text_input, max_chunk_length=chunk_size_input)
        audio_chunks = []

        # Generate audio chunks with progress information
        print(
            f"🎙️ Generating ChatterboxMultilingualTTS audio for {len(text_chunks)} chunk(s) in {language_id}..."
        )
        if len(text_chunks) == 1:
            print("📊 Progress information will appear below during generation...")

        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                print(f"📝 Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                wav = CHATTERBOX_MULTILINGUAL_MODEL.generate(
                    chunk,
                    language_id=language_id,
                    audio_prompt_path=audio_prompt_path_input,
                    exaggeration=exaggeration_input,
                    temperature=temperature_input,
                    cfg_weight=cfgw_input,
                    repetition_penalty=repetition_penalty_input,
                    min_p=min_p_input,
                    top_p=top_p_input,
                )
                audio_chunks.append(wav.squeeze(0).numpy())

            if len(text_chunks) > 1:
                print(f"✅ Chunk {i+1}/{len(text_chunks)} completed")

        # Concatenate chunks
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            silence_samples = int(CHATTERBOX_MULTILINGUAL_MODEL.sr * 0.05)
            silence = np.zeros(silence_samples)

            concatenated_chunks = []
            for i, chunk in enumerate(audio_chunks):
                concatenated_chunks.append(chunk)
                if i < len(audio_chunks) - 1:
                    concatenated_chunks.append(silence)

            final_audio = np.concatenate(concatenated_chunks)

        # Apply effects
        if effects_settings:
            final_audio = apply_audio_effects(
                final_audio, CHATTERBOX_MULTILINGUAL_MODEL.sr, effects_settings
            )

        # Save audio file in specified format (skip if requested, e.g., for audiobook chunks)
        if skip_file_saving:
            status_message = f"✅ Generated with ChatterboxMultilingualTTS ({language_id})"
        else:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"chatterbox_mtl_{language_id}_{timestamp}"
                filepath, filename = save_audio_with_format(
                    final_audio,
                    CHATTERBOX_MULTILINGUAL_MODEL.sr,
                    audio_format,
                    output_folder,
                    filename_base,
                )
                status_message = f"✅ Generated with ChatterboxMultilingualTTS ({language_id}) - Saved as: {filename}"
            except Exception as e:
                print(f"Warning: Could not save audio file: {e}")
                status_message = f"✅ Generated with ChatterboxMultilingualTTS ({language_id}) (file saving failed)"

        return (CHATTERBOX_MULTILINGUAL_MODEL.sr, final_audio), status_message

    except Exception as e:
        return None, f"❌ ChatterboxMultilingualTTS error: {str(e)}"


# ===== FISH SPEECH TTS FUNCTIONS =====
def enhance_audio_clarity(audio_data, sample_rate, enhancement_strength=1.0):
    """Enhance audio clarity and reduce muffled sound for Fish Speech."""
    if not AUDIO_PROCESSING_AVAILABLE:
        return audio_data

    try:
        from scipy.signal import butter, filtfilt

        enhanced_audio = audio_data.copy()

        # 1. Apply more aggressive high-frequency emphasis to combat muffled sound
        nyquist = sample_rate / 2

        # Multiple frequency bands for better enhancement
        # High-mid boost (2-6 kHz) - critical for speech intelligibility
        high_mid_freq = min(2000, nyquist * 0.4)
        norm_freq = high_mid_freq / nyquist
        b, a = butter(2, norm_freq, btype="high")
        high_mid_content = filtfilt(b, a, audio_data)
        enhanced_audio += high_mid_content * (
            0.25 * enhancement_strength
        )  # Adjustable boost for speech clarity

        # High frequency boost (4-8 kHz) - for brightness and presence
        high_freq = min(4000, nyquist * 0.6)
        norm_freq = high_freq / nyquist
        b, a = butter(2, norm_freq, btype="high")
        high_freq_content = filtfilt(b, a, audio_data)
        enhanced_audio += high_freq_content * (
            0.35 * enhancement_strength
        )  # Adjustable boost for brightness

        # 2. Gentle de-emphasis in low-mids to reduce muddiness (200-800 Hz)
        if sample_rate >= 1600:  # Only if sample rate allows
            low_mid_low = max(200, nyquist * 0.02)
            low_mid_high = min(800, nyquist * 0.1)
            norm_low = low_mid_low / nyquist
            norm_high = low_mid_high / nyquist

            if norm_high > norm_low and norm_high < 0.95:
                b, a = butter(2, [norm_low, norm_high], btype="band")
                muddy_content = filtfilt(b, a, audio_data)
                enhanced_audio -= muddy_content * 0.15  # Reduce muddiness

        # 3. Improved multi-band compression for better dynamics
        # Gentle compression on the whole signal
        threshold_low = 0.15  # Lower threshold for more compression
        ratio_low = 2.5  # Gentler ratio

        threshold_high = 0.6  # Higher threshold for peak limiting
        ratio_high = 6.0  # Stronger ratio for peaks

        abs_audio = np.abs(enhanced_audio)
        sign = np.sign(enhanced_audio)

        # Two-stage compression
        # Stage 1: Gentle compression for overall dynamics
        compressed = np.where(
            abs_audio > threshold_low,
            threshold_low + (abs_audio - threshold_low) / ratio_low,
            abs_audio,
        )

        # Stage 2: Peak limiting for loud parts
        compressed = np.where(
            compressed > threshold_high,
            threshold_high + (compressed - threshold_high) / ratio_high,
            compressed,
        )

        enhanced_audio = sign * compressed

        # 4. Final normalization with soft knee limiting
        peak = np.max(np.abs(enhanced_audio))
        if peak > 0.85:  # Start limiting earlier
            # Soft knee limiting
            target_peak = 0.85
            enhanced_audio = enhanced_audio * (target_peak / peak)

            # Apply soft saturation to remaining peaks
            enhanced_audio = np.tanh(enhanced_audio * 1.2) * 0.85

        return enhanced_audio

    except Exception as e:
        print(f"Audio enhancement error: {e}")
        return audio_data


def enhance_audio_clarity_minimal(audio_data, sample_rate, enhancement_strength=1.0):
    """Minimal audio enhancement that preserves Fish Speech's natural character."""
    if not AUDIO_PROCESSING_AVAILABLE or enhancement_strength <= 0:
        return audio_data

    try:
        from scipy.signal import butter, filtfilt

        # Very gentle high-frequency presence boost only (preserve Fish Speech quality)
        enhanced_audio = audio_data.copy()

        # Only apply subtle high-frequency emphasis if requested
        if enhancement_strength > 0.5:
            nyquist = sample_rate / 2

            # Gentle presence boost around 3-5kHz (speech clarity frequencies)
            presence_freq = min(3500, nyquist * 0.3)
            norm_freq = presence_freq / nyquist

            if norm_freq < 0.95:
                b, a = butter(1, norm_freq, btype="high")  # Very gentle 1st order
                presence_content = filtfilt(b, a, audio_data)
                # Much more subtle enhancement
                boost_amount = 0.1 * enhancement_strength  # Max 10% boost
                enhanced_audio += presence_content * boost_amount

        # Gentle soft limiting to prevent any artifacts
        peak = np.max(np.abs(enhanced_audio))
        if peak > 0.98:
            enhanced_audio = enhanced_audio * (0.98 / peak)

        return enhanced_audio

    except Exception as e:
        print(f"Minimal enhancement error: {e}")
        return audio_data


def normalize_audio(audio_data, target_level=-3.0, prevent_clipping=True):
    """Normalize audio to prevent clipping and improve quality - Fish Speech optimized."""
    try:
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32767.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483647.0
            else:
                audio_data = audio_data.astype(np.float32)

        # Find RMS level for better normalization (more perceptually accurate than peak)
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))

        if peak == 0 or rms == 0:
            return audio_data

        # Use RMS-based normalization for more natural sound
        target_linear = 10 ** (target_level / 20.0)

        # Calculate normalization based on RMS but limited by peak
        rms_factor = target_linear / rms
        peak_factor = 0.9 / peak  # Keep peaks under 0.9

        # Use the smaller factor to prevent clipping
        normalization_factor = min(rms_factor, peak_factor)

        # Apply normalization
        normalized_audio = audio_data * normalization_factor

        # Advanced soft limiting for Fish Speech
        if prevent_clipping:
            # Multi-stage soft limiting
            # Stage 1: Soft knee starting at 0.8
            threshold1 = 0.8
            knee_width = 0.1

            abs_audio = np.abs(normalized_audio)
            sign = np.sign(normalized_audio)

            # Soft knee compression
            knee_ratio = 3.0
            compressed = np.where(
                abs_audio > threshold1,
                threshold1 + (abs_audio - threshold1) / knee_ratio,
                abs_audio,
            )

            # Stage 2: Hard limiting at 0.95 with smooth tanh saturation
            threshold2 = 0.95
            final_audio = np.where(
                compressed > threshold2,
                np.tanh((compressed - threshold2) * 2) * 0.05 + threshold2,
                compressed,
            )

            normalized_audio = sign * final_audio

        # Final safety clip (should rarely be needed now)
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

        return normalized_audio

    except Exception as e:
        print(f"Normalization error: {e}")
        return np.clip(audio_data, -1.0, 1.0)


def generate_fish_speech_tts(
    text_input: str,
    fish_ref_audio: str = None,
    fish_ref_text: str = None,
    fish_temperature: float = 0.8,
    fish_top_p: float = 0.8,
    fish_repetition_penalty: float = 1.1,
    fish_max_tokens: int = 1024,
    fish_seed: int = None,
    effects_settings=None,
    audio_format: str = "wav",
    skip_file_saving: bool = False,
):
    """Generate TTS audio using Fish Speech - Proper implementation with chunking support."""
    if not FISH_SPEECH_AVAILABLE:
        return None, "❌ Fish Speech not available - check installation"

    if not MODEL_STATUS["fish_speech"]["loaded"] or FISH_SPEECH_ENGINE is None:
        return None, "❌ Fish Speech not loaded - please load the model first"

    try:
        from fish_speech.text.spliter import split_text

        cleaned_text = _sanitize_fish_speech_text(text_input)
        if not cleaned_text:
            return None, "❌ Fish Speech input is empty after text cleanup"

        # Prepare reference audio if provided
        references = []
        if fish_ref_audio and os.path.exists(fish_ref_audio):
            ref_audio_bytes = audio_to_bytes(fish_ref_audio)
            ref_text = fish_ref_text or ""  # Use provided text or empty string
            references.append(ServeReferenceAudio(audio=ref_audio_bytes, text=ref_text))

        # Split text into appropriate chunks using Fish Speech's own text splitter
        # This is crucial for handling long texts properly
        chunk_length = 200  # Fish Speech default chunk length for text splitting
        text_chunks = split_text(cleaned_text, chunk_length)

        if not text_chunks:
            return None, "❌ No valid text chunks generated"

        print(f"Fish Speech - Processing {len(text_chunks)} text chunks")
        for i, chunk in enumerate(text_chunks):
            print(f"  Chunk {i+1}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")

        safe_max_tokens = max(128, min(int(fish_max_tokens or 1024), 1024))
        safe_top_p = float(np.clip(fish_top_p, 0.1, 0.98))
        safe_repetition_penalty = float(np.clip(fish_repetition_penalty, 1.0, 2.0))
        safe_temperature = float(np.clip(fish_temperature, 0.1, 1.5))

        all_audio_segments = []

        # IMPORTANT: Generate a consistent seed for all chunks if no seed provided
        # This ensures voice consistency across chunks when no reference audio is used
        if fish_seed is None and not references:
            # Generate a random seed for this session to maintain consistency
            import time

            fish_seed = int(time.time()) % 1000000
            print(f"Fish Speech - Using consistent seed {fish_seed} for voice consistency")

        # If we have a reference audio from the first chunk, use it for subsequent chunks
        # This helps maintain voice consistency
        chunk_references = references.copy()

        # Process each chunk separately
        for i, chunk_text in enumerate(text_chunks):
            print(f"Fish Speech - Processing chunk {i+1}/{len(text_chunks)}")

            # Create TTS request for this chunk
            request = ServeTTSRequest(
                text=chunk_text,
                references=chunk_references,  # Use accumulated references
                reference_id=None,
                format="wav",
                max_new_tokens=safe_max_tokens,
                chunk_length=chunk_length,  # Internal chunking within Fish Speech
                top_p=safe_top_p,
                repetition_penalty=safe_repetition_penalty,
                temperature=safe_temperature,
                streaming=False,
                use_memory_cache="off",
                seed=fish_seed,  # Use consistent seed across all chunks
                normalize=False,
            )

            # Generate audio for this chunk
            results = list(FISH_SPEECH_ENGINE.inference(request))

            # Find the final result for this chunk
            chunk_final_result = None
            for result in results:
                if result.code == "final":
                    chunk_final_result = result
                    break
                elif result.code == "error":
                    return None, f"❌ Fish Speech error in chunk {i+1}: {str(result.error)}"

            if chunk_final_result is None or chunk_final_result.error is not None:
                error_msg = (
                    str(chunk_final_result.error)
                    if chunk_final_result
                    else f"No audio generated for chunk {i+1}"
                )
                return None, f"❌ Fish Speech error: {error_msg}"

            # Extract audio data for this chunk
            sample_rate, chunk_audio_data = chunk_final_result.audio

            # Convert to float32
            if chunk_audio_data.dtype != np.float32:
                chunk_audio_data = chunk_audio_data.astype(np.float32)

            all_audio_segments.append(chunk_audio_data)
            print(f"Fish Speech - Chunk {i+1} generated: {len(chunk_audio_data)} samples")

            # EXPERIMENTAL: If no reference was provided and this is the first chunk,
            # save a portion of it to use as reference for subsequent chunks
            # This helps maintain voice consistency
            if i == 0 and not references and len(text_chunks) > 1:
                try:
                    # Create a temporary file for the reference
                    import tempfile

                    temp_ref_fd, temp_ref_path = tempfile.mkstemp(suffix=".wav")

                    try:
                        # Save first 3 seconds as reference (or entire chunk if shorter)
                        ref_samples = min(len(chunk_audio_data), sample_rate * 3)
                        ref_audio = chunk_audio_data[:ref_samples]

                        # Close the file descriptor before writing
                        os.close(temp_ref_fd)

                        # Write the audio data
                        write(temp_ref_path, sample_rate, (ref_audio * 32767).astype(np.int16))

                        # Create reference for next chunks
                        ref_audio_bytes = audio_to_bytes(temp_ref_path)
                        chunk_references = [
                            ServeReferenceAudio(audio=ref_audio_bytes, text=chunk_text[:100])
                        ]
                        print(f"Fish Speech - Using first chunk as reference for voice consistency")

                    finally:
                        # Clean up temp file - use try/except to handle Windows file locking
                        try:
                            if os.path.exists(temp_ref_path):
                                os.unlink(temp_ref_path)
                        except Exception:
                            # If we can't delete it immediately on Windows, it will be cleaned up later
                            pass

                except Exception as e:
                    print(f"Fish Speech - Could not create reference from first chunk: {e}")
                    # Continue without reference - will still use consistent seed

        # Concatenate all audio segments with small silence between chunks
        if len(all_audio_segments) == 1:
            final_audio = all_audio_segments[0]
        else:
            # Add small silence between chunks (100ms)
            silence_samples = int(sample_rate * 0.1)
            silence = np.zeros(silence_samples, dtype=np.float32)

            concatenated_segments = []
            for i, segment in enumerate(all_audio_segments):
                concatenated_segments.append(segment)
                if i < len(all_audio_segments) - 1:  # Don't add silence after last segment
                    concatenated_segments.append(silence)

            final_audio = np.concatenate(concatenated_segments)

        # Simple safety normalization only if audio is clipped
        peak = np.max(np.abs(final_audio))
        if peak > 1.0:
            final_audio = final_audio / peak
            print(f"Fish Speech - Normalized clipped audio (peak was {peak:.3f})")

        print(f"Fish Speech - Final audio: {len(final_audio)} samples, peak: {peak:.3f}")

        # Apply user-requested effects
        if effects_settings:
            final_audio = apply_audio_effects(final_audio, sample_rate, effects_settings)

        # Save audio file in specified format (skip if requested, e.g., for audiobook chunks)
        if skip_file_saving:
            status_message = f"✅ Generated with Fish Speech ({len(text_chunks)} chunks processed)"
        else:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"fish_speech_output_{timestamp}"
                filepath, filename = save_audio_with_format(
                    final_audio, sample_rate, audio_format, output_folder, filename_base
                )
                status_message = f"✅ Generated with Fish Speech ({len(text_chunks)} chunks processed) - Saved as: {filename}"
            except Exception as e:
                print(f"Warning: Could not save audio file: {e}")
                status_message = f"✅ Generated with Fish Speech ({len(text_chunks)} chunks processed) (file saving failed)"

        return (sample_rate, final_audio), status_message

    except Exception as e:
        import traceback

        traceback.print_exc()
        message = str(e)
        if "device-side assert" in message or "index out of bounds" in message:
            return None, (
                "❌ Fish Speech CUDA index error. Try narration text without style tags "
                "(for example remove '(calm)'/'(break)') and reload Fish Speech before retrying."
            )
        return None, f"❌ Fish Speech error: {str(e)}"


# ===== KOKORO TTS FUNCTIONS =====
def get_custom_voices():
    """Get custom voices from the custom_voices folder."""
    custom_voices = {}
    custom_voices_folder = os.path.join(os.getcwd(), "custom_voices")
    if os.path.exists(custom_voices_folder):
        for file in os.listdir(custom_voices_folder):
            file_path = os.path.join(custom_voices_folder, file)
            if file.endswith(".pt") and os.path.isfile(file_path):
                voice_id = os.path.splitext(file)[0]
                custom_voices[f"👤 Custom: {voice_id}"] = f"custom_{voice_id}"
    return custom_voices


def upload_custom_voice(files, voice_name):
    """Upload a custom voice file to the custom_voices folder."""
    if not voice_name or not voice_name.strip():
        return "Please provide a name for your custom voice."

    # Sanitize voice name (remove spaces and special characters)
    voice_name = "".join(c for c in voice_name if c.isalnum() or c == "_")

    if not voice_name:
        return "Invalid voice name. Please use alphanumeric characters."

    # Check if any files were uploaded
    if not files:
        return "Please upload a .pt voice file."

    # In Gradio, the file object structure depends on the file_count parameter
    # For file_count="single", files is the file path as a string
    file_path = files

    # Check if the uploaded file is a .pt file
    if not file_path.endswith(".pt"):
        return "Please upload a valid .pt voice file."

    # Copy the file to the custom_voices folder with the new name
    target_file = os.path.join(custom_voices_folder, f"{voice_name}.pt")

    # If file already exists, remove it
    if os.path.exists(target_file):
        os.remove(target_file)

    # Copy the uploaded file
    shutil.copy(file_path, target_file)

    # Try to load the voice to verify it works
    voice_id = f"custom_{voice_name}"

    try:
        # Load the .pt file directly
        voice_pack = torch.load(target_file, weights_only=True)

        # Verify that the voice pack is usable with the model
        # Check if it's a tensor or a list/tuple of tensors
        if not isinstance(voice_pack, (torch.Tensor, list, tuple)):
            raise ValueError(
                "The voice file is not in the expected format (should be a tensor or list of tensors)"
            )

        # If it's a list or tuple, check that it contains tensors
        if isinstance(voice_pack, (list, tuple)) and (
            len(voice_pack) == 0 or not isinstance(voice_pack[0], torch.Tensor)
        ):
            raise ValueError("The voice file does not contain valid tensor data")

        loaded_voices[voice_id] = voice_pack
        return f"Custom voice '{voice_name}' uploaded and loaded successfully!"
    except Exception as e:
        # If loading fails, remove the file
        if os.path.exists(target_file):
            os.remove(target_file)
        return f"Error loading custom voice: {str(e)}"


def upload_and_refresh(files, voice_name):
    """Handle custom voice upload and refresh lists."""
    result = upload_custom_voice(files, voice_name)

    # If upload was successful, clear the input fields and update voice choices
    if "successfully" in result:
        updated_choices = update_kokoro_voice_choices()
        new_choices = [(k, v) for k, v in updated_choices.items()]
        # Return updates for main voice selector and all 5 conversation mode voice selectors
        return (
            result,
            get_custom_voice_list(),
            "",
            None,
            gr.update(choices=new_choices),  # Main kokoro_voice
            gr.update(choices=new_choices),  # speaker_1_kokoro_voice
            gr.update(choices=new_choices),  # speaker_2_kokoro_voice
            gr.update(choices=new_choices),  # speaker_3_kokoro_voice
            gr.update(choices=new_choices),  # speaker_4_kokoro_voice
            gr.update(choices=new_choices),
        )  # speaker_5_kokoro_voice
    else:
        # Return no updates for voice selectors on error
        return (
            result,
            get_custom_voice_list(),
            voice_name,
            files,
            gr.update(),  # Main kokoro_voice
            gr.update(),  # speaker_1_kokoro_voice
            gr.update(),  # speaker_2_kokoro_voice
            gr.update(),  # speaker_3_kokoro_voice
            gr.update(),  # speaker_4_kokoro_voice
            gr.update(),
        )  # speaker_5_kokoro_voice


def get_custom_voice_list():
    """Get the list of custom voices for the dataframe."""
    # Load any manually added custom voices first
    load_manual_custom_voices()

    custom_voices = get_custom_voices()
    if not custom_voices:
        return [["No custom voices found", "N/A"]]
    return [[name.replace("👤 Custom: ", ""), "Loaded"] for name in custom_voices.keys()]


def load_manual_custom_voices():
    """Load custom voices that were manually added to the custom_voices folder."""
    if not os.path.exists(custom_voices_folder):
        return

    custom_voices = get_custom_voices()
    for voice_name, voice_id in custom_voices.items():
        # Check if this voice is already loaded
        if voice_id not in loaded_voices:
            try:
                # Extract the actual filename from the voice_id
                voice_filename = voice_id[7:]  # Remove "custom_" prefix (7 characters)
                voice_file = f"{voice_filename}.pt"
                voice_path = os.path.join(custom_voices_folder, voice_file)

                if os.path.exists(voice_path):
                    # Load the .pt file directly
                    voice_pack = torch.load(voice_path, weights_only=True)

                    # Verify that the voice pack is usable
                    if isinstance(voice_pack, (torch.Tensor, list, tuple)):
                        loaded_voices[voice_id] = voice_pack
                        print(f"✅ Loaded manually added custom voice: {voice_name}")
                    else:
                        print(f"⚠️ Invalid voice format for {voice_name}")
                else:
                    print(f"⚠️ Voice file not found: {voice_path}")
            except Exception as e:
                print(f"❌ Error loading custom voice {voice_name}: {str(e)}")


def refresh_kokoro_voice_list():
    """Refresh the Kokoro voice list to include new custom voices."""
    # Load any manually added custom voices first
    load_manual_custom_voices()

    updated_choices = update_kokoro_voice_choices()
    new_choices = [(k, v) for k, v in updated_choices.items()]
    return gr.update(choices=new_choices)


def refresh_all_kokoro_voices():
    """Refresh all Kokoro voice selectors including conversation mode."""
    # Load any manually added custom voices first
    load_manual_custom_voices()

    updated_choices = update_kokoro_voice_choices()
    new_choices = [(k, v) for k, v in updated_choices.items()]
    # Return 5 identical updates for the 5 conversation mode voice selectors
    return [gr.update(choices=new_choices) for _ in range(5)]


def update_kokoro_voice_choices():
    """Update choices with custom voices."""
    updated_choices = KOKORO_CHOICES.copy()
    custom_voices = get_custom_voices()
    updated_choices.update(custom_voices)
    return updated_choices


def preload_kokoro_voices():
    """Preload Kokoro voices."""
    if not KOKORO_AVAILABLE or not MODEL_STATUS["kokoro"]["loaded"]:
        return

    print("Preloading Kokoro voices...")
    for voice_name, voice_id in KOKORO_CHOICES.items():
        try:
            pipeline = KOKORO_PIPELINES[voice_id[0]]
            voice_pack = pipeline.load_voice(voice_id)
            loaded_voices[voice_id] = voice_pack
            print(f"Loaded: {voice_name}")
        except Exception as e:
            print(f"Error loading {voice_name}: {e}")

    # Load custom voices (both uploaded and manually added)
    load_manual_custom_voices()

    print(f"All voices preloaded successfully. Total voices in cache: {len(loaded_voices)}")


def generate_kokoro_tts(
    text,
    voice="af_heart",
    speed=1,
    effects_settings=None,
    audio_format="wav",
    skip_file_saving=False,
):
    """Generate TTS audio using Kokoro TTS."""
    if not KOKORO_AVAILABLE:
        return None, "❌ Kokoro TTS not available - check installation"

    if not MODEL_STATUS["kokoro"]["loaded"] or not KOKORO_PIPELINES:
        return None, "❌ Kokoro TTS not loaded - please load the model first"

    try:
        # Remove hard character limit and implement chunking instead
        # Split text into chunks (using smaller chunks for Kokoro to maintain quality)
        text_chunks = split_text_into_chunks(
            text, max_chunk_length=800
        )  # Kokoro works well with smaller chunks
        audio_chunks = []

        # Get voice
        if voice.startswith("custom_"):
            voice_pack = loaded_voices.get(voice)
            if voice_pack is None:
                # Try to load the custom voice if it exists in the folder but isn't cached
                try:
                    # Extract the actual filename from the voice_id
                    # voice format: "custom_filename" -> we want "filename.pt"
                    voice_filename = voice[7:]  # Remove "custom_" prefix (7 characters)
                    voice_file = f"{voice_filename}.pt"
                    voice_path = os.path.join(custom_voices_folder, voice_file)

                    if os.path.exists(voice_path):
                        voice_pack = torch.load(voice_path, weights_only=True)

                        # Verify that the voice pack is usable
                        if isinstance(voice_pack, (torch.Tensor, list, tuple)):
                            loaded_voices[voice] = voice_pack
                            print(f"✅ Auto-loaded custom voice: {voice}")
                        else:
                            return None, f"❌ Invalid voice format for {voice}"
                    else:
                        return None, f"❌ Custom voice file not found: {voice_file}"
                except Exception as e:
                    return None, f"❌ Error loading custom voice {voice}: {str(e)}"

            if voice_pack is None:
                return None, f"❌ Custom voice {voice} not found"
            # Use American English pipeline for custom voices
            pipeline = KOKORO_PIPELINES["a"]
        else:
            voice_pack = loaded_voices.get(voice)
            if voice_pack is None:
                pipeline = KOKORO_PIPELINES[voice[0]]
                voice_pack = pipeline.load_voice(voice)
                loaded_voices[voice] = voice_pack
            else:
                # Get the correct pipeline for pre-trained voices
                pipeline = KOKORO_PIPELINES[voice[0]]

        # Generate audio for each chunk
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")

            # Use the pipeline as a callable (correct API)
            if voice.startswith("custom_"):
                audio_generator = pipeline(chunk, voice=voice_pack, speed=speed)
            else:
                audio_generator = pipeline(chunk, voice=voice, speed=speed)

            # Collect all audio chunks for this text chunk
            chunk_audio_parts = []
            for _, _, audio in audio_generator:
                # Convert tensor to numpy if needed
                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                chunk_audio_parts.append(audio)

            # Concatenate parts for this chunk
            if len(chunk_audio_parts) == 1:
                chunk_audio = chunk_audio_parts[0]
            else:
                chunk_audio = np.concatenate(chunk_audio_parts)

            audio_chunks.append(chunk_audio)

        # Concatenate all chunks with smooth transitions
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            # Add small silence between chunks for smooth transitions (shorter than ChatterboxTTS)
            silence_samples = int(24000 * 0.1)  # 100ms silence between chunks
            silence = np.zeros(silence_samples)

            concatenated_chunks = []
            for i, chunk in enumerate(audio_chunks):
                concatenated_chunks.append(chunk)
                if i < len(audio_chunks) - 1:  # Don't add silence after last chunk
                    concatenated_chunks.append(silence)

            final_audio = np.concatenate(concatenated_chunks)

        # Convert tensor to numpy if needed (redundant check, but safe)
        if hasattr(final_audio, "cpu"):
            final_audio = final_audio.cpu().numpy()

        # Apply effects
        if effects_settings:
            final_audio = apply_audio_effects(final_audio, 24000, effects_settings)

        # Save audio file in specified format (skip if requested, e.g., for audiobook chunks)
        if skip_file_saving:
            status_message = f"✅ Generated with Kokoro TTS ({len(text_chunks)} chunks)"
        else:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"kokoro_output_{timestamp}"
                filepath, filename = save_audio_with_format(
                    final_audio, 24000, audio_format, output_folder, filename_base
                )
                status_message = f"✅ Generated with Kokoro TTS ({len(text_chunks)} chunks) - Saved as: {filename}"
            except Exception as e:
                print(f"Warning: Could not save audio file: {e}")
                status_message = (
                    f"✅ Generated with Kokoro TTS ({len(text_chunks)} chunks) (file saving failed)"
                )

        return (24000, final_audio), status_message

    except Exception as e:
        return None, f"❌ Kokoro error: {str(e)}"


def generate_indextts_tts(
    text_input: str,
    indextts_ref_audio: Optional[str] = None,
    indextts_temperature: float = 0.8,
    indextts_seed: Optional[int] = None,
    effects_settings=None,
    audio_format: str = "wav",
    skip_file_saving: bool = False,
):
    """Generate speech using IndexTTS model."""

    if not INDEXTTS_AVAILABLE:
        return None, "❌ IndexTTS not available - check installation"

    if not MODEL_STATUS["indextts"]["loaded"] or INDEXTTS_MODEL is None:
        return None, "❌ IndexTTS model not loaded. Please load the model first."

    model = INDEXTTS_MODEL
    if model is None:
        return None, "❌ IndexTTS model not loaded. Please load the model first."

    if not text_input.strip():
        return None, "❌ Please enter text to synthesize"

    # Use sample audio as fallback if no reference audio provided
    if not indextts_ref_audio or not os.path.exists(indextts_ref_audio):
        sample_audio_path = os.path.join("sample", "Sample.wav")
        if os.path.exists(sample_audio_path):
            indextts_ref_audio = sample_audio_path
            print(f"🎯 Using default sample audio: {sample_audio_path}")
        else:
            return (
                None,
                "❌ Please provide a valid reference audio file or ensure sample/Sample.wav exists",
            )

    try:
        # Set seed for reproducibility
        if indextts_seed is not None and indextts_seed != 0:
            set_seed(indextts_seed)

        # Prepare temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_output_path = temp_file.name

        # Set generation parameters (IndexTTS uses different parameter names)
        generation_kwargs = {
            "max_text_tokens_per_sentence": 120,
        }

        # Generate speech using IndexTTS
        print(f"🎯 Generating speech with IndexTTS...")
        print(f"   📝 Text: {text_input.strip()[:100]}...")
        print(f"   🎵 Reference audio: {indextts_ref_audio}")
        print(f"   📁 Output path: {temp_output_path}")

        # Add timeout to prevent hanging (cross-platform)
        import threading
        import time

        generation_result: list[Optional[str]] = [None]  # Use list to store result from thread
        generation_error: list[Optional[str]] = [None]

        def generate_audio():
            try:
                with suppress_specific_warnings():
                    model.infer(
                        audio_prompt=indextts_ref_audio,
                        text=text_input.strip(),
                        output_path=temp_output_path,
                        **generation_kwargs,
                    )
                generation_result[0] = "success"
            except Exception as e:
                generation_error[0] = str(e)

        # Start generation in a separate thread
        thread = threading.Thread(target=generate_audio)
        thread.daemon = True
        thread.start()

        # Wait for completion with timeout
        thread.join(timeout=120)  # 2 minute timeout

        if thread.is_alive():
            return None, "❌ IndexTTS generation timed out after 2 minutes. The model may be stuck."

        if generation_error[0]:
            return None, f"❌ IndexTTS generation failed: {generation_error[0]}"

        if generation_result[0] != "success":
            return None, "❌ IndexTTS generation failed for unknown reason"

        print(f"✅ IndexTTS generation completed")

        # Load the generated audio
        if os.path.exists(temp_output_path):
            sample_rate, audio_data = wavfile.read(temp_output_path)

            # Clean up temporary file
            os.unlink(temp_output_path)

            # Convert to float32 for processing
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0

            # Apply audio effects if specified
            if effects_settings:
                audio_data = apply_audio_effects(audio_data, sample_rate, effects_settings)

            # Normalize audio
            audio_data = normalize_audio(audio_data)

            # Save file if not skipping
            if not skip_file_saving:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_base = f"indextts_output_{timestamp}"
                    output_folder = get_runtime_output_dir("outputs")

                    file_path, filename = save_audio_with_format(
                        audio_data, sample_rate, audio_format, output_folder, filename_base
                    )

                    # Calculate duration
                    duration = len(audio_data) / sample_rate

                    # Create enhanced status message
                    status_message = f"✅ IndexTTS synthesis completed\n"
                    status_message += f"📁 Saved as: {filename}\n"
                    status_message += f"⏱️ Duration: {duration:.2f}s\n"
                    status_message += f"📊 Sample Rate: {sample_rate}Hz"

                except Exception as save_error:
                    print(f"⚠️ Warning: Could not save IndexTTS audio file: {save_error}")
                    status_message = "✅ IndexTTS synthesis completed (file saving failed)"
            else:
                status_message = "✅ IndexTTS synthesis completed"

            return (sample_rate, audio_data), status_message
        else:
            return None, "❌ Failed to generate audio - output file not created"

    except Exception as e:
        error_msg = f"❌ IndexTTS generation failed: {str(e)}"
        print(error_msg)
        return None, error_msg


# ===== INDEXTTS2 FUNCTIONS =====
def generate_indextts2_unified_tts(
    text_input: str,
    indextts2_ref_audio: str = None,
    indextts2_emotion_mode: str = "audio_reference",
    indextts2_emotion_audio: str = None,
    indextts2_emotion_description: str = "",
    indextts2_emo_alpha: float = 1.0,
    indextts2_happy: float = 0.0,
    indextts2_angry: float = 0.0,
    indextts2_sad: float = 0.0,
    indextts2_afraid: float = 0.0,
    indextts2_disgusted: float = 0.0,
    indextts2_melancholic: float = 0.0,
    indextts2_surprised: float = 0.0,
    indextts2_calm: float = 1.0,
    indextts2_temperature: float = 0.8,
    indextts2_top_p: float = 0.9,
    indextts2_top_k: int = 50,
    indextts2_repetition_penalty: float = 1.1,
    indextts2_max_mel_tokens: int = 1500,
    indextts2_seed: int = None,
    indextts2_use_random: bool = True,
    effects_settings: dict = None,
    audio_format: str = "wav",
    skip_file_saving: bool = False,
):
    """Generate TTS audio using IndexTTS2 with advanced emotion control."""
    if not INDEXTTS2_AVAILABLE:
        return None, "❌ IndexTTS2 not available - check installation"

    if not MODEL_STATUS["indextts2"]["loaded"]:
        return None, "❌ IndexTTS2 model not loaded - please load the model first"

    if not text_input or not text_input.strip():
        return None, "❌ Please enter text to synthesize"

    if not indextts2_ref_audio:
        return None, "❌ Reference audio is required for IndexTTS2"

    try:
        print(f"🎯 Starting IndexTTS2 synthesis...")
        print(f"   Text: {text_input[:50]}...")
        print(f"   Emotion mode: {indextts2_emotion_mode}")

        # Prepare emotion vectors for vector control mode
        emotion_vectors = None
        if indextts2_emotion_mode == "vector_control":
            emotion_vectors = {
                "happy": indextts2_happy,
                "angry": indextts2_angry,
                "sad": indextts2_sad,
                "afraid": indextts2_afraid,
                "disgusted": indextts2_disgusted,
                "melancholic": indextts2_melancholic,
                "surprised": indextts2_surprised,
                "calm": indextts2_calm,
            }

        # Generate audio using IndexTTS2
        result = generate_indextts2_tts(
            text=text_input,
            reference_audio=indextts2_ref_audio,
            emotion_mode=indextts2_emotion_mode,
            emotion_audio=indextts2_emotion_audio,
            emotion_vectors=emotion_vectors,
            emotion_description=indextts2_emotion_description,
            temperature=indextts2_temperature,
            top_p=indextts2_top_p,
            top_k=indextts2_top_k,
            repetition_penalty=indextts2_repetition_penalty,
            max_mel_tokens=indextts2_max_mel_tokens,
            seed=indextts2_seed,
            use_random=indextts2_use_random,
            emo_alpha=indextts2_emo_alpha,
            effects_settings=effects_settings,
            audio_format=audio_format,
            skip_file_saving=skip_file_saving,
        )

        if result[0] is None:
            return None, result[1]

        audio_output, status_message = result

        if isinstance(audio_output, str):
            # File path returned
            return audio_output, status_message
        elif isinstance(audio_output, tuple):
            # (sample_rate, audio_data) tuple returned
            sample_rate, audio_data = audio_output

            # Apply effects if specified
            if effects_settings and not skip_file_saving:
                try:
                    audio_data = apply_audio_effects(audio_data, sample_rate, effects_settings)
                except Exception as e:
                    print(f"⚠️ Error applying effects: {e}")

            return (sample_rate, audio_data), status_message
        else:
            return None, "❌ Unexpected audio format returned"

    except Exception as e:
        error_msg = f"❌ IndexTTS2 generation failed: {str(e)}"
        print(error_msg)
        import traceback

        traceback.print_exc()
        return None, error_msg


# ===== F5-TTS FUNCTIONS =====
def generate_f5_tts(
    text_input: str,
    f5_ref_audio: Optional[str] = None,
    f5_ref_text: Optional[str] = None,
    f5_speed: float = 1.0,
    f5_cross_fade: float = 0.15,
    f5_remove_silence: bool = False,
    f5_seed: Optional[int] = None,
    effects_settings=None,
    audio_format: str = "wav",
    skip_file_saving: bool = False,
):
    """Generate TTS audio using F5-TTS."""
    if not F5_TTS_AVAILABLE:
        return None, "❌ F5-TTS not available - check installation"

    handler = get_f5_tts_handler()

    # Check if model is loaded
    if handler.model is None:
        return None, "❌ F5-TTS not loaded - please load a model first"

    print(
        f"F5-TTS generate called - Model loaded: {handler.model is not None}, Current model: {handler.current_model}"
    )

    try:
        print(f"🎵 Generating F5-TTS audio...")

        # Generate audio
        result = handler.generate_speech(
            text=text_input,
            ref_audio_path=f5_ref_audio,
            ref_text=f5_ref_text,
            speed=f5_speed,
            cross_fade_duration=f5_cross_fade,
            remove_silence=f5_remove_silence,
            seed=f5_seed if f5_seed != 0 else None,
        )

        if result[0] is None:
            return None, result[1]

        sample_rate, audio_data = result[0]

        # Apply effects if requested
        if effects_settings:
            audio_data = apply_audio_effects(audio_data, sample_rate, effects_settings)

        # Save audio file if not skipping
        if not skip_file_saving:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"f5_tts_output_{timestamp}"
            filepath, filename = save_audio_with_format(
                audio_data, sample_rate, audio_format, output_folder, filename_base
            )
            status_message = f"✅ Generated with F5-TTS - Saved as: {filename}"
        else:
            status_message = f"✅ Generated with F5-TTS"

        return (sample_rate, audio_data), status_message

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"❌ F5-TTS error: {str(e)}"


# ===== VOICE PRESET FUNCTIONS =====
def _preset_store_template():
    return {"version": 1, "presets": {}}


def _normalize_preset_name(name: str) -> str:
    if name is None:
        return ""
    return str(name).strip()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w\-. ]+", "_", str(value).strip())
    return cleaned[:80].strip() or "preset"


def migrate_legacy_presets_if_needed() -> bool:
    if os.path.exists(APP_STATE_PRESETS_FILE):
        return False

    if not os.path.exists(PRESETS_FILE):
        return False

    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as legacy_file:
            legacy_data = json.load(legacy_file)

        migrated = _preset_store_template()
        if isinstance(legacy_data, dict):
            for name, data in legacy_data.items():
                normalized = _normalize_preset_name(name)
                if not normalized:
                    continue

                audio_path = ""
                if isinstance(data, dict):
                    if isinstance(data.get("audio_path"), str):
                        audio_path = data.get("audio_path", "")
                    elif isinstance(data.get("settings"), dict):
                        audio_path = (
                            data.get("settings", {}).get("chatterbox_ref_audio")
                            or data.get("settings", {}).get("chatterbox_turbo_ref_audio")
                            or data.get("settings", {}).get("chatterbox_mtl_ref_audio")
                            or ""
                        )

                migrated["presets"][normalized] = {
                    "audio_path": audio_path,
                    "created": datetime.now().isoformat(),
                }

        save_voice_preset_store(migrated)
        print("✅ Migrated legacy voice_presets.json to app_state/presets.json")
        return True
    except Exception as error:
        print(f"⚠️ Failed to migrate legacy presets: {error}")
        return False


def load_voice_preset_store() -> dict:
    ensure_app_state_dirs()
    migrate_legacy_presets_if_needed()

    try:
        if os.path.exists(APP_STATE_PRESETS_FILE):
            with open(APP_STATE_PRESETS_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    data.setdefault("version", 1)
                    data.setdefault("presets", {})
                    if isinstance(data["presets"], dict):
                        return data
    except Exception as error:
        print(f"⚠️ Error loading app_state presets: {error}")

    return _preset_store_template()


def save_voice_preset_store(store: dict) -> bool:
    ensure_app_state_dirs()
    try:
        with open(APP_STATE_PRESETS_FILE, "w", encoding="utf-8") as file:
            json.dump(store, file, indent=2, ensure_ascii=False)
        return True
    except Exception as error:
        print(f"⚠️ Error saving app_state presets: {error}")
        return False


def load_voice_presets() -> dict:
    """Compatibility helper: returns the presets map only."""
    return load_voice_preset_store().get("presets", {})


def save_voice_presets(presets: dict) -> bool:
    """Compatibility helper: accepts a presets map and writes canonical store."""
    store = load_voice_preset_store()
    store["presets"] = presets if isinstance(presets, dict) else {}
    return save_voice_preset_store(store)


def get_voice_preset_choices() -> list:
    presets = load_voice_preset_store().get("presets", {})
    return [""] + sorted(presets.keys())


def get_voice_preset_dropdown_update(selected: str = ""):
    choices = get_voice_preset_choices()
    selected_value = selected if selected in choices else ""
    return gr.update(choices=choices, value=selected_value)


def load_speaker_profile_store() -> dict:
    """Load the conversation speaker profile store from disk."""
    default_store = {"version": 1, "profiles": {}}
    ensure_app_state_dirs()
    if not APP_STATE_SPEAKER_PROFILES_FILE.exists():
        return default_store

    try:
        with open(APP_STATE_SPEAKER_PROFILES_FILE, "r", encoding="utf-8") as file:
            store = json.load(file)
        if not isinstance(store, dict):
            return default_store
        if "profiles" not in store or not isinstance(store.get("profiles"), dict):
            store["profiles"] = {}
        store.setdefault("version", 1)
        return store
    except Exception as error:
        logger.error(f"Failed to load speaker profiles: {error}")
        return default_store


def save_speaker_profile_store(store: dict) -> bool:
    """Save the conversation speaker profile store to disk."""
    ensure_app_state_dirs()
    try:
        with open(APP_STATE_SPEAKER_PROFILES_FILE, "w", encoding="utf-8") as file:
            json.dump(store, file, indent=2, ensure_ascii=False)
        return True
    except Exception as error:
        logger.error(f"Failed to save speaker profiles: {error}")
        return False


def get_speaker_profile_choices() -> list[str]:
    """Return list of saved conversation speaker profile names."""
    store = load_speaker_profile_store()
    return sorted(store.get("profiles", {}).keys())


def _is_app_state_voice_path(audio_path: str) -> bool:
    if not isinstance(audio_path, str) or not audio_path.strip():
        return False

    abs_audio = os.path.abspath(audio_path)
    abs_voices_dir = os.path.abspath(APP_STATE_VOICES_DIR)
    voices_prefix = abs_voices_dir + os.sep
    return abs_audio == abs_voices_dir or abs_audio.startswith(voices_prefix)


def _normalize_speaker_profile_name(profile_name: str) -> str:
    return str(profile_name or "").strip()


def _speaker_profile_status_update(message: str):
    return gr.update(value=message, visible=True)


def _clone_conversation_speaker_settings(
    speaker_settings_state: dict | None,
    speakers: list[str] | None,
) -> dict[str, dict[str, Any]]:
    normalized_speakers = [
        str(speaker).strip() for speaker in speakers or [] if str(speaker).strip()
    ]
    updated_state: dict[str, dict[str, Any]] = {}

    existing_state = (
        copy.deepcopy(speaker_settings_state) if isinstance(speaker_settings_state, dict) else {}
    )

    if not normalized_speakers:
        for speaker_name, settings in existing_state.items():
            if isinstance(speaker_name, str) and isinstance(settings, dict):
                updated_state[speaker_name] = settings
        return updated_state

    default_settings = create_default_speaker_settings(normalized_speakers)
    for speaker_name in normalized_speakers:
        merged_settings = dict(default_settings.get(speaker_name, {}))
        existing_settings = existing_state.get(speaker_name, {})
        if isinstance(existing_settings, dict):
            merged_settings.update(existing_settings)
        updated_state[speaker_name] = merged_settings
    return updated_state


def update_conversation_speaker_setting(
    speaker_settings_state: dict | None,
    speakers: list[str] | None,
    speaker_slot_index: int,
    key: str,
    value: Any,
) -> dict[str, dict[str, Any]]:
    """Persist a conversation speaker-panel change into the shared state object."""
    normalized_speakers = [
        str(speaker).strip() for speaker in speakers or [] if str(speaker).strip()
    ]
    updated_state = _clone_conversation_speaker_settings(
        speaker_settings_state,
        normalized_speakers,
    )

    if not (0 <= speaker_slot_index < len(normalized_speakers)):
        return updated_state

    speaker_name = normalized_speakers[speaker_slot_index]
    updated_state.setdefault(speaker_name, {})[key] = value
    return updated_state


def update_conversation_tts_engine(
    selected_engine: str,
    speaker_settings_state: dict | None,
    speakers: list[str] | None,
) -> dict[str, dict[str, Any]]:
    """Mirror the active conversation engine into each saved speaker entry."""
    normalized_speakers = [
        str(speaker).strip() for speaker in speakers or [] if str(speaker).strip()
    ]
    updated_state = _clone_conversation_speaker_settings(
        speaker_settings_state,
        normalized_speakers,
    )
    normalized_engine = str(selected_engine or "").strip()
    if not normalized_engine:
        return updated_state

    for speaker_name in normalized_speakers:
        updated_state.setdefault(speaker_name, {})["tts_engine"] = normalized_engine
    return updated_state


def _speaker_profile_audio_referenced_elsewhere(
    store: dict,
    profile_name: str,
    audio_path: str,
) -> bool:
    if not isinstance(audio_path, str) or not audio_path.strip():
        return False

    target_audio = os.path.abspath(audio_path)
    for other_profile_name, profile_entry in store.get("profiles", {}).items():
        if other_profile_name == profile_name or not isinstance(profile_entry, dict):
            continue
        speakers = profile_entry.get("speakers", {})
        if not isinstance(speakers, dict):
            continue
        for speaker_settings in speakers.values():
            if not isinstance(speaker_settings, dict):
                continue
            other_audio = str(speaker_settings.get("ref_audio", "") or "").strip()
            if other_audio and os.path.abspath(other_audio) == target_audio:
                return True
    return False


def _build_speaker_profile_component_values(
    speaker_settings: dict[str, dict[str, Any]],
) -> tuple[list[str | None], list[str]]:
    audio_values: list[str | None] = []
    ref_text_values: list[str] = []

    for speaker_name in list(speaker_settings.keys())[:5]:
        settings = speaker_settings.get(speaker_name, {})
        if not isinstance(settings, dict):
            settings = {}

        audio_path = str(settings.get("ref_audio", "") or "").strip()
        audio_values.append(audio_path if audio_path and os.path.exists(audio_path) else None)
        ref_text_values.append(str(settings.get("fish_ref_text", "") or ""))

    while len(audio_values) < 5:
        audio_values.append(None)
    while len(ref_text_values) < 5:
        ref_text_values.append("")

    return audio_values, ref_text_values


def get_preset_audio_path(preset_name: str) -> str:
    normalized = _normalize_preset_name(preset_name)
    if not normalized:
        return ""

    presets = load_voice_preset_store().get("presets", {})
    preset_entry = presets.get(normalized, {})
    audio_path = preset_entry.get("audio_path", "") if isinstance(preset_entry, dict) else ""

    if isinstance(audio_path, str) and audio_path and os.path.exists(audio_path):
        return audio_path
    return ""


def _copy_preset_audio_into_app_state(preset_name: str, source_audio_path: str) -> str:
    ensure_app_state_dirs()
    source_path = os.path.abspath(source_audio_path)
    extension = os.path.splitext(source_path)[1] or ".wav"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_name(preset_name)}_{timestamp}{extension}"
    destination = os.path.join(APP_STATE_VOICES_DIR, filename)
    shutil.copy2(source_path, destination)
    return destination


def on_refresh_presets():
    choices = get_voice_preset_choices()
    return gr.update(choices=choices), "✅ Preset list refreshed"


def on_select_preset(selected_name: str):
    clear_audio_values = [None for _ in range(10)]
    normalized = _normalize_preset_name(selected_name)
    if not normalized:
        return "", "ℹ️ No preset selected", "Speakers Name", *clear_audio_values

    audio_path = get_preset_audio_path(normalized)
    speaker_name_guess = normalized.split("_", 1)[0] if "_" in normalized else normalized
    speaker_name_guess = speaker_name_guess.strip() or "Speakers Name"

    if audio_path:
        audio_values = [audio_path for _ in range(10)]
        return normalized, f"✅ Selected preset **{normalized}**", speaker_name_guess, *audio_values
    return (
        normalized,
        f"⚠️ Selected preset **{normalized}**, but audio path is missing",
        speaker_name_guess,
        *clear_audio_values,
    )


def on_save_preset(preset_name: str, audio_path: str, copy_into_app: bool):
    normalized_name = _normalize_preset_name(preset_name)
    if not normalized_name:
        return gr.update(), "❌ Please enter a preset name", gr.update(value="")

    if not audio_path or not os.path.exists(audio_path):
        return (
            gr.update(),
            "❌ Please upload a valid reference audio file",
            gr.update(value=normalized_name),
        )

    try:
        final_audio_path = os.path.abspath(audio_path)
        if copy_into_app:
            final_audio_path = _copy_preset_audio_into_app_state(normalized_name, audio_path)

        store = load_voice_preset_store()
        store["presets"][normalized_name] = {
            "audio_path": final_audio_path,
            "created": datetime.now().isoformat(),
        }
        if not save_voice_preset_store(store):
            return gr.update(), "❌ Failed to save preset", gr.update(value=normalized_name)

        choices = get_voice_preset_choices()
        return (
            gr.update(choices=choices, value=normalized_name),
            f"✅ Preset **{normalized_name}** saved",
            gr.update(value=normalized_name),
        )
    except Exception as error:
        return gr.update(), f"❌ Failed to save preset: {error}", gr.update(value=normalized_name)


def on_delete_preset(selected_name: str):
    normalized_name = _normalize_preset_name(selected_name)
    if not normalized_name:
        return gr.update(), "ℹ️ Select a preset to delete", gr.update(value="")

    store = load_voice_preset_store()
    presets = store.get("presets", {})
    if normalized_name not in presets:
        choices = get_voice_preset_choices()
        return (
            gr.update(choices=choices, value=""),
            f"⚠️ Preset **{normalized_name}** not found",
            gr.update(value=""),
        )

    preset_entry = presets.pop(normalized_name)
    deleted_audio = False
    audio_path = preset_entry.get("audio_path", "") if isinstance(preset_entry, dict) else ""
    try:
        if isinstance(audio_path, str) and audio_path:
            abs_audio = os.path.abspath(audio_path)
            abs_voices_dir = os.path.abspath(APP_STATE_VOICES_DIR)
            if abs_audio.startswith(abs_voices_dir) and os.path.exists(abs_audio):
                os.remove(abs_audio)
                deleted_audio = True
    except Exception as error:
        print(f"⚠️ Failed to remove preset audio file: {error}")

    save_voice_preset_store(store)
    choices = get_voice_preset_choices()
    if deleted_audio:
        message = f"✅ Preset **{normalized_name}** deleted (audio file removed)"
    else:
        message = f"✅ Preset **{normalized_name}** deleted"
    return gr.update(choices=choices, value=""), message, gr.update(value="")


def save_current_preset(preset_name, tts_engine, **settings):
    """Legacy API shim: store a preset using available reference audio from settings."""
    normalized_name = _normalize_preset_name(preset_name)
    if not normalized_name:
        return "❌ Please enter a preset name", gr.update()

    candidate_audio = (
        settings.get("chatterbox_ref_audio")
        or settings.get("chatterbox_mtl_ref_audio")
        or settings.get("chatterbox_turbo_ref_audio")
        or settings.get("fish_ref_audio")
        or settings.get("indextts_ref_audio")
        or settings.get("f5_ref_audio")
        or settings.get("qwen_ref_audio")
        or settings.get("voxcpm_ref_audio")
        or ""
    )

    if not candidate_audio:
        return "❌ No reference audio found in current settings", gr.update()

    dropdown_update, message, _ = on_save_preset(normalized_name, candidate_audio, True)
    return message, dropdown_update


def on_save_speaker_profile(profile_name: str, speaker_settings_state: dict):
    """Save the current conversation speaker settings as a reusable profile."""
    normalized_name = _normalize_speaker_profile_name(profile_name)
    if not normalized_name:
        return gr.update(), _speaker_profile_status_update("❌ Please enter a profile name")

    if not isinstance(speaker_settings_state, dict) or not speaker_settings_state:
        return (
            gr.update(),
            _speaker_profile_status_update(
                "❌ No conversation speaker settings are available to save"
            ),
        )

    try:
        saved_speakers: dict[str, dict[str, Any]] = {}
        for speaker_name, settings in copy.deepcopy(speaker_settings_state).items():
            if (
                not isinstance(speaker_name, str)
                or not speaker_name.strip()
                or not isinstance(settings, dict)
            ):
                continue

            normalized_speaker_name = speaker_name.strip()
            saved_settings = dict(settings)
            ref_audio = str(saved_settings.get("ref_audio", "") or "").strip()
            if ref_audio and os.path.exists(ref_audio) and not _is_app_state_voice_path(ref_audio):
                saved_settings["ref_audio"] = _copy_preset_audio_into_app_state(
                    normalized_speaker_name,
                    ref_audio,
                )
            elif ref_audio:
                saved_settings["ref_audio"] = os.path.abspath(ref_audio)
            else:
                saved_settings["ref_audio"] = ""

            saved_speakers[normalized_speaker_name] = saved_settings

        if not saved_speakers:
            return (
                gr.update(),
                _speaker_profile_status_update(
                    "❌ No valid speaker settings were found to save in this profile"
                ),
            )

        store = load_speaker_profile_store()
        store.setdefault("profiles", {})
        store["profiles"][normalized_name] = {
            "created": datetime.now().isoformat(),
            "speakers": saved_speakers,
        }

        if not save_speaker_profile_store(store):
            return (
                gr.update(),
                _speaker_profile_status_update("❌ Failed to save speaker profile"),
            )

        return (
            gr.update(choices=get_speaker_profile_choices(), value=normalized_name),
            _speaker_profile_status_update(
                f"✅ Speaker profile '{normalized_name}' saved ({len(saved_speakers)} speakers)"
            ),
        )
    except Exception as error:
        logger.error(f"Failed to save speaker profile '{normalized_name}': {error}")
        return (
            gr.update(),
            _speaker_profile_status_update(f"❌ Failed to save speaker profile: {error}"),
        )


def on_load_speaker_profile(
    profile_name: str,
    current_settings_state: dict,
    selected_speaker_index: int | None,
    speakers: list[str] | None,
):
    """Load a saved conversation speaker profile into the active speaker slot."""
    normalized_name = _normalize_speaker_profile_name(profile_name)
    normalized_speakers = [
        str(speaker).strip() for speaker in speakers or [] if str(speaker).strip()
    ]
    updated_state = _clone_conversation_speaker_settings(
        current_settings_state,
        normalized_speakers,
    )
    try:
        normalized_index = (
            int(selected_speaker_index) if selected_speaker_index not in (None, "") else None
        )
    except (TypeError, ValueError):
        normalized_index = None
    if normalized_index is None or not (0 <= normalized_index < len(normalized_speakers)):
        normalized_index = 0 if normalized_speakers else None

    if not normalized_name:
        return (
            updated_state,
            *[gr.update() for _ in range(10)],
            gr.update(value=""),
            _speaker_profile_status_update("ℹ️ Select a speaker profile to load"),
        )

    store = load_speaker_profile_store()
    profile_entry = store.get("profiles", {}).get(normalized_name)
    if not isinstance(profile_entry, dict):
        return (
            updated_state,
            *[gr.update() for _ in range(10)],
            gr.update(value=""),
            _speaker_profile_status_update(f"⚠️ Speaker profile '{normalized_name}' not found"),
        )

    saved_speakers = profile_entry.get("speakers", {})
    if not isinstance(saved_speakers, dict):
        saved_speakers = {}

    loaded_settings: dict[str, Any] = {}
    target_speaker_name = None
    if normalized_index is not None and normalized_index < len(normalized_speakers):
        target_speaker_name = normalized_speakers[normalized_index]
        target_saved_settings = saved_speakers.get(target_speaker_name)
        if isinstance(target_saved_settings, dict):
            loaded_settings = copy.deepcopy(target_saved_settings)

    if not loaded_settings:
        for saved_settings in saved_speakers.values():
            if isinstance(saved_settings, dict):
                loaded_settings = copy.deepcopy(saved_settings)
                break

    missing_audio_count = 0
    ref_audio = str(loaded_settings.get("ref_audio", "") or "").strip()
    if ref_audio and not os.path.exists(ref_audio):
        loaded_settings["ref_audio"] = ""
        missing_audio_count += 1

    if target_speaker_name is not None:
        updated_state.setdefault(target_speaker_name, {}).update(loaded_settings)
        updated_state[target_speaker_name]["selected_profile"] = normalized_name
    elif loaded_settings:
        normalized_profile_state = _clone_conversation_speaker_settings(
            saved_speakers,
            list(saved_speakers.keys()),
        )
        for speaker_name, speaker_settings in normalized_profile_state.items():
            if not isinstance(speaker_settings, dict):
                continue
            saved_audio = str(speaker_settings.get("ref_audio", "") or "").strip()
            if saved_audio and not os.path.exists(saved_audio):
                speaker_settings["ref_audio"] = ""
                missing_audio_count += 1
        updated_state = normalized_profile_state

    audio_values, ref_text_values = _build_speaker_profile_component_values(updated_state)

    status_message = f"✅ Loaded speaker profile '{normalized_name}'"
    if missing_audio_count:
        status_message += f" ({missing_audio_count} missing audio file(s) skipped)"

    return (
        updated_state,
        *audio_values,
        *ref_text_values,
        gr.update(value=normalized_name),
        _speaker_profile_status_update(status_message),
    )


def on_delete_speaker_profile(profile_name: str):
    """Delete a saved conversation speaker profile."""
    normalized_name = _normalize_speaker_profile_name(profile_name)
    if not normalized_name:
        return (
            gr.update(),
            _speaker_profile_status_update("ℹ️ Select a speaker profile to delete"),
        )

    store = load_speaker_profile_store()
    profiles = store.get("profiles", {})
    if normalized_name not in profiles:
        return (
            gr.update(choices=get_speaker_profile_choices(), value=None),
            _speaker_profile_status_update(f"⚠️ Speaker profile '{normalized_name}' not found"),
        )

    deleted_audio_count = 0
    profile_entry = profiles.pop(normalized_name)
    if isinstance(profile_entry, dict):
        speakers = profile_entry.get("speakers", {})
        if isinstance(speakers, dict):
            for speaker_settings in speakers.values():
                if not isinstance(speaker_settings, dict):
                    continue
                audio_path = str(speaker_settings.get("ref_audio", "") or "").strip()
                if not audio_path or not _is_app_state_voice_path(audio_path):
                    continue
                if _speaker_profile_audio_referenced_elsewhere(store, normalized_name, audio_path):
                    continue
                try:
                    abs_audio_path = os.path.abspath(audio_path)
                    if os.path.exists(abs_audio_path):
                        os.remove(abs_audio_path)
                        deleted_audio_count += 1
                except Exception as error:
                    logger.error(f"Failed to remove speaker profile audio '{audio_path}': {error}")

    if not save_speaker_profile_store(store):
        return (
            gr.update(),
            _speaker_profile_status_update("❌ Failed to delete speaker profile"),
        )

    status_message = f"✅ Speaker profile '{normalized_name}' deleted"
    if deleted_audio_count:
        status_message += f" ({deleted_audio_count} audio file(s) removed)"

    return (
        gr.update(choices=get_speaker_profile_choices(), value=None),
        _speaker_profile_status_update(status_message),
    )


# ===== EBOOK TO AUDIOBOOK FUNCTIONS =====
def analyze_ebook_file(file_path: str):
    """Analyze an uploaded eBook file and return information."""
    if not EBOOK_CONVERTER_AVAILABLE:
        return {
            "success": False,
            "error": "eBook converter not available. Please install required dependencies.",
            "chapters": [],
            "metadata": None,
        }

    if not file_path:
        return {"success": False, "error": "No file uploaded", "chapters": [], "metadata": None}

    try:
        info = analyze_ebook(file_path)
        return info
    except Exception as e:
        return {"success": False, "error": str(e), "chapters": [], "metadata": None}


def convert_ebook_to_audiobook(
    file_path: str,
    tts_engine: str,
    selected_chapters: list,
    max_chunk_length: int = 500,
    # Audio format parameter
    audio_format: str = "wav",
    # ChatterboxTTS parameters
    chatterbox_ref_audio: str = None,
    chatterbox_exaggeration: float = 0.5,
    chatterbox_temperature: float = 0.8,
    chatterbox_cfg_weight: float = 0.5,
    chatterbox_seed: int = 0,
    # Chatterbox Multilingual parameters
    chatterbox_mtl_ref_audio: str = None,
    chatterbox_mtl_language: str = "en",
    chatterbox_mtl_exaggeration: float = 0.5,
    chatterbox_mtl_temperature: float = 0.8,
    chatterbox_mtl_cfg_weight: float = 0.5,
    chatterbox_mtl_repetition_penalty: float = 2.0,
    chatterbox_mtl_min_p: float = 0.05,
    chatterbox_mtl_top_p: float = 1.0,
    chatterbox_mtl_seed: int = 0,
    # Chatterbox Turbo parameters
    chatterbox_turbo_ref_audio: str = None,
    chatterbox_turbo_exaggeration: float = 0.5,
    chatterbox_turbo_temperature: float = 0.8,
    chatterbox_turbo_cfg_weight: float = 0.5,
    chatterbox_turbo_repetition_penalty: float = 1.2,
    chatterbox_turbo_min_p: float = 0.05,
    chatterbox_turbo_top_p: float = 1.0,
    chatterbox_turbo_seed: int = 0,
    # Kokoro parameters
    kokoro_voice: str = "af_heart",
    kokoro_speed: float = 1.0,
    # Fish Speech parameters
    fish_ref_audio: str = None,
    fish_ref_text: str = None,
    fish_temperature: float = 0.8,
    fish_top_p: float = 0.8,
    fish_repetition_penalty: float = 1.1,
    fish_max_tokens: int = 1024,
    fish_seed: int = None,
    # IndexTTS parameters
    indextts_ref_audio: str = None,
    indextts_temperature: float = 0.8,
    indextts_seed: int = None,
    # IndexTTS2 parameters
    indextts2_ref_audio: str = None,
    indextts2_emotion_mode: str = "audio_reference",
    indextts2_emotion_audio: str = None,
    indextts2_emotion_description: str = "",
    indextts2_emo_alpha: float = 1.0,
    indextts2_happy: float = 0.0,
    indextts2_angry: float = 0.0,
    indextts2_sad: float = 0.0,
    indextts2_afraid: float = 0.0,
    indextts2_disgusted: float = 0.0,
    indextts2_melancholic: float = 0.0,
    indextts2_surprised: float = 0.0,
    indextts2_calm: float = 1.0,
    indextts2_temperature: float = 0.8,
    indextts2_top_p: float = 0.9,
    indextts2_top_k: int = 50,
    indextts2_repetition_penalty: float = 1.1,
    indextts2_max_mel_tokens: int = 1500,
    indextts2_seed: int = None,
    indextts2_use_random: bool = True,
    # F5-TTS parameters
    f5_ref_audio: str = None,
    f5_ref_text: str = None,
    f5_speed: float = 1.0,
    f5_cross_fade: float = 0.15,
    f5_remove_silence: bool = False,
    f5_seed: int = 0,
    # Higgs Audio parameters
    higgs_ref_audio: str = None,
    higgs_ref_text: str = None,
    higgs_voice_preset: str = "EMPTY",
    higgs_system_prompt: str = "",
    higgs_temperature: float = 1.0,
    higgs_top_p: float = 0.95,
    higgs_top_k: int = 50,
    higgs_max_tokens: int = 1024,
    higgs_ras_win_len: int = 7,
    higgs_ras_win_max_num_repeat: int = 2,
    # KittenTTS parameters
    kitten_voice: str = "expr-voice-2-f",
    # Qwen TTS parameters
    qwen_ref_audio: str = None,
    qwen_ref_text: str = "",
    qwen_language: str = "Auto",
    qwen_xvector_only: bool = False,
    qwen_clone_model_size: str = "1.7B",
    qwen_seed: int = -1,
    # Effects parameters
    gain_db: float = 0,
    enable_eq: bool = False,
    eq_bass: float = 0,
    eq_mid: float = 0,
    eq_treble: float = 0,
    enable_reverb: bool = False,
    reverb_room: float = 0.3,
    reverb_damping: float = 0.5,
    reverb_wet: float = 0.3,
    enable_echo: bool = False,
    echo_delay: float = 0.3,
    echo_decay: float = 0.5,
    enable_pitch: bool = False,
    pitch_semitones: float = 0,
    # Advanced eBook settings
    chunk_gap: float = 1.0,
    chapter_gap: float = 2.0,
):
    """Convert eBook to audiobook using selected TTS engine."""
    if not EBOOK_CONVERTER_AVAILABLE:
        return None, "❌ eBook converter not available"

    if not file_path:
        return None, "❌ No eBook file provided"

    try:
        # Convert eBook to text chunks with VoxCPM-specific optimization
        if tts_engine == "VoxCPM":
            # Use smaller chunks for VoxCPM to avoid badcase issues
            voxcpm_optimized_chunk_length = min(max_chunk_length, 350)
            print(f"🎤 Using VoxCPM-optimized chunk length: {voxcpm_optimized_chunk_length}")
            text_chunks, metadata = convert_ebook_to_text_chunks(
                file_path, voxcpm_optimized_chunk_length
            )
        else:
            text_chunks, metadata = convert_ebook_to_text_chunks(file_path, max_chunk_length)

        if not text_chunks:
            return None, "❌ No text content found in eBook"

        # Filter chunks based on selected chapters if specified
        if selected_chapters:
            # Convert selected chapter indices to set for faster lookup
            selected_indices = set(selected_chapters)
            text_chunks = [
                chunk for chunk in text_chunks if chunk["chapter_index"] in selected_indices
            ]

        if not text_chunks:
            return None, "❌ No chapters selected for conversion"

        # Prepare effects settings
        effects_settings = (
            {
                "gain_db": gain_db,
                "enable_eq": enable_eq,
                "eq_bass": eq_bass,
                "eq_mid": eq_mid,
                "eq_treble": eq_treble,
                "enable_reverb": enable_reverb,
                "reverb_room": reverb_room,
                "reverb_damping": reverb_damping,
                "reverb_wet": reverb_wet,
                "enable_echo": enable_echo,
                "echo_delay": echo_delay,
                "echo_decay": echo_decay,
                "enable_pitch": enable_pitch,
                "pitch_semitones": pitch_semitones,
            }
            if any([gain_db != 0, enable_eq, enable_reverb, enable_echo, enable_pitch])
            else None
        )

        # Generate audio for each chunk
        audio_segments = []
        total_chunks = len(text_chunks)

        # For Fish Speech without reference audio: generate a consistent seed for the entire audiobook
        audiobook_fish_seed = fish_seed
        if tts_engine == "Fish Speech" and audiobook_fish_seed is None and not fish_ref_audio:
            import time

            audiobook_fish_seed = int(time.time()) % 1000000
            print(
                f"🐟 Using consistent seed {audiobook_fish_seed} for entire audiobook voice consistency"
            )

        # For maintaining voice consistency across chunks in Fish Speech
        fish_chunk_reference_audio = fish_ref_audio
        fish_chunk_reference_text = fish_ref_text

        # For Qwen TTS: generate a consistent seed for the entire audiobook if seed is -1
        audiobook_qwen_seed = qwen_seed
        if tts_engine == "Qwen Voice Clone" and audiobook_qwen_seed == -1:
            import random

            audiobook_qwen_seed = random.randint(0, 2147483647)
            print(
                f"🎭 Using consistent seed {audiobook_qwen_seed} for entire audiobook voice consistency"
            )

        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{total_chunks}: {chunk['title']}")

            # Generate TTS for this chunk
            if tts_engine == "ChatterboxTTS":
                audio_result, status = generate_chatterbox_tts(
                    chunk["content"],
                    chatterbox_ref_audio,
                    chatterbox_exaggeration,
                    chatterbox_temperature,
                    chatterbox_seed,
                    chatterbox_cfg_weight,
                    max_chunk_length,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,  # Skip saving individual chunks
                )
            elif tts_engine == "Chatterbox Multilingual":
                print(
                    f"🌍 Using Chatterbox Multilingual with ref audio: {chatterbox_mtl_ref_audio}"
                )
                if not chatterbox_mtl_ref_audio:
                    print("⚠️ No reference audio provided - using default voice")
                audio_result, status = generate_chatterbox_multilingual_tts(
                    chunk["content"],
                    chatterbox_mtl_language,
                    chatterbox_mtl_ref_audio,
                    chatterbox_mtl_exaggeration,
                    chatterbox_mtl_temperature,
                    chatterbox_mtl_seed,
                    chatterbox_mtl_cfg_weight,
                    chatterbox_mtl_repetition_penalty,
                    chatterbox_mtl_min_p,
                    chatterbox_mtl_top_p,
                    max_chunk_length,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,
                )
            elif tts_engine == "Chatterbox Turbo":
                print(f"🚀 Using Chatterbox Turbo with ref audio: {chatterbox_turbo_ref_audio}")
                if not chatterbox_turbo_ref_audio:
                    print("⚠️ No reference audio provided - using default voice")
                audio_result, status = generate_chatterbox_turbo_tts(
                    chunk["content"],
                    chatterbox_turbo_ref_audio,
                    chatterbox_turbo_exaggeration,
                    chatterbox_turbo_temperature,
                    chatterbox_turbo_cfg_weight,
                    chatterbox_turbo_repetition_penalty,
                    chatterbox_turbo_min_p,
                    chatterbox_turbo_top_p,
                    chatterbox_turbo_seed,
                    max_chunk_length,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,
                )
            elif tts_engine == "Kokoro TTS":
                audio_result, status = generate_kokoro_tts(
                    chunk["content"],
                    kokoro_voice,
                    kokoro_speed,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,  # Skip saving individual chunks
                )
            elif tts_engine == "Fish Speech":
                # Use consistent seed and potentially updated reference for Fish Speech
                audio_result, status = generate_fish_speech_tts(
                    chunk["content"],
                    fish_chunk_reference_audio,
                    fish_chunk_reference_text,
                    fish_temperature,
                    fish_top_p,
                    fish_repetition_penalty,
                    fish_max_tokens,
                    audiobook_fish_seed,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,  # Skip saving individual chunks
                )
            elif tts_engine == "IndexTTS":
                audio_result, status = generate_indextts_tts(
                    chunk["content"],
                    indextts_ref_audio,
                    indextts_temperature,
                    indextts_seed,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,  # Skip saving individual chunks
                )
            elif tts_engine == "F5-TTS":
                audio_result, status = generate_f5_tts(
                    chunk["content"],
                    f5_ref_audio,
                    f5_ref_text,
                    f5_speed,
                    f5_cross_fade,
                    f5_remove_silence,
                    f5_seed,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,  # Skip saving individual chunks
                )
            elif tts_engine == "Higgs Audio":
                audio_result, status = generate_higgs_audio_tts(
                    chunk["content"],
                    higgs_ref_audio,
                    higgs_ref_text,
                    higgs_voice_preset,
                    higgs_system_prompt,
                    higgs_temperature,
                    higgs_top_p,
                    higgs_top_k,
                    higgs_max_tokens,
                    higgs_ras_win_len,
                    higgs_ras_win_max_num_repeat,
                    100,  # chunk_length
                    effects_settings,
                    "wav",
                    skip_file_saving=True,  # Skip saving individual chunks
                )
            elif tts_engine == "IndexTTS2":
                audio_result, status = generate_indextts2_unified_tts(
                    chunk["content"],
                    indextts2_ref_audio,
                    indextts2_emotion_mode,
                    indextts2_emotion_audio,
                    indextts2_emotion_description,
                    indextts2_emo_alpha,
                    indextts2_happy,
                    indextts2_angry,
                    indextts2_sad,
                    indextts2_afraid,
                    indextts2_disgusted,
                    indextts2_melancholic,
                    indextts2_surprised,
                    indextts2_calm,
                    indextts2_temperature,
                    indextts2_top_p,
                    indextts2_top_k,
                    indextts2_repetition_penalty,
                    indextts2_max_mel_tokens,
                    indextts2_seed,
                    indextts2_use_random,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,
                )
            elif tts_engine == "KittenTTS":
                # Use selected voice for eBook conversion
                audio_result, status = generate_kitten_tts(
                    chunk["content"], kitten_voice, effects_settings, "wav", skip_file_saving=True
                )
            elif tts_engine == "Qwen Voice Clone":
                # Use Qwen Voice Clone for eBook conversion
                audio_result, status = generate_qwen_voice_clone_tts(
                    chunk["content"],
                    qwen_ref_audio,
                    qwen_ref_text,
                    qwen_language,
                    qwen_xvector_only,
                    qwen_clone_model_size,
                    max_chunk_length,
                    0.0,  # chunk_gap handled by ebook converter
                    audiobook_qwen_seed,
                    effects_settings,
                    "wav",
                    skip_file_saving=True,
                )
            else:
                return None, f"❌ Invalid TTS engine: {tts_engine}"

            if audio_result is None:
                return None, f"❌ Failed to generate audio for chunk {i+1}: {status}"

            sample_rate, audio_data = audio_result
            audio_segments.append((sample_rate, audio_data, chunk["title"]))

            # For Fish Speech: Use first chunk as reference for subsequent chunks if no reference provided
            if (
                tts_engine == "Fish Speech"
                and i == 0
                and not fish_ref_audio
                and total_chunks > 1
                and len(audio_data) > 0
            ):
                try:
                    import tempfile

                    # Create a temporary reference from the first chunk
                    temp_ref_fd, temp_ref_path = tempfile.mkstemp(suffix=".wav")

                    try:
                        # Save first 5 seconds as reference (or entire chunk if shorter)
                        ref_samples = min(len(audio_data), sample_rate * 5)
                        ref_audio = audio_data[:ref_samples]

                        # Close the file descriptor before writing
                        os.close(temp_ref_fd)

                        # Write the audio data
                        write(temp_ref_path, sample_rate, (ref_audio * 32767).astype(np.int16))

                        # Update reference for next chunks
                        fish_chunk_reference_audio = temp_ref_path
                        fish_chunk_reference_text = chunk["content"][
                            :200
                        ]  # First 200 chars as reference text
                        print(f"🐟 Using first audiobook chunk as reference for voice consistency")

                    except Exception as e:
                        print(f"Warning: Could not create reference from first chunk: {e}")
                        # Try to clean up if possible
                        try:
                            if os.path.exists(temp_ref_path):
                                os.unlink(temp_ref_path)
                        except:
                            pass

                except Exception as e:
                    print(f"Warning: Could not setup chunk reference: {e}")

        # Concatenate all audio segments
        if not audio_segments:
            return None, "❌ No audio generated"

        # Clean up temporary reference file if it was created
        if (
            tts_engine == "Fish Speech"
            and fish_chunk_reference_audio != fish_ref_audio
            and fish_chunk_reference_audio
            and os.path.exists(fish_chunk_reference_audio)
        ):
            try:
                os.unlink(fish_chunk_reference_audio)
                print("🧹 Cleaned up temporary reference file")
            except Exception as e:
                print(f"Warning: Could not clean up temp reference: {e}")

        # Use the sample rate from the first segment
        final_sample_rate = audio_segments[0][0]

        # Create silence arrays for different gap types
        chunk_silence_samples = int(final_sample_rate * chunk_gap)
        chapter_silence_samples = int(final_sample_rate * chapter_gap)
        chunk_silence = np.zeros(chunk_silence_samples, dtype=np.float32)
        chapter_silence = np.zeros(chapter_silence_samples, dtype=np.float32)

        concatenated_audio = []
        current_chapter_index = None

        for i, (sr, audio_data, title) in enumerate(audio_segments):
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            concatenated_audio.append(audio_data)

            # Add appropriate silence between segments (except after the last one)
            if i < len(audio_segments) - 1:
                # Get the chapter index for current and next chunk
                current_chunk_chapter = text_chunks[i]["chapter_index"]
                next_chunk_chapter = text_chunks[i + 1]["chapter_index"]

                # Use chapter gap if moving to a new chapter, otherwise use chunk gap
                if current_chunk_chapter != next_chunk_chapter:
                    concatenated_audio.append(chapter_silence)
                else:
                    concatenated_audio.append(chunk_silence)

        final_audio = np.concatenate(concatenated_audio)

        # Save the complete audiobook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        book_title = metadata["title"].replace(" ", "_")
        filename_base = f"audiobook_{book_title}_{timestamp}"

        # Normalize and save
        if np.max(np.abs(final_audio)) > 0:
            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95

        # Save in specified format
        try:
            filepath, filename = save_audio_with_format(
                final_audio, final_sample_rate, audio_format, audiobooks_folder, filename_base
            )
        except Exception as e:
            print(f"Warning: Could not save audiobook in {audio_format} format: {e}")
            # Fallback to WAV
            filename = f"{filename_base}.wav"
            filepath = os.path.join(audiobooks_folder, filename)
            write(filepath, final_sample_rate, (final_audio * 32767).astype(np.int16))

        # Calculate total duration and file size
        total_duration = len(final_audio) / final_sample_rate / 60  # in minutes
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)  # in MB

        ebook_script_text = "\n\n".join(
            [
                chunk.get("content", "")
                for chunk in text_chunks
                if isinstance(chunk, dict) and chunk.get("content")
            ]
        )

        metadata_path, _ = write_generation_sidecar_metadata(
            filepath,
            {
                "mode": "ebook_audiobook",
                "engine": tts_engine,
                "book_title": metadata.get("title", "Unknown"),
                "chapters_processed": len(audio_segments),
                "chunk_gap_seconds": chunk_gap,
                "chapter_gap_seconds": chapter_gap,
                "selected_chapters": selected_chapters,
                "audio_format": audio_format,
                "duration_minutes": float(total_duration),
                "file_size_mb": float(file_size_mb),
            },
            ebook_script_text,
            original_text=ebook_script_text,
            transformed_text=ebook_script_text,
        )

        status_message = f"✅ Audiobook generated successfully!\n"
        status_message += f"📖 Book: {metadata['title']}\n"
        status_message += f"📊 Chapters processed: {len(audio_segments)}\n"
        status_message += f"⏱️ Total duration: {total_duration:.1f} minutes\n"
        status_message += f"📁 File size: {file_size_mb:.1f} MB\n"
        status_message += f"🔇 Chunk gap: {chunk_gap}s | Chapter gap: {chapter_gap}s\n"
        status_message += f"💾 Saved as: {filename}\n"
        status_message += f"📂 Location: {os.path.abspath(filepath)}\n\n"
        if metadata_path:
            status_message += f"🧾 Metadata: {os.path.abspath(metadata_path)}\n\n"

        # For large files (>50MB or >30 minutes), don't return the audio data to avoid browser issues
        if file_size_mb > 50 or total_duration > 30:
            status_message += "⚠️ Large audiobook detected!\n"
            status_message += "🎧 File too large for browser playback - please use the download link or check the audiobooks folder.\n"
            status_message += (
                "💡 You can play the file with any audio player (VLC, Windows Media Player, etc.)"
            )
            return filepath, status_message  # Return file path instead of audio data
        else:
            status_message += "🎧 Audio preview available below (for smaller files)"
            return (final_sample_rate, final_audio), status_message

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"❌ Error converting eBook: {str(e)}"


def get_ebook_info_display(analysis_result):
    """Format eBook analysis result for display."""
    if not analysis_result["success"]:
        return f"❌ Error: {analysis_result['error']}"

    metadata = analysis_result["metadata"]
    chapters = analysis_result["chapters"]

    info_text = f"📖 **{metadata['title']}**\n\n"
    info_text += f"📄 Format: {metadata['format'].upper()}\n"
    info_text += f"📊 File size: {metadata['file_size'] / 1024 / 1024:.1f} MB\n"
    info_text += f"📚 Total chapters: {metadata['total_chapters']}\n"
    info_text += f"📝 Total words: {metadata['total_words']:,}\n"
    info_text += (
        f"⏱️ Estimated duration: {analysis_result['total_estimated_duration']:.1f} minutes\n\n"
    )

    info_text += "**📋 Chapters:**\n"
    for i, chapter in enumerate(chapters[:10]):  # Show first 10 chapters
        info_text += f"{i+1}. {chapter['title']} ({chapter['word_count']} words, ~{chapter['estimated_duration']:.1f} min)\n"

    if len(chapters) > 10:
        info_text += f"... and {len(chapters) - 10} more chapters\n"

    return info_text


# # ===== VOXCPM FUNCTIONS =====
def generate_voxcpm_unified_tts(
    text_input: str,
    voxcpm_ref_audio: Optional[str] = None,
    voxcpm_ref_text: Optional[str] = None,
    voxcpm_cfg_value: float = 2.0,
    voxcpm_inference_timesteps: int = 10,
    voxcpm_normalize: bool = True,
    voxcpm_denoise: bool = True,
    voxcpm_retry_badcase: bool = True,
    voxcpm_retry_badcase_max_times: int = 3,
    voxcpm_retry_badcase_ratio_threshold: float = 6.0,
    voxcpm_seed: int | None = None,
    effects_settings: Optional[dict[str, Any]] = None,
    audio_format: str = "wav",
):
    """Generate TTS audio using VoxCPM with voice cloning capabilities."""
    if not VOXCPM_AVAILABLE:
        return None, "❌ VoxCPM not available"

    if not text_input.strip():
        return None, "❌ Please enter text to synthesize"

    try:
        print(f"🎯 Generating VoxCPM TTS...")
        print(f"   Text: {text_input[:50]}...")

        # Generate speech using VoxCPM handler
        handler = get_voxcpm_handler()
        audio_array, message = handler.generate_speech(
            text=text_input,
            reference_audio=voxcpm_ref_audio,
            reference_text=voxcpm_ref_text,
            cfg_value=voxcpm_cfg_value,
            inference_timesteps=voxcpm_inference_timesteps,
            normalize=voxcpm_normalize,
            denoise=voxcpm_denoise,
            retry_badcase=voxcpm_retry_badcase,
            retry_badcase_max_times=voxcpm_retry_badcase_max_times,
            retry_badcase_ratio_threshold=voxcpm_retry_badcase_ratio_threshold,
            seed=voxcpm_seed,
        )

        if audio_array is None:
            return None, message

        # VoxCPM returns numpy array, we need to format it for the unified interface
        sample_rate = handler.sample_rate  # 16000

        # Apply effects if specified
        if effects_settings:
            try:
                print("🎛️ Applying audio effects...")
                audio_array = apply_audio_effects(audio_array, sample_rate, effects_settings)
            except Exception as e:
                print(f"⚠️ Error applying effects: {e}")
                # Continue without effects

        # Convert format if needed
        if audio_format.lower() == "mp3":
            try:
                from pydub import AudioSegment
                import tempfile

                print("🔄 Converting to MP3...")

                # Create temporary WAV file
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav.close()

                try:
                    # Save as high-quality WAV first
                    # Convert int16 to float32 if needed
                    if audio_array.dtype == np.int16:
                        audio_float = audio_array.astype(np.float32) / 32767.0
                    else:
                        audio_float = audio_array

                    sf.write(temp_wav.name, audio_float, sample_rate)

                    # Convert WAV to MP3 with high quality settings
                    audio_segment = AudioSegment.from_wav(temp_wav.name)

                    # Create a new temporary MP3 file
                    temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    temp_mp3.close()

                    # Export with high quality settings
                    audio_segment.export(
                        temp_mp3.name,
                        format="mp3",
                        bitrate="320k",  # High quality
                        parameters=["-q:a", "0"],  # Highest quality
                    )

                    # Read back the MP3 as audio array
                    converted_segment = AudioSegment.from_mp3(temp_mp3.name)
                    audio_array = np.array(
                        converted_segment.get_array_of_samples(), dtype=np.float32
                    )

                    # Convert to proper format for return
                    if converted_segment.channels == 2:
                        audio_array = audio_array.reshape((-1, 2))
                        audio_array = audio_array.mean(axis=1)  # Convert to mono

                    # Normalize to int16 range
                    audio_array = (audio_array / np.max(np.abs(audio_array)) * 32767).astype(
                        np.int16
                    )

                finally:
                    # Clean up temporary files
                    try:
                        os.unlink(temp_wav.name)
                        os.unlink(temp_mp3.name)
                    except:
                        pass

            except Exception as e:
                print(f"⚠️ MP3 conversion failed: {e}")
                print("   Falling back to WAV format")
                # Fall back to WAV - no format change needed

        # Save to file (using the same pattern as other engines)
        try:
            # Create outputs directory if it doesn't exist
            output_folder = Path(get_runtime_output_dir("outputs"))
            output_folder.mkdir(exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if audio_format.lower() == "mp3":
                filename = f"voxcpm_tts_{timestamp}.mp3"
            else:
                filename = f"voxcpm_tts_{timestamp}.wav"

            filepath = output_folder / filename

            # Save the audio file
            if audio_format.lower() == "mp3":
                # Audio should already be converted to MP3 format above
                # But we need to save it as MP3 file
                if audio_array.dtype == np.int16:
                    audio_float = audio_array.astype(np.float32) / 32767.0
                else:
                    audio_float = audio_array

                # Save as temporary WAV first, then convert to MP3 for final file
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav.close()

                try:
                    sf.write(temp_wav.name, audio_float, sample_rate)
                    audio_segment = AudioSegment.from_wav(temp_wav.name)
                    audio_segment.export(
                        str(filepath), format="mp3", bitrate="320k", parameters=["-q:a", "0"]
                    )
                finally:
                    try:
                        os.unlink(temp_wav.name)
                    except:
                        pass
            else:
                # Save as WAV
                if audio_array.dtype == np.int16:
                    audio_float = audio_array.astype(np.float32) / 32767.0
                else:
                    audio_float = audio_array
                sf.write(str(filepath), audio_float, sample_rate)

            # Calculate duration
            duration = len(audio_array) / sample_rate

            success_message = f"✅ VoxCPM TTS generated successfully\n"
            success_message += f"📁 Saved as: {filename}\n"
            success_message += f"⏱️ Duration: {duration:.2f}s\n"
            success_message += f"📊 Sample Rate: {sample_rate}Hz"

        except Exception as save_error:
            print(f"⚠️ Warning: Could not save audio file: {save_error}")
            success_message = "✅ VoxCPM TTS generated successfully (file saving failed)"

        # Return in the format expected by the conversation handler: (sample_rate, audio_array)
        return (sample_rate, audio_array), success_message

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_msg = f"❌ VoxCPM TTS generation error: {str(e)}"
        print(error_msg)
        return None, error_msg


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

    This expands rule-based patterns so downstream TTS engines and optional LLM
    styling receive more speakable text.

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

    def phone_replacer(match):
        g1, g2, g3 = match.groups()
        return f"{_spell_digits(g1)}, {_spell_digits(g2)}, {_spell_digits(g3)}"

    text = re.sub(r"\b(\d{3})-(\d{3})-(\d{4})\b", phone_replacer, text)

    def currency_replacer(match):
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

    def percent_replacer(match):
        digits = match.group(1)
        return f"{_int_to_words(int(digits)) if digits.isdigit() else digits} percent"

    text = re.sub(r"\b(\d+)%\b", percent_replacer, text)

    def date_replacer(match):
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

    def time_replacer(match):
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

    def url_replacer(match):
        value = match.group(0)
        spoken = value.replace("https://", "").replace("http://", "")
        spoken = spoken.replace(".", " dot ").replace("/", " slash ")
        return re.sub(r"\s+", " ", spoken).strip()

    text = re.sub(
        r"\bhttps?://[^\s]+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?", url_replacer, text
    )

    return text


def _apply_local_narration_transform(source_text: str, mode: str) -> tuple:
    """Apply local (non-LLM) narration transform with mode-aware status reporting.

    Args:
        source_text: Raw source text to transform.
        mode: One of "Minimal", "Polish", or "Vivid".

    Returns:
        Tuple of (transformed_text, status_message).
    """
    result = deterministic_normalize(source_text)
    if mode == "Vivid":
        status = (
            "Narration transform: deterministic normalization applied; "
            "Vivid expressive styling requires an LLM \u2014 connect a provider for full effect"
        )
    else:
        status = "Narration transform: deterministic normalization applied"
    return result, status


# Provider configuration — structured dict replaces per-function tuple lookups
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


def _get_provider_config(provider_name: str) -> dict:
    return LLM_PROVIDER_CONFIGS.get(provider_name, _DEFAULT_PROVIDER_CONFIG)


def get_llm_provider_static_defaults(provider_name: str):
    cfg = _get_provider_config(provider_name)
    suggestions = list(LLM_PROVIDER_MODEL_SUGGESTIONS.get(provider_name, []))
    default_model = cfg["default_model"]
    if default_model and default_model not in suggestions:
        suggestions.insert(0, default_model)
    return cfg["base_url"], "", gr.update(choices=suggestions, value=default_model)


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
        hint += "\nOr use: Azure Portal \u2192 AI Foundry \u2192 Project \u2192 Keys"
    elif provider_name == "GitHub Models (OpenAI-compatible)":
        hint += "\nGenerate at: github.com/settings/tokens (Fine-grained or Classic)"
    return hint


def resolve_llm_api_key(provider_name: str, api_key: str):
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip(), "ui"

    env_name = get_llm_provider_env_var(provider_name)
    env_value = os.environ.get(env_name, "")
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip(), "env"

    return "", "missing"


def _resolve_api_key_internal(provider_name: str, api_key: str):
    return resolve_llm_api_key(provider_name, api_key)


def fetch_provider_models(
    provider_name: str,
    base_url: str,
    api_key: str = "",
    timeout_sec: float = 5.0,
) -> tuple[list[str], str]:
    """Fetch model IDs from provider model discovery endpoints."""
    import json as _json
    import urllib.error
    import urllib.parse
    import urllib.request

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
            payload = _json.loads(response.read().decode("utf-8"))

        if isinstance(payload, dict):
            items = payload.get("data", [])
            if not items:
                items = payload.get("models", [])
        else:
            items = []

        models = []
        seen = set()
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
    import time
    import urllib.request

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


def on_llm_provider_change(provider_name: str):
    """Update provider defaults, optionally start LM Studio, and fetch live models."""
    cfg = _get_provider_config(provider_name)
    base_url = cfg["base_url"]
    default_model = cfg["default_model"]

    start_status = ""
    if provider_name == "LM Studio OpenAI Server":
        start_status = try_start_lm_studio()

    models, fetch_status = fetch_provider_models(provider_name, base_url)
    if not models:
        models = list(LLM_PROVIDER_MODEL_SUGGESTIONS.get(provider_name, []))

    if default_model and default_model not in models:
        models.insert(0, default_model)

    value = models[0] if models else default_model
    status_parts = [part for part in [start_status, fetch_status] if part]
    status = " | ".join(status_parts) if status_parts else ""

    return base_url, "", gr.update(choices=models, value=value), status


def refresh_llm_models(provider_name: str, base_url: str, api_key: str):
    """Manual refresh for provider model discovery."""
    models, status = fetch_provider_models(provider_name, base_url, api_key)

    if not models:
        cfg = _get_provider_config(provider_name)
        models = list(LLM_PROVIDER_MODEL_SUGGESTIONS.get(provider_name, []))
        default_model = cfg["default_model"]
        if default_model and default_model not in models:
            models.insert(0, default_model)

    value = models[0] if models else ""
    return gr.update(choices=models, value=value), status


def on_transform_preview(
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
):
    transformed_text, status = apply_llm_transform_to_textbox(
        source_text=source_text,
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
    )
    banner = format_provenance(status)
    return (
        gr.update(visible=True),
        source_text,
        transformed_text,
        gr.update(value=banner, visible=True),
        gr.update(visible=True),
        status,
    )


def on_accept_transform(transformed_text: str):
    return (
        transformed_text,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def on_reject_transform():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


# ===== MAIN GENERATION FUNCTION =====
def generate_unified_tts(
    # Common parameters
    text_input: str,
    tts_engine: str,
    # Audio format parameter
    audio_format: str = "wav",
    # ChatterboxTTS parameters
    chatterbox_ref_audio: str = None,
    chatterbox_exaggeration: float = 0.5,
    chatterbox_temperature: float = 0.8,
    chatterbox_cfg_weight: float = 0.5,
    chatterbox_chunk_size: int = 300,
    chatterbox_seed: int = 0,
    # Chatterbox Multilingual parameters
    chatterbox_mtl_ref_audio: str = None,
    chatterbox_mtl_language: str = "en",
    chatterbox_mtl_exaggeration: float = 0.5,
    chatterbox_mtl_temperature: float = 0.8,
    chatterbox_mtl_cfg_weight: float = 0.5,
    chatterbox_mtl_repetition_penalty: float = 2.0,
    chatterbox_mtl_min_p: float = 0.05,
    chatterbox_mtl_top_p: float = 1.0,
    chatterbox_mtl_chunk_size: int = 300,
    chatterbox_mtl_seed: int = 0,
    # Chatterbox Turbo parameters
    chatterbox_turbo_ref_audio: str = None,
    chatterbox_turbo_exaggeration: float = 0.5,
    chatterbox_turbo_temperature: float = 0.8,
    chatterbox_turbo_cfg_weight: float = 0.5,
    chatterbox_turbo_repetition_penalty: float = 1.2,
    chatterbox_turbo_min_p: float = 0.05,
    chatterbox_turbo_top_p: float = 1.0,
    chatterbox_turbo_chunk_size: int = 300,
    chatterbox_turbo_seed: int = 0,
    # Kokoro parameters
    kokoro_voice: str = "af_heart",
    kokoro_speed: float = 1.0,
    # Fish Speech parameters
    fish_ref_audio: str = None,
    fish_ref_text: str = None,
    fish_temperature: float = 0.8,
    fish_top_p: float = 0.8,
    fish_repetition_penalty: float = 1.1,
    fish_max_tokens: int = 1024,
    fish_seed: int = None,
    # IndexTTS parameters
    indextts_ref_audio: str = None,
    indextts_temperature: float = 0.8,
    indextts_seed: int = None,
    # IndexTTS2 parameters
    indextts2_ref_audio: str = None,
    indextts2_emotion_mode: str = "audio_reference",
    indextts2_emotion_audio: str = None,
    indextts2_emotion_description: str = "",
    indextts2_emo_alpha: float = 1.0,
    indextts2_happy: float = 0.0,
    indextts2_angry: float = 0.0,
    indextts2_sad: float = 0.0,
    indextts2_afraid: float = 0.0,
    indextts2_disgusted: float = 0.0,
    indextts2_melancholic: float = 0.0,
    indextts2_surprised: float = 0.0,
    indextts2_calm: float = 1.0,
    indextts2_temperature: float = 0.8,
    indextts2_top_p: float = 0.9,
    indextts2_top_k: int = 50,
    indextts2_repetition_penalty: float = 1.1,
    indextts2_max_mel_tokens: int = 1500,
    indextts2_seed: int = None,
    indextts2_use_random: bool = True,
    # F5-TTS parameters
    f5_ref_audio: str = None,
    f5_ref_text: str = None,
    f5_speed: float = 1.0,
    f5_cross_fade: float = 0.15,
    f5_remove_silence: bool = False,
    f5_seed: int = 0,
    # Higgs Audio parameters
    higgs_ref_audio: str = None,
    higgs_ref_text: str = None,
    higgs_voice_preset: str = "EMPTY",
    higgs_system_prompt: str = "",
    higgs_temperature: float = 1.0,
    higgs_top_p: float = 0.95,
    higgs_top_k: int = 50,
    higgs_max_tokens: int = 1024,
    higgs_ras_win_len: int = 7,
    higgs_ras_win_max_num_repeat: int = 2,
    # KittenTTS parameters
    kitten_voice: str = "expr-voice-2-f",
    # VoxCPM parameters
    voxcpm_ref_audio: str = None,
    voxcpm_ref_text: str = None,
    voxcpm_cfg_value: float = 2.0,
    voxcpm_inference_timesteps: int = 10,
    voxcpm_normalize: bool = True,
    voxcpm_denoise: bool = True,
    voxcpm_retry_badcase: bool = True,
    voxcpm_retry_badcase_max_times: int = 3,
    voxcpm_retry_badcase_ratio_threshold: float = 6.0,
    voxcpm_seed: int = None,
    # Qwen TTS parameters
    qwen_mode: str = "voice_clone",
    qwen_voice_description: str = "",
    qwen_ref_audio: str = None,
    qwen_ref_text: str = "",
    qwen_xvector_only: bool = False,
    qwen_clone_model_size: str = "1.7B",
    qwen_chunk_size: int = 200,
    qwen_chunk_gap: float = 0.0,
    qwen_speaker: str = "Ryan",
    qwen_custom_model_size: str = "1.7B",
    qwen_style_instruct: str = "",
    qwen_language: str = "Auto",
    qwen_seed: int = -1,
    # Optional LLM narration transform parameters
    llm_transform_enabled: bool = False,
    llm_provider: str = "LM Studio OpenAI Server",
    llm_base_url: str = "http://localhost:1234/v1",
    llm_api_key: str = "",
    llm_model_id: str = "qwen/qwen3-30b-a3b-instruct-2507",
    llm_mode: str = "NORMALIZE",
    llm_locale: str = "en-US",
    llm_style: str = "cinematic_audiobook",
    llm_max_tag_density: float = 0.35,
    llm_system_prompt: str = "",
    llm_timeout_seconds: int = 60,
    llm_temperature: float = 0.2,
    llm_top_p: float = 0.9,
    llm_max_tokens: int = 1024,
    llm_allow_local_fallback: bool = True,
    # Effects parameters
    gain_db: float = 0,
    enable_eq: bool = False,
    eq_bass: float = 0,
    eq_mid: float = 0,
    eq_treble: float = 0,
    enable_reverb: bool = False,
    reverb_room: float = 0.3,
    reverb_damping: float = 0.5,
    reverb_wet: float = 0.3,
    enable_echo: bool = False,
    echo_delay: float = 0.3,
    echo_decay: float = 0.5,
    enable_pitch: bool = False,
    pitch_semitones: float = 0,
):
    """Unified TTS generation function."""

    if not text_input.strip():
        return None, "❌ Please enter text to synthesize"

    # Prepare effects settings
    effects_settings = (
        {
            "gain_db": gain_db,
            "enable_eq": enable_eq,
            "eq_bass": eq_bass,
            "eq_mid": eq_mid,
            "eq_treble": eq_treble,
            "enable_reverb": enable_reverb,
            "reverb_room": reverb_room,
            "reverb_damping": reverb_damping,
            "reverb_wet": reverb_wet,
            "enable_echo": enable_echo,
            "echo_delay": echo_delay,
            "echo_decay": echo_decay,
            "enable_pitch": enable_pitch,
            "pitch_semitones": pitch_semitones,
        }
        if any([gain_db != 0, enable_eq, enable_reverb, enable_echo, enable_pitch])
        else None
    )

    if tts_engine == "ChatterboxTTS":
        return generate_chatterbox_tts(
            text_input,
            chatterbox_ref_audio,
            chatterbox_exaggeration,
            chatterbox_temperature,
            chatterbox_seed,
            chatterbox_cfg_weight,
            chatterbox_chunk_size,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "Chatterbox Multilingual":
        return generate_chatterbox_multilingual_tts(
            text_input,
            chatterbox_mtl_language,
            chatterbox_mtl_ref_audio,
            chatterbox_mtl_exaggeration,
            chatterbox_mtl_temperature,
            chatterbox_mtl_seed,
            chatterbox_mtl_cfg_weight,
            chatterbox_mtl_repetition_penalty,
            chatterbox_mtl_min_p,
            chatterbox_mtl_top_p,
            chatterbox_mtl_chunk_size,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "Chatterbox Turbo":
        return generate_chatterbox_turbo_tts(
            text_input,
            chatterbox_turbo_ref_audio,
            chatterbox_turbo_exaggeration,
            chatterbox_turbo_temperature,
            chatterbox_turbo_cfg_weight,
            chatterbox_turbo_repetition_penalty,
            chatterbox_turbo_min_p,
            chatterbox_turbo_top_p,
            chatterbox_turbo_seed,
            chatterbox_turbo_chunk_size,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "Kokoro TTS":
        return generate_kokoro_tts(
            text_input, kokoro_voice, kokoro_speed, effects_settings, audio_format
        )
    elif tts_engine == "Fish Speech":
        return generate_fish_speech_tts(
            text_input,
            fish_ref_audio,
            fish_ref_text,
            fish_temperature,
            fish_top_p,
            fish_repetition_penalty,
            fish_max_tokens,
            fish_seed,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "IndexTTS":
        return generate_indextts_tts(
            text_input,
            indextts_ref_audio,
            indextts_temperature,
            indextts_seed,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "IndexTTS2":
        return generate_indextts2_unified_tts(
            text_input,
            indextts2_ref_audio,
            indextts2_emotion_mode,
            indextts2_emotion_audio,
            indextts2_emotion_description,
            indextts2_emo_alpha,
            indextts2_happy,
            indextts2_angry,
            indextts2_sad,
            indextts2_afraid,
            indextts2_disgusted,
            indextts2_melancholic,
            indextts2_surprised,
            indextts2_calm,
            indextts2_temperature,
            indextts2_top_p,
            indextts2_top_k,
            indextts2_repetition_penalty,
            indextts2_max_mel_tokens,
            indextts2_seed,
            indextts2_use_random,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "F5-TTS":
        return generate_f5_tts(
            text_input,
            f5_ref_audio,
            f5_ref_text,
            f5_speed,
            f5_cross_fade,
            f5_remove_silence,
            f5_seed,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "Higgs Audio":
        return generate_higgs_audio_tts(
            text_input,
            higgs_ref_audio,
            higgs_ref_text,
            higgs_voice_preset,
            higgs_system_prompt,
            higgs_temperature,
            higgs_top_p,
            higgs_top_k,
            higgs_max_tokens,
            higgs_ras_win_len,
            higgs_ras_win_max_num_repeat,
            150,  # chunk_length
            effects_settings,
            audio_format,
        )
    elif tts_engine == "KittenTTS":
        return generate_kitten_tts(text_input, kitten_voice, effects_settings, audio_format)
    elif tts_engine == "VoxCPM":
        return generate_voxcpm_unified_tts(
            text_input,
            voxcpm_ref_audio,
            voxcpm_ref_text,
            voxcpm_cfg_value,
            voxcpm_inference_timesteps,
            voxcpm_normalize,
            voxcpm_denoise,
            voxcpm_retry_badcase,
            voxcpm_retry_badcase_max_times,
            voxcpm_retry_badcase_ratio_threshold,
            voxcpm_seed,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "Qwen Voice Design":
        return generate_qwen_voice_design_tts(
            text_input,
            qwen_language,
            qwen_voice_description,
            qwen_seed,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "Qwen Voice Clone":
        return generate_qwen_voice_clone_tts(
            text_input,
            qwen_ref_audio,
            qwen_ref_text,
            qwen_language,
            qwen_xvector_only,
            qwen_clone_model_size,
            qwen_chunk_size,
            qwen_chunk_gap,
            qwen_seed,
            effects_settings,
            audio_format,
        )
    elif tts_engine == "Qwen Custom Voice":
        return generate_qwen_custom_voice_tts(
            text_input,
            qwen_speaker,
            qwen_language,
            qwen_style_instruct,
            qwen_custom_model_size,
            qwen_seed,
            effects_settings,
            audio_format,
        )
    else:
        return None, "❌ Invalid TTS engine selected"


def _is_seed_empty(seed_value, tts_engine: str) -> bool:
    if seed_value is None:
        return True
    if isinstance(seed_value, str) and not seed_value.strip():
        return True

    try:
        numeric_seed = int(seed_value)
    except Exception:
        return False

    if (
        tts_engine == "Qwen Voice Clone"
        or tts_engine == "Qwen Voice Design"
        or tts_engine == "Qwen Custom Voice"
    ):
        return numeric_seed <= 0
    return numeric_seed == 0


def _generate_seed() -> int:
    return random.randint(1, 2147483647)


def _collect_engine_metadata_controls(tts_engine: str, base_args: list, param_idx: dict) -> dict:
    controls = {}
    for param_name, metadata_key in ENGINE_METADATA_CONTROL_MAP.get(tts_engine, []):
        if param_name not in param_idx:
            continue
        value = base_args[param_idx[param_name]]
        if value is None:
            continue
        controls[metadata_key] = value
    return controls


def _safe_project_name(project_name: str) -> str:
    if not isinstance(project_name, str):
        return "default"
    cleaned = re.sub(r"[^\w\-. ]+", "_", project_name.strip())
    return cleaned[:80].strip() or "default"


def _safe_speaker_name(speaker_name: str) -> str:
    if not isinstance(speaker_name, str):
        return "speaker"
    cleaned = re.sub(r"[^\w\-. ]+", "_", speaker_name.strip())
    return cleaned[:80].strip() or "speaker"


def _safe_optional_label(value: str) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = re.sub(r"[^\w\-. ]+", "_", value.strip())
    return cleaned[:80].strip()


def _build_autosave_run_base(
    project_name: str, speaker_name: str, engine_name: str, preset_name: str = ""
) -> str:
    settings = load_app_state_settings()
    template = settings.get("filename_template", DEFAULT_AUTOSAVE_SETTINGS["filename_template"])
    if (
        not isinstance(template, str)
        or not template.strip()
        or template
        in {
            LEGACY_DEFAULT_FILENAME_TEMPLATE,
            PREVIOUS_DEFAULT_FILENAME_TEMPLATE,
            PRESET_ONLY_FILENAME_TEMPLATE,
        }
    ):
        template = DEFAULT_AUTOSAVE_SETTINGS["filename_template"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_preset = _safe_optional_label(preset_name)
    safe_speaker = _safe_speaker_name(speaker_name)
    preset_label = safe_preset if safe_preset else "no_preset"

    values = {
        "project": _safe_project_name(project_name),
        "speaker": safe_speaker,
        "preset": safe_preset,
        "voice": preset_label,
        "engine": _safe_project_name(engine_name or "tts"),
        "timestamp": timestamp,
    }

    try:
        candidate = str(template).format(**values)
    except Exception:
        candidate = f"{values['project']}_{preset_label}_{values['timestamp']}"

    if "{preset" in template and not safe_preset:
        candidate = f"{values['project']}_{preset_label}_{values['timestamp']}"

    candidate = re.sub(r"[^\w\-.]+", "_", candidate).strip("._")
    return candidate[:160] or f"tts_{timestamp}"


def autosave_generation_artifacts(
    audio_output_data,
    text_input: str,
    audio_format: str,
    project_name: str,
    speaker_name: str,
    metadata: dict,
    source_audio_path: str = None,
    store_audio_copy: bool = False,
    original_text_input: str = None,
    transformed_text_input: str = None,
):
    ensure_app_state_dirs()

    if (
        not audio_output_data
        or not isinstance(audio_output_data, tuple)
        or len(audio_output_data) != 2
    ):
        return None, "No audio output available to save"

    sample_rate, audio_data = audio_output_data
    project = _safe_project_name(project_name)
    autosave_root = get_runtime_output_dir("autosave")
    project_root = os.path.join(autosave_root, project)
    audio_dir = os.path.join(project_root, "audio")
    scripts_dir = os.path.join(project_root, "scripts")
    meta_dir = os.path.join(project_root, "meta")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    engine_name = (metadata or {}).get("engine", "tts")
    run_base = _build_autosave_run_base(
        project, speaker_name, engine_name, (metadata or {}).get("preset", "")
    )

    audio_copy_created = True
    if source_audio_path and not store_audio_copy and os.path.exists(source_audio_path):
        audio_path = os.path.abspath(source_audio_path)
        audio_filename = os.path.basename(audio_path)
        audio_copy_created = False
    else:
        audio_path, audio_filename = save_audio_with_format(
            audio_data,
            int(sample_rate),
            audio_format,
            audio_dir,
            run_base,
        )

    original_script_text = (
        original_text_input if isinstance(original_text_input, str) else (text_input or "")
    )
    transformed_script_text = (
        transformed_text_input if isinstance(transformed_text_input, str) else (text_input or "")
    )

    script_path = os.path.join(scripts_dir, f"{run_base}.txt")
    original_script_path = os.path.join(scripts_dir, f"{run_base}.original.txt")
    transformed_script_path = os.path.join(scripts_dir, f"{run_base}.transformed.txt")

    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.write(transformed_script_text)

    with open(original_script_path, "w", encoding="utf-8") as script_file:
        script_file.write(original_script_text)

    with open(transformed_script_path, "w", encoding="utf-8") as script_file:
        script_file.write(transformed_script_text)

    text_hash = hashlib.sha256(transformed_script_text.encode("utf-8")).hexdigest()
    original_text_hash = hashlib.sha256(original_script_text.encode("utf-8")).hexdigest()
    transformed_text_hash = hashlib.sha256(transformed_script_text.encode("utf-8")).hexdigest()

    full_metadata = {
        "timestamp": datetime.now().isoformat(),
        "text_hash_sha256": text_hash,
        "text_versions": {
            "original": {
                "sha256": original_text_hash,
                "chars": len(original_script_text),
            },
            "transformed": {
                "sha256": transformed_text_hash,
                "chars": len(transformed_script_text),
            },
        },
        "paths": {
            "audio": os.path.abspath(audio_path),
            "script": os.path.abspath(script_path),
            "script_original": os.path.abspath(original_script_path),
            "script_transformed": os.path.abspath(transformed_script_path),
            "meta": "",
        },
        "audio_copy_created": audio_copy_created,
        **(metadata or {}),
    }

    meta_path = os.path.join(meta_dir, f"{run_base}.json")
    full_metadata["paths"]["meta"] = os.path.abspath(meta_path)

    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(full_metadata, meta_file, indent=2, ensure_ascii=False)

    return {
        "audio_path": os.path.abspath(audio_path),
        "audio_filename": audio_filename,
        "script_path": os.path.abspath(script_path),
        "original_script_path": os.path.abspath(original_script_path),
        "transformed_script_path": os.path.abspath(transformed_script_path),
        "meta_path": os.path.abspath(meta_path),
    }, None


def generate_unified_tts_wrapped(*all_args):
    """Wrapper around generate_unified_tts: preset resolution, deterministic seed, and autosave."""
    signature_params = list(inspect.signature(generate_unified_tts).parameters.keys())
    base_count = len(signature_params)
    if len(all_args) < base_count:
        return None, "❌ Internal error: incomplete generation arguments", "🎲 Last Seed: N/A", None

    base_args = list(all_args[:base_count])
    extra_args = list(all_args[base_count:])

    # Current ordering: speaker_name, voice_preset, autosave_enabled, autosave_project_name,
    #                   autosave_store_audio_copy, keep_legacy_output_copy, last_seed_state
    # Backward compatibility retained for prior orderings.
    speaker_name = ""
    voice_preset = ""
    autosave_enabled = False
    autosave_project_name = "default"
    autosave_store_audio_copy = True
    keep_legacy_output_copy = True
    last_seed_state = None

    if len(extra_args) >= 7 and isinstance(extra_args[2], bool):
        speaker_name = extra_args[0]
        voice_preset = extra_args[1]
        autosave_enabled = bool(extra_args[2])
        autosave_project_name = extra_args[3]
        autosave_store_audio_copy = bool(extra_args[4])
        keep_legacy_output_copy = bool(extra_args[5])
        last_seed_state = extra_args[6]
    elif len(extra_args) >= 6 and isinstance(extra_args[2], bool):
        speaker_name = extra_args[0]
        voice_preset = extra_args[1]
        autosave_enabled = bool(extra_args[2])
        autosave_project_name = extra_args[3]
        autosave_store_audio_copy = bool(extra_args[4])
        last_seed_state = extra_args[5]
    elif len(extra_args) >= 6:
        speaker_name = extra_args[1]
        voice_preset = extra_args[2]
        autosave_enabled = bool(extra_args[3])
        autosave_project_name = extra_args[4]
        last_seed_state = extra_args[5]
    elif len(extra_args) >= 5:
        speaker_name = extra_args[0]
        voice_preset = extra_args[1]
        autosave_enabled = bool(extra_args[2])
        autosave_project_name = extra_args[3]
        last_seed_state = extra_args[4]
    else:
        speaker_name = ""
        voice_preset = extra_args[0] if len(extra_args) > 0 else ""
        autosave_enabled = bool(extra_args[1]) if len(extra_args) > 1 else False
        autosave_project_name = extra_args[2] if len(extra_args) > 2 else "default"
        autosave_store_audio_copy = bool(extra_args[3]) if len(extra_args) > 3 else True
        keep_legacy_output_copy = bool(extra_args[4]) if len(extra_args) > 4 else True
        last_seed_state = extra_args[5] if len(extra_args) > 5 else None

    resolved_project = _safe_project_name(autosave_project_name or "default")
    resolved_speaker = _safe_speaker_name(speaker_name or "speaker")

    param_idx = {name: index for index, name in enumerate(signature_params)}
    text_input = base_args[param_idx["text_input"]]
    original_text_input = text_input
    tts_engine = base_args[param_idx["tts_engine"]]
    audio_format = base_args[param_idx["audio_format"]]

    llm_transform_status = "LLM transform: disabled"
    if "llm_transform_enabled" in param_idx:
        transformed_text, llm_transform_status = apply_llm_narration_transform(
            source_text=text_input,
            enabled=bool(base_args[param_idx["llm_transform_enabled"]]),
            provider_name=base_args[param_idx["llm_provider"]],
            base_url=base_args[param_idx["llm_base_url"]],
            api_key=base_args[param_idx["llm_api_key"]],
            model_id=base_args[param_idx["llm_model_id"]],
            mode=base_args[param_idx["llm_mode"]],
            locale=base_args[param_idx["llm_locale"]],
            style=base_args[param_idx["llm_style"]],
            max_tag_density=base_args[param_idx["llm_max_tag_density"]],
            system_prompt=base_args[param_idx["llm_system_prompt"]],
            timeout_seconds=base_args[param_idx["llm_timeout_seconds"]],
            temperature=base_args[param_idx["llm_temperature"]],
            top_p=base_args[param_idx["llm_top_p"]],
            max_tokens=base_args[param_idx["llm_max_tokens"]],
            allow_local_fallback=bool(base_args[param_idx["llm_allow_local_fallback"]]),
            engine=tts_engine,
        )
        base_args[param_idx["text_input"]] = transformed_text
        text_input = transformed_text

    llm_transform_applied = bool(
        isinstance(original_text_input, str)
        and isinstance(text_input, str)
        and original_text_input != text_input
    )

    preset_name_used = ""
    preset_audio = get_preset_audio_path(voice_preset)
    engine_ref_param = {
        "ChatterboxTTS": "chatterbox_ref_audio",
        "Chatterbox Multilingual": "chatterbox_mtl_ref_audio",
        "Chatterbox Turbo": "chatterbox_turbo_ref_audio",
    }

    ref_param_name = engine_ref_param.get(tts_engine)
    if ref_param_name and preset_audio:
        base_args[param_idx[ref_param_name]] = preset_audio
        preset_name_used = voice_preset

    engine_seed_param = {
        "ChatterboxTTS": "chatterbox_seed",
        "Chatterbox Multilingual": "chatterbox_mtl_seed",
        "Chatterbox Turbo": "chatterbox_turbo_seed",
        "Fish Speech": "fish_seed",
        "IndexTTS": "indextts_seed",
        "IndexTTS2": "indextts2_seed",
        "F5-TTS": "f5_seed",
        "VoxCPM": "voxcpm_seed",
        "Qwen Voice Design": "qwen_seed",
        "Qwen Voice Clone": "qwen_seed",
        "Qwen Custom Voice": "qwen_seed",
    }

    used_seed = last_seed_state
    seed_param_name = engine_seed_param.get(tts_engine)
    if seed_param_name:
        seed_index = param_idx[seed_param_name]
        current_seed = base_args[seed_index]
        if _is_seed_empty(current_seed, tts_engine):
            used_seed = _generate_seed()
            base_args[seed_index] = used_seed
        else:
            try:
                used_seed = int(current_seed)
            except Exception:
                used_seed = current_seed

    active_output_dir = get_runtime_output_dir("outputs")
    before_audio_files = _list_audio_files(active_output_dir)

    generation_output, generation_status = generate_unified_tts(*base_args)

    after_audio_files = _list_audio_files(active_output_dir)
    generated_audio_file = _detect_new_audio_file(before_audio_files, after_audio_files)

    status_lines = [str(generation_status or "")]
    status_lines.append(f"\nEngine: {tts_engine}")
    status_lines.append(f"Seed: {used_seed if used_seed is not None else 'N/A'}")
    status_lines.append(f"Preset: {preset_name_used if preset_name_used else 'None'}")
    status_lines.append(f"Project: {resolved_project}")
    status_lines.append(f"Speaker: {resolved_speaker}")
    status_lines.append(llm_transform_status)
    status_lines.append(
        f"Autosave copy audio: {'Yes' if autosave_store_audio_copy else 'No (metadata+script only)'}"
    )
    status_lines.append(f"Legacy outputs copy: {'Yes' if keep_legacy_output_copy else 'No'}")

    autosave_error = None
    autosave_paths = None
    if autosave_enabled and generation_output is not None:
        engine_controls = _collect_engine_metadata_controls(tts_engine, base_args, param_idx)
        metadata = {
            "engine": tts_engine,
            "seed": used_seed,
            "preset": preset_name_used,
            "project": resolved_project,
            "speaker": resolved_speaker,
            "audio_format": audio_format,
            "llm_transform": {
                "status": llm_transform_status,
                "applied": llm_transform_applied,
            },
            **engine_controls,
        }
        try:
            autosave_paths, autosave_error = autosave_generation_artifacts(
                generation_output,
                text_input,
                audio_format,
                resolved_project,
                resolved_speaker,
                metadata,
                source_audio_path=generated_audio_file,
                store_audio_copy=autosave_store_audio_copy,
                original_text_input=original_text_input,
                transformed_text_input=text_input,
            )
            if autosave_paths:
                status_lines.append(f"Autosave audio: {autosave_paths['audio_path']}")
                status_lines.append(f"Autosave script: {autosave_paths['script_path']}")
                status_lines.append(
                    f"Autosave original script: {autosave_paths['original_script_path']}"
                )
                status_lines.append(
                    f"Autosave transformed script: {autosave_paths['transformed_script_path']}"
                )
                status_lines.append(f"Autosave meta: {autosave_paths['meta_path']}")
        except Exception as error:
            autosave_error = str(error)

    if autosave_error:
        status_lines.append(f"⚠️ Autosave failed: {autosave_error}")

    if (
        autosave_enabled
        and generation_output is not None
        and generated_audio_file
        and not keep_legacy_output_copy
        and autosave_paths
    ):
        generated_audio_abs = os.path.abspath(generated_audio_file)
        autosave_audio_abs = os.path.abspath(autosave_paths.get("audio_path", ""))
        if generated_audio_abs != autosave_audio_abs and os.path.exists(generated_audio_abs):
            try:
                os.remove(generated_audio_abs)
                status_lines.append(f"Legacy output removed: {generated_audio_abs}")
            except Exception as error:
                status_lines.append(f"⚠️ Failed to remove legacy output copy: {error}")
        elif generated_audio_abs == autosave_audio_abs:
            status_lines.append("ℹ️ Legacy output cleanup skipped (autosave references same file)")

    seed_label = f"🎲 Last Seed: {used_seed if used_seed is not None else 'N/A'}"
    return generation_output, "\n".join(status_lines), seed_label, used_seed


# ===== GRADIO INTERFACE =====
def create_gradio_interface():
    """Create the unified Gradio interface."""
    current_storage_settings = load_app_state_settings()
    current_llm_settings = get_initial_llm_panel_settings(current_storage_settings)
    llm_content_type_info = (
        "Choose the prompt preset that replaces the base system prompt. "
        + " ".join(
            f"{name}: {preset['description']}" for name, preset in CONTENT_TYPE_PRESETS.items()
        )
    )
    lexicon_path = Path(__file__).parent / "app_state" / "lexicon.json"
    current_storage_mode, current_storage_path = resolve_output_storage_settings(
        current_storage_settings
    )
    storage_mode_value = (
        "Custom Path" if current_storage_mode == "custom" else "Project Folders (default)"
    )
    storage_status_default = (
        f"ℹ️ Output storage mode: {storage_mode_value}\n"
        f"Outputs: {os.path.abspath(get_runtime_output_dir('outputs', current_storage_settings))}\n"
        f"Audiobooks: {os.path.abspath(get_runtime_output_dir('audiobooks', current_storage_settings))}\n"
        f"Autosave: {os.path.abspath(get_runtime_output_dir('autosave', current_storage_settings))}"
    )

    def on_preset_change(preset_name: str):
        return get_llm_outcome_preset_values(preset_name)

    def _to_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        return default

    def _lexicon_terms_to_rows(protected_terms: list[ProtectedTerm]) -> list[list[Any]]:
        return [
            [protected_term.term, protected_term.case_sensitive]
            for protected_term in protected_terms
            if protected_term.term
        ]

    def _lexicon_overrides_to_rows(overrides: list[PronunciationOverride]) -> list[list[Any]]:
        return [
            [override.word, override.phonetic, override.case_sensitive]
            for override in overrides
            if override.word and override.phonetic
        ]

    def _load_glossary_rows() -> tuple[list[list[Any]], list[list[Any]], str]:
        if not lexicon_path.exists():
            return [], [], f"ℹ️ Glossary file not found yet: {lexicon_path.name}"

        try:
            protected_terms, overrides = load_lexicon(lexicon_path)
        except (FileNotFoundError, ValueError, OSError) as error:
            return [], [], f"❌ Failed to load glossary: {error}"

        protected_rows = _lexicon_terms_to_rows(protected_terms)
        override_rows = _lexicon_overrides_to_rows(overrides)
        return (
            protected_rows,
            override_rows,
            f"✅ Loaded glossary from {lexicon_path.name} ({len(protected_rows)} terms, {len(override_rows)} overrides)",
        )

    glossary_protected_rows, glossary_override_rows, glossary_status_message = _load_glossary_rows()

    # Kokoro voices will be preloaded when the model is loaded

    gr_themes = getattr(gr, "themes")

    with gr.Blocks(
        title="✨ ULTIMATE TTS STUDIO PRO ✨",
        theme=gr_themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="gray",
            font=[gr_themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ).set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
        ),
        js="""
        () => {
            // Force dark mode on page load
            document.body.classList.remove('light');
            document.body.classList.add('dark');

            // Override any theme preference detection
            const style = document.createElement('style');
            style.textContent = `
                .light { display: none !important; }
                body, .gradio-container { color-scheme: dark !important; }
            `;
            document.head.appendChild(style);

            // Set localStorage to persist dark mode
            localStorage.setItem('theme', 'dark');

            // Watch for theme changes and revert to dark
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.attributeName === 'class' && document.body.classList.contains('light')) {
                        document.body.classList.remove('light');
                        document.body.classList.add('dark');
                    }
                });
            });
            observer.observe(document.body, { attributes: true });
        }
        """,
        css="""
        /* CSS Variables for Theme Support */
        :root {
            --text-primary: rgba(241, 245, 249, 0.96);
            --text-secondary: rgba(203, 213, 225, 0.82);
            --text-muted: rgba(148, 163, 184, 0.78);
            --bg-primary: rgba(15, 23, 42, 0.56);
            --bg-secondary: rgba(15, 23, 42, 0.38);
            --border-color: rgba(148, 163, 184, 0.22);
            --accent-color: #8b5cf6;
            --accent-color-2: #22d3ee;
            --accent-color-3: #f472b6;
            --gradient-bg: linear-gradient(160deg, #050816 0%, #0f172a 48%, #1e1b4b 100%);
            --panel-radius: 14px;
            --space-1: 8px;
            --space-2: 12px;
            --space-3: 16px;
            --space-4: 20px;
        }

        /* Force Dark Mode Variables for Light Mode */
        .light :root,
        [data-theme="light"] :root,
        .gradio-container.light,
        .gradio-container[data-theme="light"],
        body.light,
        body[data-theme="light"] {
            --text-primary: rgba(241, 245, 249, 0.96) !important;
            --text-secondary: rgba(203, 213, 225, 0.82) !important;
            --text-muted: rgba(148, 163, 184, 0.78) !important;
            --bg-primary: rgba(15, 23, 42, 0.56) !important;
            --bg-secondary: rgba(15, 23, 42, 0.38) !important;
            --border-color: rgba(148, 163, 184, 0.22) !important;
            --accent-color: #8b5cf6 !important;
            --accent-color-2: #22d3ee !important;
            --accent-color-3: #f472b6 !important;
            --gradient-bg: linear-gradient(160deg, #050816 0%, #0f172a 48%, #1e1b4b 100%) !important;

            /* Override Gradio default light theme vars */
            --body-background-fill: transparent !important;
            --background-fill-primary: rgba(255, 255, 255, 0.05) !important;
            --background-fill-secondary: rgba(255, 255, 255, 0.03) !important;
            --block-background-fill: rgba(255, 255, 255, 0.05) !important;
            --block-border-color: rgba(255, 255, 255, 0.1) !important;
            --block-label-text-color: rgba(255, 255, 255, 0.9) !important;
            --input-background-fill: rgba(255, 255, 255, 0.05) !important;
            --input-border-color: rgba(255, 255, 255, 0.1) !important;
            --input-placeholder-color: rgba(255, 255, 255, 0.5) !important;
            --neutral-50: #1f2937 !important;
            --neutral-100: #374151 !important;
            --neutral-200: #4b5563 !important;
            --neutral-300: #9ca3af !important;
            --neutral-400: #d1d5db !important;
            --neutral-500: #e5e7eb !important;
            --neutral-600: #f3f4f6 !important;
            --neutral-700: #f9fafb !important;
            --neutral-800: #ffffff !important;
            --neutral-900: #ffffff !important;
            --neutral-950: #ffffff !important;
        }

        /* Force background for light mode */
        body.light, .gradio-container.light {
            background: var(--gradient-bg) !important;
            color: white !important;
        }

        /* Global Styles */
        .gradio-container {
            max-width: 1680px !important;
            margin: 0 auto !important;
            background: var(--gradient-bg) !important;
            min-height: 100vh;
            font-family: 'Inter', system-ui, sans-serif !important;
            padding: 0 var(--space-3) 36px var(--space-3) !important;
            line-height: 1.45 !important;
        }

        /* Animated Background - Responsive */
        .gradio-container::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image:
                radial-gradient(circle at 16% 80%, rgba(139, 92, 246, 0.18) 0%, transparent 52%),
                radial-gradient(circle at 82% 18%, rgba(34, 211, 238, 0.14) 0%, transparent 52%),
                radial-gradient(circle at 48% 38%, rgba(244, 114, 182, 0.12) 0%, transparent 56%);
            animation: gradientShift 20s ease infinite;
            pointer-events: none;
            z-index: 0;
            animation-play-state: paused;
        }

        /* Light mode background adjustment removed */

        @keyframes gradientShift {
            0%, 100% { transform: rotate(0deg) scale(1); }
            50% { transform: rotate(180deg) scale(1.1); }
        }

        /* Main Title Styling - Compact */
        .main-title {
            text-align: center;
            color: #f8fafc !important;
            font-size: 2.8em;
            font-weight: 900;
            margin: 15px 0 10px 0;
            text-shadow:
                0 0 16px rgba(139, 92, 246, 0.45),
                0 0 30px rgba(34, 211, 238, 0.2);
            letter-spacing: -1px;
            position: relative;
            z-index: 1;
            line-height: 1.1;
        }

        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .subtitle {
            text-align: center;
            color: var(--text-primary);
            font-size: 1.0em;
            margin-bottom: 20px;
            font-weight: 300;
            letter-spacing: 0.3px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
            line-height: 1.3;
        }

        /* Glassmorphism Cards - Compact */
        .card, .settings-card, .gr-group {
            background: var(--bg-primary) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
            border-radius: var(--panel-radius) !important;
            padding: var(--space-3) !important;
            margin: var(--space-1) 0 !important;
            border: 1px solid var(--border-color) !important;
            box-shadow:
                0 2px 10px 0 rgba(31, 38, 135, 0.2),
                inset 0 0 0 1px var(--border-color) !important;
            transition: all 0.3s ease !important;
            position: relative;
            overflow: visible;
            z-index: 1;
        }

        .card:hover, .settings-card:hover, .gr-group:hover {
            box-shadow:
                0 6px 18px 0 rgba(31, 38, 135, 0.28),
                inset 0 0 0 1px var(--border-color) !important;
            background: var(--bg-secondary) !important;
        }

        /* Gradient Borders */
        .card::before, .settings-card::before, .gr-group::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 20px;
            padding: 2px;
            background: linear-gradient(45deg, var(--accent-color), #6d28d9, var(--accent-color-3), var(--accent-color-2));
            -webkit-mask:
                linear-gradient(#fff 0 0) content-box,
                linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none; /* ensure overlay doesn't block clicks */
        }

        .card:hover::before, .settings-card:hover::before, .gr-group:hover::before {
            opacity: 0.5;
        }

        /* Generate Button - Compact */
        .generate-btn {
            background: linear-gradient(135deg, var(--accent-color) 0%, #6d28d9 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 13px 28px !important;
            font-size: 1.0em !important;
            font-weight: 700 !important;
            color: white !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
            box-shadow:
                0 6px 16px rgba(139, 92, 246, 0.38),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transition: all 0.3s ease !important;
            text-transform: none;
            letter-spacing: 0.2px;
            position: relative;
            overflow: hidden;
            margin: var(--space-3) auto !important;
            display: block !important;
            width: min(100%, 320px) !important;
        }

        .generate-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.5s ease;
        }

        .generate-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow:
                0 10px 26px rgba(139, 92, 246, 0.55),
                inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        }

        .generate-btn:hover::before {
            left: 100%;
        }

        .generate-btn:active {
            transform: translateY(-1px) scale(1.02) !important;
        }

        /* Input Fields */
        .gr-textbox, .gr-dropdown, .gr-slider, .gr-audio, .gr-number {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            transition: all 0.3s ease !important;
        }

        .gr-textbox:focus, .gr-dropdown:focus, .gr-number:focus {
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
            background: var(--bg-secondary) !important;
        }

        /* Dropdown specific fixes */
        .gr-dropdown {
            position: relative !important;
            z-index: 9999 !important;
        }

        /* Ensure dropdown container allows overflow and is stable */
        .gr-dropdown > div,
        .gr-dropdown .wrap {
            position: relative !important;
        }

        /* Dropdown menu styling - Multiple selectors for better compatibility */
        .gr-dropdown .choices,
        .gr-dropdown ul[role="listbox"],
        .gr-dropdown .dropdown-menu,
        .gr-dropdown [role="listbox"],
        .gr-dropdown .svelte-select-list {
            background: var(--bg-primary) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 2px solid var(--accent-color) !important;
            border-radius: 12px !important;
            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.8),
                0 0 0 1px rgba(102, 126, 234, 0.3) !important;
            z-index: 99999 !important;
            position: absolute !important;
            top: 100% !important;
            left: 0 !important;
            right: 0 !important;
            max-height: 300px !important;
            overflow-y: auto !important;
            margin-top: 4px !important;
            width: 100% !important;
            transform: none !important;
            transition: none !important;
            pointer-events: auto !important;
        }

        /* Dropdown items */
        .gr-dropdown .choices .item,
        .gr-dropdown li[role="option"],
        .gr-dropdown .dropdown-item,
        .gr-dropdown .svelte-select-list .item {
            color: var(--text-primary) !important;
            padding: 12px 16px !important;
            transition: all 0.2s ease !important;
            background: transparent !important;
            border: none !important;
            cursor: pointer !important;
            white-space: nowrap !important;
            font-size: 0.9em !important;
        }

        .gr-dropdown .choices .item:hover,
        .gr-dropdown li[role="option"]:hover,
        .gr-dropdown .dropdown-item:hover,
        .gr-dropdown .svelte-select-list .item:hover {
            background: rgba(102, 126, 234, 0.4) !important;
            color: white !important;
            transform: translateX(2px) !important;
        }

        .gr-dropdown .choices .item.selected,
        .gr-dropdown li[role="option"][aria-selected="true"],
        .gr-dropdown .dropdown-item.selected,
        .gr-dropdown .svelte-select-list .item.selected {
            background: rgba(102, 126, 234, 0.6) !important;
            color: white !important;
            font-weight: 600 !important;
        }

        /* Fix dropdown container overflow - Apply to all parent containers */
        .gr-group,
        .gr-column,
        .gr-row,
        .gr-accordion,
        .gradio-container {
            overflow: visible !important;
        }

        /* Specific fix for accordion content */

        .gr-accordion .gr-accordion-header {
            color: #f8fafc !important;
            font-weight: 740 !important;
            letter-spacing: 0.25px !important;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.20), rgba(34, 211, 238, 0.08)) !important;
        }
        .gr-accordion .gr-accordion-content {
            overflow: visible !important;
        }

        /* When a card/group contains a dropdown, increase its z-index and disable transforms */
        .card:has(.gr-dropdown),
        .settings-card:has(.gr-dropdown),
        .gr-group:has(.gr-dropdown) {
            z-index: 9998 !important;
            overflow: visible !important;
        }

        .card:has(.gr-dropdown):hover,
        .settings-card:has(.gr-dropdown):hover,
        .gr-group:has(.gr-dropdown):hover {
            transform: none !important;
        }

        /* Prevent parent hover effects from affecting dropdown */
        .gr-group:hover .gr-dropdown,
        .card:hover .gr-dropdown,
        .settings-card:hover .gr-dropdown {
            transform: none !important;
        }

        /* Ensure dropdown trigger button is properly styled */
        .gr-dropdown button,
        .gr-dropdown .dropdown-toggle {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            padding: 10px 15px !important;
            width: 100% !important;
            text-align: left !important;
            position: relative !important;
            z-index: 1 !important;
            transform: none !important;
        }

        /* Prevent any parent transforms from affecting dropdown positioning */
        .gr-dropdown,
        .gr-dropdown > *,
        .gr-dropdown .wrap,
        .gr-dropdown .wrap > * {
            transform: none !important;
            will-change: auto !important;
        }

        .gr-dropdown button:hover,
        .gr-dropdown .dropdown-toggle:hover {
            border-color: var(--accent-color) !important;
            background: var(--bg-secondary) !important;
        }

        /* Arrow icon styling */
        .gr-dropdown button::after,
        .gr-dropdown .dropdown-toggle::after {
            content: '▼' !important;
            float: right !important;
            transition: transform 0.2s ease !important;
        }

        .gr-dropdown button[aria-expanded="true"]::after,
        .gr-dropdown .dropdown-toggle.open::after {
            transform: rotate(180deg) !important;
        }

        /* Labels */
        label, .gr-label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
            font-size: 0.95em !important;
            margin-bottom: 8px !important;
            text-transform: none;
            letter-spacing: 0;
        }

        /* Workspace hierarchy */
        .workspace-row {
            align-items: flex-start !important;
            gap: 16px !important;
        }

        .main-workspace {
            min-width: 0 !important;
        }

        .right-rail {
            min-width: 0 !important;
            position: sticky;
            top: 12px;
            align-self: flex-start;
        }

        #preset_autosave_panel {
            width: 100%;
            margin-top: 8px;
            border: 1px solid rgba(102, 126, 234, 0.25) !important;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25) !important;
        }

        #preset_autosave_panel .gr-accordion-header,
        #engine_selector_panel .gr-accordion-header,
        #audiobook_results_panel .gr-accordion-header,
        #engine_settings_panel .gr-accordion-header,
        #model_manager_panel .gr-accordion-header {
            font-size: 0.96em !important;
            font-weight: 760 !important;
            letter-spacing: 0.3px !important;
            color: #f1f5f9 !important;
            text-shadow: 0 0 10px rgba(139, 92, 246, 0.3);
        }

        #engine_selector_panel {
            width: 100%;
            margin-top: 8px;
            border: 1px solid rgba(139, 92, 246, 0.42) !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22) !important;
        }

        #engine_settings_toggle {
            margin-top: 10px;
        }

        #engine_settings_toggle button {
            width: 100% !important;
            justify-content: center !important;
            border-radius: 12px !important;
            border: 1px solid rgba(139, 92, 246, 0.38) !important;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.24), rgba(34, 211, 238, 0.12)) !important;
            color: #f8fafc !important;
            font-weight: 760 !important;
            letter-spacing: 0.25px !important;
            text-shadow: 0 0 8px rgba(139, 92, 246, 0.28);
            cursor: default !important;
        }

        #text_synthesize_input textarea {
            resize: vertical !important;
            min-height: 210px !important;
            max-height: 72vh !important;
            line-height: 1.5 !important;
        }

        /* Sliders */
        .gr-slider input[type="range"] {
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            height: 8px !important;
        }

        .gr-slider input[type="range"]::-webkit-slider-thumb {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            border: 2px solid white !important;
            width: 20px !important;
            height: 20px !important;
            border-radius: 50% !important;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.5) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }

        .gr-slider input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2) !important;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.7) !important;
        }

        /* Accordion Styling - Compact */
        .gr-accordion {
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            margin: 12px 0 !important;
        }

        .gr-accordion-header {
            background: var(--bg-primary) !important;
            padding: 12px 15px !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            transition: all 0.3s ease !important;
            font-size: 0.95em !important;
        }

        .gradio-container .tabs {
            margin-top: var(--space-2) !important;
        }

        .gradio-container .tab-nav,
        .gradio-container .tab-nav button {
            border-radius: 10px !important;
        }

        .gradio-container .tab-nav button {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            padding: 8px 12px !important;
        }

        .gradio-container .tab-nav button[aria-selected="true"] {
            background: rgba(139, 92, 246, 0.24) !important;
            border-color: rgba(139, 92, 246, 0.5) !important;
            color: var(--text-primary) !important;
        }

        .gr-accordion-header:hover {
            background: var(--bg-secondary) !important;
        }

        /* Radio Buttons */
        .gr-radio {
            gap: 15px !important;
        }

        .gr-radio label {
            background: var(--bg-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 15px 25px !important;
            transition: all 0.3s ease !important;
            cursor: pointer !important;
            position: relative !important;
            overflow: hidden !important;
        }

        .gr-radio label:hover {
            background: var(--bg-secondary) !important;
            border-color: var(--accent-color) !important;
            transform: translateY(-2px) !important;
        }

        .gr-radio input[type="radio"]:checked + label {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)) !important;
            border-color: rgba(102, 126, 234, 0.5) !important;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
        }

        /* Voice Grid Layout */
        .voice-grid {
            display: grid !important;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
            gap: 10px !important;
            max-height: 400px !important;
            overflow-y: auto !important;
            padding: 10px !important;
            background: var(--bg-secondary) !important;
            border-radius: 15px !important;
            border: 1px solid var(--border-color) !important;
        }

        /* Conversation Mode Voice Grid - More compact */
        .conversation-voice-grid .voice-grid {
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) !important;
            gap: 8px !important;
            max-height: 300px !important;
            padding: 8px !important;
        }

        /* Make conversation voice labels smaller */
        .conversation-voice-grid .voice-grid label {
            font-size: 0.8em !important;
            padding: 6px 10px !important;
            min-height: 35px !important;
        }

        .voice-grid .gr-radio {
            display: contents !important;
        }

        .voice-grid label {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
            margin: 0 !important;
            font-size: 0.85em !important;
            text-align: center !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            min-height: 40px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .voice-grid label:hover {
            background: rgba(102, 126, 234, 0.2) !important;
            border-color: var(--accent-color) !important;
            transform: scale(1.02) !important;
        }

        /* Multiple selectors to ensure Gradio compatibility */
        .voice-grid input[type="radio"]:checked + label,
        .voice-grid input:checked + label,
        .voice-grid .gr-radio input:checked + label,
        .voice-grid [data-testid="radio"] input:checked + label {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            border: 2px solid #667eea !important;
            color: white !important;
            font-weight: 700 !important;
            box-shadow:
                0 4px 20px rgba(102, 126, 234, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            transform: scale(1.05) !important;
            z-index: 10 !important;
            position: relative !important;
        }

        /* Ensure selected state persists even on hover */
        .voice-grid input[type="radio"]:checked + label:hover,
        .voice-grid input:checked + label:hover,
        .voice-grid .gr-radio input:checked + label:hover,
        .voice-grid [data-testid="radio"] input:checked + label:hover {
            background: linear-gradient(135deg, #5a67d8, #6b46c1) !important;
            transform: scale(1.05) !important;
            border: 2px solid #5a67d8 !important;
        }

        .voice-grid input[type="radio"] {
            display: none !important;
        }

        /* Add a checkmark or indicator for selected voice */
        .voice-grid input[type="radio"]:checked + label::after,
        .voice-grid input:checked + label::after,
        .voice-grid .gr-radio input:checked + label::after,
        .voice-grid [data-testid="radio"] input:checked + label::after {
            content: '✓' !important;
            position: absolute !important;
            top: 4px !important;
            right: 6px !important;
            font-size: 14px !important;
            font-weight: bold !important;
            color: white !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8) !important;
        }

        /* Ensure the label has relative positioning for the checkmark */
        .voice-grid label {
            position: relative !important;
        }

        /* Add a subtle glow animation for selected voice */
        .voice-grid input[type="radio"]:checked + label,
        .voice-grid input:checked + label,
        .voice-grid .gr-radio input:checked + label,
        .voice-grid [data-testid="radio"] input:checked + label {
            animation: selectedGlow 2s ease-in-out infinite alternate !important;
        }

        /* Force override any Gradio default styles */
        .voice-grid .gr-radio label[data-selected="true"],
        .voice-grid label[aria-checked="true"],
        .voice-grid label.selected,
        .voice-grid label.voice-selected {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            border: 2px solid #667eea !important;
            color: white !important;
            font-weight: 700 !important;
            transform: scale(1.05) !important;
            box-shadow:
                0 4px 20px rgba(102, 126, 234, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            z-index: 10 !important;
            position: relative !important;
        }

        /* Checkmark for custom selected class */
        .voice-grid label.voice-selected::after,
        .voice-grid label[data-selected="true"]::after {
            content: '✓' !important;
            position: absolute !important;
            top: 4px !important;
            right: 6px !important;
            font-size: 14px !important;
            font-weight: bold !important;
            color: white !important;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8) !important;
        }

        /* Animation for custom selected class */
        .voice-grid label.voice-selected,
        .voice-grid label[data-selected="true"] {
            animation: selectedGlow 2s ease-in-out infinite alternate !important;
        }

        @keyframes selectedGlow {
            0% {
                box-shadow:
                    0 4px 20px rgba(102, 126, 234, 0.6),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
            }
            100% {
                box-shadow:
                    0 6px 30px rgba(102, 126, 234, 0.8),
                    inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
            }
        }

        /* Checkboxes */
        .gr-checkbox {
            background: var(--bg-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }

        .gr-checkbox:checked {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            border-color: transparent !important;
        }

        /* Audio Component */
        .gr-audio {
            background: var(--bg-secondary) !important;
            border-radius: 15px !important;
            padding: 20px !important;
        }

        /* Section Headers - Compact */
        h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            margin: 10px 0 8px 0 !important;
            position: relative !important;
            padding-left: 12px !important;
            font-size: 1.05em !important;
            line-height: 1.2 !important;
        }

        h2::before, h3::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 2px;
            height: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 2px;
        }

        /* Settings Group Headers - Smaller and more subtle */
        .gr-group h3, .settings-card h3 {
            font-size: 1.05em !important;
            font-weight: 500 !important;
            margin: 10px 0 8px 0 !important;
            padding-left: 10px !important;
            color: var(--text-primary) !important;
        }

        .gr-group h3::before, .settings-card h3::before {
            width: 2px !important;
            height: 50% !important;
        }

        /* Info Text */
        .gr-info {
            color: var(--text-muted) !important;
            font-size: 0.85em !important;
            font-style: italic !important;
        }

        /* Markdown Styling */
        .gr-markdown {
            color: var(--text-primary) !important;
            line-height: 1.6 !important;
        }

        .gr-markdown h3 {
            font-size: 1.05em !important;
            font-weight: 500 !important;
            margin: 8px 0 6px 0 !important;
            padding-left: 8px !important;
            color: var(--text-primary) !important;
        }

        .gr-markdown h3::before {
            width: 2px !important;
            height: 45% !important;
        }

        .gr-markdown h4 {
            font-size: 0.95em !important;
            font-weight: 500 !important;
            margin: 6px 0 4px 0 !important;
            padding-left: 6px !important;
            color: var(--text-secondary) !important;
        }

        .gr-markdown h4::before {
            width: 1.5px !important;
            height: 40% !important;
        }

        .gr-markdown strong {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        .gr-markdown code {
            background: var(--bg-primary) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: 'Fira Code', monospace !important;
        }

        /* Status Output */
        .gr-textbox[readonly] {
            background: rgba(255, 255, 255, 0.04) !important;
            border-color: rgba(255, 255, 255, 0.14) !important;
            color: var(--text-primary) !important;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, var(--accent-color), #6d28d9);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #6d28d9, var(--accent-color));
        }

        /* Loading Animation */
        .gr-loading {
            color: #667eea !important;
        }

        /* Feature Cards - Compact */
        .feature-card {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.12), rgba(34, 211, 238, 0.07));
            border: 1px solid rgba(139, 92, 246, 0.24);
            border-radius: 12px;
            padding: 12px;
            margin: 5px;
            transition: all 0.3s ease;
            text-align: center;
        }

        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 24px rgba(139, 92, 246, 0.28);
        }

        /* Glow Effects */
        .glow {
            box-shadow:
                0 0 10px rgba(139, 92, 246, 0.34),
                0 0 20px rgba(34, 211, 238, 0.16);
        }

        /* Responsive Design - Compact */
        @media (max-width: 1200px) {
            .workspace-row {
                flex-direction: column !important;
            }

            .right-rail {
                position: static !important;
                top: auto !important;
                width: 100% !important;
            }
        }

        @media (max-width: 768px) {
            .main-title {
                font-size: 2.2em;
            }

            .subtitle {
                font-size: 0.9em;
                margin-bottom: 15px;
            }

            .generate-btn {
                width: 100% !important;
                padding: 12px 25px !important;
                font-size: 1.0em !important;
            }

            .card, .settings-card, .gr-group {
                padding: 12px !important;
                margin: 5px 0 !important;
            }

            .feature-card {
                padding: 8px;
                margin: 3px;
            }
        }

        /* Additional light mode fixes removed */

        /* Dark Theme Overrides */
        .dark {
            --tw-bg-opacity: 0 !important;
        }

        /* Custom Animations */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
        """
        + """
        <script>
        // Theme detection and handling
        function updateTheme() {
            const container = document.querySelector('.gradio-container');
            if (container) {
                // Always enforce dark mode by removing light class/attribute
                container.classList.remove('light');
                container.removeAttribute('data-theme');
            }
        }

        // Run on load
        document.addEventListener('DOMContentLoaded', updateTheme);

        // Watch for theme changes
        const observer = new MutationObserver(updateTheme);
        observer.observe(document.body, {
            attributes: true,
            attributeFilter: ['class', 'data-theme'],
            subtree: true
        });

        // Also watch for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', updateTheme);
        }

        // Voice selection highlighting fix
        function setupVoiceSelection() {
            const voiceGrids = document.querySelectorAll('.voice-grid');
            voiceGrids.forEach(grid => {
                const radioInputs = grid.querySelectorAll('input[type="radio"]');
                radioInputs.forEach(input => {
                    input.addEventListener('change', function() {
                        // Remove selected class from all labels in this grid
                        const allLabels = grid.querySelectorAll('label');
                        allLabels.forEach(label => {
                            label.classList.remove('voice-selected');
                            label.removeAttribute('data-selected');
                        });

                        // Add selected class to the current label
                        if (this.checked) {
                            const label = this.nextElementSibling;
                            if (label && label.tagName === 'LABEL') {
                                label.classList.add('voice-selected');
                                label.setAttribute('data-selected', 'true');
                            }
                        }
                    });
                });
            });
        }

        // Run voice selection setup after DOM loads and when content changes
        document.addEventListener('DOMContentLoaded', setupVoiceSelection);

        // Also run when new content is added (Gradio dynamic updates)
        const contentObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    setupVoiceSelection();
                    setupTabSwitching();
                    setupEbookPanelExpansion();
                }
            });
        });

        contentObserver.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Tab switching functionality
        function setupTabSwitching() {
            const tabs = document.querySelectorAll('.gradio-tabs .tab-nav button');

            function findButtonByText(text) {
                return Array.from(document.querySelectorAll('button')).find(btn => btn.textContent.includes(text));
            }

            function findTextboxByLabel(labelText) {
                const labels = Array.from(document.querySelectorAll('label'));
                const label = labels.find(l => l.textContent.includes(labelText));
                if (label) {
                    const textbox = label.parentElement.querySelector('textarea');
                    return textbox;
                }
                return null;
            }

            tabs.forEach((tab, index) => {
                tab.addEventListener('click', function() {
                    setTimeout(() => {
                        const generateSpeechBtn = findButtonByText('🚀 Generate Speech');
                        const generateConversationBtn = findButtonByText('🎭 Generate Conversation');
                        const statusOutput = findTextboxByLabel('📊 Status');
                        const conversationInfo = findTextboxByLabel('📊 Conversation Summary');

                        if (tab.textContent.includes('TEXT TO SYNTHESIZE')) {
                            // Single voice mode
                            if (generateSpeechBtn) generateSpeechBtn.closest('.gradio-column').style.display = 'block';
                            if (generateConversationBtn) generateConversationBtn.closest('.gradio-column').style.display = 'none';
                            if (statusOutput) statusOutput.closest('.gradio-textbox').style.display = 'block';
                            if (conversationInfo) conversationInfo.closest('.gradio-textbox').style.display = 'none';
                        } else if (tab.textContent.includes('CONVERSATION MODE')) {
                            // Conversation mode
                            if (generateSpeechBtn) generateSpeechBtn.closest('.gradio-column').style.display = 'none';
                            if (generateConversationBtn) generateConversationBtn.closest('.gradio-column').style.display = 'block';
                            if (statusOutput) statusOutput.closest('.gradio-textbox').style.display = 'none';
                            if (conversationInfo) conversationInfo.closest('.gradio-textbox').style.display = 'block';
                        }
                    }, 100);
                });
            });
        }

        function setupEbookPanelExpansion() {
            const convertButton = document.querySelector('#convert_ebook_btn button, #convert_ebook_btn');
            const resultsPanel = document.querySelector('#audiobook_results_panel');
            if (!convertButton || !resultsPanel) return;
            if (convertButton.dataset.expandBound === 'true') return;

            const openResultsPanel = () => {
                const header = resultsPanel.querySelector('.gr-accordion-header');
                if (header && header.getAttribute('aria-expanded') === 'false') {
                    header.click();
                }
                setTimeout(() => {
                    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 120);
            };

            convertButton.addEventListener('click', openResultsPanel);
            convertButton.dataset.expandBound = 'true';
        }

        // Initialize tab switching on page load
        document.addEventListener('DOMContentLoaded', function() {
            setupTabSwitching();
            setupEbookPanelExpansion();
        });
        </script>
        """,
    ) as demo:

        # Header with enhanced styling
        gr.Markdown(
            """
        <div class="fade-in">
            <div style="display:flex; align-items:center; justify-content:center; gap:14px; flex-wrap:wrap; margin-bottom:2px;">
                <div class="main-title" style="margin: 8px 0;">
                ✨ ULTIMATE TTS STUDIO PRO ✨
                </div>
                <a href="https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition" target="_blank" style="text-decoration:none;">
                    <button style="background:linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); border:1px solid rgba(255,255,255,0.16); border-radius:12px; padding:8px 14px; color:white; font-weight:700; font-size:13px; cursor:pointer; box-shadow:0 4px 12px rgba(139, 92, 246, 0.35); white-space:nowrap;">
                        ⭐ Star on GitHub
                    </button>
                </a>
            </div>
            <div class="subtitle">
            🎭 ChatterboxTTS + Kokoro TTS + Fish Speech + IndexTTS + F5-TTS + VoxCPM | SUP3R EDITION 🚀<br/>
            <strong>Advanced Text-to-Speech with Multiple Engines, Voice Presets, Audio Effects & Export Options</strong>
            </div>
        </div>

        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr)); gap:10px; margin:12px 0;">
            <div class="feature-card">
                <h3 style="margin: 0 0 5px 0; padding: 0; font-size: 0.9em;">🎤 Voice Cloning</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 0.8em;">Clone any voice with ChatterboxTTS</p>
            </div>
            <div class="feature-card">
                <h3 style="margin: 0 0 5px 0; padding: 0; font-size: 0.9em;">🗣️ Pre-trained Voices</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 0.8em;">30+ high-quality Kokoro voices</p>
            </div>
            <div class="feature-card">
                <h3 style="margin: 0 0 5px 0; padding: 0; font-size: 0.9em;">📚 eBook Conversion</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 0.8em;">Convert books to audiobooks</p>
            </div>
            <div class="feature-card">
                <h3 style="margin: 0 0 5px 0; padding: 0; font-size: 0.9em;">🎵 Audio Effects</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 0.8em;">Professional audio enhancement</p>
            </div>
        </div>
        """
        )

        # Model Management Section - Compact Version
        with gr.Accordion(
            "🧩 Model Manager", open=False, elem_classes=["fade-in"], elem_id="model_manager_panel"
        ):
            gr.Markdown("*Load only the models you need to save memory.*", elem_classes=["fade-in"])

            # Compact model status display
            model_status_display = gr.Markdown(
                value=get_model_status(),
                elem_classes=["fade-in"],
                visible=False,  # Hide the detailed status by default
            )

            # F5-TTS Management in collapsible accordion
            with gr.Accordion("🎵 F5-TTS Model Management", open=False, elem_classes=["fade-in"]):
                if F5_TTS_AVAILABLE:
                    f5_model_status = gr.Markdown(
                        value="Loading model status...", elem_classes=["fade-in"]
                    )

                    with gr.Row():
                        # Get model choices dynamically from F5TTSHandler
                        from f5_tts_handler import get_f5_tts_handler

                        f5_handler = get_f5_tts_handler()
                        f5_model_choices = list(f5_handler.AVAILABLE_MODELS.keys())

                        f5_model_select = gr.Dropdown(
                            choices=f5_model_choices,
                            value="F5-TTS Base",
                            label="🎯 Select Model",
                            elem_classes=["fade-in"],
                        )

                        with gr.Column():
                            f5_download_btn = gr.Button(
                                "📥 Download Model", variant="secondary", elem_classes=["fade-in"]
                            )
                            f5_load_btn = gr.Button(
                                "🚀 Load Model", variant="primary", elem_classes=["fade-in"]
                            )
                            f5_unload_btn = gr.Button(
                                "🗑️ Unload Model", variant="secondary", elem_classes=["fade-in"]
                            )

                    f5_download_status = gr.Textbox(
                        label="📊 Download Status",
                        interactive=False,
                        elem_classes=["fade-in"],
                        visible=False,
                    )
                else:
                    gr.Markdown(
                        "⚠️ F5-TTS not available - please install with: `pip install f5-tts`"
                    )
                    # Create dummy components for F5-TTS model management
                    f5_model_select = gr.Dropdown(visible=False, value="F5-TTS Base", choices=[])
                    f5_download_btn = gr.Button(visible=False)
                    f5_load_btn = gr.Button(visible=False)
                    f5_unload_btn = gr.Button(visible=False)
                    f5_model_status = gr.Markdown(visible=False, value="")
                    f5_download_status = gr.Textbox(visible=False, value="")

            # Qwen TTS Management in collapsible accordion
            with gr.Accordion("🎙️ Qwen TTS Model Management", open=False, elem_classes=["fade-in"]):
                if QWEN_TTS_AVAILABLE:
                    qwen_model_status = gr.Markdown(
                        value="Loading model status...", elem_classes=["fade-in"]
                    )

                    with gr.Row():
                        qwen_model_type = gr.Dropdown(
                            choices=["Base", "VoiceDesign", "CustomVoice"],
                            value="Base",
                            label="🎯 Model Type",
                            info="Base=Voice Clone, VoiceDesign=Create voices, CustomVoice=Predefined speakers",
                            elem_classes=["fade-in"],
                        )
                        qwen_model_size = gr.Dropdown(
                            choices=["0.6B", "1.7B"],
                            value="1.7B",
                            label="📊 Model Size",
                            elem_classes=["fade-in"],
                        )

                    with gr.Row():
                        qwen_download_btn = gr.Button(
                            "📥 Download Model", variant="secondary", elem_classes=["fade-in"]
                        )
                        load_qwen_btn = gr.Button(
                            "🚀 Load Model", variant="primary", elem_classes=["fade-in"]
                        )
                        unload_qwen_btn = gr.Button(
                            "🗑️ Unload Model", variant="secondary", elem_classes=["fade-in"]
                        )

                    qwen_download_status = gr.Textbox(
                        label="📊 Status", interactive=False, elem_classes=["fade-in"], visible=True
                    )

                    qwen_status = gr.Markdown(
                        value="⭕ Not loaded", visible=False  # Hidden, used for internal state
                    )

                    gr.Markdown(
                        """
                    <div style='margin-top: 10px; padding: 10px; background: rgba(102, 126, 234, 0.05); border-radius: 8px; border-left: 3px solid #667eea;'>
                        <p style='margin: 0; font-size: 0.85em; opacity: 0.8;'>
                            <strong>📋 Model Info:</strong><br/>
                            • <strong>Base (Voice Clone):</strong> 0.6B or 1.7B - Clone voices from reference audio ✅ Supports chunking<br/>
                            • <strong>VoiceDesign:</strong> 1.7B only - Create voices from text descriptions<br/>
                            • <strong>CustomVoice:</strong> 0.6B or 1.7B - Use predefined speakers (Aiden, Dylan, Eric, etc.)
                        </p>
                    </div>
                    """
                    )
                else:
                    gr.Markdown(
                        "⚠️ Qwen TTS not available - check qwen_tts module and transformers version"
                    )
                    # Create dummy components
                    qwen_model_type = gr.Dropdown(visible=False, value="Base", choices=["Base"])
                    qwen_model_size = gr.Dropdown(visible=False, value="1.7B", choices=["1.7B"])
                    qwen_download_btn = gr.Button(visible=False)
                    load_qwen_btn = gr.Button(visible=False)
                    unload_qwen_btn = gr.Button(visible=False)
                    qwen_model_status = gr.Markdown(visible=False, value="")
                    qwen_download_status = gr.Textbox(visible=False, value="")
                    qwen_status = gr.Markdown(visible=False, value="❌ Not available")

            with gr.Row():
                # ChatterboxTTS Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🎤 **ChatterboxTTS**", elem_classes=["fade-in"])
                        chatterbox_status = gr.Markdown(
                            value="⭕ Not loaded" if CHATTERBOX_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_chatterbox_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=CHATTERBOX_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_chatterbox_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=CHATTERBOX_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

                # ChatterboxTTS Multilingual Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🌍 **Chatterbox Multi**", elem_classes=["fade-in"])
                        chatterbox_mtl_status = gr.Markdown(
                            value=(
                                "⭕ Not loaded"
                                if CHATTERBOX_MULTILINGUAL_AVAILABLE
                                else "❌ Not available"
                            ),
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_chatterbox_mtl_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=CHATTERBOX_MULTILINGUAL_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_chatterbox_mtl_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=CHATTERBOX_MULTILINGUAL_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

                # Chatterbox Turbo Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🚀 **Chatterbox Turbo**", elem_classes=["fade-in"])
                        chatterbox_turbo_status = gr.Markdown(
                            value=(
                                "⭕ Not loaded"
                                if CHATTERBOX_TURBO_AVAILABLE
                                else "❌ Not available"
                            ),
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_chatterbox_turbo_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=CHATTERBOX_TURBO_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_chatterbox_turbo_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=CHATTERBOX_TURBO_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

                # Kokoro TTS Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🗣️ **Kokoro TTS**", elem_classes=["fade-in"])
                        kokoro_status = gr.Markdown(
                            value="⭕ Not loaded" if KOKORO_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_kokoro_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=KOKORO_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_kokoro_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=KOKORO_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

                # Fish Speech Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🐟 **Fish Speech**", elem_classes=["fade-in"])
                        fish_status = gr.Markdown(
                            value="⭕ Not loaded" if FISH_SPEECH_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_fish_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=FISH_SPEECH_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_fish_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=FISH_SPEECH_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

                # IndexTTS Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🎯 **IndexTTS**", elem_classes=["fade-in"])
                        indextts_status = gr.Markdown(
                            value="⭕ Not loaded" if INDEXTTS_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_indextts_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=INDEXTTS_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_indextts_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=INDEXTTS_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

                # IndexTTS2 Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🎯 **IndexTTS2**", elem_classes=["fade-in"])
                        indextts2_status = gr.Markdown(
                            value="⭕ Not loaded" if INDEXTTS2_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_indextts2_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=INDEXTTS2_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_indextts2_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=INDEXTTS2_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

            # Second row for Higgs Audio and KittenTTS
            with gr.Row():
                # Higgs Audio Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🎙️ **Higgs Audio**", elem_classes=["fade-in"])
                        higgs_status = gr.Markdown(
                            value="⭕ Not loaded" if HIGGS_AUDIO_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_higgs_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=HIGGS_AUDIO_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_higgs_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=HIGGS_AUDIO_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

                # VoxCPM Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🎤 **VoxCPM 1.5**", elem_classes=["fade-in"])
                        voxcpm_status = gr.Markdown(
                            value="⭕ Not loaded" if VOXCPM_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_voxcpm_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=VOXCPM_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_voxcpm_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=VOXCPM_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

                # KittenTTS Management - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🐱 **KittenTTS**", elem_classes=["fade-in"])
                        kitten_status = gr.Markdown(
                            value="⭕ Not loaded" if KITTEN_TTS_AVAILABLE else "❌ Not available",
                            elem_classes=["fade-in"],
                        )
                    with gr.Row():
                        load_kitten_btn = gr.Button(
                            "🔄 Load",
                            variant="primary",
                            size="sm",
                            visible=KITTEN_TTS_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )
                        unload_kitten_btn = gr.Button(
                            "🗑️ Unload",
                            variant="secondary",
                            size="sm",
                            visible=KITTEN_TTS_AVAILABLE,
                            elem_classes=["fade-in"],
                            scale=1,
                        )

            # Third row for System Cleanup
            with gr.Row():
                # System Cleanup - Compact
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("🧹 **System Cleanup**", elem_classes=["fade-in"])
                        cleanup_status = gr.Markdown(
                            value="💾 Temp files ready", elem_classes=["fade-in"]
                        )
                    with gr.Row():
                        clear_temp_btn = gr.Button(
                            "🧹 Clear Temp Files",
                            variant="secondary",
                            size="sm",
                            elem_classes=["fade-in"],
                            scale=2,
                        )

        # Assistant status bar - compact row above workspace
        with gr.Row(elem_classes=["assistant-status-bar", "fade-in"]):
            assistant_status_indicator = gr.Markdown(
                value="🤖 Assistant: Not configured",
                elem_classes=["fade-in"],
            )

        # Main input section with tabs for single voice, conversation mode, and eBook conversion
        with gr.Row(elem_classes=["workspace-row"]):
            with gr.Column(scale=3, elem_classes=["main-workspace"]):
                # Tabs for different input modes
                with gr.Tabs(elem_classes=["fade-in"]) as input_tabs:
                    # Single Voice Tab
                    with gr.TabItem("📝 TEXT TO SYNTHESIZE", id="single_voice"):
                        # Text input with enhanced styling
                        text = gr.Textbox(
                            value="Hello! This is a demonstration of the ultimate TTS studio. You can choose between Chatterbox TTS. Fish Speech, VoxCPM, Index TTS and Index TTS 2, Higgs audio TTS and F5 TTS for custom voice cloning or Kitten TTS and Kokoro TTS for high-quality pre-trained voices and VibeVoice for podcast.",
                            label="📝 Text to synthesize",
                            lines=8,
                            max_lines=28,
                            placeholder="Enter your text here...",
                            elem_classes=["fade-in"],
                            elem_id="text_synthesize_input",
                        )

                        with gr.Accordion(
                            "🧠 Narration Transform (LLM)", open=False, elem_classes=["fade-in"]
                        ):
                            llm_transform_enabled = gr.Checkbox(
                                value=False,
                                label="Enable LLM narration transform before synthesis",
                                info="Applies optional text rewrite/normalization before sending text to the selected TTS engine.",
                            )

                            with gr.Row():
                                llm_provider = gr.Dropdown(
                                    label="LLM Provider",
                                    choices=list(LLM_PROVIDER_CONFIGS.keys()),
                                    value=current_llm_settings["provider"],
                                    info="Choose which AI service processes your text. Local providers (LM Studio, Ollama) run on your PC. Cloud providers need an API key.",
                                )

                            with gr.Row():
                                llm_model_id = gr.Dropdown(
                                    label="Model ID",
                                    choices=current_llm_settings["model_choices"],
                                    value=current_llm_settings["model_id"],
                                    info="The specific AI model to use. Click Refresh to see models available from your provider.",
                                    allow_custom_value=True,
                                    scale=4,
                                )
                                llm_refresh_models_btn = gr.Button(
                                    "🔄 Refresh Models",
                                    variant="secondary",
                                    scale=1,
                                    elem_classes=["fade-in"],
                                )

                            with gr.Row():
                                llm_base_url = gr.Textbox(
                                    label="Base URL",
                                    value=current_llm_settings["base_url"],
                                    placeholder="OpenAI-compatible base URL",
                                    info="The API endpoint URL. Auto-filled when you select a provider. Only change this if you have a custom setup.",
                                )
                                llm_api_key = gr.Textbox(
                                    label="API Key (required for cloud providers)",
                                    value=current_llm_settings["api_key"],
                                    type="password",
                                    placeholder="Optional in UI. Prefer shell env var for safety: GOOGLE_API_KEY or OPENAI_API_KEY",
                                    info="Required for cloud providers (Gemini, GitHub, Foundry). Can also be set as an environment variable for security.",
                                )

                            with gr.Row():
                                llm_content_type = gr.Dropdown(
                                    label="Content Type",
                                    choices=get_content_type_preset_names(),
                                    value=current_llm_settings["content_type"],
                                    info=llm_content_type_info,
                                )

                            with gr.Row():
                                llm_mode = gr.Radio(
                                    choices=["Minimal", "Polish", "Vivid"],
                                    value="Polish",
                                    label="Transform Mode",
                                    info="Minimal: fix only ambiguous items the automatic pass cannot resolve. Polish: normalize plus smooth phrasing and remove markdown. Vivid: Polish plus sparse emotional audio cues and expressive punctuation.",
                                )
                                llm_locale = gr.Dropdown(
                                    label="Locale",
                                    choices=[
                                        "en-US",
                                        "en-GB",
                                        "en-AU",
                                        "es-ES",
                                        "es-MX",
                                        "fr-FR",
                                        "de-DE",
                                        "it-IT",
                                        "pt-BR",
                                        "ja-JP",
                                        "ko-KR",
                                        "zh-CN",
                                        "zh-TW",
                                        "ru-RU",
                                        "ar-SA",
                                        "hi-IN",
                                        "nl-NL",
                                        "sv-SE",
                                        "pl-PL",
                                        "tr-TR",
                                    ],
                                    value="en-US",
                                    allow_custom_value=True,
                                    info="Target language and region. Affects how numbers, dates, and currency are spoken (e.g., en-US vs en-GB).",
                                )
                                llm_style = gr.Dropdown(
                                    label="Style",
                                    choices=[
                                        "cinematic_audiobook",
                                        "conversational",
                                        "news_broadcast",
                                        "documentary",
                                        "dramatic_reading",
                                        "bedtime_story",
                                        "podcast",
                                        "lecture",
                                        "meditation",
                                        "commercial",
                                    ],
                                    value="cinematic_audiobook",
                                    allow_custom_value=True,
                                    info="Sets the narration tone. The AI adapts pacing, emphasis, and delivery to match the chosen style.",
                                )

                            with gr.Row():
                                llm_max_tag_density = gr.Slider(
                                    0.0,
                                    1.0,
                                    step=0.05,
                                    value=0.35,
                                    label="Max tag density (tags per sentence)",
                                    info="Controls how many audio/voice direction tags the AI adds. Lower = cleaner text, higher = more expressive markup.",
                                )
                                llm_timeout_seconds = gr.Slider(
                                    10,
                                    180,
                                    step=5,
                                    value=60,
                                    label="LLM timeout (seconds)",
                                    info="Maximum wait time for the AI to respond. Increase for slower connections or larger texts.",
                                )

                            llm_allow_local_fallback = gr.Checkbox(
                                value=True,
                                label="Use local fallback when LLM is unavailable",
                                info="If the AI provider is unavailable, automatically use built-in text cleanup instead of failing.",
                            )

                            llm_preset = gr.Dropdown(
                                choices=list(LLM_OUTCOME_PRESETS.keys()),
                                value=current_llm_settings["preset"],
                                label="Outcome Preset",
                                info="Controls creativity vs faithfulness. Advanced users can override below.",
                            )

                            with gr.Accordion("Advanced LLM Parameters", open=False):
                                with gr.Row():
                                    llm_temperature = gr.Slider(
                                        0.0,
                                        1.2,
                                        step=0.05,
                                        value=current_llm_settings["temperature"],
                                        label="LLM temperature",
                                        info="Controls creativity. Low (0.0-0.3) = predictable and consistent. High (0.8+) = more creative and varied.",
                                    )
                                    llm_top_p = gr.Slider(
                                        0.1,
                                        1.0,
                                        step=0.05,
                                        value=current_llm_settings["top_p"],
                                        label="LLM top-p",
                                        info="Controls word choice diversity. Lower values make output more focused. Usually best left at 0.9.",
                                    )
                                    llm_max_tokens = gr.Slider(
                                        128,
                                        4096,
                                        step=64,
                                        value=current_llm_settings["max_tokens"],
                                        label="LLM max tokens",
                                        info="Maximum length of the AI's response. Increase for longer texts to avoid truncation.",
                                    )

                            llm_system_prompt = gr.Textbox(
                                label="LLM System Prompt",
                                lines=8,
                                max_lines=20,
                                value=current_llm_settings["system_prompt"],
                                info="Instructions that tell the AI how to transform your text. Edit to customize behavior, or click Reset to restore default.",
                            )

                            with gr.Row():
                                llm_test_btn = gr.Button(
                                    "🔌 Test LLM Connection", variant="secondary"
                                )
                                llm_apply_btn = gr.Button(
                                    "✨ Apply Transform to Text Box", variant="primary"
                                )
                                llm_prompt_reset_btn = gr.Button(
                                    "♻️ Reset Default Prompt", variant="secondary"
                                )

                            llm_connection_status = gr.Textbox(
                                label="LLM Connection Status",
                                lines=4,
                                interactive=False,
                                value="LLM transform disabled by default.",
                            )

                            with gr.Row(visible=False) as preview_row:
                                with gr.Column(scale=1):
                                    gr.Markdown("### Original")
                                    original_preview = gr.Textbox(
                                        label="Original Text",
                                        lines=10,
                                        max_lines=20,
                                        interactive=False,
                                    )
                                with gr.Column(scale=1):
                                    gr.Markdown("### Transformed")
                                    transformed_preview = gr.Textbox(
                                        label="Transformed Text",
                                        lines=10,
                                        max_lines=20,
                                        interactive=False,
                                    )

                            provenance_banner = gr.Markdown(visible=False)

                            with gr.Row(visible=False) as action_row:
                                accept_btn = gr.Button("✓ Accept", variant="primary", size="sm")
                                reject_btn = gr.Button("✗ Reject", variant="secondary", size="sm")

                            with gr.Accordion(
                                "📖 Pronunciation Glossary",
                                open=False,
                                elem_classes=["fade-in"],
                            ):
                                gr.Markdown(
                                    "Terms that will be preserved unchanged through AI narration transform.",
                                    elem_classes=["fade-in"],
                                )

                                protected_terms_df = gr.Dataframe(
                                    headers=["Term", "Case Sensitive"],
                                    datatype=["str", "bool"],
                                    value=glossary_protected_rows,
                                    interactive=True,
                                    wrap=True,
                                    label="Protected Terms",
                                    elem_classes=["fade-in"],
                                )

                                with gr.Row():
                                    protected_term_input = gr.Textbox(
                                        label="New Term",
                                        placeholder="Enter a term to preserve",
                                        elem_classes=["fade-in"],
                                        scale=3,
                                    )
                                    protected_term_case_sensitive = gr.Checkbox(
                                        value=True,
                                        label="Case Sensitive",
                                        elem_classes=["fade-in"],
                                        scale=1,
                                    )

                                with gr.Row():
                                    add_protected_term_btn = gr.Button(
                                        "➕ Add Term",
                                        variant="secondary",
                                        elem_classes=["fade-in"],
                                    )
                                    remove_protected_term_btn = gr.Button(
                                        "🗑️ Remove Selected",
                                        variant="secondary",
                                        elem_classes=["fade-in"],
                                    )

                                gr.Markdown(
                                    "Word-to-phonetic mappings applied before TTS synthesis.",
                                    elem_classes=["fade-in"],
                                )

                                pronunciation_overrides_df = gr.Dataframe(
                                    headers=["Word", "Phonetic", "Case Sensitive"],
                                    datatype=["str", "str", "bool"],
                                    value=glossary_override_rows,
                                    interactive=True,
                                    wrap=True,
                                    label="Pronunciation Overrides",
                                    elem_classes=["fade-in"],
                                )

                                with gr.Row():
                                    override_word_input = gr.Textbox(
                                        label="Word",
                                        placeholder="Enter the source word",
                                        elem_classes=["fade-in"],
                                        scale=2,
                                    )
                                    override_phonetic_input = gr.Textbox(
                                        label="Phonetic",
                                        placeholder="Enter the phonetic spelling",
                                        elem_classes=["fade-in"],
                                        scale=2,
                                    )
                                    override_case_sensitive = gr.Checkbox(
                                        value=False,
                                        label="Case Sensitive",
                                        elem_classes=["fade-in"],
                                        scale=1,
                                    )

                                with gr.Row():
                                    add_override_btn = gr.Button(
                                        "➕ Add Override",
                                        variant="secondary",
                                        elem_classes=["fade-in"],
                                    )
                                    remove_override_btn = gr.Button(
                                        "🗑️ Remove Selected",
                                        variant="secondary",
                                        elem_classes=["fade-in"],
                                    )

                                with gr.Row():
                                    save_glossary_btn = gr.Button(
                                        "💾 Save Glossary",
                                        variant="primary",
                                        elem_classes=["fade-in"],
                                    )
                                    load_glossary_btn = gr.Button(
                                        "📂 Load Glossary",
                                        variant="secondary",
                                        elem_classes=["fade-in"],
                                    )
                                    clear_glossary_btn = gr.Button(
                                        "🗑️ Clear All",
                                        variant="secondary",
                                        elem_classes=["fade-in"],
                                    )

                                glossary_status = gr.Textbox(
                                    label="Glossary Status",
                                    lines=3,
                                    interactive=False,
                                    value=glossary_status_message,
                                    elem_classes=["fade-in"],
                                )

                    # Conversation Mode Tab
                    with gr.TabItem("🎭 CONVERSATION MODE", id="conversation_mode"):
                        conversation_speakers_state = gr.State(value=[])
                        conversation_rows_state = gr.State(value=[])
                        conversation_selected_speaker_state = gr.State(value=None)
                        conversation_selected_line_state = gr.State(value=None)
                        conversation_speaker_settings_state = gr.State(value={})

                        kitten_conversation_voice_choices = [
                            "expr-voice-2-m",
                            "expr-voice-2-f",
                            "expr-voice-3-m",
                            "expr-voice-3-f",
                            "expr-voice-4-m",
                            "expr-voice-4-f",
                            "expr-voice-5-m",
                            "expr-voice-5-f",
                        ]

                        with gr.Column():
                            conversation_script = gr.Textbox(
                                label="📝 Conversation Script",
                                placeholder="""Enter conversation in this format:

Alice: Hello there! How are you doing today?
Bob: I'm doing great, thanks for asking! How about you?
Alice: I'm wonderful! I just got back from vacation.
Bob: That sounds amazing! Where did you go?
Alice: I went to Japan. It was absolutely incredible!""",
                                lines=8,
                                info="Format: 'SpeakerName: Text' - Each line should start with speaker name followed by a colon.",
                                elem_classes=["fade-in"],
                            )

                            with gr.Row():
                                analyze_script_btn = gr.Button(
                                    "🔍 Analyze Script",
                                    variant="secondary",
                                    elem_classes=["fade-in"],
                                )
                                ai_format_script_btn = gr.Button(
                                    "✨ AI Format",
                                    variant="secondary",
                                    elem_classes=["fade-in"],
                                )
                                example_script_btn = gr.Button(
                                    "📋 Load Example",
                                    variant="secondary",
                                    elem_classes=["fade-in"],
                                )
                                clear_script_btn = gr.Button(
                                    "🗑️ Clear Script",
                                    variant="secondary",
                                    elem_classes=["fade-in"],
                                )

                            with gr.Group(
                                visible=False, elem_classes=["fade-in"]
                            ) as conversation_workspace:
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        with gr.Row():
                                            speaker_profile_selector = gr.Dropdown(
                                                label="Speaker Profile",
                                                choices=get_speaker_profile_choices(),
                                                value=None,
                                                interactive=True,
                                                scale=3,
                                            )
                                            save_speaker_profile_btn = gr.Button(
                                                "💾 Save Profile",
                                                scale=1,
                                            )
                                            delete_speaker_profile_btn = gr.Button(
                                                "🗑️ Delete",
                                                scale=1,
                                            )
                                            cast_characters_btn = gr.Button(
                                                "🎭 Cast Characters",
                                                variant="secondary",
                                                scale=1,
                                                elem_classes=["fade-in"],
                                            )

                                        speaker_profile_name_input = gr.Textbox(
                                            label="Profile Name",
                                            placeholder="Enter a name for this speaker profile...",
                                            elem_classes=["fade-in"],
                                        )

                                        speaker_profile_status = gr.Textbox(
                                            label="Profile Status",
                                            interactive=False,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        )

                                        detected_speakers = gr.Textbox(
                                            label="🔍 Character Roster Summary",
                                            value="No speakers detected",
                                            interactive=False,
                                            lines=8,
                                            elem_classes=["fade-in"],
                                        )
                                        character_roster = gr.Radio(
                                            label="👥 Character Roster",
                                            choices=[],
                                            value=None,
                                            info="Select a character to edit their voice settings.",
                                            elem_classes=["fade-in"],
                                        )
                                        gr.Markdown(
                                            """
                                        <div style='padding: 10px; background: rgba(102, 126, 234, 0.05); border-radius: 8px; border-left: 3px solid #667eea;'>
                                            <p style='margin: 0; font-size: 0.85em; opacity: 0.8;'>
                                                <strong>💡 Guided flow:</strong><br/>
                                                1. Analyze or AI-format the script<br/>
                                                2. Pick a character in the roster<br/>
                                                3. Configure the voice for that character<br/>
                                                4. Select a script line below to edit it in context
                                            </p>
                                        </div>
                                        """
                                        )

                                    with gr.Column(scale=2):
                                        selected_character_header = gr.Markdown(
                                            "### Select a character",
                                            elem_classes=["fade-in"],
                                        )
                                        selected_character_hint = gr.Markdown(
                                            "Analyze a script to populate the roster, then choose a speaker to edit.",
                                            elem_classes=["fade-in"],
                                        )
                                        selected_character_capabilities = gr.Markdown(
                                            "",
                                            elem_classes=["fade-in"],
                                        )

                                        with gr.Group(
                                            visible=False, elem_classes=["fade-in"]
                                        ) as speaker_1_group:
                                            gr.Markdown("**🎤 Voice Clone Setup**")
                                            speaker_1_audio = gr.Audio(
                                                sources=["upload", "microphone"],
                                                type="filepath",
                                                label="Voice Sample",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_1_transcribe_btn = gr.Button(
                                                "📝 Transcribe", size="sm"
                                            )
                                            speaker_1_ref_text = gr.Textbox(
                                                label="Reference Text",
                                                placeholder="Transcribed text will appear here...",
                                                lines=2,
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Group(
                                            visible=False, elem_classes=["fade-in"]
                                        ) as speaker_2_group:
                                            gr.Markdown("**🎤 Voice Clone Setup**")
                                            speaker_2_audio = gr.Audio(
                                                sources=["upload", "microphone"],
                                                type="filepath",
                                                label="Voice Sample",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_2_transcribe_btn = gr.Button(
                                                "📝 Transcribe", size="sm"
                                            )
                                            speaker_2_ref_text = gr.Textbox(
                                                label="Reference Text",
                                                placeholder="Transcribed text will appear here...",
                                                lines=2,
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Group(
                                            visible=False, elem_classes=["fade-in"]
                                        ) as speaker_3_group:
                                            gr.Markdown("**🎤 Voice Clone Setup**")
                                            speaker_3_audio = gr.Audio(
                                                sources=["upload", "microphone"],
                                                type="filepath",
                                                label="Voice Sample",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_3_transcribe_btn = gr.Button(
                                                "📝 Transcribe", size="sm"
                                            )
                                            speaker_3_ref_text = gr.Textbox(
                                                label="Reference Text",
                                                placeholder="Transcribed text will appear here...",
                                                lines=2,
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Group(
                                            visible=False, elem_classes=["fade-in"]
                                        ) as speaker_4_group:
                                            gr.Markdown("**🎤 Voice Clone Setup**")
                                            speaker_4_audio = gr.Audio(
                                                sources=["upload", "microphone"],
                                                type="filepath",
                                                label="Voice Sample",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_4_transcribe_btn = gr.Button(
                                                "📝 Transcribe", size="sm"
                                            )
                                            speaker_4_ref_text = gr.Textbox(
                                                label="Reference Text",
                                                placeholder="Transcribed text will appear here...",
                                                lines=2,
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Group(
                                            visible=False, elem_classes=["fade-in"]
                                        ) as speaker_5_group:
                                            gr.Markdown("**🎤 Voice Clone Setup**")
                                            speaker_5_audio = gr.Audio(
                                                sources=["upload", "microphone"],
                                                type="filepath",
                                                label="Voice Sample",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_5_transcribe_btn = gr.Button(
                                                "📝 Transcribe", size="sm"
                                            )
                                            speaker_5_ref_text = gr.Textbox(
                                                label="Reference Text",
                                                placeholder="Transcribed text will appear here...",
                                                lines=2,
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Accordion(
                                            "🗣️ Speaker 1 Kokoro Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_1_kokoro_accordion:
                                            speaker_1_kokoro_voice = gr.Radio(
                                                choices=[
                                                    (k, v)
                                                    for k, v in update_kokoro_voice_choices().items()
                                                ],
                                                value="af_heart",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🗣️ Speaker 2 Kokoro Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_2_kokoro_accordion:
                                            speaker_2_kokoro_voice = gr.Radio(
                                                choices=[
                                                    (k, v)
                                                    for k, v in update_kokoro_voice_choices().items()
                                                ],
                                                value="am_adam",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🗣️ Speaker 3 Kokoro Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_3_kokoro_accordion:
                                            speaker_3_kokoro_voice = gr.Radio(
                                                choices=[
                                                    (k, v)
                                                    for k, v in update_kokoro_voice_choices().items()
                                                ],
                                                value="bf_emma",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🗣️ Speaker 4 Kokoro Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_4_kokoro_accordion:
                                            speaker_4_kokoro_voice = gr.Radio(
                                                choices=[
                                                    (k, v)
                                                    for k, v in update_kokoro_voice_choices().items()
                                                ],
                                                value="bm_lewis",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🗣️ Speaker 5 Kokoro Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_5_kokoro_accordion:
                                            speaker_5_kokoro_voice = gr.Radio(
                                                choices=[
                                                    (k, v)
                                                    for k, v in update_kokoro_voice_choices().items()
                                                ],
                                                value="af_sarah",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🐱 Speaker 1 KittenTTS Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_1_kitten_accordion:
                                            speaker_1_kitten_voice = gr.Radio(
                                                choices=kitten_conversation_voice_choices,
                                                value="expr-voice-2-f",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🐱 Speaker 2 KittenTTS Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_2_kitten_accordion:
                                            speaker_2_kitten_voice = gr.Radio(
                                                choices=kitten_conversation_voice_choices,
                                                value="expr-voice-2-m",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🐱 Speaker 3 KittenTTS Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_3_kitten_accordion:
                                            speaker_3_kitten_voice = gr.Radio(
                                                choices=kitten_conversation_voice_choices,
                                                value="expr-voice-3-f",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🐱 Speaker 4 KittenTTS Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_4_kitten_accordion:
                                            speaker_4_kitten_voice = gr.Radio(
                                                choices=kitten_conversation_voice_choices,
                                                value="expr-voice-3-m",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🐱 Speaker 5 KittenTTS Voice",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_5_kitten_accordion:
                                            speaker_5_kitten_voice = gr.Radio(
                                                choices=kitten_conversation_voice_choices,
                                                value="expr-voice-4-f",
                                                label="",
                                                elem_classes=["voice-grid"],
                                                show_label=False,
                                            )

                                        with gr.Accordion(
                                            "🎭 Speaker 1 IndexTTS2 Emotions",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_1_indextts2_accordion:
                                            speaker_1_emotion_mode = gr.Radio(
                                                choices=[
                                                    ("🎵 Audio Reference", "audio_reference"),
                                                    ("🎛️ Manual Control", "vector_control"),
                                                    ("📝 Text Description", "text_description"),
                                                ],
                                                value="audio_reference",
                                                label="Emotion Control Mode",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_1_emotion_audio = gr.Audio(
                                                sources=["upload"],
                                                type="filepath",
                                                label="🎵 Emotion Reference Audio",
                                                visible=True,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_1_emotion_description = gr.Textbox(
                                                label="📝 Emotion Description",
                                                placeholder="e.g., 'happy and excited', 'sad and melancholic'",
                                                visible=False,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_1_emotion_vectors = gr.Group(
                                                visible=False, elem_classes=["fade-in"]
                                            )
                                            with speaker_1_emotion_vectors:
                                                gr.Markdown("**🎛️ Emotion Intensity Controls**")
                                                with gr.Row():
                                                    speaker_1_happy = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😊 Happy"
                                                    )
                                                    speaker_1_sad = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😢 Sad"
                                                    )
                                                with gr.Row():
                                                    speaker_1_angry = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😠 Angry"
                                                    )
                                                    speaker_1_afraid = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😨 Afraid"
                                                    )
                                                with gr.Row():
                                                    speaker_1_surprised = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😲 Surprised"
                                                    )
                                                    speaker_1_calm = gr.Slider(
                                                        0, 1, 1, step=0.1, label="😌 Calm"
                                                    )

                                        with gr.Accordion(
                                            "🎭 Speaker 2 IndexTTS2 Emotions",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_2_indextts2_accordion:
                                            speaker_2_emotion_mode = gr.Radio(
                                                choices=[
                                                    ("🎵 Audio Reference", "audio_reference"),
                                                    ("🎛️ Manual Control", "vector_control"),
                                                    ("📝 Text Description", "text_description"),
                                                ],
                                                value="audio_reference",
                                                label="Emotion Control Mode",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_2_emotion_audio = gr.Audio(
                                                sources=["upload"],
                                                type="filepath",
                                                label="🎵 Emotion Reference Audio",
                                                visible=True,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_2_emotion_description = gr.Textbox(
                                                label="📝 Emotion Description",
                                                placeholder="e.g., 'happy and excited', 'sad and melancholic'",
                                                visible=False,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_2_emotion_vectors = gr.Group(
                                                visible=False, elem_classes=["fade-in"]
                                            )
                                            with speaker_2_emotion_vectors:
                                                gr.Markdown("**🎛️ Emotion Intensity Controls**")
                                                with gr.Row():
                                                    speaker_2_happy = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😊 Happy"
                                                    )
                                                    speaker_2_sad = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😢 Sad"
                                                    )
                                                with gr.Row():
                                                    speaker_2_angry = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😠 Angry"
                                                    )
                                                    speaker_2_afraid = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😨 Afraid"
                                                    )
                                                with gr.Row():
                                                    speaker_2_surprised = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😲 Surprised"
                                                    )
                                                    speaker_2_calm = gr.Slider(
                                                        0, 1, 1, step=0.1, label="😌 Calm"
                                                    )

                                        with gr.Accordion(
                                            "🎭 Speaker 3 IndexTTS2 Emotions",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_3_indextts2_accordion:
                                            speaker_3_emotion_mode = gr.Radio(
                                                choices=[
                                                    ("🎵 Audio Reference", "audio_reference"),
                                                    ("🎛️ Manual Control", "vector_control"),
                                                    ("📝 Text Description", "text_description"),
                                                ],
                                                value="audio_reference",
                                                label="Emotion Control Mode",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_3_emotion_audio = gr.Audio(
                                                sources=["upload"],
                                                type="filepath",
                                                label="🎵 Emotion Reference Audio",
                                                visible=True,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_3_emotion_description = gr.Textbox(
                                                label="📝 Emotion Description",
                                                placeholder="e.g., 'happy and excited', 'sad and melancholic'",
                                                visible=False,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_3_emotion_vectors = gr.Group(
                                                visible=False, elem_classes=["fade-in"]
                                            )
                                            with speaker_3_emotion_vectors:
                                                gr.Markdown("**🎛️ Emotion Intensity Controls**")
                                                with gr.Row():
                                                    speaker_3_happy = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😊 Happy"
                                                    )
                                                    speaker_3_sad = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😢 Sad"
                                                    )
                                                with gr.Row():
                                                    speaker_3_angry = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😠 Angry"
                                                    )
                                                    speaker_3_afraid = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😨 Afraid"
                                                    )
                                                with gr.Row():
                                                    speaker_3_surprised = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😲 Surprised"
                                                    )
                                                    speaker_3_calm = gr.Slider(
                                                        0, 1, 1, step=0.1, label="😌 Calm"
                                                    )

                                        with gr.Accordion(
                                            "🎭 Speaker 4 IndexTTS2 Emotions",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_4_indextts2_accordion:
                                            speaker_4_emotion_mode = gr.Radio(
                                                choices=[
                                                    ("🎵 Audio Reference", "audio_reference"),
                                                    ("🎛️ Manual Control", "vector_control"),
                                                    ("📝 Text Description", "text_description"),
                                                ],
                                                value="audio_reference",
                                                label="Emotion Control Mode",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_4_emotion_audio = gr.Audio(
                                                sources=["upload"],
                                                type="filepath",
                                                label="🎵 Emotion Reference Audio",
                                                visible=True,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_4_emotion_description = gr.Textbox(
                                                label="📝 Emotion Description",
                                                placeholder="e.g., 'happy and excited', 'sad and melancholic'",
                                                visible=False,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_4_emotion_vectors = gr.Group(
                                                visible=False, elem_classes=["fade-in"]
                                            )
                                            with speaker_4_emotion_vectors:
                                                gr.Markdown("**🎛️ Emotion Intensity Controls**")
                                                with gr.Row():
                                                    speaker_4_happy = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😊 Happy"
                                                    )
                                                    speaker_4_sad = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😢 Sad"
                                                    )
                                                with gr.Row():
                                                    speaker_4_angry = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😠 Angry"
                                                    )
                                                    speaker_4_afraid = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😨 Afraid"
                                                    )
                                                with gr.Row():
                                                    speaker_4_surprised = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😲 Surprised"
                                                    )
                                                    speaker_4_calm = gr.Slider(
                                                        0, 1, 1, step=0.1, label="😌 Calm"
                                                    )

                                        with gr.Accordion(
                                            "🎭 Speaker 5 IndexTTS2 Emotions",
                                            open=True,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as speaker_5_indextts2_accordion:
                                            speaker_5_emotion_mode = gr.Radio(
                                                choices=[
                                                    ("🎵 Audio Reference", "audio_reference"),
                                                    ("🎛️ Manual Control", "vector_control"),
                                                    ("📝 Text Description", "text_description"),
                                                ],
                                                value="audio_reference",
                                                label="Emotion Control Mode",
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_5_emotion_audio = gr.Audio(
                                                sources=["upload"],
                                                type="filepath",
                                                label="🎵 Emotion Reference Audio",
                                                visible=True,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_5_emotion_description = gr.Textbox(
                                                label="📝 Emotion Description",
                                                placeholder="e.g., 'happy and excited', 'sad and melancholic'",
                                                visible=False,
                                                elem_classes=["fade-in"],
                                            )
                                            speaker_5_emotion_vectors = gr.Group(
                                                visible=False, elem_classes=["fade-in"]
                                            )
                                            with speaker_5_emotion_vectors:
                                                gr.Markdown("**🎛️ Emotion Intensity Controls**")
                                                with gr.Row():
                                                    speaker_5_happy = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😊 Happy"
                                                    )
                                                    speaker_5_sad = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😢 Sad"
                                                    )
                                                with gr.Row():
                                                    speaker_5_angry = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😠 Angry"
                                                    )
                                                    speaker_5_afraid = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😨 Afraid"
                                                    )
                                                with gr.Row():
                                                    speaker_5_surprised = gr.Slider(
                                                        0, 1, 0, step=0.1, label="😲 Surprised"
                                                    )
                                                    speaker_5_calm = gr.Slider(
                                                        0, 1, 1, step=0.1, label="😌 Calm"
                                                    )

                                conversation_lines_df = gr.Dataframe(
                                    headers=["Line #", "Speaker", "Text"],
                                    datatype=["number", "str", "str"],
                                    value=[],
                                    interactive=False,
                                    wrap=True,
                                    elem_classes=["fade-in"],
                                    label="📜 Script Overview",
                                )

                                with gr.Group(
                                    visible=False, elem_classes=["fade-in"]
                                ) as line_editor_group:
                                    gr.Markdown("### ✏️ Selected Line Editor")
                                    with gr.Row():
                                        line_number_display = gr.Textbox(
                                            label="Line #",
                                            interactive=False,
                                            value="",
                                            elem_classes=["fade-in"],
                                            scale=1,
                                        )
                                        line_speaker_editor = gr.Dropdown(
                                            label="Speaker",
                                            choices=[],
                                            value=None,
                                            elem_classes=["fade-in"],
                                            scale=1,
                                        )
                                    line_text_editor = gr.Textbox(
                                        label="Line Text",
                                        lines=3,
                                        placeholder="Edit the selected line text here...",
                                        elem_classes=["fade-in"],
                                    )
                                    line_context_preview = gr.Markdown(
                                        "Select a line above to edit it in context.",
                                        elem_classes=["fade-in"],
                                    )
                                    with gr.Row():
                                        save_line_edit_btn = gr.Button(
                                            "💾 Save Line",
                                            variant="primary",
                                            elem_classes=["fade-in"],
                                        )
                                        revert_line_edit_btn = gr.Button(
                                            "↩️ Revert",
                                            variant="secondary",
                                            elem_classes=["fade-in"],
                                        )
                                    line_editor_status = gr.Markdown(
                                        "Select a line from the table to begin editing.",
                                        elem_classes=["fade-in"],
                                    )

                                with gr.Row():
                                    conversation_pause = gr.Slider(
                                        -0.5,
                                        2.0,
                                        step=0.1,
                                        value=0.8,
                                        label="🔇 Speaker Change Pause (s)",
                                        info="Pause duration when speakers change (negative = overlap)",
                                        elem_classes=["fade-in"],
                                    )
                                    speaker_transition_pause = gr.Slider(
                                        -0.5,
                                        1.0,
                                        step=0.1,
                                        value=0.3,
                                        label="⏸️ Same Speaker Pause (s)",
                                        info="Pause when same speaker continues (negative = overlap)",
                                        elem_classes=["fade-in"],
                                    )

                                gr.Markdown(
                                    """
                                <div style='margin-top: 10px; padding: 10px; background: rgba(102, 126, 234, 0.05); border-radius: 8px; border-left: 3px solid #667eea;'>
                                    <p style='margin: 0; font-size: 0.85em; opacity: 0.8;'>
                                        <strong>💡 Voice setup notes:</strong><br/>
                                        • Voice-clone engines use uploaded speaker samples and optional reference text<br/>
                                        • Kokoro TTS and KittenTTS use built-in voices, so no sample upload is required<br/>
                                        • IndexTTS2 combines voice cloning with emotion control for the selected speaker<br/>
                                        • The Generate Conversation button still uses the same backend dispatch as before
                                    </p>
                                </div>
                                """
                                )

                        conversation_component_state_inputs = [
                            speaker_1_audio,
                            speaker_2_audio,
                            speaker_3_audio,
                            speaker_4_audio,
                            speaker_5_audio,
                            speaker_1_ref_text,
                            speaker_2_ref_text,
                            speaker_3_ref_text,
                            speaker_4_ref_text,
                            speaker_5_ref_text,
                            speaker_1_kokoro_voice,
                            speaker_2_kokoro_voice,
                            speaker_3_kokoro_voice,
                            speaker_4_kokoro_voice,
                            speaker_5_kokoro_voice,
                            speaker_1_kitten_voice,
                            speaker_2_kitten_voice,
                            speaker_3_kitten_voice,
                            speaker_4_kitten_voice,
                            speaker_5_kitten_voice,
                            speaker_1_emotion_mode,
                            speaker_2_emotion_mode,
                            speaker_3_emotion_mode,
                            speaker_4_emotion_mode,
                            speaker_5_emotion_mode,
                        ]

                    # eBook to Audiobook Tab
                    with gr.TabItem("📚 EBOOK TO AUDIOBOOK", id="ebook_mode"):
                        if EBOOK_CONVERTER_AVAILABLE:
                            gr.Markdown(
                                """
                            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                                        padding: 15px; border-radius: 12px; margin-bottom: 15px;'>
                                <h3 style='margin: 0 0 8px 0; padding: 0; font-size: 1.1em;'>📖 Convert eBooks to Audiobooks</h3>
                                <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>
                                    Upload your eBook (.epub, .pdf, .txt, .html) and convert it to an audiobook using any TTS engine.
                                    .html files work best for automatic chapter detection.
                                </p>
                            </div>
                            """
                            )

                            with gr.Row():
                                with gr.Column(scale=2):
                                    # File upload
                                    ebook_file = gr.File(
                                        label="📁 Upload eBook File",
                                        file_types=[
                                            ".epub",
                                            ".pdf",
                                            ".txt",
                                            ".html",
                                            ".htm",
                                            ".rtf",
                                            ".fb2",
                                            ".odt",
                                        ],
                                        elem_classes=["fade-in"],
                                    )

                                    # Analysis button and results
                                    with gr.Row():
                                        analyze_btn = gr.Button(
                                            "🔍 Analyze eBook",
                                            variant="secondary",
                                            elem_classes=["fade-in"],
                                        )
                                        convert_ebook_btn = gr.Button(
                                            "🎧 Convert to Audiobook",
                                            variant="primary",
                                            elem_classes=["fade-in"],
                                            elem_id="convert_ebook_btn",
                                        )
                                        clear_ebook_btn = gr.Button(
                                            "🗑️ Clear", variant="secondary", elem_classes=["fade-in"]
                                        )

                                    # eBook information display
                                    ebook_info = gr.Markdown(
                                        value="Upload an eBook file and click 'Analyze eBook' to see details.\n\n💡 **Tip:** To use voice cloning, upload reference audio in the **ChatterboxTTS tab** before converting.",
                                        elem_classes=["fade-in"],
                                    )

                                    # Chapter selection
                                    chapter_selection = gr.CheckboxGroup(
                                        label="📋 Select Chapters to Convert (leave empty for all)",
                                        choices=[],
                                        value=[],
                                        visible=False,
                                        elem_classes=["fade-in"],
                                    )

                                with gr.Column(scale=1):
                                    # Conversion settings
                                    gr.Markdown("**⚙️ Conversion Settings**")

                                    ebook_tts_engine = gr.Radio(
                                        choices=[
                                            ("🎤 ChatterboxTTS", "ChatterboxTTS"),
                                            (
                                                "🌍 Chatterbox Multilingual",
                                                "Chatterbox Multilingual",
                                            ),
                                            ("🚀 Chatterbox Turbo", "Chatterbox Turbo"),
                                            ("🗣️ Kokoro TTS", "Kokoro TTS"),
                                            ("🐟 Fish Speech", "Fish Speech"),
                                            ("🎯 IndexTTS", "IndexTTS"),
                                            ("🎯 IndexTTS2", "IndexTTS2"),
                                            ("🎵 F5-TTS", "F5-TTS"),
                                            ("🎙️ Higgs Audio", "Higgs Audio"),
                                            ("🐱 KittenTTS", "KittenTTS"),
                                            ("🎙️ Qwen Voice Clone", "Qwen Voice Clone"),
                                        ],
                                        value=(
                                            "ChatterboxTTS"
                                            if CHATTERBOX_AVAILABLE
                                            else (
                                                "Chatterbox Turbo"
                                                if CHATTERBOX_TURBO_AVAILABLE
                                                else (
                                                    "Kokoro TTS"
                                                    if KOKORO_AVAILABLE
                                                    else (
                                                        "Fish Speech"
                                                        if FISH_SPEECH_AVAILABLE
                                                        else (
                                                            "IndexTTS"
                                                            if INDEXTTS_AVAILABLE
                                                            else (
                                                                "F5-TTS"
                                                                if F5_TTS_AVAILABLE
                                                                else (
                                                                    "Higgs Audio"
                                                                    if HIGGS_AUDIO_AVAILABLE
                                                                    else (
                                                                        "KittenTTS"
                                                                        if KITTEN_TTS_AVAILABLE
                                                                        else "Qwen Voice Clone"
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        ),
                                        label="🎯 TTS Engine for Audiobook",
                                        elem_classes=["fade-in"],
                                    )

                                    # Audio Format for eBook conversion
                                    ebook_audio_format = gr.Radio(
                                        choices=[
                                            ("🎵 WAV - Uncompressed (High Quality)", "wav"),
                                            ("🎶 MP3 - Compressed (Smaller Size)", "mp3"),
                                        ],
                                        value="wav",
                                        label="🎵 Audiobook Format",
                                        info="Choose format: WAV for best quality, MP3 for smaller file size",
                                        elem_classes=["fade-in"],
                                    )

                                    ebook_chunk_length = gr.Slider(
                                        300,
                                        800,
                                        step=50,
                                        label="📄 Text Chunk Length",
                                        value=500,
                                        info="Characters per TTS chunk",
                                        elem_classes=["fade-in"],
                                    )

                                    # Chunk timing controls for eBook conversion
                                    with gr.Accordion(
                                        "⏱️ Chunk Timing Controls",
                                        open=True,
                                        elem_classes=["fade-in"],
                                    ):
                                        gr.Markdown(
                                            """
                                        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                                                    padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                                            <p style='margin: 0; opacity: 0.8; font-size: 0.85em;'>
                                                🔇 Control the silence duration between chunks and chapters in your audiobook
                                            </p>
                                        </div>
                                        """
                                        )

                                        ebook_chunk_gap = gr.Slider(
                                            0.0,
                                            3.0,
                                            step=0.1,
                                            label="🔇 Gap Between Chunks (seconds)",
                                            value=1.0,
                                            info="Silence duration between text chunks within the same chapter",
                                            elem_classes=["fade-in"],
                                        )

                                        ebook_chapter_gap = gr.Slider(
                                            0.0,
                                            5.0,
                                            step=0.1,
                                            label="📖 Gap Between Chapters (seconds)",
                                            value=2.0,
                                            info="Silence duration when transitioning between chapters",
                                            elem_classes=["fade-in"],
                                        )

                            # Supported formats info
                            supported_formats = (
                                get_supported_formats() if EBOOK_CONVERTER_AVAILABLE else {}
                            )
                            gr.Markdown(
                                f"""
                            <div style='margin-top: 15px; padding: 12px; background: rgba(102, 126, 234, 0.05); border-radius: 8px; border-left: 3px solid #667eea;'>
                                <p style='margin: 0; font-size: 0.85em; opacity: 0.8;'>
                                    <strong>📋 Supported Formats:</strong> {', '.join(supported_formats.keys()) if supported_formats else 'N/A'}<br/>
                                    <strong>💡 Best Results:</strong> .html files work best for automatic chapter detection.<br/>
                                    <strong>⚡ Performance:</strong> Large books may take several minutes to convert depending on length and TTS engine.<br/>
                                    <strong>📁 Large Files:</strong> Audiobooks >50MB or >30min will be saved to the audiobooks folder with a download link (browser can't play very large files).<br/>
                                    <strong>🎧 Playback:</strong> Use VLC, Windows Media Player, or any audio player for large audiobooks.<br/>
                                    <strong>🐟 Fish Speech:</strong> Maintains consistent voice throughout the entire audiobook using smart seed management and reference cloning.
                                </p>
                            </div>
                            """
                            )
                        else:
                            # Placeholder when eBook converter is not available
                            gr.Markdown(
                                """
                            <div style='text-align: center; padding: 40px; opacity: 0.5;'>
                                <h3>📚 eBook to Audiobook Converter</h3>
                                <p>⚠️ Not available - please install required dependencies:</p>
                                <code>pip install ebooklib PyPDF2 beautifulsoup4 chardet</code>
                            </div>
                            """
                            )
                            # Create dummy components to maintain interface consistency
                            ebook_file = gr.File(visible=False, value=None)
                            analyze_btn = gr.Button(visible=False)
                            convert_ebook_btn = gr.Button(visible=False)
                            clear_ebook_btn = gr.Button(visible=False)
                            ebook_info = gr.Markdown(visible=False, value="")
                            chapter_selection = gr.CheckboxGroup(
                                visible=False, choices=[], value=[]
                            )
                            ebook_tts_engine = gr.Radio(visible=False, choices=[], value=None)
                            ebook_audio_format = gr.Radio(
                                visible=False,
                                choices=[("WAV", "wav"), ("MP3", "mp3")],
                                value="wav",
                            )
                            ebook_chunk_length = gr.Slider(visible=False, value=500)
                            ebook_chunk_gap = gr.Slider(visible=False, value=1.0)
                            ebook_chapter_gap = gr.Slider(visible=False, value=2.0)

                    # VibeVoice Tab
                    with gr.TabItem("🎙️ VIBEVOICE", id="vibevoice_mode"):
                        if VIBEVOICE_AVAILABLE:
                            gr.Markdown(
                                """
                            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                                        padding: 15px; border-radius: 12px; margin-bottom: 15px;'>
                                <h3 style='margin: 0 0 8px 0; padding: 0; font-size: 1.1em;'>🎙️ VibeVoice Podcast Generation</h3>
                                <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>
                                    Generate high-quality multi-speaker podcasts and conversations using VibeVoice's advanced TTS technology.
                                    Upload voice samples and create natural-sounding dialogues.
                                </p>
                            </div>
                            """
                            )

                            with gr.Row():
                                with gr.Column(scale=2):
                                    # Script input
                                    vibevoice_script = gr.Textbox(
                                        label="📝 Podcast Script",
                                        placeholder="Enter your podcast script here. Each line will be assigned to speakers in rotation.\n\nExample:\nWelcome to our podcast today!\nThanks for having me, it's great to be here.\nLet's dive into our topic...",
                                        lines=8,
                                        elem_classes=["fade-in"],
                                    )

                                    # Number of speakers
                                    vibevoice_num_speakers = gr.Slider(
                                        minimum=1,
                                        maximum=4,
                                        step=1,
                                        value=2,
                                        label="🎤 Number of Speakers",
                                        elem_classes=["fade-in"],
                                    )

                                    # Speaker voice selections
                                    with gr.Group():
                                        with gr.Row():
                                            gr.Markdown("**🎭 Speaker Voice Selection**")
                                            vibevoice_refresh_voices_btn = gr.Button(
                                                "🔄 Refresh Voices",
                                                variant="secondary",
                                                size="sm",
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Accordion(
                                            "🎤 Speaker 1 Voice",
                                            open=False,
                                            elem_classes=["fade-in"],
                                        ) as vibevoice_speaker_1_accordion:
                                            vibevoice_speaker_1 = gr.Radio(
                                                choices=(
                                                    get_vibevoice_voices()
                                                    if VIBEVOICE_AVAILABLE
                                                    else []
                                                ),
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Accordion(
                                            "🎤 Speaker 2 Voice",
                                            open=False,
                                            elem_classes=["fade-in"],
                                        ) as vibevoice_speaker_2_accordion:
                                            vibevoice_speaker_2 = gr.Radio(
                                                choices=(
                                                    get_vibevoice_voices()
                                                    if VIBEVOICE_AVAILABLE
                                                    else []
                                                ),
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Accordion(
                                            "🎤 Speaker 3 Voice",
                                            open=False,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as vibevoice_speaker_3_accordion:
                                            vibevoice_speaker_3 = gr.Radio(
                                                choices=(
                                                    get_vibevoice_voices()
                                                    if VIBEVOICE_AVAILABLE
                                                    else []
                                                ),
                                                elem_classes=["fade-in"],
                                            )

                                        with gr.Accordion(
                                            "🎤 Speaker 4 Voice",
                                            open=False,
                                            visible=False,
                                            elem_classes=["fade-in"],
                                        ) as vibevoice_speaker_4_accordion:
                                            vibevoice_speaker_4 = gr.Radio(
                                                choices=(
                                                    get_vibevoice_voices()
                                                    if VIBEVOICE_AVAILABLE
                                                    else []
                                                ),
                                                elem_classes=["fade-in"],
                                            )

                                with gr.Column(scale=1):
                                    # VibeVoice settings
                                    gr.Markdown("**⚙️ VibeVoice Settings**")

                                    vibevoice_cfg_scale = gr.Slider(
                                        minimum=0.1,
                                        maximum=3.0,
                                        step=0.1,
                                        value=1.3,
                                        label="🎛️ CFG Scale",
                                        info="Controls generation quality vs diversity",
                                        elem_classes=["fade-in"],
                                    )

                                    vibevoice_seed = gr.Number(
                                        label="🎲 Seed (optional)",
                                        value=44,
                                        precision=0,
                                        info="Leave empty for random generation",
                                        elem_classes=["fade-in"],
                                    )

                                    vibevoice_audio_format = gr.Radio(
                                        choices=[("WAV", "wav"), ("MP3", "mp3")],
                                        value="wav",
                                        label="🎵 Audio Format",
                                        info="Choose output audio format",
                                        elem_classes=["fade-in"],
                                    )

                                    # Model management
                                    with gr.Accordion("🤖 Model Management", open=False):
                                        vibevoice_model_status = gr.Markdown(
                                            value=(
                                                get_vibevoice_status()
                                                if VIBEVOICE_AVAILABLE
                                                else "❌ VibeVoice not available"
                                            ),
                                            elem_classes=["fade-in"],
                                        )

                                        # Model download section
                                        with gr.Group():
                                            gr.Markdown("**📥 Download Models**")
                                            vibevoice_model_select = gr.Radio(
                                                choices=[
                                                    ("VIBEVOICE-1.5B (COMPACT)", "VibeVoice-1.5B"),
                                                    (
                                                        "VIBEVOICE-7B (HIGH QUALITY)",
                                                        "VibeVoice-Large",
                                                    ),
                                                ],
                                                value="VibeVoice-1.5B",
                                                label="Select Model to Download",
                                                elem_classes=["fade-in"],
                                            )

                                            vibevoice_download_btn = gr.Button(
                                                "📥 Download Model",
                                                variant="primary",
                                                interactive=True,
                                                elem_classes=["fade-in"],
                                            )

                                            vibevoice_download_status = gr.Markdown(
                                                value="", elem_classes=["fade-in"]
                                            )

                                        # Model loading section
                                        with gr.Group():
                                            gr.Markdown("**🔄 Load/Unload Models**")
                                            # Downloaded models selector and refresh
                                            with gr.Row():
                                                vibevoice_downloaded_models = gr.Radio(
                                                    label="📦 Downloaded Models",
                                                    choices=(
                                                        scan_vibevoice_models()
                                                        if VIBEVOICE_AVAILABLE
                                                        else []
                                                    ),
                                                    elem_classes=["fade-in"],
                                                    scale=3,
                                                )
                                                vibevoice_refresh_models_btn = gr.Button(
                                                    "🔄 Refresh Models",
                                                    variant="secondary",
                                                    elem_classes=["fade-in"],
                                                    scale=1,
                                                )
                                            vibevoice_model_path = gr.Textbox(
                                                label="📁 Model Path",
                                                value="models/VibeVoice-1.5B",
                                                elem_classes=["fade-in"],
                                            )

                                            vibevoice_flash_attention = gr.Checkbox(
                                                label="⚡ Use Flash Attention",
                                                value=False,
                                                info="Set this BEFORE loading the model. Requires compatible GPU, may not work on all systems.",
                                                elem_classes=["fade-in"],
                                            )

                                            with gr.Row():
                                                vibevoice_load_btn = gr.Button(
                                                    "🔄 Load Model",
                                                    variant="secondary",
                                                    elem_classes=["fade-in"],
                                                )
                                                vibevoice_unload_btn = gr.Button(
                                                    "🗑️ Unload Model",
                                                    variant="secondary",
                                                    elem_classes=["fade-in"],
                                                )
                                                gr.Markdown(
                                                    value=(
                                                        "**Note:** Unload works only before the first generation. "
                                                        "After generating once, restart the app to fully unload."
                                                    ),
                                                    elem_classes=["fade-in"],
                                                )

                                    # Custom voice upload
                                    with gr.Accordion(
                                        "🎤 Add Custom Voice (3 to 10 seconds)", open=False
                                    ):
                                        custom_voice_file = gr.File(
                                            label="📁 Upload Voice Sample",
                                            file_types=[".wav", ".mp3", ".flac", ".ogg"],
                                            elem_classes=["fade-in"],
                                        )
                                        vibevoice_custom_voice_name = gr.Textbox(
                                            label="🏷️ Voice Name",
                                            placeholder="Enter a name for this voice",
                                            elem_classes=["fade-in"],
                                        )
                                        add_voice_btn = gr.Button(
                                            "➕ Add Voice",
                                            variant="secondary",
                                            elem_classes=["fade-in"],
                                        )
                                        add_voice_status = gr.Markdown(
                                            value="", elem_classes=["fade-in"]
                                        )

                            # Generate button
                            vibevoice_generate_btn = gr.Button(
                                "🎙️ Generate Podcast",
                                variant="primary",
                                size="lg",
                                elem_classes=["generate-btn", "fade-in"],
                            )

                            # Output
                            vibevoice_output = gr.Audio(
                                label="🎧 Generated Podcast",
                                show_download_button=True,
                                elem_classes=["fade-in", "glow"],
                            )

                            vibevoice_status = gr.Textbox(
                                label="📊 Generation Status",
                                lines=6,
                                interactive=False,
                                elem_classes=["fade-in"],
                            )

                        else:
                            # Placeholder when VibeVoice is not available
                            gr.Markdown(
                                """
                            <div style='text-align: center; padding: 40px; opacity: 0.5;'>
                                <h3>🎙️ VibeVoice Podcast Generator</h3>
                                <p>⚠️ Not available - please install VibeVoice dependencies</p>
                            </div>
                            """
                            )
                            # Create dummy components
                            vibevoice_script = gr.Textbox(visible=False)
                            vibevoice_num_speakers = gr.Slider(visible=False, value=2)
                            vibevoice_refresh_voices_btn = gr.Button(visible=False)
                            vibevoice_speaker_1 = gr.Radio(visible=False, choices=[])
                            vibevoice_speaker_2 = gr.Radio(visible=False, choices=[])
                            vibevoice_speaker_3 = gr.Radio(visible=False, choices=[])
                            vibevoice_speaker_4 = gr.Radio(visible=False, choices=[])
                            vibevoice_cfg_scale = gr.Slider(visible=False, value=1.3)
                            vibevoice_seed = gr.Number(visible=False)
                            vibevoice_audio_format = gr.Radio(visible=False, value="wav")
                            vibevoice_model_status = gr.Markdown(visible=False)
                            vibevoice_model_select = gr.Radio(visible=False, choices=[])
                            vibevoice_download_btn = gr.Button(visible=False)
                            vibevoice_download_status = gr.Markdown(visible=False)
                            vibevoice_model_path = gr.Textbox(visible=False)
                            vibevoice_load_btn = gr.Button(visible=False)
                            vibevoice_unload_btn = gr.Button(visible=False)
                            custom_voice_file = gr.File(visible=False)
                            vibevoice_custom_voice_name = gr.Textbox(visible=False)
                            add_voice_btn = gr.Button(visible=False)
                            add_voice_status = gr.Markdown(visible=False)
                            vibevoice_generate_btn = gr.Button(visible=False)
                            vibevoice_output = gr.Audio(visible=False)
                            vibevoice_status = gr.Textbox(visible=False)

                    with gr.TabItem("🤖 ASSISTANT", id="assistant_mode"):
                        gr.Markdown(
                            """
                        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1));
                                    padding: 15px; border-radius: 12px; margin-bottom: 15px;'>
                            <h3 style='margin: 0 0 8px 0; padding: 0; font-size: 1.1em;'>🤖 TTS Studio Assistant</h3>
                            <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>
                                Ask questions about TTS engines, voice configuration, troubleshooting,
                                and workflow guidance. Powered by your configured LLM provider.
                            </p>
                        </div>
                        """
                        )

                        assistant_chatbot = gr.Chatbot(
                            label="💬 Assistant Chat",
                            height=400,
                            type="messages",
                            elem_classes=["fade-in"],
                        )

                        with gr.Row():
                            assistant_msg_input = gr.Textbox(
                                placeholder="Ask the assistant about TTS engines, voices, settings...",
                                label="Message",
                                lines=2,
                                scale=4,
                                elem_classes=["fade-in"],
                            )
                            assistant_send_btn = gr.Button(
                                "📤 Send",
                                variant="primary",
                                scale=1,
                                elem_classes=["fade-in"],
                            )

                        with gr.Row():
                            assistant_clear_btn = gr.Button(
                                "🗑️ Clear Chat",
                                variant="secondary",
                                size="sm",
                                elem_classes=["fade-in"],
                            )

                        with gr.Accordion(
                            "⚙️ Assistant LLM Settings", open=False, elem_classes=["fade-in"]
                        ):
                            assistant_llm_settings = get_initial_assistant_llm_settings()

                            assistant_llm_provider = gr.Dropdown(
                                choices=list(LLM_PROVIDER_CONFIGS.keys()),
                                value=assistant_llm_settings["provider"],
                                label="🔌 LLM Provider",
                                elem_classes=["fade-in"],
                            )

                            with gr.Row():
                                assistant_llm_base_url = gr.Textbox(
                                    value=assistant_llm_settings["base_url"],
                                    label="🌐 Base URL",
                                    scale=3,
                                    elem_classes=["fade-in"],
                                )
                                assistant_llm_api_key = gr.Textbox(
                                    value="",
                                    label="🔑 API Key",
                                    type="password",
                                    scale=2,
                                    info="Session-only. Not saved. Use env vars for persistence.",
                                    elem_classes=["fade-in"],
                                )

                            assistant_llm_model_id = gr.Dropdown(
                                choices=assistant_llm_settings["model_choices"],
                                value=assistant_llm_settings["model_id"],
                                label="🧠 Model",
                                allow_custom_value=True,
                                elem_classes=["fade-in"],
                            )

                            assistant_llm_system_prompt = gr.Textbox(
                                value=(
                                    assistant_llm_settings["system_prompt"]
                                    or DEFAULT_ASSISTANT_SYSTEM_PROMPT
                                ),
                                label="📝 System Prompt",
                                lines=4,
                                elem_classes=["fade-in"],
                            )

                            with gr.Row():
                                assistant_llm_test_btn = gr.Button(
                                    "🔗 Test Connection",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["fade-in"],
                                )
                                assistant_llm_save_btn = gr.Button(
                                    "💾 Save Settings",
                                    variant="secondary",
                                    size="sm",
                                    elem_classes=["fade-in"],
                                )

                            assistant_llm_status = gr.Markdown(value="", elem_classes=["fade-in"])

                    with gr.TabItem("📋 JOBS", id="jobs_mode"):
                        gr.Markdown(
                            """
                        <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.1));
                                    padding: 15px; border-radius: 12px; margin-bottom: 15px;'>
                            <h3 style='margin: 0 0 8px 0; padding: 0; font-size: 1.1em;'>📋 Job Queue</h3>
                            <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>
                                Monitor background TTS jobs, view queue status, and manage long-running
                                synthesis tasks without blocking the main UI.
                            </p>
                        </div>
                        """
                        )

                        job_queue_display = gr.Dataframe(
                            headers=[
                                "ID",
                                "Status",
                                "Engine",
                                "Created",
                                "Elapsed",
                                "Text Preview",
                            ],
                            datatype=["str", "str", "str", "str", "str", "str"],
                            value=[["—", "No jobs", "—", "—", "—", "—"]],
                            label="🔄 Active & Recent Jobs",
                            interactive=False,
                            wrap=True,
                            elem_classes=["fade-in"],
                        )

                        with gr.Row():
                            job_refresh_btn = gr.Button(
                                "🔄 Refresh",
                                variant="secondary",
                                size="sm",
                                elem_classes=["fade-in"],
                            )
                            job_auto_refresh = gr.Checkbox(
                                value=False,
                                label="Auto-refresh (every 3s)",
                                elem_classes=["fade-in"],
                            )

                        with gr.Row():
                            job_id_input = gr.Textbox(
                                label="Job ID",
                                placeholder="Enter a full or truncated job ID",
                                scale=3,
                                elem_classes=["fade-in"],
                            )
                            job_cancel_btn = gr.Button(
                                "❌ Cancel Job",
                                variant="stop",
                                size="sm",
                                scale=1,
                                elem_classes=["fade-in"],
                            )
                            job_retry_btn = gr.Button(
                                "🔁 Retry Job",
                                variant="secondary",
                                size="sm",
                                scale=1,
                                elem_classes=["fade-in"],
                            )

                        job_detail_output = gr.Markdown(
                            value="Select a job to view details.",
                            elem_classes=["fade-in"],
                        )

                        job_timer = gr.Timer(value=3, active=False)

            with gr.Column(scale=1, elem_classes=["right-rail"]):
                # Audio output section with glow effect
                audio_output = gr.Audio(
                    label="🎵 Generated Audio",
                    show_download_button=True,
                    elem_classes=["fade-in", "glow"],
                )

                # Status with custom styling
                status_output = gr.Markdown(value="Ready.", elem_classes=["fade-in"])

                last_seed_out = gr.Markdown(value="🎲 Last Seed: N/A", elem_classes=["fade-in"])

                last_seed_state = gr.State(value=None)

                # Conversation info output (visible for conversation mode)
                conversation_info = gr.Textbox(
                    label="📊 Conversation Summary",
                    lines=8,
                    interactive=False,
                    elem_classes=["fade-in"],
                    visible=False,
                    value="Ready for conversation generation...",
                )

                with gr.Accordion(
                    "📚 Audiobook Results",
                    open=False,
                    elem_classes=["fade-in"],
                    elem_id="audiobook_results_panel",
                ):
                    audiobook_output = gr.Audio(
                        label="🎧 Generated Audiobook",
                        show_download_button=True,
                        elem_classes=["fade-in", "glow"],
                    )

                    audiobook_download = gr.File(
                        label="📥 Download Large Audiobook", visible=False, elem_classes=["fade-in"]
                    )

                    ebook_status = gr.Textbox(
                        label="📊 eBook Conversion Status",
                        lines=6,
                        interactive=False,
                        elem_classes=["fade-in"],
                    )

        # Generate buttons - separate for single voice and conversation modes
        with gr.Row():
            with gr.Column():
                generate_btn = gr.Button(
                    "🚀 Generate Speech",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn", "fade-in"],
                    visible=True,
                )
            with gr.Column():
                generate_conversation_btn = gr.Button(
                    "🎭 Generate Conversation",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn", "fade-in"],
                    visible=False,
                )

        conversation_panel_updates = [
            speaker_1_group,
            speaker_2_group,
            speaker_3_group,
            speaker_4_group,
            speaker_5_group,
            speaker_1_kokoro_accordion,
            speaker_2_kokoro_accordion,
            speaker_3_kokoro_accordion,
            speaker_4_kokoro_accordion,
            speaker_5_kokoro_accordion,
            speaker_1_kitten_accordion,
            speaker_2_kitten_accordion,
            speaker_3_kitten_accordion,
            speaker_4_kitten_accordion,
            speaker_5_kitten_accordion,
            speaker_1_indextts2_accordion,
            speaker_2_indextts2_accordion,
            speaker_3_indextts2_accordion,
            speaker_4_indextts2_accordion,
            speaker_5_indextts2_accordion,
        ]

        conversation_analysis_outputs = [
            detected_speakers,
            generate_conversation_btn,
            conversation_workspace,
            character_roster,
            selected_character_header,
            selected_character_hint,
            selected_character_capabilities,
            conversation_lines_df,
            line_editor_group,
            line_editor_status,
            line_number_display,
            line_speaker_editor,
            line_text_editor,
            line_context_preview,
            conversation_speakers_state,
            conversation_rows_state,
            conversation_selected_speaker_state,
            conversation_selected_line_state,
            conversation_speaker_settings_state,
            *conversation_panel_updates,
        ]

        with gr.Accordion(
            "🧭 Workspace Controls",
            open=False,
            elem_classes=["fade-in"],
            elem_id="preset_autosave_panel",
        ):
            with gr.Row(elem_classes=["workspace-row"]):
                with gr.Column(scale=2):
                    speaker_name_tb = gr.Textbox(
                        value="Speakers Name", label="🗣️ Speaker Name", placeholder="Speakers Name"
                    )

                    with gr.Row():
                        voice_preset_dd = gr.Dropdown(
                            label="🎙️ Voice Preset",
                            choices=get_voice_preset_choices(),
                            value="",
                            allow_custom_value=True,
                            info="Select a preset to reuse reference audio for supported engines",
                        )
                        preset_name_tb = gr.Textbox(
                            label="🏷️ Preset Name", placeholder="Speakers Name_Style"
                        )

                    with gr.Row():
                        preset_audio_file = gr.File(label="📎 Preset Audio File", type="filepath")
                        copy_into_app_chk = gr.Checkbox(
                            value=True, label="📁 Copy audio into app_state/voices"
                        )

                    with gr.Row():
                        save_preset_btn = gr.Button("💾 Save / Update Preset", variant="secondary")
                        delete_preset_btn = gr.Button("🗑️ Delete Preset", variant="stop")
                        refresh_presets_btn = gr.Button("🔄 Refresh", variant="secondary")

                    preset_status_md = gr.Markdown(value="ℹ️ Preset manager ready")

                with gr.Column(scale=2):
                    with gr.Row():
                        autosave_enabled = gr.Checkbox(
                            value=True, label="💾 Autosave project files"
                        )
                        autosave_project_name = gr.Textbox(
                            value="default",
                            label="📚 Project Name",
                            placeholder="book_title_or_project",
                        )

                    with gr.Row():
                        autosave_store_audio_copy = gr.Checkbox(
                            value=True, label="🧬 Keep structured autosave audio copy"
                        )
                        keep_legacy_output_copy = gr.Checkbox(
                            value=True, label="📦 Keep legacy output copy"
                        )

                    output_storage_mode = gr.Dropdown(
                        label="📦 Generated Output Storage",
                        choices=["Project Folders (default)", "Custom Path"],
                        value=storage_mode_value,
                        info="Set where generated outputs/autosaves are stored; preset voices remain local",
                    )
                    output_storage_path = gr.Textbox(
                        label="🛣️ Custom Output Base Path",
                        value=current_storage_path,
                        placeholder="D:/Ultimate-TTS-Outputs",
                    )

                    with gr.Row():
                        save_storage_btn = gr.Button("💾 Apply Storage", variant="primary")
                        open_output_folder_btn = gr.Button(
                            "📂 Open Output Folder", variant="secondary"
                        )
                        open_autosave_folder_btn = gr.Button(
                            "🗂️ Open Autosave Folder", variant="secondary"
                        )

                    storage_status_md = gr.Markdown(value=storage_status_default)

        with gr.Accordion(
            "🎚️ Engine Selection",
            open=False,
            elem_classes=["fade-in"],
            elem_id="engine_selector_panel",
        ):
            with gr.Row():
                tts_engine = gr.Dropdown(
                    choices=[
                        ("🎤 ChatterboxTTS - Voice Cloning", "ChatterboxTTS"),
                        ("🌍 Chatterbox Multilingual - 23 Languages", "Chatterbox Multilingual"),
                        ("🚀 Chatterbox Turbo - Fast Voice Cloning", "Chatterbox Turbo"),
                        ("🗣️ Kokoro TTS - Pre-trained Voices", "Kokoro TTS"),
                        ("🐟 Fish Speech - Natural TTS", "Fish Speech"),
                        ("🎯 IndexTTS - Industrial Quality", "IndexTTS"),
                        ("🎯 IndexTTS2 - Advanced Emotion Control", "IndexTTS2"),
                        ("🎵 F5-TTS - Flow Matching TTS", "F5-TTS"),
                        ("🎙️ Higgs Audio - Advanced Multimodal TTS", "Higgs Audio"),
                        ("🎤 VoxCPM - Voice Cloning TTS", "VoxCPM"),
                        ("🐱 KittenTTS - Mini Model TTS", "KittenTTS"),
                        ("🎨 Qwen Voice Design - Create Voices", "Qwen Voice Design"),
                        ("🎭 Qwen Voice Clone - Clone Voices", "Qwen Voice Clone"),
                        ("🗣️ Qwen Custom Voice - Predefined Speakers", "Qwen Custom Voice"),
                    ],
                    value=(
                        "ChatterboxTTS"
                        if CHATTERBOX_AVAILABLE
                        else (
                            "Chatterbox Turbo"
                            if CHATTERBOX_TURBO_AVAILABLE
                            else (
                                "Kokoro TTS"
                                if KOKORO_AVAILABLE
                                else (
                                    "Fish Speech"
                                    if FISH_SPEECH_AVAILABLE
                                    else (
                                        "IndexTTS"
                                        if INDEXTTS_AVAILABLE
                                        else (
                                            "F5-TTS"
                                            if F5_TTS_AVAILABLE
                                            else (
                                                "Higgs Audio"
                                                if HIGGS_AUDIO_AVAILABLE
                                                else (
                                                    "VoxCPM"
                                                    if VOXCPM_AVAILABLE
                                                    else (
                                                        "KittenTTS"
                                                        if KITTEN_TTS_AVAILABLE
                                                        else "Qwen Voice Clone"
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    label="🎯 Select TTS Engine",
                    info="Use this only if you want to override auto-selection from loaded models",
                    elem_classes=["fade-in"],
                )

                audio_format = gr.Dropdown(
                    choices=[
                        ("🎵 WAV - Uncompressed (High Quality)", "wav"),
                        ("🎶 MP3 - Compressed (Smaller Size)", "mp3"),
                    ],
                    value="wav",
                    label="🎵 Audio Output Format",
                    info="WAV for quality, MP3 for smaller files",
                    elem_classes=["fade-in"],
                )

        conversation_analyze_inputs = [
            conversation_script,
            tts_engine,
            *conversation_component_state_inputs,
        ]

        engine_settings_toggle = gr.Button(
            "🛠️ TTS Engine Settings",
            variant="secondary",
            elem_classes=["fade-in"],
            elem_id="engine_settings_toggle",
        )

        with gr.Tabs(elem_classes=["fade-in"], elem_id="engine_settings_tabs") as engine_tabs:
            # ChatterboxTTS Tab
            with gr.TabItem("🎤 ChatterboxTTS", id="chatterbox_tab"):
                if CHATTERBOX_AVAILABLE:
                    with gr.Group() as chatterbox_controls:
                        gr.Markdown("**🎤 ChatterboxTTS - Voice cloning from reference audio**")
                        gr.Markdown(
                            "*💡 Try the sample file: `sample/Sample.wav`*",
                            elem_classes=["fade-in"],
                        )

                        with gr.Row():
                            with gr.Column(scale=2):
                                chatterbox_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio File (Optional)",
                                    value=None,
                                    elem_classes=["fade-in"],
                                )

                            with gr.Column(scale=1):
                                chatterbox_exaggeration = gr.Slider(
                                    0.25,
                                    2,
                                    step=0.05,
                                    label="🎭 Exaggeration",
                                    value=0.5,
                                    info="Higher = more dramatic",
                                    elem_classes=["fade-in"],
                                )
                                chatterbox_cfg_weight = gr.Slider(
                                    0.2,
                                    1,
                                    step=0.05,
                                    label="⚡ CFG Weight",
                                    value=0.5,
                                    info="Speed vs quality",
                                    elem_classes=["fade-in"],
                                )

                        with gr.Accordion(
                            "🔧 Advanced ChatterboxTTS Settings",
                            open=False,
                            elem_classes=["fade-in"],
                        ):
                            with gr.Row():
                                chatterbox_temperature = gr.Slider(
                                    0.05,
                                    5,
                                    step=0.05,
                                    label="🌡️ Temperature",
                                    value=0.8,
                                    info="Higher = more creative",
                                )
                                chatterbox_chunk_size = gr.Slider(
                                    100,
                                    400,
                                    step=25,
                                    label="📄 Chunk Size",
                                    value=300,
                                    info="Characters per chunk",
                                )
                                chatterbox_seed = gr.Number(
                                    value=0,
                                    label="🎲 Seed (0=random)",
                                    info="For reproducible results",
                                )
                else:
                    # Placeholder when ChatterboxTTS is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🎤 ChatterboxTTS** - ⚠️ Not available - please check installation</div>"
                        )
                        # Create dummy components to maintain consistent interface
                        chatterbox_ref_audio = gr.Audio(visible=False, value=None)
                        chatterbox_exaggeration = gr.Slider(visible=False, value=0.5)
                        chatterbox_temperature = gr.Slider(visible=False, value=0.8)
                        chatterbox_cfg_weight = gr.Slider(visible=False, value=0.5)
                        chatterbox_chunk_size = gr.Slider(visible=False, value=300)
                        chatterbox_seed = gr.Number(visible=False, value=0)

            # Chatterbox Multilingual Tab
            with gr.TabItem("🌍 Chatterbox Multilingual", id="chatterbox_mtl_tab"):
                if CHATTERBOX_MULTILINGUAL_AVAILABLE:
                    with gr.Group() as chatterbox_mtl_controls:
                        gr.Markdown(
                            "**🌍 Chatterbox Multilingual - Voice cloning in 23 languages**"
                        )
                        gr.Markdown(
                            "*💡 Supports: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish*",
                            elem_classes=["fade-in"],
                        )

                        with gr.Row():
                            with gr.Column(scale=2):
                                chatterbox_mtl_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio File (Optional)",
                                    value=None,
                                    elem_classes=["fade-in"],
                                )

                            with gr.Column(scale=1):
                                with gr.Accordion(
                                    "🌍 Language", open=False, elem_classes=["fade-in"]
                                ):
                                    chatterbox_mtl_language = gr.Radio(
                                        choices=[
                                            ("🇸🇦 Arabic", "ar"),
                                            ("🇨🇳 Chinese", "zh"),
                                            ("🇩🇰 Danish", "da"),
                                            ("🇳🇱 Dutch", "nl"),
                                            ("🇬🇧 English", "en"),
                                            ("🇫🇮 Finnish", "fi"),
                                            ("🇫🇷 French", "fr"),
                                            ("🇩🇪 German", "de"),
                                            ("🇬🇷 Greek", "el"),
                                            ("🇮🇱 Hebrew", "he"),
                                            ("🇮🇳 Hindi", "hi"),
                                            ("🇮🇹 Italian", "it"),
                                            ("🇯🇵 Japanese", "ja"),
                                            ("🇰🇷 Korean", "ko"),
                                            ("🇲🇾 Malay", "ms"),
                                            ("🇳🇴 Norwegian", "no"),
                                            ("🇵🇱 Polish", "pl"),
                                            ("🇵🇹 Portuguese", "pt"),
                                            ("🇷🇺 Russian", "ru"),
                                            ("🇪🇸 Spanish", "es"),
                                            ("🇹🇿 Swahili", "sw"),
                                            ("🇸🇪 Swedish", "sv"),
                                            ("🇹🇷 Turkish", "tr"),
                                        ],
                                        value="en",
                                        label="Select target language",
                                        elem_classes=["fade-in"],
                                    )
                                chatterbox_mtl_exaggeration = gr.Slider(
                                    0.25,
                                    2,
                                    step=0.05,
                                    label="🎭 Exaggeration",
                                    value=0.5,
                                    info="Higher = more dramatic",
                                    elem_classes=["fade-in"],
                                )

                        with gr.Accordion(
                            "🔧 Advanced Multilingual Settings",
                            open=False,
                            elem_classes=["fade-in"],
                        ):
                            with gr.Row():
                                chatterbox_mtl_temperature = gr.Slider(
                                    0.05,
                                    5,
                                    step=0.05,
                                    label="🌡️ Temperature",
                                    value=0.8,
                                    info="Higher = more creative",
                                )
                                chatterbox_mtl_cfg_weight = gr.Slider(
                                    0.0,
                                    1,
                                    step=0.05,
                                    label="⚡ CFG Weight",
                                    value=0.5,
                                    info="Speed vs quality",
                                )
                                chatterbox_mtl_chunk_size = gr.Slider(
                                    100,
                                    400,
                                    step=25,
                                    label="📄 Chunk Size",
                                    value=300,
                                    info="Characters per chunk",
                                )
                            with gr.Row():
                                chatterbox_mtl_repetition_penalty = gr.Slider(
                                    1.0,
                                    3.0,
                                    step=0.1,
                                    label="🔁 Repetition Penalty",
                                    value=2.0,
                                    info="Reduce repetitions",
                                )
                                chatterbox_mtl_min_p = gr.Slider(
                                    0.0,
                                    0.5,
                                    step=0.01,
                                    label="📊 Min P",
                                    value=0.05,
                                    info="Minimum probability threshold",
                                )
                                chatterbox_mtl_top_p = gr.Slider(
                                    0.5,
                                    1.0,
                                    step=0.05,
                                    label="🎯 Top P",
                                    value=1.0,
                                    info="Nucleus sampling",
                                )
                                chatterbox_mtl_seed = gr.Number(
                                    value=0,
                                    label="🎲 Seed (0=random)",
                                    info="For reproducible results",
                                )
                else:
                    # Placeholder when Chatterbox Multilingual is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🌍 Chatterbox Multilingual** - ⚠️ Not available - please check installation</div>"
                        )
                        # Create dummy components to maintain consistent interface
                        chatterbox_mtl_ref_audio = gr.Audio(visible=False, value=None)
                        chatterbox_mtl_language = gr.Radio(
                            visible=False, value="en", choices=[("English", "en")]
                        )
                        chatterbox_mtl_exaggeration = gr.Slider(visible=False, value=0.5)
                        chatterbox_mtl_temperature = gr.Slider(visible=False, value=0.8)
                        chatterbox_mtl_cfg_weight = gr.Slider(visible=False, value=0.5)
                        chatterbox_mtl_repetition_penalty = gr.Slider(visible=False, value=2.0)
                        chatterbox_mtl_min_p = gr.Slider(visible=False, value=0.05)
                        chatterbox_mtl_top_p = gr.Slider(visible=False, value=1.0)
                        chatterbox_mtl_chunk_size = gr.Slider(visible=False, value=300)
                        chatterbox_mtl_seed = gr.Number(visible=False, value=0)

            # Chatterbox Turbo Tab
            with gr.TabItem("🚀 Chatterbox Turbo", id="chatterbox_turbo_tab"):
                if CHATTERBOX_TURBO_AVAILABLE:
                    with gr.Group() as chatterbox_turbo_controls:
                        gr.Markdown("**🚀 Chatterbox Turbo - Fast distilled voice cloning**")
                        gr.Markdown(
                            "*💡 Faster inference with similar quality to standard Chatterbox*",
                            elem_classes=["fade-in"],
                        )

                        with gr.Row():
                            with gr.Column(scale=2):
                                chatterbox_turbo_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio File (Optional)",
                                    value=None,
                                    elem_classes=["fade-in"],
                                )

                            with gr.Column(scale=1):
                                chatterbox_turbo_exaggeration = gr.Slider(
                                    0.25,
                                    2,
                                    step=0.05,
                                    label="🎭 Exaggeration",
                                    value=0.5,
                                    info="Higher = more dramatic",
                                    elem_classes=["fade-in"],
                                )

                        with gr.Accordion(
                            "🔧 Advanced Turbo Settings", open=False, elem_classes=["fade-in"]
                        ):
                            with gr.Row():
                                chatterbox_turbo_temperature = gr.Slider(
                                    0.05,
                                    5,
                                    step=0.05,
                                    label="🌡️ Temperature",
                                    value=0.8,
                                    info="Higher = more creative",
                                )
                                chatterbox_turbo_cfg_weight = gr.Slider(
                                    0.0,
                                    1,
                                    step=0.05,
                                    label="⚡ CFG Weight",
                                    value=0.5,
                                    info="Speed vs quality",
                                )
                                chatterbox_turbo_chunk_size = gr.Slider(
                                    100,
                                    400,
                                    step=25,
                                    label="📄 Chunk Size",
                                    value=300,
                                    info="Characters per chunk",
                                )
                            with gr.Row():
                                chatterbox_turbo_repetition_penalty = gr.Slider(
                                    1.0,
                                    3.0,
                                    step=0.1,
                                    label="🔁 Repetition Penalty",
                                    value=1.2,
                                    info="Reduce repetitions",
                                )
                                chatterbox_turbo_min_p = gr.Slider(
                                    0.0,
                                    0.5,
                                    step=0.01,
                                    label="📊 Min P",
                                    value=0.05,
                                    info="Minimum probability threshold",
                                )
                                chatterbox_turbo_top_p = gr.Slider(
                                    0.5,
                                    1.0,
                                    step=0.05,
                                    label="🎯 Top P",
                                    value=1.0,
                                    info="Nucleus sampling",
                                )
                                chatterbox_turbo_seed = gr.Number(
                                    value=0,
                                    label="🎲 Seed (0=random)",
                                    info="For reproducible results",
                                )
                else:
                    # Placeholder when Chatterbox Turbo is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🚀 Chatterbox Turbo** - ⚠️ Not available - please check installation</div>"
                        )
                        # Create dummy components to maintain consistent interface
                        chatterbox_turbo_ref_audio = gr.Audio(visible=False, value=None)
                        chatterbox_turbo_exaggeration = gr.Slider(visible=False, value=0.5)
                        chatterbox_turbo_temperature = gr.Slider(visible=False, value=0.8)
                        chatterbox_turbo_cfg_weight = gr.Slider(visible=False, value=0.5)
                        chatterbox_turbo_repetition_penalty = gr.Slider(visible=False, value=1.2)
                        chatterbox_turbo_min_p = gr.Slider(visible=False, value=0.05)
                        chatterbox_turbo_top_p = gr.Slider(visible=False, value=1.0)
                        chatterbox_turbo_chunk_size = gr.Slider(visible=False, value=300)
                        chatterbox_turbo_seed = gr.Number(visible=False, value=0)

            # Kokoro TTS Tab
            with gr.TabItem("🗣️ Kokoro TTS", id="kokoro_tab"):
                if KOKORO_AVAILABLE:
                    with gr.Group() as kokoro_controls:
                        gr.Markdown("**🗣️ Kokoro TTS - High-quality pre-trained voices**")

                        # Voice selection grid
                        gr.Markdown("**🎭 Select Voice**")
                        gr.Markdown("*Choose from pre-trained voices*", elem_classes=["gr-info"])

                        # Create choices as (label, value) pairs
                        kokoro_voice_choices = [
                            (k, v) for k, v in update_kokoro_voice_choices().items()
                        ]

                        kokoro_voice = gr.Radio(
                            choices=kokoro_voice_choices,
                            value=list(KOKORO_CHOICES.values())[0] if KOKORO_CHOICES else None,
                            label="",
                            elem_classes=["fade-in", "voice-grid"],
                            show_label=False,
                        )

                        # Speed control below the voice grid
                        with gr.Row():
                            kokoro_speed = gr.Slider(
                                0.5,
                                2.0,
                                step=0.1,
                                label="⚡ Speech Speed",
                                value=1.0,
                                info="Adjust speaking speed",
                                elem_classes=["fade-in"],
                            )

                        # Custom Voice Upload Section
                        with gr.Accordion(
                            "👤 Custom Voice Upload", open=False, elem_classes=["fade-in"]
                        ):
                            gr.Markdown(
                                """
                            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                                        padding: 12px; border-radius: 12px; margin-bottom: 15px;'>
                                <h3 style='margin: 0 0 5px 0; padding: 0; font-size: 1.0em;'>📁 Upload Your Custom Voices</h3>
                                <p style='margin: 0; opacity: 0.8; font-size: 0.85em;'>Add your own .pt voice files to use with Kokoro TTS</p>
                            </div>
                            """
                            )

                            with gr.Row():
                                with gr.Column(scale=2):
                                    custom_voice_name = gr.Textbox(
                                        label="👤 Custom Voice Name",
                                        placeholder="Enter a name for your custom voice",
                                        info="Use only letters, numbers, and underscores",
                                        elem_classes=["fade-in"],
                                    )

                                    custom_voice_files = gr.File(
                                        label="📁 Upload Voice File (.pt)",
                                        file_count="single",
                                        file_types=[".pt"],
                                        elem_classes=["fade-in"],
                                    )

                                    with gr.Row():
                                        upload_btn = gr.Button(
                                            "📤 Upload Voice",
                                            variant="primary",
                                            elem_classes=["fade-in"],
                                        )
                                        refresh_voices_btn = gr.Button(
                                            "🔄 Refresh Voices",
                                            variant="secondary",
                                            elem_classes=["fade-in"],
                                        )

                                    upload_status = gr.Textbox(
                                        label="📊 Upload Status",
                                        interactive=False,
                                        elem_classes=["fade-in"],
                                    )

                                with gr.Column(scale=1):
                                    gr.Markdown("**📋 Your Custom Voices**")
                                    custom_voice_list = gr.Dataframe(
                                        headers=["Voice Name", "Status"],
                                        datatype=["str", "str"],
                                        row_count=(5, "fixed"),
                                        col_count=(2, "fixed"),
                                        interactive=False,
                                        value=get_custom_voice_list(),
                                        elem_classes=["fade-in"],
                                    )

                            gr.Markdown(
                                """
                            <div style='margin-top: 10px; padding: 10px; background: rgba(102, 126, 234, 0.05); border-radius: 8px; border-left: 3px solid #667eea;'>
                                <p style='margin: 0; font-size: 0.85em; opacity: 0.8;'>
                                    <strong>💡 Tips:</strong> Upload .pt voice files compatible with Kokoro TTS.
                                    Custom voices will appear with a 👤 prefix in the voice selector above.
                                    Use the refresh button to update the voice list after uploading.
                                </p>
                            </div>
                            """
                            )
                else:
                    # Placeholder when Kokoro is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🗣️ Kokoro TTS** - ⚠️ Not available - please check installation</div>"
                        )
                        # Create dummy components
                        kokoro_voice = gr.Radio(visible=False, value=None, choices=[])
                        kokoro_speed = gr.Slider(visible=False, value=1.0)
                        # Dummy custom voice components
                        custom_voice_name = gr.Textbox(visible=False, value="")
                        custom_voice_files = gr.File(visible=False, value=None)
                        upload_btn = gr.Button(visible=False)
                        refresh_voices_btn = gr.Button(visible=False)
                        upload_status = gr.Textbox(visible=False, value="")
                        custom_voice_list = gr.Dataframe(visible=False, value=[])

            # Fish Speech Tab
            with gr.TabItem("🐟 Fish Speech", id="fish_tab"):
                if FISH_SPEECH_AVAILABLE:
                    with gr.Group() as fish_speech_controls:
                        gr.Markdown("**🐟 Fish Speech - Natural text-to-speech synthesis**")
                        gr.Markdown(
                            "*💡 Try the sample file: `sample/Sample.wav`*",
                            elem_classes=["fade-in"],
                        )

                        with gr.Row():
                            with gr.Column(scale=2):
                                fish_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio File (Optional)",
                                    value=None,
                                    elem_classes=["fade-in"],
                                )

                            with gr.Column(scale=1):
                                fish_ref_text = gr.Textbox(
                                    label="🗣️ Reference Text (Optional)",
                                    placeholder="Enter reference text here...",
                                    elem_classes=["fade-in"],
                                )

                        with gr.Accordion(
                            "🔧 Advanced Fish Speech Settings", open=False, elem_classes=["fade-in"]
                        ):
                            gr.Markdown(
                                "<p style='opacity: 0.7; margin-bottom: 15px;'>🔧 Fine-tune Fish Speech generation parameters</p>"
                            )
                            with gr.Row():
                                fish_temperature = gr.Slider(
                                    0.1,
                                    1.0,
                                    step=0.05,
                                    label="🌡️ Temperature",
                                    value=0.8,
                                    info="Higher = more creative (0.1-1.0)",
                                )
                                fish_top_p = gr.Slider(
                                    0.1,
                                    1.0,
                                    step=0.05,
                                    label="🎭 Top P",
                                    value=0.8,
                                    info="Controls diversity (0.1-1.0)",
                                )
                                fish_repetition_penalty = gr.Slider(
                                    0.9,
                                    2.0,
                                    step=0.05,
                                    label="🔄 Repetition Penalty",
                                    value=1.1,
                                    info="Reduces repetition (0.9-2.0)",
                                )
                            with gr.Row():
                                fish_max_tokens = gr.Slider(
                                    100,
                                    2000,
                                    step=100,
                                    label="🔢 Max Tokens",
                                    value=1024,
                                    info="Maximum tokens per chunk",
                                )
                                fish_seed = gr.Number(
                                    value=None,
                                    label="🎲 Seed (None=random)",
                                    info="For reproducible results",
                                )

                            gr.Markdown("### 📝 Text Processing & Voice Consistency")
                            gr.Markdown(
                                """<p style='opacity: 0.7; margin-bottom: 10px;'>
                            • Fish Speech automatically splits long texts into chunks for better quality<br/>
                            • Without reference audio: Uses consistent seed across chunks to maintain voice<br/>
                            • With reference audio: Voice cloning ensures consistency<br/>
                            • Tip: Set a specific seed value for reproducible results
                            </p>"""
                            )
                else:
                    # Placeholder when Fish Speech is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🐟 Fish Speech** - ⚠️ Not available - please check installation</div>"
                        )
                    # Create dummy components
                    fish_ref_audio = gr.Audio(visible=False, value=None)
                    fish_ref_text = gr.Textbox(visible=False, value="")
                    fish_temperature = gr.Slider(visible=False, value=0.8)
                    fish_top_p = gr.Slider(visible=False, value=0.8)
                    fish_repetition_penalty = gr.Slider(visible=False, value=1.1)
                    fish_max_tokens = gr.Slider(visible=False, value=1024)
                    fish_seed = gr.Number(visible=False, value=None)

            # IndexTTS Tab
            with gr.TabItem("🎯 IndexTTS", id="indextts_tab"):
                if INDEXTTS_AVAILABLE:
                    with gr.Group(
                        visible=True, elem_id="indextts_controls", elem_classes=["fade-in"]
                    ):
                        gr.Markdown("**🎯 IndexTTS - Industrial-level controllable TTS**")

                        with gr.Row():
                            indextts_ref_audio = gr.Audio(
                                label="🎤 Reference Audio", type="filepath"
                            )

                        with gr.Accordion(
                            "🔧 Advanced IndexTTS Settings", open=False, elem_classes=["fade-in"]
                        ):
                            gr.Markdown(
                                "<p style='opacity: 0.7; margin-bottom: 15px;'>🔧 Fine-tune IndexTTS generation parameters</p>"
                            )

                            with gr.Row():
                                indextts_temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=0.8,
                                    step=0.1,
                                    label="🌡️ Temperature",
                                    info="Controls randomness in generation (0.1=stable, 2.0=creative)",
                                )
                                indextts_seed = gr.Number(
                                    label="🎲 Seed",
                                    value=None,
                                    precision=0,
                                    info="Set seed for reproducible results (leave empty for random)",
                                )

                # Placeholder when IndexTTS is not available
                else:
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🎯 IndexTTS** - ⚠️ Not available - please check installation</div>"
                        )
                        # Create dummy components
                        indextts_ref_audio = gr.Audio(visible=False, value=None)
                        indextts_temperature = gr.Slider(visible=False, value=0.8)
                        indextts_seed = gr.Number(visible=False, value=None)

            # IndexTTS2 Tab
            with gr.TabItem("🎯 IndexTTS2", id="indextts2_tab"):
                if INDEXTTS2_AVAILABLE:
                    with gr.Group(
                        visible=True, elem_id="indextts2_controls", elem_classes=["fade-in"]
                    ):
                        gr.Markdown("**🎯 IndexTTS2 - Advanced Emotionally Expressive TTS**")
                        gr.Markdown(
                            "*💡 Zero-shot voice cloning with advanced emotion control*",
                            elem_classes=["fade-in"],
                        )

                        with gr.Row():
                            indextts2_ref_audio = gr.Audio(
                                label="🎤 Reference Audio (Required) - Voice to clone (max 15s for optimal performance)",
                                type="filepath",
                            )

                        # Emotion Control Section
                        with gr.Accordion(
                            "🎭 Emotion Control", open=True, elem_classes=["fade-in"]
                        ):
                            indextts2_emotion_mode = gr.Radio(
                                choices=[
                                    ("🎵 Audio Reference", "audio_reference"),
                                    ("🎛️ Manual Control", "vector_control"),
                                    ("📝 Text Description", "text_description"),
                                ],
                                value="audio_reference",
                                label="🎭 Emotion Control Mode - Choose how to control emotional expression",
                            )

                            # Audio Reference Mode
                            with gr.Group(visible=True) as indextts2_audio_mode:
                                indextts2_emotion_audio = gr.Audio(
                                    label="🎭 Emotion Reference Audio - Audio expressing the desired emotion",
                                    type="filepath",
                                )
                                indextts2_emo_alpha = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.1,
                                    label="🎚️ Emotion Strength - Blend between speaker voice and emotion reference",
                                )

                            # Vector Control Mode
                            with gr.Group(visible=False) as indextts2_vector_mode:
                                gr.Markdown("**🎛️ Manual Emotion Control**")
                                with gr.Row():
                                    indextts2_happy = gr.Slider(0, 1, 0, step=0.1, label="😊 Happy")
                                    indextts2_angry = gr.Slider(0, 1, 0, step=0.1, label="😠 Angry")
                                    indextts2_sad = gr.Slider(0, 1, 0, step=0.1, label="😢 Sad")
                                    indextts2_afraid = gr.Slider(
                                        0, 1, 0, step=0.1, label="😨 Afraid"
                                    )
                                with gr.Row():
                                    indextts2_disgusted = gr.Slider(
                                        0, 1, 0, step=0.1, label="🤢 Disgusted"
                                    )
                                    indextts2_melancholic = gr.Slider(
                                        0, 1, 0, step=0.1, label="😔 Melancholic"
                                    )
                                    indextts2_surprised = gr.Slider(
                                        0, 1, 0, step=0.1, label="😲 Surprised"
                                    )
                                    indextts2_calm = gr.Slider(0, 1, 1, step=0.1, label="😌 Calm")

                                # Emotion Presets
                                with gr.Row():
                                    indextts2_emotion_preset = gr.Radio(
                                        choices=(
                                            list(EMOTION_PRESETS.keys())
                                            if INDEXTTS2_AVAILABLE
                                            else []
                                        ),
                                        label="🎭 Emotion Presets - Quick emotion settings",
                                        value=None,
                                    )
                                    indextts2_apply_preset = gr.Button("Apply Preset", size="sm")

                            # Text Description Mode
                            with gr.Group(visible=False) as indextts2_text_mode:
                                indextts2_emotion_description = gr.Textbox(
                                    label="📝 Emotion Description - Describe the desired emotion in natural language",
                                    placeholder="e.g., 'excited and happy', 'sad and melancholic', 'calm and peaceful'",
                                )

                        with gr.Accordion(
                            "🔧 Advanced IndexTTS2 Settings", open=False, elem_classes=["fade-in"]
                        ):
                            gr.Markdown(
                                "<p style='opacity: 0.7; margin-bottom: 15px;'>🔧 Fine-tune IndexTTS2 generation parameters</p>"
                            )

                            with gr.Row():
                                indextts2_temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=2.0,
                                    value=0.8,
                                    step=0.1,
                                    label="🌡️ Temperature - Controls randomness in generation",
                                )
                                indextts2_top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.05,
                                    label="🎯 Top-p - Nucleus sampling parameter",
                                )

                            with gr.Row():
                                indextts2_top_k = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=50,
                                    step=1,
                                    label="🔝 Top-k - Top-k sampling parameter",
                                )
                                indextts2_repetition_penalty = gr.Slider(
                                    minimum=1.0,
                                    maximum=2.0,
                                    value=1.1,
                                    step=0.1,
                                    label="🔄 Repetition Penalty - Penalty for repetitive content",
                                )

                            with gr.Row():
                                indextts2_max_mel_tokens = gr.Slider(
                                    minimum=500,
                                    maximum=3000,
                                    value=1500,
                                    step=100,
                                    label="📏 Max Mel Tokens - Maximum length of generated audio",
                                )
                                indextts2_seed = gr.Number(
                                    label="🎲 Seed - Set seed for reproducible results",
                                    value=None,
                                    precision=0,
                                )

                            indextts2_use_random = gr.Checkbox(
                                label="🎲 Random Sampling - Enable random sampling for variation",
                                value=False,
                            )

                # Placeholder when IndexTTS2 is not available
                else:
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🎯 IndexTTS2** - ⚠️ Not available - please check installation</div>"
                        )
                        # Create dummy components
                        indextts2_ref_audio = gr.Audio(visible=False, value=None)
                        indextts2_emotion_mode = gr.Radio(
                            visible=False,
                            choices=[
                                ("🎵 Audio Reference", "audio_reference"),
                                ("🎛️ Manual Control", "vector_control"),
                                ("📝 Text Description", "text_description"),
                            ],
                            value="audio_reference",
                        )
                        indextts2_emotion_audio = gr.Audio(visible=False, value=None)
                        indextts2_emotion_description = gr.Textbox(visible=False, value="")
                        indextts2_emo_alpha = gr.Slider(visible=False, value=1.0)
                        indextts2_happy = gr.Slider(visible=False, value=0)
                        indextts2_angry = gr.Slider(visible=False, value=0)
                        indextts2_sad = gr.Slider(visible=False, value=0)
                        indextts2_afraid = gr.Slider(visible=False, value=0)
                        indextts2_disgusted = gr.Slider(visible=False, value=0)
                        indextts2_melancholic = gr.Slider(visible=False, value=0)
                        indextts2_surprised = gr.Slider(visible=False, value=0)
                        indextts2_calm = gr.Slider(visible=False, value=1)
                        indextts2_emotion_preset = gr.Radio(visible=False, choices=[])
                        indextts2_apply_preset = gr.Button(visible=False)
                        indextts2_temperature = gr.Slider(visible=False, value=0.8)
                        indextts2_top_p = gr.Slider(visible=False, value=0.9)
                        indextts2_top_k = gr.Slider(visible=False, value=50)
                        indextts2_repetition_penalty = gr.Slider(visible=False, value=1.1)
                        indextts2_max_mel_tokens = gr.Slider(visible=False, value=1500)
                        indextts2_seed = gr.Number(visible=False, value=None)
                        indextts2_use_random = gr.Checkbox(visible=False, value=True)
                        indextts2_audio_mode = gr.Group(visible=False)
                        indextts2_vector_mode = gr.Group(visible=False)
                        indextts2_text_mode = gr.Group(visible=False)

            # F5-TTS Tab
            with gr.TabItem("🎵 F5-TTS", id="f5_tab"):
                if F5_TTS_AVAILABLE:
                    with gr.Group() as f5_tts_controls:
                        gr.Markdown("**🎵 F5-TTS - Flow Matching Text-to-Speech**")
                        gr.Markdown(
                            "*💡 High-quality voice cloning - Load model from Model Management section above*",
                            elem_classes=["fade-in"],
                        )

                        # Generation settings
                        with gr.Row():
                            with gr.Column(scale=2):
                                f5_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio (Optional)",
                                    elem_classes=["fade-in"],
                                )

                                f5_ref_text = gr.Textbox(
                                    label="📝 Reference Text (Optional)",
                                    placeholder="Text spoken in reference audio",
                                    elem_classes=["fade-in"],
                                )

                            with gr.Column(scale=1):
                                f5_speed = gr.Slider(
                                    0.5,
                                    2.0,
                                    step=0.1,
                                    label="⚡ Speed",
                                    value=1.0,
                                    info="Speech speed multiplier",
                                    elem_classes=["fade-in"],
                                )

                                f5_cross_fade = gr.Slider(
                                    0.0,
                                    0.5,
                                    step=0.05,
                                    label="🔄 Cross-fade Duration",
                                    value=0.15,
                                    info="Smooth transitions (seconds)",
                                    elem_classes=["fade-in"],
                                )

                        with gr.Accordion(
                            "🔧 Advanced F5-TTS Settings", open=False, elem_classes=["fade-in"]
                        ):
                            with gr.Row():
                                f5_remove_silence = gr.Checkbox(
                                    label="🔇 Remove Silence",
                                    value=False,
                                    info="Remove silence from start/end",
                                    elem_classes=["fade-in"],
                                )

                                f5_seed = gr.Number(
                                    value=0,
                                    label="🎲 Seed (0=random)",
                                    info="For reproducible results",
                                    elem_classes=["fade-in"],
                                )
                else:
                    # Placeholder when F5-TTS is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🎵 F5-TTS** - ⚠️ Not available - please check installation</div>"
                        )
                        # Create dummy components for generation settings only
                        f5_ref_audio = gr.Audio(visible=False, value=None)
                        f5_ref_text = gr.Textbox(visible=False, value="")
                        f5_speed = gr.Slider(visible=False, value=1.0)
                        f5_cross_fade = gr.Slider(visible=False, value=0.15)
                        f5_remove_silence = gr.Checkbox(visible=False, value=False)
                        f5_seed = gr.Number(visible=False, value=0)

            # Higgs Audio Tab
            with gr.TabItem("🎙️ Higgs Audio", id="higgs_tab"):
                if HIGGS_AUDIO_AVAILABLE:
                    with gr.Group() as higgs_audio_controls:
                        gr.Markdown("**🎙️ Higgs Audio - Advanced Multimodal TTS**")
                        gr.Markdown(
                            "*💡 State-of-the-art voice cloning with multimodal capabilities (Use wav files not mp3)*",
                            elem_classes=["fade-in"],
                        )

                        # Generation settings
                        with gr.Row():
                            with gr.Column(scale=2):
                                higgs_ref_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="🎤 Reference Audio (Optional)",
                                    elem_classes=["fade-in"],
                                )

                                higgs_ref_text = gr.Textbox(
                                    label="📝 Reference Text (Optional)",
                                    placeholder="Text spoken in reference audio",
                                    elem_classes=["fade-in"],
                                )

                                higgs_voice_preset = gr.Dropdown(
                                    label="🗣️ Voice Preset",
                                    choices=["EMPTY"]
                                    + (
                                        get_higgs_audio_handler().get_available_voice_presets()[1:]
                                        if HIGGS_AUDIO_AVAILABLE
                                        else []
                                    ),
                                    value="EMPTY",
                                    info="Select a predefined voice or use custom reference audio",
                                    elem_classes=["fade-in"],
                                )

                            with gr.Column(scale=1):
                                higgs_temperature = gr.Slider(
                                    0.0,
                                    1.5,
                                    step=0.1,
                                    label="🌡️ Temperature",
                                    value=1.0,
                                    info="Creativity vs consistency",
                                    elem_classes=["fade-in"],
                                )

                                higgs_top_p = gr.Slider(
                                    0.1,
                                    1.0,
                                    step=0.05,
                                    label="🎯 Top-P",
                                    value=0.95,
                                    info="Nucleus sampling",
                                    elem_classes=["fade-in"],
                                )

                                higgs_top_k = gr.Slider(
                                    1,
                                    100,
                                    step=1,
                                    label="🔝 Top-K",
                                    value=50,
                                    info="Top-K sampling",
                                    elem_classes=["fade-in"],
                                )

                        with gr.Accordion(
                            "🔧 Advanced Higgs Audio Settings", open=False, elem_classes=["fade-in"]
                        ):
                            higgs_system_prompt = gr.Textbox(
                                label="💬 System Prompt (Optional)",
                                placeholder="Custom system prompt for generation context",
                                lines=3,
                                elem_classes=["fade-in"],
                            )

                            with gr.Row():
                                higgs_max_tokens = gr.Slider(
                                    128,
                                    4096,
                                    step=64,
                                    label="📏 Max Tokens",
                                    value=1024,
                                    info="Maximum generation length",
                                    elem_classes=["fade-in"],
                                )

                                higgs_ras_win_len = gr.Slider(
                                    0,
                                    10,
                                    step=1,
                                    label="🪟 RAS Window Length",
                                    value=7,
                                    info="Repetition avoidance window",
                                    elem_classes=["fade-in"],
                                )

                                higgs_ras_win_max_num_repeat = gr.Slider(
                                    1,
                                    10,
                                    step=1,
                                    label="🔄 RAS Max Repeats",
                                    value=2,
                                    info="Max repetitions in window",
                                    elem_classes=["fade-in"],
                                )
                else:
                    # Placeholder when Higgs Audio is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🎙️ Higgs Audio** - ⚠️ Not available - please check installation</div>"
                        )
                        # Create dummy components
                        higgs_ref_audio = gr.Audio(visible=False, value=None)
                        higgs_ref_text = gr.Textbox(visible=False, value="")
                        higgs_voice_preset = gr.Dropdown(
                            visible=False, choices=["EMPTY"], value="EMPTY"
                        )
                        higgs_system_prompt = gr.Textbox(visible=False, value="")
                        higgs_temperature = gr.Slider(visible=False, value=1.0)
                        higgs_top_p = gr.Slider(visible=False, value=0.95)
                        higgs_top_k = gr.Slider(visible=False, value=50)
                        higgs_max_tokens = gr.Slider(visible=False, value=1024)
                        higgs_ras_win_len = gr.Slider(visible=False, value=7)
                        higgs_ras_win_max_num_repeat = gr.Slider(visible=False, value=2)

            # VoxCPM Tab
            with gr.TabItem("🎤 VoxCPM", id="voxcpm_tab"):
                if VOXCPM_AVAILABLE:
                    with gr.Group() as voxcpm_controls:
                        gr.Markdown("**🎤 VoxCPM - Voice Cloning TTS**")
                        gr.Markdown(
                            "*💡 Advanced voice cloning with automatic transcription using Whisper!*",
                            elem_classes=["fade-in"],
                        )
                        gr.Markdown(
                            "📝 **Instructions:** Upload a clear reference audio (3-10 seconds) and the text will be auto-transcribed using Whisper for voice cloning."
                        )

                        # Voice cloning section
                        with gr.Row():
                            with gr.Column():
                                voxcpm_ref_audio = gr.Audio(
                                    label="🎤 Reference Audio (for voice cloning)",
                                    type="filepath",
                                    sources=["upload"],
                                )
                                voxcpm_ref_text = gr.Textbox(
                                    label="📝 Reference Text (auto-transcribed)",
                                    placeholder="Will be automatically filled when you upload audio above...",
                                    lines=2,
                                )

                        # Advanced settings
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            gr.Markdown(
                                "**CFG Value:** LM guidance on LocDiT, higher for better adherence to prompt"
                            )
                            gr.Markdown(
                                "**Inference Timesteps:** Higher for better quality, lower for faster speed"
                            )
                            gr.Markdown("**Normalize/Denoise:** Enable external processing tools")
                            gr.Markdown(
                                "**Retry:** Enable retrying for bad cases with configurable thresholds"
                            )
                            gr.Markdown(
                                "**Seed:** Random seed for reproducible generation (-1 for random)"
                            )
                            with gr.Row():
                                voxcpm_cfg_value = gr.Slider(
                                    minimum=0.5, maximum=5.0, value=2.0, step=0.1, label="CFG Value"
                                )
                                voxcpm_inference_timesteps = gr.Slider(
                                    minimum=5,
                                    maximum=50,
                                    value=10,
                                    step=1,
                                    label="Inference Timesteps",
                                )

                            with gr.Row():
                                voxcpm_normalize = gr.Checkbox(value=True, label="Normalize")
                                voxcpm_denoise = gr.Checkbox(value=True, label="Denoise")
                                voxcpm_retry_badcase = gr.Checkbox(
                                    value=True, label="Retry Bad Cases"
                                )

                            with gr.Row():
                                voxcpm_retry_badcase_max_times = gr.Number(
                                    value=3, minimum=1, maximum=10, step=1, label="Max Retry Times"
                                )
                                voxcpm_retry_badcase_ratio_threshold = gr.Number(
                                    value=6.0,
                                    minimum=1.0,
                                    maximum=10.0,
                                    step=0.5,
                                    label="Retry Ratio Threshold",
                                )
                                voxcpm_seed = gr.Number(
                                    value=-1, minimum=-1, maximum=2147483647, step=1, label="Seed"
                                )
                else:
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🎤 VoxCPM** - ⚠️ Not available - please install with: `pip install voxcpm openai-whisper`</div>"
                        )
                        # Create dummy components
                        voxcpm_ref_audio = gr.Audio(visible=False)
                        voxcpm_ref_text = gr.Textbox(visible=False)
                        voxcpm_cfg_value = gr.Slider(visible=False, value=2.0)
                        voxcpm_inference_timesteps = gr.Slider(visible=False, value=10)
                        voxcpm_normalize = gr.Checkbox(visible=False, value=True)
                        voxcpm_denoise = gr.Checkbox(visible=False, value=True)
                        voxcpm_retry_badcase = gr.Checkbox(visible=False, value=True)
                        voxcpm_retry_badcase_max_times = gr.Number(visible=False, value=3)
                        voxcpm_retry_badcase_ratio_threshold = gr.Number(visible=False, value=6.0)
                        voxcpm_seed = gr.Number(visible=False, value=-1)

            # KittenTTS Tab
            with gr.TabItem("🐱 KittenTTS", id="kitten_tab"):
                if KITTEN_TTS_AVAILABLE:
                    with gr.Group() as kitten_tts_controls:
                        gr.Markdown("**🐱 KittenTTS - Mini Model TTS**")
                        gr.Markdown(
                            "*💡 High-quality mini model with 8 built-in voices - no reference audio needed!*",
                            elem_classes=["fade-in"],
                        )

                        # Voice selection
                        with gr.Row():
                            with gr.Column():
                                kitten_voice = gr.Radio(
                                    label="🗣️ Voice Selection",
                                    choices=(
                                        KITTEN_VOICES
                                        if KITTEN_TTS_AVAILABLE
                                        else ["expr-voice-2-f"]
                                    ),
                                    value="expr-voice-2-f",
                                    info="Choose from 8 built-in voices (male/female variants)",
                                    elem_classes=["fade-in"],
                                )

                                gr.Markdown(
                                    """
                                **Voice Guide:**
                                - `expr-voice-2-f/m` - Expressive female/male voice
                                - `expr-voice-3-f/m` - Natural female/male voice
                                - `expr-voice-4-f/m` - Clear female/male voice
                                - `expr-voice-5-f/m` - Warm female/male voice
                                """,
                                    elem_classes=["fade-in"],
                                )
                else:
                    # Placeholder when KittenTTS is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🐱 KittenTTS** - ⚠️ Not available - please install with: `pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl`</div>"
                        )
                        # Create dummy component
                        kitten_voice = gr.Dropdown(
                            visible=False, choices=["expr-voice-2-f"], value="expr-voice-2-f"
                        )

            # Qwen TTS Tab
            with gr.TabItem("🎙️ Qwen TTS", id="qwen_tab"):
                if QWEN_TTS_AVAILABLE:
                    with gr.Group() as qwen_tts_controls:
                        gr.Markdown("**🎙️ Qwen3-TTS - Advanced Text-to-Speech**")
                        gr.Markdown(
                            "*💡 Three modes: Voice Design (create voices from descriptions), Voice Clone (clone from audio), Custom Voice (predefined speakers)*",
                            elem_classes=["fade-in"],
                        )

                        # Mode selection
                        with gr.Row():
                            qwen_mode = gr.Radio(
                                label="🎯 TTS Mode",
                                choices=[
                                    ("🎨 Voice Design", "voice_design"),
                                    ("🎭 Voice Clone", "voice_clone"),
                                    ("🗣️ Custom Voice", "custom_voice"),
                                ],
                                value="voice_clone",
                                info="Voice Clone supports chunking, conversation mode, and ebook mode",
                                elem_classes=["fade-in"],
                            )

                        # Currently loaded model status
                        qwen_loaded_model_status = gr.Markdown(
                            value="📦 **Loaded Model:** None - Load a model from Models tab first",
                            elem_classes=["fade-in"],
                        )

                        # Voice Design controls (visible when voice_design mode selected)
                        with gr.Group(visible=False) as qwen_voice_design_group:
                            gr.Markdown(
                                "**🎨 Voice Design Mode** - Create unique voices from natural language descriptions"
                            )
                            gr.Markdown("*⚠️ Only works in Text to Speech mode (no chunking)*")
                            qwen_voice_description = gr.Textbox(
                                label="🎭 Voice Description",
                                placeholder="Describe the voice you want, e.g., 'Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.'",
                                lines=3,
                                elem_classes=["fade-in"],
                            )

                        # Voice Clone controls (visible when voice_clone mode selected)
                        with gr.Group(visible=True) as qwen_voice_clone_group:
                            gr.Markdown(
                                "**🎭 Voice Clone Mode** - Clone voice from reference audio"
                            )
                            gr.Markdown("*✅ Supports chunking, conversation mode, and ebook mode*")
                            with gr.Row():
                                qwen_ref_audio = gr.Audio(
                                    label="🎤 Reference Audio",
                                    type="filepath",
                                    sources=["upload", "microphone"],
                                    elem_classes=["fade-in"],
                                )
                            with gr.Row():
                                qwen_ref_text = gr.Textbox(
                                    label="📝 Reference Text",
                                    placeholder="Transcript of the reference audio (or click Transcribe)...",
                                    lines=2,
                                    scale=3,
                                    elem_classes=["fade-in"],
                                )
                                qwen_transcribe_btn = gr.Button(
                                    "🎤 Transcribe", scale=1, elem_classes=["fade-in"]
                                )
                            qwen_xvector_only = gr.Checkbox(
                                label="X-vector only (no text needed, lower quality)",
                                value=False,
                                elem_classes=["fade-in"],
                            )
                            with gr.Row():
                                qwen_clone_model_size = gr.Dropdown(
                                    label="Model Size",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    elem_classes=["fade-in"],
                                )
                                qwen_chunk_size = gr.Slider(
                                    label="Chunk Size",
                                    minimum=50,
                                    maximum=500,
                                    value=200,
                                    step=10,
                                    elem_classes=["fade-in"],
                                )
                                qwen_chunk_gap = gr.Slider(
                                    label="Chunk Gap (s)",
                                    minimum=0.0,
                                    maximum=3.0,
                                    value=0.0,
                                    step=0.01,
                                    elem_classes=["fade-in"],
                                )

                        # Custom Voice controls (visible when custom_voice mode selected)
                        with gr.Group(visible=False) as qwen_custom_voice_group:
                            gr.Markdown(
                                "**🗣️ Custom Voice Mode** - Use predefined speakers with style instructions"
                            )
                            gr.Markdown("*⚠️ Only works in Text to Speech mode (no chunking)*")
                            qwen_speaker = gr.Radio(
                                label="👤 Speaker",
                                choices=QWEN_SPEAKERS if QWEN_TTS_AVAILABLE else ["Ryan"],
                                value="Ryan",
                                elem_classes=["fade-in"],
                            )
                            with gr.Row():
                                qwen_custom_model_size = gr.Dropdown(
                                    label="Model Size",
                                    choices=["0.6B", "1.7B"],
                                    value="1.7B",
                                    elem_classes=["fade-in"],
                                )
                            qwen_style_instruct = gr.Textbox(
                                label="🎭 Style Instruction (Optional, 1.7B only)",
                                placeholder="e.g., Speak in a cheerful and energetic tone",
                                lines=2,
                                elem_classes=["fade-in"],
                            )

                        # Common settings
                        with gr.Row():
                            qwen_language = gr.Dropdown(
                                label="🌍 Language",
                                choices=QWEN_LANGUAGES if QWEN_TTS_AVAILABLE else ["Auto"],
                                value="Auto",
                                elem_classes=["fade-in"],
                            )
                            qwen_seed = gr.Number(
                                label="🎲 Seed (-1 = Auto)",
                                value=-1,
                                precision=0,
                                elem_classes=["fade-in"],
                            )
                else:
                    # Placeholder when Qwen TTS is not available
                    with gr.Group():
                        gr.Markdown(
                            "<div style='text-align: center; padding: 40px; opacity: 0.5;'>**🎙️ Qwen TTS** - ⚠️ Not available - please check qwen_tts module installation</div>"
                        )
                        # Create dummy components
                        qwen_mode = gr.Radio(
                            visible=False, choices=["voice_clone"], value="voice_clone"
                        )
                        qwen_loaded_model_status = gr.Markdown(visible=False, value="")
                        qwen_voice_description = gr.Textbox(visible=False, value="")
                        qwen_ref_audio = gr.Audio(visible=False)
                        qwen_ref_text = gr.Textbox(visible=False, value="")
                        qwen_transcribe_btn = gr.Button(visible=False)
                        qwen_xvector_only = gr.Checkbox(visible=False, value=False)
                        qwen_clone_model_size = gr.Dropdown(
                            visible=False, choices=["1.7B"], value="1.7B"
                        )
                        qwen_chunk_size = gr.Slider(visible=False, value=200)
                        qwen_chunk_gap = gr.Slider(visible=False, value=0.0)
                        qwen_speaker = gr.Radio(visible=False, choices=["Ryan"], value="Ryan")
                        qwen_custom_model_size = gr.Dropdown(
                            visible=False, choices=["1.7B"], value="1.7B"
                        )
                        qwen_style_instruct = gr.Textbox(visible=False, value="")
                        qwen_language = gr.Dropdown(visible=False, choices=["Auto"], value="Auto")
                        qwen_seed = gr.Number(visible=False, value=-1)
                        qwen_voice_design_group = gr.Group(visible=False)
                        qwen_voice_clone_group = gr.Group(visible=False)
                        qwen_custom_voice_group = gr.Group(visible=False)

        # Audio Effects in a separate expandable section
        with gr.Accordion("🎵 Audio Effects Studio", open=False, elem_classes=["fade-in"]):
            gr.Markdown(
                """
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                        padding: 12px; border-radius: 12px; margin-bottom: 15px;'>
                <h3 style='margin: 0 0 5px 0; padding: 0; font-size: 1.0em;'>🎚️ Professional Audio Processing</h3>
                <p style='margin: 0; opacity: 0.8; font-size: 0.85em;'>Add studio-quality effects to enhance your generated speech</p>
            </div>
            """
            )

            # Volume and EQ Section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔊 Volume & EQ Settings")
                    gain_db = gr.Slider(
                        -20,
                        20,
                        step=0.5,
                        label="🎚️ Master Gain (dB)",
                        value=0,
                        info="Boost or reduce overall volume",
                        elem_classes=["fade-in"],
                    )

                    enable_eq = gr.Checkbox(
                        label="Enable 3-Band EQ", value=False, elem_classes=["fade-in"]
                    )
                    with gr.Row():
                        eq_bass = gr.Slider(
                            -12,
                            12,
                            step=0.5,
                            label="🔈 Bass",
                            value=0,
                            info="80-250 Hz",
                            elem_classes=["fade-in"],
                        )
                        eq_mid = gr.Slider(
                            -12,
                            12,
                            step=0.5,
                            label="🔉 Mid",
                            value=0,
                            info="250-4000 Hz",
                            elem_classes=["fade-in"],
                        )
                        eq_treble = gr.Slider(
                            -12,
                            12,
                            step=0.5,
                            label="🔊 Treble",
                            value=0,
                            info="4000+ Hz",
                            elem_classes=["fade-in"],
                        )

            # Effects Section with better layout
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🏛️ Spatial Effects")
                    enable_reverb = gr.Checkbox(
                        label="Enable Reverb", value=False, elem_classes=["fade-in"]
                    )
                    with gr.Column():
                        reverb_room = gr.Slider(
                            0.1,
                            1.0,
                            step=0.1,
                            label="Room Size",
                            value=0.3,
                            elem_classes=["fade-in"],
                        )
                        reverb_damping = gr.Slider(
                            0.1, 1.0, step=0.1, label="Damping", value=0.5, elem_classes=["fade-in"]
                        )
                        reverb_wet = gr.Slider(
                            0.1, 0.8, step=0.1, label="Wet Mix", value=0.3, elem_classes=["fade-in"]
                        )

                with gr.Column():
                    gr.Markdown("#### 🔊 Time-Based Effects")
                    enable_echo = gr.Checkbox(
                        label="Enable Echo", value=False, elem_classes=["fade-in"]
                    )
                    with gr.Column():
                        echo_delay = gr.Slider(
                            0.1,
                            1.0,
                            step=0.1,
                            label="Delay Time (s)",
                            value=0.3,
                            elem_classes=["fade-in"],
                        )
                        echo_decay = gr.Slider(
                            0.1,
                            0.9,
                            step=0.1,
                            label="Decay Amount",
                            value=0.5,
                            elem_classes=["fade-in"],
                        )

                with gr.Column():
                    gr.Markdown("#### 🎼 Pitch Effects")
                    enable_pitch = gr.Checkbox(
                        label="Enable Pitch Shift", value=False, elem_classes=["fade-in"]
                    )
                    pitch_semitones = gr.Slider(
                        -12,
                        12,
                        step=1,
                        label="Pitch (semitones)",
                        value=0,
                        info="±12 semitones = ±1 octave",
                        elem_classes=["fade-in"],
                    )

        # Footer with credits - Compact
        gr.Markdown(
            """
        <div style='text-align: center; margin-top: 20px; padding: 15px;
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
                    border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.1);'>
            <p style='opacity: 0.7; margin: 0; font-size: 0.85em;'>
                Made with ❤️ by SUP3RMASS1VE |
                <a href='https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition-Pinokio' target='_blank' style='color: #667eea; text-decoration: none;'>GitHub</a> |
                <a href='https://discord.gg/mvDcrA57AQ' target='_blank' style='color: #667eea; text-decoration: none;'>Discord</a>
            </p>
        </div>
        """
        )

        # Model management event handlers - Updated for compact interface with auto-selection
        def handle_load_chatterbox():
            success, message = init_chatterbox()
            if success:
                chatterbox_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select ChatterboxTTS engine when loaded
                selected_engine = "ChatterboxTTS"
                # Auto-switch to ChatterboxTTS tab
                selected_tab = gr.update(selected="chatterbox_tab")
            else:
                chatterbox_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return chatterbox_status_text, selected_engine, selected_engine, selected_tab
            else:
                return chatterbox_status_text, selected_engine, selected_tab

        def handle_unload_chatterbox():
            message = unload_chatterbox()
            chatterbox_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return chatterbox_status_text

        def handle_load_chatterbox_multilingual():
            success, message = init_chatterbox_multilingual()
            if success:
                chatterbox_mtl_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select Chatterbox Multilingual engine when loaded
                selected_engine = "Chatterbox Multilingual"
                # Auto-switch to Chatterbox Multilingual tab
                selected_tab = gr.update(selected="chatterbox_mtl_tab")
            else:
                chatterbox_mtl_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return chatterbox_mtl_status_text, selected_engine, selected_engine, selected_tab
            else:
                return chatterbox_mtl_status_text, selected_engine, selected_tab

        def handle_unload_chatterbox_multilingual():
            message = unload_chatterbox_multilingual()
            chatterbox_mtl_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return chatterbox_mtl_status_text

        def handle_load_chatterbox_turbo():
            success, message = init_chatterbox_turbo_model()
            if success:
                chatterbox_turbo_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select Chatterbox Turbo engine when loaded
                selected_engine = "Chatterbox Turbo"
                # Auto-switch to Chatterbox Turbo tab
                selected_tab = gr.update(selected="chatterbox_turbo_tab")
            else:
                chatterbox_turbo_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return chatterbox_turbo_status_text, selected_engine, selected_engine, selected_tab
            else:
                return chatterbox_turbo_status_text, selected_engine, selected_tab

        def handle_unload_chatterbox_turbo():
            message = unload_chatterbox_turbo_model()
            chatterbox_turbo_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return chatterbox_turbo_status_text

        def handle_load_kokoro():
            success, message = init_kokoro()
            if success:
                preload_kokoro_voices()  # Preload voices after loading model
                kokoro_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select Kokoro TTS engine when loaded
                selected_engine = "Kokoro TTS"
                # Auto-switch to Kokoro TTS tab
                selected_tab = gr.update(selected="kokoro_tab")
            else:
                kokoro_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return kokoro_status_text, selected_engine, selected_engine, selected_tab
            else:
                return kokoro_status_text, selected_engine, selected_tab

        def handle_unload_kokoro():
            message = unload_kokoro()
            kokoro_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return kokoro_status_text

        def handle_load_fish():
            success, message = init_fish_speech()
            if success:
                fish_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select Fish Speech engine when loaded
                selected_engine = "Fish Speech"
                # Auto-switch to Fish Speech tab
                selected_tab = gr.update(selected="fish_tab")
            else:
                fish_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return fish_status_text, selected_engine, selected_engine, selected_tab
            else:
                return fish_status_text, selected_engine, selected_tab

        def handle_unload_fish():
            message = unload_fish_speech()
            fish_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return fish_status_text

        def handle_load_indextts():
            success, message = init_indextts()
            if success:
                indextts_status_text = "✅ Loaded (Auto-selected)"
                selected_engine = "IndexTTS"
                # Auto-switch to IndexTTS tab
                selected_tab = gr.update(selected="indextts_tab")
            else:
                indextts_status_text = "❌ Failed to load"
                selected_engine = gr.update()
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return indextts_status_text, selected_engine, selected_engine, selected_tab
            else:
                return indextts_status_text, selected_engine, selected_tab

        def handle_unload_indextts():
            message = unload_indextts()
            indextts_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return indextts_status_text

        def handle_load_higgs():
            success, message = init_higgs_audio()
            if success:
                higgs_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select Higgs Audio engine when loaded
                selected_engine = "Higgs Audio"
                # Auto-switch to Higgs Audio tab
                selected_tab = gr.update(selected="higgs_tab")
            else:
                higgs_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return higgs_status_text, selected_engine, selected_engine, selected_tab
            else:
                return higgs_status_text, selected_engine, selected_tab

        def handle_unload_higgs():
            message = unload_higgs_audio()
            higgs_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return higgs_status_text

        def handle_load_voxcpm():
            success, message = init_voxcpm_model()
            if success:
                voxcpm_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select VoxCPM engine when loaded
                selected_engine = "VoxCPM"
                # Auto-switch to VoxCPM tab
                selected_tab = gr.update(selected="voxcpm_tab")
            else:
                voxcpm_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return voxcpm_status_text, selected_engine, selected_engine, selected_tab
            else:
                return voxcpm_status_text, selected_engine, selected_tab

        def handle_unload_voxcpm():
            message = unload_voxcpm_model()
            voxcpm_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return voxcpm_status_text

        def handle_load_kitten():
            success, message = init_kitten_tts_model()
            if success:
                kitten_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select KittenTTS engine when loaded
                selected_engine = "KittenTTS"
                # Auto-switch to KittenTTS tab
                selected_tab = gr.update(selected="kitten_tab")
            else:
                kitten_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return kitten_status_text, selected_engine, selected_engine, selected_tab
            else:
                return kitten_status_text, selected_engine, selected_tab

        def handle_unload_kitten():
            message = unload_kitten_tts_model()
            kitten_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return kitten_status_text

        # ── Assistant handlers ────────────────────────────────────────────

        def handle_assistant_send(
            user_message,
            chat_history,
            provider,
            base_url,
            api_key,
            model_id,
            system_prompt,
        ):
            """Send a message to the assistant and update the chatbot."""
            if not user_message or not str(user_message).strip():
                return chat_history, "", "⚠️ Please enter a message"

            from assistant_service import AssistantRequest, ChatMessage, chat as assistant_chat

            history_messages = []
            for msg in chat_history or []:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                else:
                    role = "user"
                    content = str(msg)
                history_messages.append(ChatMessage(role=str(role), content=str(content)))

            request = AssistantRequest(
                user_message=str(user_message).strip(),
                conversation_history=tuple(history_messages),
                provider_name=provider or "LM Studio OpenAI Server",
                base_url=base_url or "",
                api_key=api_key or "",
                model_id=model_id or "",
                system_prompt=system_prompt or "",
                temperature=0.4,
                top_p=0.9,
                max_tokens=1024,
                timeout_seconds=30,
            )

            response = assistant_chat(request)

            new_history = list(chat_history or [])
            new_history.append({"role": "user", "content": str(user_message).strip()})

            if response.error:
                error_msg = f"❌ {response.error}"
                new_history.append({"role": "assistant", "content": error_msg})
                status = f"❌ Error: {response.error}"
            else:
                new_history.append({"role": "assistant", "content": response.content})
                status = (
                    "✅ Response received "
                    f"({response.elapsed_seconds}s) — "
                    f"{response.provider_name}/{response.model_id}"
                )

            return new_history, "", status

        def handle_assistant_clear():
            """Clear the chat history."""
            return [], "", "Chat cleared"

        def handle_assistant_test_connection(provider, base_url, api_key, model_id):
            """Test assistant LLM connection."""
            from assistant_service import test_assistant_connection

            result = test_assistant_connection(
                provider_name=provider or "LM Studio OpenAI Server",
                base_url=base_url or "",
                api_key=api_key or "",
                model_id=model_id or "",
            )
            if result.startswith("✅"):
                indicator = f"🤖 Assistant: Connected ({provider})"
            else:
                indicator = "🤖 Assistant: Connection failed"
            return result, indicator

        def handle_assistant_save_settings(provider, base_url, model_id, api_key, system_prompt):
            """Save assistant LLM settings."""
            save_assistant_llm_settings(
                provider_name=provider,
                base_url=base_url,
                model_id=model_id,
                api_key=api_key,
                system_prompt=system_prompt,
            )
            return "✅ Assistant LLM settings saved"

        def handle_assistant_provider_change(provider_name):
            """Handle assistant LLM provider change using the shared provider defaults."""
            cfg = _get_provider_config(provider_name)
            suggestions = list(LLM_PROVIDER_MODEL_SUGGESTIONS.get(provider_name, []))
            if cfg["default_model"] and cfg["default_model"] not in suggestions:
                suggestions.insert(0, cfg["default_model"])
            return (
                gr.update(value=cfg["base_url"]),
                gr.update(choices=suggestions, value=cfg["default_model"]),
            )

        def _resolve_job_id(manager, raw_job_id):
            clean_id = str(raw_job_id or "").strip().rstrip(".")
            if not clean_id:
                return ""

            for job in manager.list_jobs(limit=100):
                if job.id == clean_id or job.id.startswith(clean_id):
                    return job.id
            return ""

        def _format_job_queue_rows(jobs):
            import time as _time

            if not jobs:
                return [["—", "No jobs", "—", "—", "—", "—"]]

            status_icons = {
                "pending": "⏳ Pending",
                "running": "🔄 Running",
                "completed": "✅ Done",
                "failed": "❌ Failed",
                "cancelled": "🚫 Cancelled",
            }

            rows = []
            for job in jobs:
                request = job.request or {}
                created = (
                    _time.strftime("%H:%M:%S", _time.localtime(job.created_at))
                    if job.created_at
                    else "—"
                )

                if job.completed_at and job.started_at:
                    elapsed = f"{job.completed_at - job.started_at:.1f}s"
                elif job.started_at:
                    elapsed = f"{_time.time() - job.started_at:.1f}s (running)"
                elif job.created_at:
                    elapsed = f"{_time.time() - job.created_at:.1f}s (queued)"
                else:
                    elapsed = "—"

                text = str(request.get("text", "")).replace("\n", " ").strip()
                if len(text) > 60:
                    text = f"{text[:57]}..."

                rows.append(
                    [
                        f"{job.id[:12]}..." if len(job.id) > 12 else job.id,
                        status_icons.get(job.status, job.status),
                        str(request.get("engine", "Unknown")),
                        created,
                        elapsed,
                        text or "—",
                    ]
                )

            return rows

        def handle_job_detail(job_id):
            """Show detailed information for a job."""
            if not job_id or not str(job_id).strip():
                return "Select a job to view details."

            import time as _time

            from job_manager import get_job_manager

            manager = get_job_manager()
            matched_job_id = _resolve_job_id(manager, job_id)
            if not matched_job_id:
                return f"❌ Job not found: {str(job_id).strip()}"

            try:
                info = manager.get_status(matched_job_id)
            except KeyError:
                return f"❌ Job not found: {matched_job_id}"

            status_icons = {
                "pending": "⏳ Pending",
                "running": "🔄 Running",
                "completed": "✅ Completed",
                "failed": "❌ Failed",
                "cancelled": "🚫 Cancelled",
            }

            request = info.request or {}
            lines = [f"### Job {info.id[:12]}...", f"**Full ID:** {info.id}"]
            lines.append(f"**Status:** {status_icons.get(info.status, info.status)}")
            lines.append(f"**Engine:** {request.get('engine', 'Unknown')}")
            lines.append(f"**Format:** {request.get('audio_format', 'wav')}")

            if info.created_at:
                lines.append(
                    "**Created:** "
                    f"{_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(info.created_at))}"
                )
            if info.started_at:
                lines.append(
                    "**Started:** "
                    f"{_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(info.started_at))}"
                )
            if info.completed_at:
                lines.append(
                    "**Completed:** "
                    f"{_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(info.completed_at))}"
                )

            if info.error:
                lines.append(f"**Error:** {info.error}")

            if info.result:
                lines.append("**Result:**")
                lines.append(f"```json\n{json.dumps(info.result, indent=2, default=str)}\n```")

            text = str(request.get("text", "")).strip()
            if text:
                preview = text[:300]
                if len(text) > 300:
                    preview = f"{preview}..."
                lines.append("**Text:**")
                lines.append(f"```\n{preview}\n```")

            return "\n\n".join(lines)

        def handle_job_panel_refresh(job_id):
            """Refresh the job queue and optionally update the selected job detail panel."""
            from job_manager import get_job_manager

            manager = get_job_manager()
            rows = _format_job_queue_rows(manager.list_jobs(limit=25))
            detail = (
                handle_job_detail(job_id)
                if str(job_id or "").strip()
                else "Select a job to view details."
            )
            return rows, detail

        def handle_job_cancel(job_id):
            """Cancel a pending or running job by ID."""
            from job_manager import get_job_manager

            if not job_id or not str(job_id).strip():
                rows, _ = handle_job_panel_refresh("")
                return rows, "⚠️ Enter a job ID to cancel.", str(job_id or "")

            manager = get_job_manager()
            matched_job_id = _resolve_job_id(manager, job_id)
            if not matched_job_id:
                rows, _ = handle_job_panel_refresh("")
                return rows, f"❌ Job not found: {str(job_id).strip()}", str(job_id).strip()

            try:
                cancelled = manager.cancel(matched_job_id)
            except KeyError:
                rows, _ = handle_job_panel_refresh("")
                return rows, f"❌ Job not found: {matched_job_id}", str(job_id).strip()
            except Exception as exc:
                rows, detail = handle_job_panel_refresh(matched_job_id)
                return rows, f"❌ Cancel failed: {exc}\n\n{detail}", matched_job_id

            rows, detail = handle_job_panel_refresh(matched_job_id)
            if cancelled:
                return (
                    rows,
                    f"✅ Job {matched_job_id[:12]}... cancelled.\n\n{detail}",
                    matched_job_id,
                )

            return (
                rows,
                f"⚠️ Cannot cancel job in terminal state.\n\n{detail}",
                matched_job_id,
            )

        def handle_job_retry(job_id):
            """Retry a completed, failed, or cancelled job using the original request."""
            from job_manager import JobRequest, get_job_manager

            if not job_id or not str(job_id).strip():
                rows, _ = handle_job_panel_refresh("")
                return rows, "⚠️ Enter a job ID to retry.", str(job_id or "")

            manager = get_job_manager()
            matched_job_id = _resolve_job_id(manager, job_id)
            if not matched_job_id:
                rows, _ = handle_job_panel_refresh("")
                return rows, f"❌ Job not found: {str(job_id).strip()}", str(job_id).strip()

            try:
                info = manager.get_status(matched_job_id)
            except KeyError:
                rows, _ = handle_job_panel_refresh("")
                return rows, f"❌ Job not found: {matched_job_id}", str(job_id).strip()

            if info.status in {"pending", "running"}:
                rows, detail = handle_job_panel_refresh(matched_job_id)
                return (
                    rows,
                    f"⚠️ Job {matched_job_id[:12]}... is still active and cannot be retried yet.\n\n{detail}",
                    matched_job_id,
                )

            request = info.request or {}
            new_job_id = manager.submit(
                JobRequest(
                    text=str(request.get("text", "")),
                    engine=str(request.get("engine", "Kokoro TTS")),
                    audio_format=str(request.get("audio_format", "wav")),
                    engine_params=dict(request.get("engine_params") or {}),
                )
            )

            rows, detail = handle_job_panel_refresh(new_job_id)
            return (
                rows,
                f"✅ Retried job {matched_job_id[:12]}... as {new_job_id[:12]}...\n\n{detail}",
                new_job_id,
            )

        def handle_job_auto_refresh_toggle(auto_enabled):
            """Toggle periodic timer polling for the Jobs tab."""
            return gr.Timer(active=bool(auto_enabled))

        def handle_clear_temp_files():
            """Handle clearing Gradio temporary files and reset audio components."""
            result_message = clear_gradio_temp_files()
            # Also clear the reference audio components since their temp files are gone
            chatterbox_audio_update = gr.update(value=None)
            fish_audio_update = gr.update(value=None)
            # Clear conversation mode speaker audio components too
            speaker_audio_updates = [gr.update(value=None) for _ in range(5)]
            # Return a simple, clean message instead of technical details
            simple_message = "✅ All temporary files cleared successfully"
            return (
                simple_message,
                chatterbox_audio_update,
                fish_audio_update,
                *speaker_audio_updates,
            )

        # IndexTTS2 management functions
        def handle_load_indextts2():
            success, message = init_indextts2_model()
            if success:
                indextts2_status_text = "✅ Loaded (Auto-selected)"
                # Auto-select IndexTTS2 engine when loaded
                selected_engine = "IndexTTS2"
                # Auto-switch to IndexTTS2 tab
                selected_tab = gr.update(selected="indextts2_tab")
            else:
                indextts2_status_text = "❌ Failed to load"
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return indextts2_status_text, selected_engine, selected_engine, selected_tab
            else:
                return indextts2_status_text, selected_engine, selected_tab

        def handle_unload_indextts2():
            message = unload_indextts2_model()
            indextts2_status_text = "⭕ Not loaded"
            # Don't change engine selection when unloading
            return indextts2_status_text

        # Qwen TTS management functions
        def handle_load_qwen(model_type, model_size):
            """Load selected Qwen TTS model."""
            success, message = init_qwen_tts_model(model_type, model_size)
            loaded_display = get_qwen_loaded_model_display()
            if success:
                qwen_status_text = "✅ Loaded"
                status_msg = f"✅ {model_type} ({model_size}) loaded successfully"
                # Auto-select appropriate Qwen engine when loaded
                if model_type == "Base":
                    selected_engine = "Qwen Voice Clone"
                    selected_mode = "voice_clone"
                    # Show/hide the right groups
                    design_visible = gr.update(visible=False)
                    clone_visible = gr.update(visible=True)
                    custom_visible = gr.update(visible=False)
                elif model_type == "VoiceDesign":
                    selected_engine = "Qwen Voice Design"
                    selected_mode = "voice_design"
                    design_visible = gr.update(visible=True)
                    clone_visible = gr.update(visible=False)
                    custom_visible = gr.update(visible=False)
                else:  # CustomVoice
                    selected_engine = "Qwen Custom Voice"
                    selected_mode = "custom_voice"
                    design_visible = gr.update(visible=False)
                    clone_visible = gr.update(visible=False)
                    custom_visible = gr.update(visible=True)
                # Auto-switch to Qwen TTS tab
                selected_tab = gr.update(selected="qwen_tab")
                # Update the model size dropdowns to match loaded model
                clone_size_update = gr.update(value=model_size)
                custom_size_update = gr.update(value=model_size)
                mode_update = gr.update(value=selected_mode)
            else:
                qwen_status_text = "❌ Failed to load"
                status_msg = message
                selected_engine = gr.update()  # No change to current selection
                selected_tab = gr.update()  # No tab change
                clone_size_update = gr.update()  # No change
                custom_size_update = gr.update()  # No change
                mode_update = gr.update()  # No change
                design_visible = gr.update()
                clone_visible = gr.update()
                custom_visible = gr.update()

            if EBOOK_CONVERTER_AVAILABLE:
                return (
                    qwen_status_text,
                    status_msg,
                    selected_engine,
                    selected_engine,
                    selected_tab,
                    loaded_display,
                    clone_size_update,
                    custom_size_update,
                    mode_update,
                    design_visible,
                    clone_visible,
                    custom_visible,
                )
            else:
                return (
                    qwen_status_text,
                    status_msg,
                    selected_engine,
                    selected_tab,
                    loaded_display,
                    clone_size_update,
                    custom_size_update,
                    mode_update,
                    design_visible,
                    clone_visible,
                    custom_visible,
                )

        def handle_unload_qwen():
            message = unload_qwen_tts_model()
            qwen_status_text = "⭕ Not loaded"
            status_msg = message
            loaded_display = get_qwen_loaded_model_display()
            return qwen_status_text, status_msg, loaded_display

        def handle_qwen_download(model_type, model_size):
            """Download selected Qwen TTS model."""
            if not QWEN_TTS_AVAILABLE:
                return "❌ Qwen TTS not available", "❌ Qwen TTS not available"

            handler = get_qwen_tts_handler()
            success, message = handler.download_model(model_type, model_size)
            # Return both download status and updated model status
            return message, handler.get_downloaded_models_status()

        def update_qwen_model_status():
            """Update Qwen TTS model status display."""
            if not QWEN_TTS_AVAILABLE:
                return "❌ Qwen TTS not available"

            handler = get_qwen_tts_handler()
            return handler.get_downloaded_models_status()

        def get_qwen_loaded_model_display():
            """Get the currently loaded Qwen TTS model for display."""
            if not QWEN_TTS_AVAILABLE:
                return "📦 **Loaded Model:** ❌ Qwen TTS not available"

            handler = get_qwen_tts_handler()
            if handler.current_model_key:
                model_type, model_size = handler.current_model_key
                return f"📦 **Loaded Model:** ✅ {model_type} ({model_size})"
            else:
                return "📦 **Loaded Model:** ⚠️ None - Load a model from Models tab first"

        def update_qwen_size_choices(model_type):
            """Update available sizes based on model type."""
            if model_type == "VoiceDesign":
                return gr.update(choices=["1.7B"], value="1.7B")
            else:
                return gr.update(choices=["0.6B", "1.7B"], value="1.7B")

        def handle_qwen_mode_change(mode):
            """Handle Qwen TTS mode change to show/hide appropriate controls."""
            if mode == "voice_design":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif mode == "voice_clone":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            elif mode == "custom_voice":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

        def handle_qwen_transcribe(audio):
            """Handle Qwen TTS audio transcription."""
            if not QWEN_TTS_AVAILABLE:
                return "❌ Qwen TTS not available"
            return transcribe_qwen_audio(audio)

        def _normalize_emotion_mode(mode_input):
            """Normalize Radio input into one of: audio_reference, vector_control, text_description.

            Radio widgets may emit either the display label or the internal value
            depending on how choices were configured. This normalizer accepts both.
            """
            aliases = {
                "audio_reference": [
                    "audio_reference",
                    "AUDIO_REFERENCE",
                    "Audio Reference",
                    "🎵 Audio Reference",
                ],
                "vector_control": [
                    "vector_control",
                    "VECTOR_CONTROL",
                    "Manual Control",
                    "🎛️ Manual Control",
                ],
                "text_description": [
                    "text_description",
                    "TEXT_DESCRIPTION",
                    "Text Description",
                    "📝 Text Description",
                ],
            }

            candidates = []
            if isinstance(mode_input, (list, tuple)):
                candidates = [str(x) for x in mode_input]
            else:
                candidates = [str(mode_input)]

            for candidate in candidates:
                for canonical, keys in aliases.items():
                    if candidate in keys:
                        return canonical
            # Fallback: return as-is to keep previous behavior
            return str(mode_input)

        def handle_indextts2_emotion_mode_change(mode):
            """Handle IndexTTS2 emotion mode changes"""
            normalized = _normalize_emotion_mode(mode)
            if normalized == "audio_reference":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif normalized == "vector_control":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            elif normalized == "text_description":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        def handle_conversation_emotion_mode_change(mode):
            """Handle conversation mode IndexTTS2 emotion mode changes"""
            normalized = _normalize_emotion_mode(mode)
            print(f"🎭 Conversation emotion mode changed to: {mode}")
            if normalized == "audio_reference":
                print("   → Showing audio reference controls")
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif normalized == "vector_control":
                print("   → Showing vector control sliders")
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
            elif normalized == "text_description":
                print("   → Showing text description input")
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            else:
                print("   → Defaulting to audio reference controls")
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        # Individual handlers for each speaker to avoid state conflicts
        def handle_speaker_1_emotion_mode_change(mode):
            return handle_conversation_emotion_mode_change(mode)

        def handle_speaker_2_emotion_mode_change(mode):
            return handle_conversation_emotion_mode_change(mode)

        def handle_speaker_3_emotion_mode_change(mode):
            return handle_conversation_emotion_mode_change(mode)

        def handle_speaker_4_emotion_mode_change(mode):
            return handle_conversation_emotion_mode_change(mode)

        def handle_speaker_5_emotion_mode_change(mode):
            return handle_conversation_emotion_mode_change(mode)

        def apply_indextts2_emotion_preset(preset_name):
            """Apply IndexTTS2 emotion preset"""
            if not INDEXTTS2_AVAILABLE or not preset_name or preset_name not in EMOTION_PRESETS:
                return [gr.update() for _ in range(8)]  # Return no changes

            preset = EMOTION_PRESETS[preset_name]
            return [
                gr.update(value=preset.get("happy", 0.0)),
                gr.update(value=preset.get("angry", 0.0)),
                gr.update(value=preset.get("sad", 0.0)),
                gr.update(value=preset.get("afraid", 0.0)),
                gr.update(value=preset.get("disgusted", 0.0)),
                gr.update(value=preset.get("melancholic", 0.0)),
                gr.update(value=preset.get("surprised", 0.0)),
                gr.update(value=preset.get("calm", 0.0)),
            ]

        # ChatterboxTTS management
        if CHATTERBOX_AVAILABLE:
            load_chatterbox_btn.click(
                fn=handle_load_chatterbox,
                outputs=(
                    [chatterbox_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [chatterbox_status, tts_engine, engine_tabs]
                ),
            )
            unload_chatterbox_btn.click(fn=handle_unload_chatterbox, outputs=[chatterbox_status])

        # Chatterbox Multilingual management
        if CHATTERBOX_MULTILINGUAL_AVAILABLE:
            load_chatterbox_mtl_btn.click(
                fn=handle_load_chatterbox_multilingual,
                outputs=(
                    [chatterbox_mtl_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [chatterbox_mtl_status, tts_engine, engine_tabs]
                ),
            )
            unload_chatterbox_mtl_btn.click(
                fn=handle_unload_chatterbox_multilingual, outputs=[chatterbox_mtl_status]
            )

        # Chatterbox Turbo management
        if CHATTERBOX_TURBO_AVAILABLE:
            load_chatterbox_turbo_btn.click(
                fn=handle_load_chatterbox_turbo,
                outputs=(
                    [chatterbox_turbo_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [chatterbox_turbo_status, tts_engine, engine_tabs]
                ),
            )
            unload_chatterbox_turbo_btn.click(
                fn=handle_unload_chatterbox_turbo, outputs=[chatterbox_turbo_status]
            )

        # Kokoro TTS management
        if KOKORO_AVAILABLE:
            load_kokoro_btn.click(
                fn=handle_load_kokoro,
                outputs=(
                    [kokoro_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [kokoro_status, tts_engine, engine_tabs]
                ),
            )
            unload_kokoro_btn.click(fn=handle_unload_kokoro, outputs=[kokoro_status])

            # Fish Speech management
        if FISH_SPEECH_AVAILABLE:
            load_fish_btn.click(
                fn=handle_load_fish,
                outputs=(
                    [fish_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [fish_status, tts_engine, engine_tabs]
                ),
            )
            unload_fish_btn.click(fn=handle_unload_fish, outputs=[fish_status])

        # IndexTTS management
        if INDEXTTS_AVAILABLE:
            load_indextts_btn.click(
                fn=handle_load_indextts,
                outputs=(
                    [indextts_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [indextts_status, tts_engine, engine_tabs]
                ),
            )
            unload_indextts_btn.click(fn=handle_unload_indextts, outputs=[indextts_status])

        # IndexTTS2 management
        if INDEXTTS2_AVAILABLE:
            load_indextts2_btn.click(
                fn=handle_load_indextts2,
                outputs=(
                    [indextts2_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [indextts2_status, tts_engine, engine_tabs]
                ),
            )
            unload_indextts2_btn.click(fn=handle_unload_indextts2, outputs=[indextts2_status])

            # IndexTTS2 emotion mode switching
            indextts2_emotion_mode.change(
                fn=handle_indextts2_emotion_mode_change,
                inputs=[indextts2_emotion_mode],
                outputs=[indextts2_audio_mode, indextts2_vector_mode, indextts2_text_mode],
            )

            # Conversation mode IndexTTS2 emotion mode switching - individual handlers
            speaker_1_emotion_mode.change(
                fn=handle_speaker_1_emotion_mode_change,
                inputs=[speaker_1_emotion_mode],
                outputs=[
                    speaker_1_emotion_audio,
                    speaker_1_emotion_description,
                    speaker_1_emotion_vectors,
                ],
            )
            speaker_2_emotion_mode.change(
                fn=handle_speaker_2_emotion_mode_change,
                inputs=[speaker_2_emotion_mode],
                outputs=[
                    speaker_2_emotion_audio,
                    speaker_2_emotion_description,
                    speaker_2_emotion_vectors,
                ],
            )
            speaker_3_emotion_mode.change(
                fn=handle_speaker_3_emotion_mode_change,
                inputs=[speaker_3_emotion_mode],
                outputs=[
                    speaker_3_emotion_audio,
                    speaker_3_emotion_description,
                    speaker_3_emotion_vectors,
                ],
            )
            speaker_4_emotion_mode.change(
                fn=handle_speaker_4_emotion_mode_change,
                inputs=[speaker_4_emotion_mode],
                outputs=[
                    speaker_4_emotion_audio,
                    speaker_4_emotion_description,
                    speaker_4_emotion_vectors,
                ],
            )
            speaker_5_emotion_mode.change(
                fn=handle_speaker_5_emotion_mode_change,
                inputs=[speaker_5_emotion_mode],
                outputs=[
                    speaker_5_emotion_audio,
                    speaker_5_emotion_description,
                    speaker_5_emotion_vectors,
                ],
            )

            # IndexTTS2 emotion preset application
            indextts2_apply_preset.click(
                fn=apply_indextts2_emotion_preset,
                inputs=[indextts2_emotion_preset],
                outputs=[
                    indextts2_happy,
                    indextts2_angry,
                    indextts2_sad,
                    indextts2_afraid,
                    indextts2_disgusted,
                    indextts2_melancholic,
                    indextts2_surprised,
                    indextts2_calm,
                ],
            )

        # Higgs Audio management
        if HIGGS_AUDIO_AVAILABLE:
            load_higgs_btn.click(
                fn=handle_load_higgs,
                outputs=(
                    [higgs_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [higgs_status, tts_engine, engine_tabs]
                ),
            )
            unload_higgs_btn.click(fn=handle_unload_higgs, outputs=[higgs_status])

        # VoxCPM management
        if VOXCPM_AVAILABLE:
            load_voxcpm_btn.click(
                fn=handle_load_voxcpm,
                outputs=(
                    [voxcpm_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [voxcpm_status, tts_engine, engine_tabs]
                ),
            )
            unload_voxcpm_btn.click(fn=handle_unload_voxcpm, outputs=[voxcpm_status])

        # KittenTTS management
        if KITTEN_TTS_AVAILABLE:
            load_kitten_btn.click(
                fn=handle_load_kitten,
                outputs=(
                    [kitten_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [kitten_status, tts_engine, engine_tabs]
                ),
            )
            unload_kitten_btn.click(fn=handle_unload_kitten, outputs=[kitten_status])

        # Qwen TTS management
        if QWEN_TTS_AVAILABLE:
            # Load button - pass model type and size
            load_qwen_btn.click(
                fn=handle_load_qwen,
                inputs=[qwen_model_type, qwen_model_size],
                outputs=(
                    [
                        qwen_status,
                        qwen_download_status,
                        tts_engine,
                        ebook_tts_engine,
                        engine_tabs,
                        qwen_loaded_model_status,
                        qwen_clone_model_size,
                        qwen_custom_model_size,
                        qwen_mode,
                        qwen_voice_design_group,
                        qwen_voice_clone_group,
                        qwen_custom_voice_group,
                    ]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [
                        qwen_status,
                        qwen_download_status,
                        tts_engine,
                        engine_tabs,
                        qwen_loaded_model_status,
                        qwen_clone_model_size,
                        qwen_custom_model_size,
                        qwen_mode,
                        qwen_voice_design_group,
                        qwen_voice_clone_group,
                        qwen_custom_voice_group,
                    ]
                ),
            )
            # Unload button
            unload_qwen_btn.click(
                fn=handle_unload_qwen,
                outputs=[qwen_status, qwen_download_status, qwen_loaded_model_status],
            )
            # Download button - updates both download status and model status
            qwen_download_btn.click(
                fn=handle_qwen_download,
                inputs=[qwen_model_type, qwen_model_size],
                outputs=[qwen_download_status, qwen_model_status],
            )
            # Model type change - update available sizes
            qwen_model_type.change(
                fn=update_qwen_size_choices, inputs=[qwen_model_type], outputs=[qwen_model_size]
            )
            # Mode change handler (in engine settings tab)
            qwen_mode.change(
                fn=handle_qwen_mode_change,
                inputs=[qwen_mode],
                outputs=[qwen_voice_design_group, qwen_voice_clone_group, qwen_custom_voice_group],
            )
            # Transcribe button handler
            qwen_transcribe_btn.click(
                fn=handle_qwen_transcribe, inputs=[qwen_ref_audio], outputs=[qwen_ref_text]
            )

        # F5-TTS management functions
        def update_f5_model_status():
            """Update F5-TTS model status display"""
            if not F5_TTS_AVAILABLE:
                return "❌ F5-TTS not available - please install"

            handler = get_f5_tts_handler()
            status = handler.get_model_status()

            status_text = "📊 **F5-TTS Model Status:**\n\n"
            for model_name, model_info in status.items():
                if model_info["downloaded"]:
                    if model_info["loaded"]:
                        status_text += f"✅ **{model_name}** - Loaded and ready\n"
                    else:
                        status_text += f"📦 **{model_name}** - Downloaded (click Load to use)\n"
                else:
                    status_text += f"⬇️ **{model_name}** - Not downloaded ({model_info['size']})\n"
                status_text += f"   *{model_info['description']}*\n\n"

            return status_text

        def handle_f5_download(model_name):
            """Handle F5-TTS model download"""
            if not F5_TTS_AVAILABLE:
                return (
                    gr.update(visible=True, value="❌ F5-TTS not available"),
                    update_f5_model_status(),
                )

            handler = get_f5_tts_handler()

            # Create progress callback
            progress_messages = []

            def progress_callback(message):
                progress_messages.append(message)
                return gr.update(visible=True, value="\n".join(progress_messages))

            # Show initial message
            yield gr.update(
                visible=True, value=f"Starting download of {model_name}..."
            ), update_f5_model_status()

            # Download with progress
            success, message = handler.download_model(model_name, progress_callback)

            if success:
                final_message = f"✅ {message}\n" + "\n".join(progress_messages)
            else:
                final_message = f"❌ {message}"

            yield gr.update(visible=True, value=final_message), update_f5_model_status()

        def handle_f5_load(model_name):
            """Handle F5-TTS model loading"""
            if not F5_TTS_AVAILABLE:
                return "❌ F5-TTS not available", update_f5_model_status(), gr.update(), gr.update()

            handler = get_f5_tts_handler()
            print(f"Attempting to load F5-TTS model: {model_name}")
            print(
                f"Handler before load - Model: {handler.model is not None}, Current: {handler.current_model}"
            )

            success, message = handler.load_model(model_name)
            print(f"Load result - Success: {success}, Message: {message}")
            print(
                f"Handler after load - Model: {handler.model is not None}, Current: {handler.current_model}"
            )

            if success:
                MODEL_STATUS["f5_tts"]["loaded"] = True
                MODEL_STATUS["f5_tts"]["current_model"] = model_name
                print(f"✅ F5-TTS model loaded successfully: {model_name}")
                print(f"MODEL_STATUS updated: {MODEL_STATUS['f5_tts']}")
                # Auto-select F5-TTS engine
                selected_engine = "F5-TTS"
                # Auto-switch to F5-TTS tab
                selected_tab = gr.update(selected="f5_tab")
            else:
                MODEL_STATUS["f5_tts"]["loaded"] = False
                print(f"❌ Failed to load F5-TTS model: {message}")
                selected_engine = gr.update()
                selected_tab = gr.update()  # No tab change

            if EBOOK_CONVERTER_AVAILABLE:
                return (
                    message,
                    update_f5_model_status(),
                    selected_engine,
                    selected_engine,
                    selected_tab,
                )
            else:
                return message, update_f5_model_status(), selected_engine, selected_tab

        def handle_f5_unload():
            """Handle F5-TTS model unloading"""
            if not F5_TTS_AVAILABLE:
                return "❌ F5-TTS not available", update_f5_model_status()

            handler = get_f5_tts_handler()
            message = handler.unload_model()
            MODEL_STATUS["f5_tts"]["loaded"] = False
            MODEL_STATUS["f5_tts"]["current_model"] = None

            return message, update_f5_model_status()

        def _normalize_protected_df_rows(current_df: Any) -> list[list[Any]]:
            normalized_rows: list[list[Any]] = []
            for row in current_df or []:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                term = str(row[0]).strip() if row[0] is not None else ""
                if not term:
                    continue
                normalized_rows.append([term, _to_bool(row[1], True)])
            return normalized_rows

        def _normalize_override_df_rows(current_df: Any) -> list[list[Any]]:
            normalized_rows: list[list[Any]] = []
            for row in current_df or []:
                if not isinstance(row, (list, tuple)) or len(row) < 3:
                    continue
                word = str(row[0]).strip() if row[0] is not None else ""
                phonetic = str(row[1]).strip() if row[1] is not None else ""
                if not word or not phonetic:
                    continue
                normalized_rows.append([word, phonetic, _to_bool(row[2], False)])
            return normalized_rows

        def handle_load_glossary():
            protected_rows, override_rows, status = _load_glossary_rows()
            return protected_rows, override_rows, status

        def handle_save_glossary(protected_df: Any, overrides_df: Any):
            protected_rows = _normalize_protected_df_rows(protected_df)
            override_rows = _normalize_override_df_rows(overrides_df)

            protected_terms = [
                ProtectedTerm(term=row[0], case_sensitive=row[1]) for row in protected_rows
            ]
            overrides = [
                PronunciationOverride(word=row[0], phonetic=row[1], case_sensitive=row[2])
                for row in override_rows
            ]

            try:
                save_lexicon(lexicon_path, protected_terms, overrides)
            except OSError as error:
                return protected_rows, override_rows, f"❌ Failed to save glossary: {error}"

            return (
                protected_rows,
                override_rows,
                f"✅ Saved glossary to {lexicon_path.name} ({len(protected_rows)} terms, {len(override_rows)} overrides)",
            )

        def handle_add_protected_term(term_text: str, case_sensitive: bool, current_df: Any):
            protected_rows = _normalize_protected_df_rows(current_df)
            normalized_term = term_text.strip()
            if not normalized_term:
                return protected_rows, "⚠️ Enter a term before adding it."

            protected_rows.append([normalized_term, bool(case_sensitive)])
            return protected_rows, f"✅ Added protected term: {normalized_term}"

        def handle_remove_protected_term(current_df: Any):
            protected_rows = _normalize_protected_df_rows(current_df)
            if not protected_rows:
                return protected_rows, "ℹ️ No protected terms to remove."

            removed_term = protected_rows.pop()[0]
            return protected_rows, f"✅ Removed protected term: {removed_term}"

        def handle_add_pronunciation_override(
            word: str,
            phonetic: str,
            case_sensitive: bool,
            current_df: Any,
        ):
            override_rows = _normalize_override_df_rows(current_df)
            normalized_word = word.strip()
            normalized_phonetic = phonetic.strip()
            if not normalized_word or not normalized_phonetic:
                return override_rows, "⚠️ Enter both a word and phonetic spelling before adding it."

            override_rows.append([normalized_word, normalized_phonetic, bool(case_sensitive)])
            return override_rows, f"✅ Added override for: {normalized_word}"

        def handle_remove_pronunciation_override(current_df: Any):
            override_rows = _normalize_override_df_rows(current_df)
            if not override_rows:
                return override_rows, "ℹ️ No pronunciation overrides to remove."

            removed_word = override_rows.pop()[0]
            return override_rows, f"✅ Removed override for: {removed_word}"

        def handle_clear_glossary():
            return [], [], "🧹 Cleared glossary tables. Save to persist the empty glossary."

        # F5-TTS event handlers
        if F5_TTS_AVAILABLE:
            # Initial status update
            demo.load(fn=update_f5_model_status, outputs=[f5_model_status])

            f5_download_btn.click(
                fn=handle_f5_download,
                inputs=[f5_model_select],
                outputs=[f5_download_status, f5_model_status],
            )

            f5_load_btn.click(
                fn=handle_f5_load,
                inputs=[f5_model_select],
                outputs=(
                    [f5_download_status, f5_model_status, tts_engine, ebook_tts_engine, engine_tabs]
                    if EBOOK_CONVERTER_AVAILABLE
                    else [f5_download_status, f5_model_status, tts_engine, engine_tabs]
                ),
            )

            f5_unload_btn.click(fn=handle_f5_unload, outputs=[f5_download_status, f5_model_status])

        # Qwen TTS initial status update
        if QWEN_TTS_AVAILABLE:
            demo.load(fn=update_qwen_model_status, outputs=[qwen_model_status])
            demo.load(fn=get_qwen_loaded_model_display, outputs=[qwen_loaded_model_status])

        demo.load(
            fn=handle_load_glossary,
            outputs=[protected_terms_df, pronunciation_overrides_df, glossary_status],
        )

        # Cleanup management
        clear_temp_btn.click(
            fn=handle_clear_temp_files,
            outputs=[
                cleanup_status,
                chatterbox_ref_audio,
                fish_ref_audio,
                speaker_1_audio,
                speaker_2_audio,
                speaker_3_audio,
                speaker_4_audio,
                speaker_5_audio,
            ],
        )

        demo.load(fn=get_voice_preset_dropdown_update, outputs=[voice_preset_dd])

        # Voice preset controls
        refresh_presets_btn.click(
            fn=on_refresh_presets, outputs=[voice_preset_dd, preset_status_md]
        )

        voice_preset_dd.change(
            fn=on_select_preset,
            inputs=[voice_preset_dd],
            outputs=[
                preset_name_tb,
                preset_status_md,
                speaker_name_tb,
                chatterbox_ref_audio,
                chatterbox_mtl_ref_audio,
                chatterbox_turbo_ref_audio,
                fish_ref_audio,
                indextts_ref_audio,
                indextts2_ref_audio,
                f5_ref_audio,
                higgs_ref_audio,
                voxcpm_ref_audio,
                qwen_ref_audio,
            ],
        )

        save_preset_btn.click(
            fn=on_save_preset,
            inputs=[preset_name_tb, preset_audio_file, copy_into_app_chk],
            outputs=[voice_preset_dd, preset_status_md, preset_name_tb],
        )

        save_storage_btn.click(
            fn=save_output_storage_settings,
            inputs=[output_storage_mode, output_storage_path],
            outputs=[storage_status_md],
        )

        output_storage_mode.change(
            fn=choose_custom_output_storage_path,
            inputs=[output_storage_mode, output_storage_path],
            outputs=[output_storage_path, storage_status_md],
        )

        open_output_folder_btn.click(fn=open_active_output_folder, outputs=[storage_status_md])

        open_autosave_folder_btn.click(fn=open_active_autosave_folder, outputs=[storage_status_md])

        delete_preset_btn.click(
            fn=on_delete_preset,
            inputs=[voice_preset_dd],
            outputs=[voice_preset_dd, preset_status_md, preset_name_tb],
        )

        llm_provider.change(
            fn=on_llm_provider_change,
            inputs=[llm_provider],
            outputs=[llm_base_url, llm_api_key, llm_model_id, llm_connection_status],
        ).then(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        llm_content_type.change(
            fn=get_content_type_system_prompt,
            inputs=[llm_content_type],
            outputs=[llm_system_prompt],
        ).then(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        llm_model_id.change(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        llm_base_url.change(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        llm_api_key.change(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        llm_system_prompt.change(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        llm_preset.change(
            fn=on_preset_change,
            inputs=[llm_preset],
            outputs=[llm_temperature, llm_top_p, llm_max_tokens],
        ).then(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        llm_refresh_models_btn.click(
            fn=refresh_llm_models,
            inputs=[llm_provider, llm_base_url, llm_api_key],
            outputs=[llm_model_id, llm_connection_status],
        ).then(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        llm_test_btn.click(
            fn=test_llm_connection,
            inputs=[llm_provider, llm_base_url, llm_api_key, llm_model_id, llm_timeout_seconds],
            outputs=[llm_connection_status],
        )

        llm_prompt_reset_btn.click(
            fn=lambda: (DEFAULT_CONTENT_TYPE_PRESET, DEFAULT_LLM_NARRATION_SYSTEM_PROMPT),
            outputs=[llm_content_type, llm_system_prompt],
        ).then(
            fn=save_llm_panel_settings,
            inputs=[
                llm_provider,
                llm_base_url,
                llm_model_id,
                llm_api_key,
                llm_content_type,
                llm_system_prompt,
                llm_preset,
            ],
        )

        assistant_send_btn.click(
            fn=handle_assistant_send,
            inputs=[
                assistant_msg_input,
                assistant_chatbot,
                assistant_llm_provider,
                assistant_llm_base_url,
                assistant_llm_api_key,
                assistant_llm_model_id,
                assistant_llm_system_prompt,
            ],
            outputs=[assistant_chatbot, assistant_msg_input, assistant_llm_status],
        )

        assistant_msg_input.submit(
            fn=handle_assistant_send,
            inputs=[
                assistant_msg_input,
                assistant_chatbot,
                assistant_llm_provider,
                assistant_llm_base_url,
                assistant_llm_api_key,
                assistant_llm_model_id,
                assistant_llm_system_prompt,
            ],
            outputs=[assistant_chatbot, assistant_msg_input, assistant_llm_status],
        )

        assistant_clear_btn.click(
            fn=handle_assistant_clear,
            inputs=[],
            outputs=[assistant_chatbot, assistant_msg_input, assistant_llm_status],
        )

        assistant_llm_test_btn.click(
            fn=handle_assistant_test_connection,
            inputs=[
                assistant_llm_provider,
                assistant_llm_base_url,
                assistant_llm_api_key,
                assistant_llm_model_id,
            ],
            outputs=[assistant_llm_status, assistant_status_indicator],
        )

        assistant_llm_save_btn.click(
            fn=handle_assistant_save_settings,
            inputs=[
                assistant_llm_provider,
                assistant_llm_base_url,
                assistant_llm_model_id,
                assistant_llm_api_key,
                assistant_llm_system_prompt,
            ],
            outputs=[assistant_llm_status],
        )

        assistant_llm_provider.change(
            fn=handle_assistant_provider_change,
            inputs=[assistant_llm_provider],
            outputs=[assistant_llm_base_url, assistant_llm_model_id],
        )

        demo.load(
            fn=handle_job_panel_refresh,
            inputs=[job_id_input],
            outputs=[job_queue_display, job_detail_output],
        )

        job_refresh_btn.click(
            fn=handle_job_panel_refresh,
            inputs=[job_id_input],
            outputs=[job_queue_display, job_detail_output],
        )

        job_timer.tick(
            fn=handle_job_panel_refresh,
            inputs=[job_id_input],
            outputs=[job_queue_display, job_detail_output],
        )

        job_auto_refresh.change(
            fn=handle_job_auto_refresh_toggle,
            inputs=[job_auto_refresh],
            outputs=[job_timer],
        )

        job_cancel_btn.click(
            fn=handle_job_cancel,
            inputs=[job_id_input],
            outputs=[job_queue_display, job_detail_output, job_id_input],
        )

        job_retry_btn.click(
            fn=handle_job_retry,
            inputs=[job_id_input],
            outputs=[job_queue_display, job_detail_output, job_id_input],
        )

        job_id_input.change(
            fn=handle_job_detail,
            inputs=[job_id_input],
            outputs=[job_detail_output],
        )

        add_protected_term_btn.click(
            fn=handle_add_protected_term,
            inputs=[protected_term_input, protected_term_case_sensitive, protected_terms_df],
            outputs=[protected_terms_df, glossary_status],
        )

        remove_protected_term_btn.click(
            fn=handle_remove_protected_term,
            inputs=[protected_terms_df],
            outputs=[protected_terms_df, glossary_status],
        )

        add_override_btn.click(
            fn=handle_add_pronunciation_override,
            inputs=[
                override_word_input,
                override_phonetic_input,
                override_case_sensitive,
                pronunciation_overrides_df,
            ],
            outputs=[pronunciation_overrides_df, glossary_status],
        )

        remove_override_btn.click(
            fn=handle_remove_pronunciation_override,
            inputs=[pronunciation_overrides_df],
            outputs=[pronunciation_overrides_df, glossary_status],
        )

        save_glossary_btn.click(
            fn=handle_save_glossary,
            inputs=[protected_terms_df, pronunciation_overrides_df],
            outputs=[protected_terms_df, pronunciation_overrides_df, glossary_status],
        )

        load_glossary_btn.click(
            fn=handle_load_glossary,
            outputs=[protected_terms_df, pronunciation_overrides_df, glossary_status],
        )

        clear_glossary_btn.click(
            fn=handle_clear_glossary,
            outputs=[protected_terms_df, pronunciation_overrides_df, glossary_status],
        )

        llm_apply_btn.click(
            fn=on_transform_preview,
            inputs=[
                text,
                llm_provider,
                llm_base_url,
                llm_api_key,
                llm_model_id,
                llm_mode,
                llm_locale,
                llm_style,
                llm_max_tag_density,
                llm_system_prompt,
                llm_timeout_seconds,
                llm_temperature,
                llm_top_p,
                llm_max_tokens,
                llm_allow_local_fallback,
                tts_engine,
            ],
            outputs=[
                preview_row,
                original_preview,
                transformed_preview,
                provenance_banner,
                action_row,
                llm_connection_status,
            ],
        )

        accept_btn.click(
            fn=on_accept_transform,
            inputs=[transformed_preview],
            outputs=[text, preview_row, provenance_banner, action_row],
        )

        reject_btn.click(
            fn=on_reject_transform,
            outputs=[preview_row, provenance_banner, action_row],
        )

        # Main generation event handler
        generate_btn.click(
            fn=generate_unified_tts_wrapped,
            inputs=[
                text,
                tts_engine,
                audio_format,
                chatterbox_ref_audio,
                chatterbox_exaggeration,
                chatterbox_temperature,
                chatterbox_cfg_weight,
                chatterbox_chunk_size,
                chatterbox_seed,
                chatterbox_mtl_ref_audio,
                chatterbox_mtl_language,
                chatterbox_mtl_exaggeration,
                chatterbox_mtl_temperature,
                chatterbox_mtl_cfg_weight,
                chatterbox_mtl_repetition_penalty,
                chatterbox_mtl_min_p,
                chatterbox_mtl_top_p,
                chatterbox_mtl_chunk_size,
                chatterbox_mtl_seed,
                chatterbox_turbo_ref_audio,
                chatterbox_turbo_exaggeration,
                chatterbox_turbo_temperature,
                chatterbox_turbo_cfg_weight,
                chatterbox_turbo_repetition_penalty,
                chatterbox_turbo_min_p,
                chatterbox_turbo_top_p,
                chatterbox_turbo_chunk_size,
                chatterbox_turbo_seed,
                kokoro_voice,
                kokoro_speed,
                fish_ref_audio,
                fish_ref_text,
                fish_temperature,
                fish_top_p,
                fish_repetition_penalty,
                fish_max_tokens,
                fish_seed,
                indextts_ref_audio,
                indextts_temperature,
                indextts_seed,
                indextts2_ref_audio,
                indextts2_emotion_mode,
                indextts2_emotion_audio,
                indextts2_emotion_description,
                indextts2_emo_alpha,
                indextts2_happy,
                indextts2_angry,
                indextts2_sad,
                indextts2_afraid,
                indextts2_disgusted,
                indextts2_melancholic,
                indextts2_surprised,
                indextts2_calm,
                indextts2_temperature,
                indextts2_top_p,
                indextts2_top_k,
                indextts2_repetition_penalty,
                indextts2_max_mel_tokens,
                indextts2_seed,
                indextts2_use_random,
                f5_ref_audio,
                f5_ref_text,
                f5_speed,
                f5_cross_fade,
                f5_remove_silence,
                f5_seed,
                higgs_ref_audio,
                higgs_ref_text,
                higgs_voice_preset,
                higgs_system_prompt,
                higgs_temperature,
                higgs_top_p,
                higgs_top_k,
                higgs_max_tokens,
                higgs_ras_win_len,
                higgs_ras_win_max_num_repeat,
                kitten_voice,
                voxcpm_ref_audio,
                voxcpm_ref_text,
                voxcpm_cfg_value,
                voxcpm_inference_timesteps,
                voxcpm_normalize,
                voxcpm_denoise,
                voxcpm_retry_badcase,
                voxcpm_retry_badcase_max_times,
                voxcpm_retry_badcase_ratio_threshold,
                voxcpm_seed,
                qwen_mode,
                qwen_voice_description,
                qwen_ref_audio,
                qwen_ref_text,
                qwen_xvector_only,
                qwen_clone_model_size,
                qwen_chunk_size,
                qwen_chunk_gap,
                qwen_speaker,
                qwen_custom_model_size,
                qwen_style_instruct,
                qwen_language,
                qwen_seed,
                llm_transform_enabled,
                llm_provider,
                llm_base_url,
                llm_api_key,
                llm_model_id,
                llm_mode,
                llm_locale,
                llm_style,
                llm_max_tag_density,
                llm_system_prompt,
                llm_timeout_seconds,
                llm_temperature,
                llm_top_p,
                llm_max_tokens,
                llm_allow_local_fallback,
                gain_db,
                enable_eq,
                eq_bass,
                eq_mid,
                eq_treble,
                enable_reverb,
                reverb_room,
                reverb_damping,
                reverb_wet,
                enable_echo,
                echo_delay,
                echo_decay,
                enable_pitch,
                pitch_semitones,
                speaker_name_tb,
                voice_preset_dd,
                autosave_enabled,
                autosave_project_name,
                autosave_store_audio_copy,
                keep_legacy_output_copy,
                last_seed_state,
            ],
            outputs=[audio_output, status_output, last_seed_out, last_seed_state],
        )

        # Conversation Mode Event Handlers
        def _conversation_engine_family(selected_engine: str) -> str:
            if selected_engine == "Kokoro TTS":
                return "kokoro"
            if selected_engine == "KittenTTS":
                return "kitten"
            if selected_engine == "IndexTTS2":
                return "indextts2"
            return "voice_clone"

        def _coerce_conversation_index(index_value) -> int | None:
            try:
                if index_value is None or index_value == "":
                    return None
                return int(index_value)
            except (TypeError, ValueError):
                return None

        def _build_conversation_table_rows(conversation_rows):
            return [
                [line_index + 1, row.get("speaker", ""), row.get("text", "")]
                for line_index, row in enumerate(conversation_rows)
            ]

        def _serialize_conversation_rows(conversation_rows) -> str:
            serialized_lines = []
            for row in conversation_rows:
                speaker = str(row.get("speaker", "") or "").strip()
                text = str(row.get("text", "") or "").strip().replace("\r", " ")
                clean_text = " ".join(segment for segment in text.splitlines() if segment.strip())
                if speaker and clean_text:
                    serialized_lines.append(f"{speaker}: {clean_text}")
            return "\n".join(serialized_lines)

        def _build_conversation_line_context(conversation_rows, selected_index: int | None) -> str:
            if selected_index is None or not (0 <= selected_index < len(conversation_rows)):
                return "Select a line above to edit it in context."

            previous_line = conversation_rows[selected_index - 1] if selected_index > 0 else None
            current_line = conversation_rows[selected_index]
            next_line = (
                conversation_rows[selected_index + 1]
                if selected_index + 1 < len(conversation_rows)
                else None
            )

            def _render_line(prefix: str, line: dict | None) -> str:
                if not line:
                    return f"- {prefix}: none"
                return f"- {prefix}: **{line['speaker']}** — {line['text']}"

            return "\n".join(
                [
                    "**Context**",
                    _render_line("Previous", previous_line),
                    _render_line("Current", current_line),
                    _render_line("Next", next_line),
                ]
            )

        def _conversation_indicator_for_speaker(
            speaker_name: str,
            speaker_index: int,
            selected_engine: str,
            voice_samples,
            ref_texts,
            kokoro_voices,
            kitten_voices,
            emotion_modes,
            speaker_settings,
        ) -> str:
            engine_family = _conversation_engine_family(selected_engine)
            default_settings = speaker_settings.get(speaker_name, {}) if speaker_settings else {}

            if engine_family == "kokoro":
                voice_value = (
                    kokoro_voices[speaker_index]
                    if speaker_index < len(kokoro_voices)
                    else default_settings.get("kokoro_voice", "af_heart")
                )
                return f"Kokoro: {voice_value or 'unassigned'}"

            if engine_family == "kitten":
                voice_value = (
                    kitten_voices[speaker_index]
                    if speaker_index < len(kitten_voices)
                    else "expr-voice-2-f"
                )
                return f"Kitten: {voice_value or 'unassigned'}"

            if engine_family == "indextts2":
                emotion_mode = (
                    emotion_modes[speaker_index]
                    if speaker_index < len(emotion_modes)
                    else "audio_reference"
                )
                emotion_label = {
                    "audio_reference": "Emotion audio",
                    "vector_control": "Manual emotion",
                    "text_description": "Text emotion",
                }.get(emotion_mode, "Emotion control")
                voice_label = (
                    "voice sample uploaded"
                    if speaker_index < len(voice_samples) and voice_samples[speaker_index]
                    else "awaiting voice sample"
                )
                return f"IndexTTS2: {emotion_label}, {voice_label}"

            has_sample = speaker_index < len(voice_samples) and bool(voice_samples[speaker_index])
            has_ref_text = speaker_index < len(ref_texts) and bool(
                str(ref_texts[speaker_index] or "").strip()
            )
            if has_sample:
                return "Voice sample: uploaded"
            if has_ref_text:
                return "Reference text: ready"
            return "Voice sample: pending"

        def _build_conversation_roster_choices(
            speakers,
            selected_engine,
            voice_samples,
            ref_texts,
            kokoro_voices,
            kitten_voices,
            emotion_modes,
            speaker_settings,
        ):
            choices = []
            for speaker_index, speaker_name in enumerate(speakers):
                indicator = _conversation_indicator_for_speaker(
                    speaker_name,
                    speaker_index,
                    selected_engine,
                    voice_samples,
                    ref_texts,
                    kokoro_voices,
                    kitten_voices,
                    emotion_modes,
                    speaker_settings,
                )
                choices.append((f"{speaker_name} — {indicator}", str(speaker_index)))
            return choices

        def _build_conversation_capabilities(selected_engine: str) -> str:
            capability_flags = ENGINE_EXPRESSIVENESS.get(
                selected_engine,
                _DEFAULT_ENGINE_EXPRESSIVENESS,
            )
            enabled_capabilities = []
            if capability_flags.get("ellipsis_pause"):
                enabled_capabilities.append("ellipsis pause cues")
            if capability_flags.get("bracket_cues"):
                enabled_capabilities.append("bracket cues")
            if capability_flags.get("emotion_vectors"):
                enabled_capabilities.append("emotion vectors")
            if capability_flags.get("ssml"):
                enabled_capabilities.append("SSML")
            if capability_flags.get("allcaps_emphasis"):
                enabled_capabilities.append("all-caps emphasis")

            capability_text = (
                ", ".join(enabled_capabilities) if enabled_capabilities else "basic line delivery"
            )
            return (
                f"**Engine profile:** {selected_engine} supports {capability_text}. "
                "The guided editor keeps the backend generation flow unchanged."
            )

        def _build_selected_character_hint(selected_engine: str, speaker_name: str) -> str:
            engine_family = _conversation_engine_family(selected_engine)
            if engine_family == "kokoro":
                return (
                    f"Configure the built-in Kokoro voice for **{speaker_name}**. "
                    "No uploaded sample is required."
                )
            if engine_family == "kitten":
                return (
                    f"Choose the KittenTTS preset voice for **{speaker_name}**. "
                    "No uploaded sample is required."
                )
            if engine_family == "indextts2":
                return f"Upload a voice sample for **{speaker_name}** and set the active emotion control mode."
            return (
                f"Upload or record a voice sample for **{speaker_name}**. "
                "Reference text helps engines that support guided cloning."
            )

        def _build_conversation_summary(
            conversation_rows,
            speakers,
            selected_engine: str,
            warnings: list[str] | None = None,
        ) -> str:
            line_count = len(conversation_rows)
            speaker_count = len(speakers)
            line_counts: dict[str, int] = {}
            for row in conversation_rows:
                speaker_name = str(row.get("speaker", "") or "")
                line_counts[speaker_name] = line_counts.get(speaker_name, 0) + 1

            speaker_breakdown = "\n".join(
                f"• {speaker}: {line_counts.get(speaker, 0)} lines" for speaker in speakers
            )

            warning_prefix = ""
            if warnings:
                warning_prefix = "\n\n".join(warnings).strip() + "\n\n"

            return (
                f"{warning_prefix}"
                f"Detected {speaker_count} speakers across {line_count} lines.\n"
                f"Active engine: {selected_engine}\n\n"
                f"{speaker_breakdown}\n\n"
                "Select a character from the roster to configure their voice."
            )

        def _build_conversation_panel_updates(
            speakers,
            selected_engine: str,
            selected_speaker_index: int | None,
        ):
            engine_family = _conversation_engine_family(selected_engine)
            audio_updates = []
            kokoro_updates = []
            kitten_updates = []
            indextts2_updates = []

            for slot_index in range(5):
                speaker_visible = (
                    slot_index < len(speakers) and slot_index == selected_speaker_index
                )
                speaker_name = (
                    speakers[slot_index]
                    if slot_index < len(speakers)
                    else f"Speaker {slot_index + 1}"
                )

                audio_updates.append(
                    gr.update(
                        visible=speaker_visible and engine_family in {"voice_clone", "indextts2"}
                    )
                )
                kokoro_updates.append(
                    gr.update(
                        visible=speaker_visible and engine_family == "kokoro",
                        label=f"🗣️ {speaker_name} Kokoro Voice",
                    )
                )
                kitten_updates.append(
                    gr.update(
                        visible=speaker_visible and engine_family == "kitten",
                        label=f"🐱 {speaker_name} KittenTTS Voice",
                    )
                )
                indextts2_updates.append(
                    gr.update(
                        visible=speaker_visible and engine_family == "indextts2",
                        label=f"🎭 {speaker_name} IndexTTS2 Emotions",
                    )
                )

            return audio_updates + kokoro_updates + kitten_updates + indextts2_updates

        def _conversation_empty_response(status_message: str):
            hidden_updates = [gr.update(visible=False) for _ in range(20)]
            return (
                status_message,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                gr.update(value="### Select a character"),
                gr.update(
                    value="Analyze a script to populate the roster, then choose a speaker to edit."
                ),
                gr.update(value=""),
                gr.update(value=[], visible=False),
                gr.update(visible=False),
                gr.update(value="Select a line from the table to begin editing."),
                gr.update(value=""),
                gr.update(choices=[], value=None),
                gr.update(value=""),
                gr.update(value="Select a line above to edit it in context."),
                [],
                [],
                None,
                None,
                {},
                *hidden_updates,
            )

        def _build_conversation_analysis_response(
            script_text,
            selected_engine,
            voice_samples=None,
            ref_texts=None,
            kokoro_voices=None,
            kitten_voices=None,
            emotion_modes=None,
            selected_speaker_index: int | None = None,
        ):
            if not isinstance(script_text, str) or not script_text.strip():
                return _conversation_empty_response("No script provided")

            conversation_rows, parse_error = parse_conversation_script(script_text)
            if parse_error:
                return _conversation_empty_response(f"Script parsing failed: {parse_error}")

            speakers = get_speaker_names_from_script(script_text)
            if not speakers:
                return _conversation_empty_response(
                    "No speakers detected. Please check the Speaker: Text format."
                )

            speaker_warnings = detect_suspect_speaker_names(speakers)

            voice_samples = list(voice_samples or [None] * 5)
            ref_texts = list(ref_texts or [""] * 5)
            kokoro_voices = list(kokoro_voices or ["af_heart"] * 5)
            kitten_voices = list(kitten_voices or ["expr-voice-2-f"] * 5)
            emotion_modes = list(emotion_modes or ["audio_reference"] * 5)

            normalized_selected_index = selected_speaker_index
            if normalized_selected_index is None or not (
                0 <= normalized_selected_index < len(speakers)
            ):
                normalized_selected_index = 0

            speaker_settings = create_default_speaker_settings(speakers)
            roster_choices = _build_conversation_roster_choices(
                speakers,
                selected_engine,
                voice_samples,
                ref_texts,
                kokoro_voices,
                kitten_voices,
                emotion_modes,
                speaker_settings,
            )
            selected_speaker_name = speakers[normalized_selected_index]
            conversation_table = _build_conversation_table_rows(conversation_rows)

            return (
                _build_conversation_summary(
                    conversation_rows,
                    speakers,
                    selected_engine,
                    warnings=speaker_warnings,
                ),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(choices=roster_choices, value=str(normalized_selected_index)),
                gr.update(value=f"### {selected_speaker_name}"),
                gr.update(
                    value=_build_selected_character_hint(selected_engine, selected_speaker_name)
                ),
                gr.update(value=_build_conversation_capabilities(selected_engine)),
                gr.update(value=conversation_table, visible=True),
                gr.update(visible=False),
                gr.update(value="Select a line from the table to begin editing."),
                gr.update(value=""),
                gr.update(choices=speakers, value=selected_speaker_name),
                gr.update(value=""),
                gr.update(value="Select a line above to edit it in context."),
                speakers,
                conversation_rows,
                normalized_selected_index,
                None,
                speaker_settings,
                *_build_conversation_panel_updates(
                    speakers,
                    selected_engine,
                    normalized_selected_index,
                ),
            )

        def handle_analyze_script(
            script_text,
            selected_engine,
            *component_values,
        ):
            """Analyze the script and populate the guided conversation editor."""
            voice_samples = component_values[0:5]
            ref_texts = component_values[5:10]
            kokoro_voices = component_values[10:15]
            kitten_voices = component_values[15:20]
            emotion_modes = component_values[20:25]
            return _build_conversation_analysis_response(
                script_text,
                selected_engine,
                voice_samples=voice_samples,
                ref_texts=ref_texts,
                kokoro_voices=kokoro_voices,
                kitten_voices=kitten_voices,
                emotion_modes=emotion_modes,
            )

        def handle_example_script(selected_engine, *component_values):
            """Load the default example and analyze it into the guided editor."""
            example_script = """Alice: Hello there! How are you doing today?
Bob: I'm doing great, thanks for asking! How about you?
Alice: I'm wonderful! I just got back from vacation.
Bob: That sounds amazing! Where did you go?
Alice: I went to Japan. It was absolutely incredible!
Bob: Japan must have been fascinating! What was your favorite part?
Alice: The food was unbelievable, and the people were so kind.
Bob: I'd love to visit Japan someday. Any recommendations?
Alice: Definitely visit Kyoto and try authentic ramen!"""
            analysis_response = handle_analyze_script(
                example_script, selected_engine, *component_values
            )
            return example_script, *analysis_response

        def handle_clear_script():
            """Reset the conversation tab to its initial empty state."""
            return "", *_conversation_empty_response("No speakers detected")

        def handle_select_speaker(
            speaker_index,
            speakers,
            selected_engine,
            *component_values,
        ):
            """Select a roster entry and show only that speaker's detail controls."""
            voice_samples = component_values[0:5]
            ref_texts = component_values[5:10]
            kokoro_voices = component_values[10:15]
            kitten_voices = component_values[15:20]
            emotion_modes = component_values[20:25]
            speaker_settings = component_values[25] if len(component_values) > 25 else {}

            if not speakers:
                hidden_updates = [gr.update(visible=False) for _ in range(20)]
                return (
                    gr.update(choices=[], value=None),
                    gr.update(value="### Select a character"),
                    gr.update(
                        value="Analyze a script to populate the roster, then choose a speaker to edit."
                    ),
                    gr.update(value=""),
                    None,
                    gr.update(value=None),
                    gr.update(value=""),
                    *hidden_updates,
                )

            normalized_index = _coerce_conversation_index(speaker_index)
            if normalized_index is None or not (0 <= normalized_index < len(speakers)):
                normalized_index = 0

            selected_speaker_name = speakers[normalized_index]
            selected_profile_name = ""
            if isinstance(speaker_settings, dict):
                current_speaker_settings = speaker_settings.get(selected_speaker_name, {})
                if isinstance(current_speaker_settings, dict):
                    selected_profile_name = str(
                        current_speaker_settings.get("selected_profile", "") or ""
                    ).strip()
            roster_choices = _build_conversation_roster_choices(
                speakers,
                selected_engine,
                voice_samples,
                ref_texts,
                kokoro_voices,
                kitten_voices,
                emotion_modes,
                speaker_settings,
            )

            return (
                gr.update(choices=roster_choices, value=str(normalized_index)),
                gr.update(value=f"### {selected_speaker_name}"),
                gr.update(
                    value=_build_selected_character_hint(selected_engine, selected_speaker_name)
                ),
                gr.update(value=_build_conversation_capabilities(selected_engine)),
                normalized_index,
                (
                    gr.update(value=selected_profile_name)
                    if selected_profile_name
                    else gr.update()
                ),
                gr.update(value=selected_profile_name),
                *_build_conversation_panel_updates(speakers, selected_engine, normalized_index),
            )

        def handle_select_line(evt: gr.SelectData, conversation_rows, speakers):
            """Populate the line editor from the selected dataframe row."""
            if not conversation_rows:
                return (
                    gr.update(visible=False),
                    gr.update(value="No script lines available."),
                    gr.update(value=""),
                    gr.update(choices=[], value=None),
                    gr.update(value=""),
                    gr.update(value="Select a line above to edit it in context."),
                    None,
                )

            row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            row_index = _coerce_conversation_index(row_index)
            if row_index is None or not (0 <= row_index < len(conversation_rows)):
                return (
                    gr.update(visible=False),
                    gr.update(value="Select a valid line from the table."),
                    gr.update(value=""),
                    gr.update(choices=speakers or [], value=None),
                    gr.update(value=""),
                    gr.update(value="Select a line above to edit it in context."),
                    None,
                )

            current_row = conversation_rows[row_index]
            return (
                gr.update(visible=True),
                gr.update(value=f"Editing line {row_index + 1}. Save to rewrite the script text."),
                gr.update(value=str(row_index + 1)),
                gr.update(choices=speakers or [], value=current_row.get("speaker", "")),
                gr.update(value=current_row.get("text", "")),
                gr.update(value=_build_conversation_line_context(conversation_rows, row_index)),
                row_index,
            )

        def handle_save_line_edit(
            selected_line_index,
            edited_speaker,
            edited_text,
            conversation_rows,
            current_script,
            speakers,
        ):
            """Apply the line editor changes back into the raw script text."""
            normalized_index = _coerce_conversation_index(selected_line_index)
            if (
                normalized_index is None
                or not conversation_rows
                or not (0 <= normalized_index < len(conversation_rows))
            ):
                return (
                    current_script,
                    gr.update(
                        value=_build_conversation_table_rows(conversation_rows or []), visible=True
                    ),
                    gr.update(value="Select a line from the table before saving edits."),
                    gr.update(value=""),
                    gr.update(choices=speakers or [], value=None),
                    gr.update(value=str(edited_text or "")),
                    gr.update(value="Select a line above to edit it in context."),
                    conversation_rows,
                    selected_line_index,
                )

            clean_speaker = str(edited_speaker or "").strip()
            clean_text = str(edited_text or "").strip()
            if not clean_speaker:
                return (
                    current_script,
                    gr.update(
                        value=_build_conversation_table_rows(conversation_rows), visible=True
                    ),
                    gr.update(value="Select a speaker before saving."),
                    gr.update(value=str(normalized_index + 1)),
                    gr.update(choices=speakers or [], value=None),
                    gr.update(value=clean_text),
                    gr.update(
                        value=_build_conversation_line_context(conversation_rows, normalized_index)
                    ),
                    conversation_rows,
                    selected_line_index,
                )
            if not clean_text:
                return (
                    current_script,
                    gr.update(
                        value=_build_conversation_table_rows(conversation_rows), visible=True
                    ),
                    gr.update(value="Line text cannot be empty."),
                    gr.update(value=str(normalized_index + 1)),
                    gr.update(choices=speakers or [], value=clean_speaker),
                    gr.update(value=""),
                    gr.update(
                        value=_build_conversation_line_context(conversation_rows, normalized_index)
                    ),
                    conversation_rows,
                    selected_line_index,
                )

            updated_rows = [dict(row) for row in conversation_rows]
            updated_rows[normalized_index] = {
                "speaker": clean_speaker,
                "text": " ".join(segment for segment in clean_text.splitlines() if segment.strip()),
            }
            updated_script = _serialize_conversation_rows(updated_rows)
            updated_context = _build_conversation_line_context(updated_rows, normalized_index)

            return (
                updated_script,
                gr.update(value=_build_conversation_table_rows(updated_rows), visible=True),
                gr.update(
                    value=f"Saved line {normalized_index + 1} back into the conversation script."
                ),
                gr.update(value=str(normalized_index + 1)),
                gr.update(choices=speakers or [], value=clean_speaker),
                gr.update(value=updated_rows[normalized_index]["text"]),
                gr.update(value=updated_context),
                updated_rows,
                normalized_index,
            )

        def handle_revert_line_edit(selected_line_index, conversation_rows, speakers):
            """Restore the line editor fields from the current parsed row state."""
            normalized_index = _coerce_conversation_index(selected_line_index)
            if (
                normalized_index is None
                or not conversation_rows
                or not (0 <= normalized_index < len(conversation_rows))
            ):
                return (
                    gr.update(value="Select a line from the table to begin editing."),
                    gr.update(value=""),
                    gr.update(choices=speakers or [], value=None),
                    gr.update(value=""),
                    gr.update(value="Select a line above to edit it in context."),
                )

            current_row = conversation_rows[normalized_index]
            return (
                gr.update(value=f"Reverted edits for line {normalized_index + 1}."),
                gr.update(value=str(normalized_index + 1)),
                gr.update(choices=speakers or [], value=current_row.get("speaker", "")),
                gr.update(value=current_row.get("text", "")),
                gr.update(
                    value=_build_conversation_line_context(conversation_rows, normalized_index)
                ),
            )

        def handle_ai_format_script(
            script_text,
            provider_name,
            base_url,
            api_key,
            model_id,
            timeout_seconds,
        ):
            """Use the configured LLM provider to normalize free-form dialogue into Speaker: Text format."""
            if not isinstance(script_text, str) or not script_text.strip():
                return script_text, "❌ Enter script text before using AI Format."

            provider_config = _get_provider_config(provider_name)
            resolved_api_key, _api_key_source = resolve_llm_api_key(provider_name, api_key)
            narration_script, error_message = format_conversation_with_llm(
                text=script_text,
                base_url=str(base_url or "").strip(),
                api_key=resolved_api_key,
                model_id=str(model_id or "").strip(),
                timeout_seconds=int(timeout_seconds),
                extra_headers=dict(provider_config.get("headers", {})),
                auth_style=provider_config.get("auth_style", "bearer"),
            )
            if error_message:
                return script_text, f"❌ {error_message}"

            formatted_rows = [
                {"speaker": line.speaker, "text": line.text} for line in narration_script.lines
            ]
            formatted_script = _serialize_conversation_rows(formatted_rows)
            return (
                formatted_script,
                (
                    f"✨ AI formatted {len(narration_script.lines)} lines across "
                    f"{len(narration_script.speakers)} speakers."
                ),
            )

        def handle_cast_characters(
            speakers_state,
            provider_name,
            base_url,
            api_key,
            model_id,
            timeout_seconds,
        ):
            """Use the configured LLM to generate voice profiles for discovered speakers."""
            if (
                not speakers_state
                or not isinstance(speakers_state, list)
                or len(speakers_state) == 0
            ):
                return _speaker_profile_status_update(
                    "❌ Analyze a script first to discover speakers."
                )

            clean_base_url = str(base_url or "").strip()
            clean_model_id = str(model_id or "").strip()
            if not clean_base_url or not clean_model_id:
                return _speaker_profile_status_update(
                    "❌ Configure the LLM provider base URL and model ID before casting."
                )

            provider_config = _get_provider_config(provider_name)
            resolved_api_key, _ = resolve_llm_api_key(provider_name, api_key)
            if provider_config.get("requires_api_key") and not resolved_api_key:
                return _speaker_profile_status_update(
                    f"❌ {provider_name} API key required before casting.\n"
                    + get_llm_shell_key_setup_hint(provider_name)
                )

            result_text, error_msg = generate_voice_casting(
                speaker_names=speakers_state,
                base_url=clean_base_url,
                api_key=resolved_api_key,
                model_id=clean_model_id,
                timeout_seconds=int(timeout_seconds or 60),
                extra_headers=dict(provider_config.get("headers", {})),
                auth_style=provider_config.get("auth_style", "bearer"),
            )

            if error_msg:
                return _speaker_profile_status_update(f"❌ Cast failed: {error_msg}")

            return _speaker_profile_status_update(f"🎭 Voice Casting Results:\n\n{result_text}")

        def handle_tts_engine_change(selected_engine):
            """Handle TTS engine selection changes and update UI accordingly."""
            print(f"🎯 TTS Engine changed to: {selected_engine}")

            # Enable conversation mode for all engines now including Kokoro TTS
            conversation_info_text = "Ready for conversation generation..."
            return (
                gr.update(visible=False),  # Keep conversation button hidden until script analyzed
                gr.update(visible=True, value=conversation_info_text),  # Reset conversation info
                gr.update(interactive=True),  # Enable conversation script
                gr.update(interactive=True),  # Enable analyze button
                gr.update(interactive=True),  # Enable AI format button
                gr.update(interactive=True),  # Enable example button
                gr.update(interactive=True),  # Enable clear button
                gr.update(interactive=True),  # Enable pause slider
                gr.update(interactive=True),  # Enable transition pause slider
            )

        def handle_generate_conversation_advanced(
            script_text,
            pause_duration,
            transition_pause,
            audio_format,
            voice_samples,
            ref_texts,
            kokoro_voices,
            kitten_voices,
            selected_engine,
            emotion_modes=None,
            emotion_audios=None,
            emotion_descriptions=None,
            emotion_vectors=None,
        ):
            """Generate the multi-voice conversation with voice samples or Kokoro voice selections."""
            print(f"🎭 Conversation handler called with engine: {selected_engine}")

            if not script_text.strip():
                return None, "❌ No conversation script provided"

            try:
                # For Kokoro TTS, use the selected voices instead of voice samples
                if selected_engine == "Kokoro TTS":
                    result = generate_conversation_audio_kokoro(
                        script_text,
                        kokoro_voices,
                        selected_engine=selected_engine,
                        conversation_pause_duration=pause_duration,
                        speaker_transition_pause=transition_pause,
                        effects_settings=None,
                        audio_format=audio_format,
                    )
                elif selected_engine == "KittenTTS":
                    # For KittenTTS, use the selected voices
                    result = generate_conversation_audio_kitten(
                        script_text,
                        kitten_voices,
                        selected_engine=selected_engine,
                        conversation_pause_duration=pause_duration,
                        speaker_transition_pause=transition_pause,
                        effects_settings=None,
                        audio_format=audio_format,
                    )
                elif selected_engine == "IndexTTS2":
                    # For IndexTTS2, use emotion controls
                    result = generate_conversation_audio_indextts2(
                        script_text,
                        voice_samples,
                        emotion_modes or [],
                        emotion_audios or [],
                        emotion_descriptions or [],
                        emotion_vectors or [],
                        selected_engine=selected_engine,
                        conversation_pause_duration=pause_duration,
                        speaker_transition_pause=transition_pause,
                        effects_settings=None,
                        audio_format=audio_format,
                    )
                else:
                    # Use the original function for other engines
                    result = generate_conversation_audio_simple(
                        script_text,
                        voice_samples,
                        ref_texts=ref_texts,
                        selected_engine=selected_engine,
                        conversation_pause_duration=pause_duration,
                        speaker_transition_pause=transition_pause,
                        effects_settings=None,
                        audio_format=audio_format,
                    )

                if result[0] is None:
                    print(f"❌ Conversation generation failed: {result[1]}")
                    return None, result[1]

                audio_data, summary = result
                summary_text = format_conversation_info(summary)

                print(f"✅ Conversation generated successfully")
                return audio_data, summary_text

            except Exception as e:
                import traceback

                traceback.print_exc()
                error_msg = f"❌ Generation error: {str(e)}"
                print(f"❌ Exception in conversation handler: {error_msg}")
                return None, error_msg

        def handle_generate_conversation_simple(
            script_text,
            pause_duration,
            transition_pause,
            audio_format,
            voice_samples,
            ref_texts,
            selected_engine,
        ):
            """Generate the multi-voice conversation with voice samples - Simplified version."""
            print(f"🎭 Conversation handler called with engine: {selected_engine}")

            if not script_text.strip():
                return None, "❌ No conversation script provided"

            try:
                # Generate the conversation audio using the simplified function
                result = generate_conversation_audio_simple(
                    script_text,
                    voice_samples,
                    ref_texts=ref_texts,
                    selected_engine=selected_engine,
                    conversation_pause_duration=pause_duration,
                    speaker_transition_pause=transition_pause,
                    effects_settings=None,  # Effects will be applied from the main UI
                    audio_format=audio_format,
                )

                if result[0] is None:
                    print(f"❌ Conversation generation failed: {result[1]}")
                    return None, result[1]  # Return error message

                audio_data, summary = result
                summary_text = format_conversation_info(summary)

                print(
                    f"✅ Conversation generated successfully, returning summary: {summary_text[:100]}..."
                )
                return audio_data, summary_text

            except Exception as e:
                import traceback

                traceback.print_exc()
                error_msg = f"❌ Generation error: {str(e)}"
                print(f"❌ Exception in conversation handler: {error_msg}")
                return None, error_msg

        # Wire up conversation mode event handlers
        analyze_script_btn.click(
            fn=handle_analyze_script,
            inputs=conversation_analyze_inputs,
            outputs=conversation_analysis_outputs,
        )

        example_script_btn.click(
            fn=handle_example_script,
            inputs=[tts_engine, *conversation_component_state_inputs],
            outputs=[conversation_script, *conversation_analysis_outputs],
        )

        clear_script_btn.click(
            fn=handle_clear_script,
            outputs=[conversation_script, *conversation_analysis_outputs],
        )

        speaker_profile_selector.change(
            fn=on_load_speaker_profile,
            inputs=[
                speaker_profile_selector,
                conversation_speaker_settings_state,
                conversation_selected_speaker_state,
                conversation_speakers_state,
            ],
            outputs=[
                conversation_speaker_settings_state,
                speaker_1_audio,
                speaker_2_audio,
                speaker_3_audio,
                speaker_4_audio,
                speaker_5_audio,
                speaker_1_ref_text,
                speaker_2_ref_text,
                speaker_3_ref_text,
                speaker_4_ref_text,
                speaker_5_ref_text,
                speaker_profile_name_input,
                speaker_profile_status,
            ],
        )

        save_speaker_profile_btn.click(
            fn=on_save_speaker_profile,
            inputs=[speaker_profile_name_input, conversation_speaker_settings_state],
            outputs=[speaker_profile_selector, speaker_profile_status],
        )

        delete_speaker_profile_btn.click(
            fn=on_delete_speaker_profile,
            inputs=[speaker_profile_selector],
            outputs=[speaker_profile_selector, speaker_profile_status],
        )

        cast_characters_btn.click(
            fn=handle_cast_characters,
            inputs=[
                conversation_speakers_state,
                llm_provider,
                llm_base_url,
                llm_api_key,
                llm_model_id,
                llm_timeout_seconds,
            ],
            outputs=[speaker_profile_status],
        )

        ai_format_script_btn.click(
            fn=handle_ai_format_script,
            inputs=[
                conversation_script,
                llm_provider,
                llm_base_url,
                llm_api_key,
                llm_model_id,
                llm_timeout_seconds,
            ],
            outputs=[conversation_script, conversation_info],
        ).then(
            fn=handle_analyze_script,
            inputs=conversation_analyze_inputs,
            outputs=conversation_analysis_outputs,
        )

        character_roster.change(
            fn=handle_select_speaker,
            inputs=[
                character_roster,
                conversation_speakers_state,
                tts_engine,
                *conversation_component_state_inputs,
                conversation_speaker_settings_state,
            ],
            outputs=[
                character_roster,
                selected_character_header,
                selected_character_hint,
                selected_character_capabilities,
                conversation_selected_speaker_state,
                speaker_profile_selector,
                speaker_profile_name_input,
                *conversation_panel_updates,
            ],
        )

        conversation_lines_df.select(
            fn=handle_select_line,
            inputs=[conversation_rows_state, conversation_speakers_state],
            outputs=[
                line_editor_group,
                line_editor_status,
                line_number_display,
                line_speaker_editor,
                line_text_editor,
                line_context_preview,
                conversation_selected_line_state,
            ],
        )

        save_line_edit_btn.click(
            fn=handle_save_line_edit,
            inputs=[
                conversation_selected_line_state,
                line_speaker_editor,
                line_text_editor,
                conversation_rows_state,
                conversation_script,
                conversation_speakers_state,
            ],
            outputs=[
                conversation_script,
                conversation_lines_df,
                line_editor_status,
                line_number_display,
                line_speaker_editor,
                line_text_editor,
                line_context_preview,
                conversation_rows_state,
                conversation_selected_line_state,
            ],
        )

        revert_line_edit_btn.click(
            fn=handle_revert_line_edit,
            inputs=[
                conversation_selected_line_state,
                conversation_rows_state,
                conversation_speakers_state,
            ],
            outputs=[
                line_editor_status,
                line_number_display,
                line_speaker_editor,
                line_text_editor,
                line_context_preview,
            ],
        )

        generate_conversation_btn.click(
            fn=lambda script, pause, trans_pause, audio_fmt, s1, s2, s3, s4, s5, rt1, rt2, rt3, rt4, rt5, kv1, kv2, kv3, kv4, kv5, ktv1, ktv2, ktv3, ktv4, ktv5, engine, em1, ea1, ed1, h1, s1_sad, a1, af1, su1, c1, em2, ea2, ed2, h2, s2_sad, a2, af2, su2, c2, em3, ea3, ed3, h3, s3_sad, a3, af3, su3, c3, em4, ea4, ed4, h4, s4_sad, a4, af4, su4, c4, em5, ea5, ed5, h5, s5_sad, a5, af5, su5, c5: handle_generate_conversation_advanced(
                script,
                pause,
                trans_pause,
                audio_fmt,
                [s1, s2, s3, s4, s5],
                [rt1, rt2, rt3, rt4, rt5],
                [kv1, kv2, kv3, kv4, kv5],
                [ktv1, ktv2, ktv3, ktv4, ktv5],
                engine,
                # IndexTTS2 emotion parameters
                [em1, em2, em3, em4, em5],  # emotion_modes
                [ea1, ea2, ea3, ea4, ea5],  # emotion_audios
                [ed1, ed2, ed3, ed4, ed5],  # emotion_descriptions
                [
                    {
                        "happy": h1,
                        "sad": s1_sad,
                        "angry": a1,
                        "afraid": af1,
                        "surprised": su1,
                        "calm": c1,
                    },
                    {
                        "happy": h2,
                        "sad": s2_sad,
                        "angry": a2,
                        "afraid": af2,
                        "surprised": su2,
                        "calm": c2,
                    },
                    {
                        "happy": h3,
                        "sad": s3_sad,
                        "angry": a3,
                        "afraid": af3,
                        "surprised": su3,
                        "calm": c3,
                    },
                    {
                        "happy": h4,
                        "sad": s4_sad,
                        "angry": a4,
                        "afraid": af4,
                        "surprised": su4,
                        "calm": c4,
                    },
                    {
                        "happy": h5,
                        "sad": s5_sad,
                        "angry": a5,
                        "afraid": af5,
                        "surprised": su5,
                        "calm": c5,
                    },
                ],  # emotion_vectors
            ),
            inputs=[
                conversation_script,
                conversation_pause,
                speaker_transition_pause,
                audio_format,  # Use the same audio format selector as single voice mode
                speaker_1_audio,
                speaker_2_audio,
                speaker_3_audio,
                speaker_4_audio,
                speaker_5_audio,
                speaker_1_ref_text,
                speaker_2_ref_text,
                speaker_3_ref_text,
                speaker_4_ref_text,
                speaker_5_ref_text,
                speaker_1_kokoro_voice,
                speaker_2_kokoro_voice,
                speaker_3_kokoro_voice,
                speaker_4_kokoro_voice,
                speaker_5_kokoro_voice,
                speaker_1_kitten_voice,
                speaker_2_kitten_voice,
                speaker_3_kitten_voice,
                speaker_4_kitten_voice,
                speaker_5_kitten_voice,
                tts_engine,  # Use the main TTS engine selector
                # IndexTTS2 emotion controls
                speaker_1_emotion_mode,
                speaker_1_emotion_audio,
                speaker_1_emotion_description,
                speaker_1_happy,
                speaker_1_sad,
                speaker_1_angry,
                speaker_1_afraid,
                speaker_1_surprised,
                speaker_1_calm,
                speaker_2_emotion_mode,
                speaker_2_emotion_audio,
                speaker_2_emotion_description,
                speaker_2_happy,
                speaker_2_sad,
                speaker_2_angry,
                speaker_2_afraid,
                speaker_2_surprised,
                speaker_2_calm,
                speaker_3_emotion_mode,
                speaker_3_emotion_audio,
                speaker_3_emotion_description,
                speaker_3_happy,
                speaker_3_sad,
                speaker_3_angry,
                speaker_3_afraid,
                speaker_3_surprised,
                speaker_3_calm,
                speaker_4_emotion_mode,
                speaker_4_emotion_audio,
                speaker_4_emotion_description,
                speaker_4_happy,
                speaker_4_sad,
                speaker_4_angry,
                speaker_4_afraid,
                speaker_4_surprised,
                speaker_4_calm,
                speaker_5_emotion_mode,
                speaker_5_emotion_audio,
                speaker_5_emotion_description,
                speaker_5_happy,
                speaker_5_sad,
                speaker_5_angry,
                speaker_5_afraid,
                speaker_5_surprised,
                speaker_5_calm,
            ],
            outputs=[audio_output, conversation_info],  # Use same audio output as single voice mode
        )

        # Speaker transcribe button handlers for conversation mode
        speaker_1_transcribe_btn.click(
            fn=handle_qwen_transcribe, inputs=[speaker_1_audio], outputs=[speaker_1_ref_text]
        )
        speaker_2_transcribe_btn.click(
            fn=handle_qwen_transcribe, inputs=[speaker_2_audio], outputs=[speaker_2_ref_text]
        )
        speaker_3_transcribe_btn.click(
            fn=handle_qwen_transcribe, inputs=[speaker_3_audio], outputs=[speaker_3_ref_text]
        )
        speaker_4_transcribe_btn.click(
            fn=handle_qwen_transcribe, inputs=[speaker_4_audio], outputs=[speaker_4_ref_text]
        )
        speaker_5_transcribe_btn.click(
            fn=handle_qwen_transcribe, inputs=[speaker_5_audio], outputs=[speaker_5_ref_text]
        )

        speaker_1_audio.change(
            fn=lambda audio, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                0,
                "ref_audio",
                str(audio or ""),
            ),
            inputs=[
                speaker_1_audio,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_2_audio.change(
            fn=lambda audio, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                1,
                "ref_audio",
                str(audio or ""),
            ),
            inputs=[
                speaker_2_audio,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_3_audio.change(
            fn=lambda audio, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                2,
                "ref_audio",
                str(audio or ""),
            ),
            inputs=[
                speaker_3_audio,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_4_audio.change(
            fn=lambda audio, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                3,
                "ref_audio",
                str(audio or ""),
            ),
            inputs=[
                speaker_4_audio,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_5_audio.change(
            fn=lambda audio, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                4,
                "ref_audio",
                str(audio or ""),
            ),
            inputs=[
                speaker_5_audio,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )

        speaker_1_ref_text.change(
            fn=lambda text_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                0,
                "fish_ref_text",
                str(text_value or ""),
            ),
            inputs=[
                speaker_1_ref_text,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_2_ref_text.change(
            fn=lambda text_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                1,
                "fish_ref_text",
                str(text_value or ""),
            ),
            inputs=[
                speaker_2_ref_text,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_3_ref_text.change(
            fn=lambda text_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                2,
                "fish_ref_text",
                str(text_value or ""),
            ),
            inputs=[
                speaker_3_ref_text,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_4_ref_text.change(
            fn=lambda text_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                3,
                "fish_ref_text",
                str(text_value or ""),
            ),
            inputs=[
                speaker_4_ref_text,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_5_ref_text.change(
            fn=lambda text_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                4,
                "fish_ref_text",
                str(text_value or ""),
            ),
            inputs=[
                speaker_5_ref_text,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )

        speaker_1_kokoro_voice.change(
            fn=lambda voice_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                0,
                "kokoro_voice",
                voice_value,
            ),
            inputs=[
                speaker_1_kokoro_voice,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_2_kokoro_voice.change(
            fn=lambda voice_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                1,
                "kokoro_voice",
                voice_value,
            ),
            inputs=[
                speaker_2_kokoro_voice,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_3_kokoro_voice.change(
            fn=lambda voice_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                2,
                "kokoro_voice",
                voice_value,
            ),
            inputs=[
                speaker_3_kokoro_voice,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_4_kokoro_voice.change(
            fn=lambda voice_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                3,
                "kokoro_voice",
                voice_value,
            ),
            inputs=[
                speaker_4_kokoro_voice,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )
        speaker_5_kokoro_voice.change(
            fn=lambda voice_value, settings, speakers: update_conversation_speaker_setting(
                settings,
                speakers,
                4,
                "kokoro_voice",
                voice_value,
            ),
            inputs=[
                speaker_5_kokoro_voice,
                conversation_speaker_settings_state,
                conversation_speakers_state,
            ],
            outputs=[conversation_speaker_settings_state],
        )

        tts_engine.change(
            fn=update_conversation_tts_engine,
            inputs=[tts_engine, conversation_speaker_settings_state, conversation_speakers_state],
            outputs=[conversation_speaker_settings_state],
        )

        # Function to switch tabs based on TTS engine selection
        def switch_engine_tab(selected_engine):
            """Switch to the appropriate tab when TTS engine is selected."""
            tab_mapping = {
                "ChatterboxTTS": "chatterbox_tab",
                "Kokoro TTS": "kokoro_tab",
                "Fish Speech": "fish_tab",
                "IndexTTS": "indextts_tab",
                "F5-TTS": "f5_tab",
                "Higgs Audio": "higgs_tab",
                "KittenTTS": "kitten_tab",
            }

            if selected_engine in tab_mapping:
                return gr.update(selected=tab_mapping[selected_engine])
            return gr.update()

        # VoxCPM auto-transcription when reference audio is uploaded
        if VOXCPM_AVAILABLE:

            def handle_voxcpm_transcription(audio_path):
                """Auto-transcribe VoxCPM reference audio using Whisper"""
                if not audio_path:
                    return ""
                try:
                    transcription = transcribe_voxcpm_audio(audio_path)
                    return transcription
                except Exception as e:
                    print(f"❌ VoxCPM transcription error: {e}")
                    return ""

            voxcpm_ref_audio.change(
                fn=handle_voxcpm_transcription,
                inputs=[voxcpm_ref_audio],
                outputs=[voxcpm_ref_text],
                show_progress="minimal",
            )

        # Handle TTS engine changes to enable/disable conversation mode and switch tabs
        tts_engine.change(
            fn=handle_tts_engine_change,
            inputs=[tts_engine],
            outputs=[
                generate_conversation_btn,  # Show/hide conversation button
                conversation_info,  # Update conversation info text
                conversation_script,  # Enable/disable script input
                analyze_script_btn,  # Enable/disable analyze button
                ai_format_script_btn,  # Enable/disable AI format button
                example_script_btn,  # Enable/disable example button
                clear_script_btn,  # Enable/disable clear button
                conversation_pause,  # Enable/disable pause slider
                speaker_transition_pause,  # Enable/disable transition pause slider
            ],
        ).then(
            fn=handle_analyze_script,
            inputs=conversation_analyze_inputs,
            outputs=conversation_analysis_outputs,
        ).then(
            fn=switch_engine_tab, inputs=[tts_engine], outputs=[engine_tabs]
        )

        # eBook conversion event handlers
        if EBOOK_CONVERTER_AVAILABLE:

            def handle_ebook_analysis(file_path):
                """Handle eBook file analysis."""
                if not file_path:
                    return "Please upload an eBook file first.", gr.update(
                        choices=[], visible=False
                    )

                analysis_result = analyze_ebook_file(file_path)
                info_display = get_ebook_info_display(analysis_result)

                if analysis_result["success"]:
                    # Create chapter choices for selection
                    chapter_choices = [
                        (f"{i+1}. {ch['title']}", i)
                        for i, ch in enumerate(analysis_result["chapters"])
                    ]
                    return info_display, gr.update(choices=chapter_choices, visible=True)
                else:
                    return info_display, gr.update(choices=[], visible=False)

            def handle_ebook_conversion(
                file_path,
                tts_engine_choice,
                selected_chapters,
                chunk_length,
                ebook_format,
                # All the TTS parameters need to be passed through
                cb_ref_audio,
                cb_exag,
                cb_temp,
                cb_cfg,
                cb_seed,
                # Chatterbox Multilingual parameters
                cb_mtl_ref_audio,
                cb_mtl_lang,
                cb_mtl_exag,
                cb_mtl_temp,
                cb_mtl_cfg,
                cb_mtl_rep_pen,
                cb_mtl_min_p,
                cb_mtl_top_p,
                cb_mtl_seed,
                # Chatterbox Turbo parameters
                cb_turbo_ref_audio,
                cb_turbo_exag,
                cb_turbo_temp,
                cb_turbo_cfg,
                cb_turbo_rep_pen,
                cb_turbo_min_p,
                cb_turbo_top_p,
                cb_turbo_seed,
                kok_voice,
                kok_speed,
                fish_ref_audio,
                fish_ref_text,
                fish_temp,
                fish_top_p,
                fish_rep_pen,
                fish_max_tok,
                fish_seed_val,
                # IndexTTS parameters
                idx_ref_audio,
                idx_temp,
                idx_seed,
                # IndexTTS2 parameters
                indextts2_ref_audio,
                indextts2_emotion_mode,
                indextts2_emotion_audio,
                indextts2_emotion_description,
                indextts2_emo_alpha,
                indextts2_happy,
                indextts2_angry,
                indextts2_sad,
                indextts2_afraid,
                indextts2_disgusted,
                indextts2_melancholic,
                indextts2_surprised,
                indextts2_calm,
                indextts2_temperature,
                indextts2_top_p,
                indextts2_top_k,
                indextts2_repetition_penalty,
                indextts2_max_mel_tokens,
                indextts2_seed,
                indextts2_use_random,
                # F5-TTS parameters
                f5_ref_audio,
                f5_ref_text,
                f5_speed,
                f5_cross_fade,
                f5_remove_silence,
                f5_seed_val,
                # Higgs Audio parameters
                higgs_ref_audio,
                higgs_ref_text,
                higgs_voice_preset,
                higgs_system_prompt,
                higgs_temperature,
                higgs_top_p,
                higgs_top_k,
                higgs_max_tokens,
                higgs_ras_win_len,
                higgs_ras_win_max_num_repeat,
                # KittenTTS parameters
                kitten_voice_param,
                # Qwen TTS parameters
                qwen_ref_audio_param,
                qwen_ref_text_param,
                qwen_language_param,
                qwen_xvector_only_param,
                qwen_clone_model_size_param,
                qwen_seed_param,
                # VoxCPM parameters
                voxcpm_ref_audio,
                voxcpm_ref_text,
                voxcpm_cfg_value,
                voxcpm_inference_timesteps,
                voxcpm_normalize,
                voxcpm_denoise,
                voxcpm_retry_badcase,
                voxcpm_retry_badcase_max_times,
                voxcpm_retry_badcase_ratio_threshold,
                voxcpm_seed,
                gain,
                eq_en,
                eq_b,
                eq_m,
                eq_t,
                rev_en,
                rev_room,
                rev_damp,
                rev_wet,
                echo_en,
                echo_del,
                echo_dec,
                pitch_en,
                pitch_semi,
                # Advanced eBook settings
                chunk_gap,
                chapter_gap,
            ):
                """Handle eBook to audiobook conversion."""
                if not file_path:
                    return None, None, "Please upload an eBook file first."

                result = convert_ebook_to_audiobook(
                    file_path,
                    tts_engine_choice,
                    selected_chapters,
                    chunk_length,
                    ebook_format,
                    cb_ref_audio,
                    cb_exag,
                    cb_temp,
                    cb_cfg,
                    cb_seed,
                    # Chatterbox Multilingual parameters
                    cb_mtl_ref_audio,
                    cb_mtl_lang,
                    cb_mtl_exag,
                    cb_mtl_temp,
                    cb_mtl_cfg,
                    cb_mtl_rep_pen,
                    cb_mtl_min_p,
                    cb_mtl_top_p,
                    cb_mtl_seed,
                    # Chatterbox Turbo parameters
                    cb_turbo_ref_audio,
                    cb_turbo_exag,
                    cb_turbo_temp,
                    cb_turbo_cfg,
                    cb_turbo_rep_pen,
                    cb_turbo_min_p,
                    cb_turbo_top_p,
                    cb_turbo_seed,
                    kok_voice,
                    kok_speed,
                    fish_ref_audio,
                    fish_ref_text,
                    fish_temp,
                    fish_top_p,
                    fish_rep_pen,
                    fish_max_tok,
                    fish_seed_val,
                    # IndexTTS parameters
                    idx_ref_audio,
                    idx_temp,
                    idx_seed,
                    # IndexTTS2 parameters
                    indextts2_ref_audio,
                    indextts2_emotion_mode,
                    indextts2_emotion_audio,
                    indextts2_emotion_description,
                    indextts2_emo_alpha,
                    indextts2_happy,
                    indextts2_angry,
                    indextts2_sad,
                    indextts2_afraid,
                    indextts2_disgusted,
                    indextts2_melancholic,
                    indextts2_surprised,
                    indextts2_calm,
                    indextts2_temperature,
                    indextts2_top_p,
                    indextts2_top_k,
                    indextts2_repetition_penalty,
                    indextts2_max_mel_tokens,
                    indextts2_seed,
                    indextts2_use_random,
                    # F5-TTS parameters
                    f5_ref_audio,
                    f5_ref_text,
                    f5_speed,
                    f5_cross_fade,
                    f5_remove_silence,
                    f5_seed_val,
                    # Higgs Audio parameters
                    higgs_ref_audio,
                    higgs_ref_text,
                    higgs_voice_preset,
                    higgs_system_prompt,
                    higgs_temperature,
                    higgs_top_p,
                    higgs_top_k,
                    higgs_max_tokens,
                    higgs_ras_win_len,
                    higgs_ras_win_max_num_repeat,
                    # KittenTTS parameters
                    kitten_voice_param,
                    # Qwen TTS parameters
                    qwen_ref_audio_param,
                    qwen_ref_text_param,
                    qwen_language_param,
                    qwen_xvector_only_param,
                    qwen_clone_model_size_param,
                    qwen_seed_param,
                    gain,
                    eq_en,
                    eq_b,
                    eq_m,
                    eq_t,
                    rev_en,
                    rev_room,
                    rev_damp,
                    rev_wet,
                    echo_en,
                    echo_del,
                    echo_dec,
                    pitch_en,
                    pitch_semi,
                    # Advanced eBook settings
                    chunk_gap,
                    chapter_gap,
                )

                if result[0] is None:
                    # Error case
                    return None, gr.update(visible=False), result[1]

                audio_result, status_message = result

                # Check if result is a file path (large file) or audio data (small file)
                if isinstance(audio_result, str):
                    # Large file - return file path for download
                    return None, gr.update(value=audio_result, visible=True), status_message
                else:
                    # Small file - return audio data for playback
                    return audio_result, gr.update(visible=False), status_message

            def handle_clear_ebook():
                """Clear all eBook-related inputs and outputs."""
                return (
                    None,  # ebook_file
                    "Upload an eBook file and click 'Analyze eBook' to see details.",  # ebook_info
                    gr.update(choices=[], value=[], visible=False),  # chapter_selection
                    None,  # audiobook_output
                    gr.update(visible=False),  # audiobook_download
                    "",  # ebook_status
                )

            # Connect eBook analysis
            analyze_btn.click(
                fn=handle_ebook_analysis,
                inputs=[ebook_file],
                outputs=[ebook_info, chapter_selection],
            )

            # Connect eBook conversion
            convert_ebook_btn.click(
                fn=handle_ebook_conversion,
                inputs=[
                    ebook_file,
                    ebook_tts_engine,
                    chapter_selection,
                    ebook_chunk_length,
                    ebook_audio_format,
                    # ChatterboxTTS parameters
                    chatterbox_ref_audio,
                    chatterbox_exaggeration,
                    chatterbox_temperature,
                    chatterbox_cfg_weight,
                    chatterbox_seed,
                    # Chatterbox Multilingual parameters
                    chatterbox_mtl_ref_audio,
                    chatterbox_mtl_language,
                    chatterbox_mtl_exaggeration,
                    chatterbox_mtl_temperature,
                    chatterbox_mtl_cfg_weight,
                    chatterbox_mtl_repetition_penalty,
                    chatterbox_mtl_min_p,
                    chatterbox_mtl_top_p,
                    chatterbox_mtl_seed,
                    # Chatterbox Turbo parameters
                    chatterbox_turbo_ref_audio,
                    chatterbox_turbo_exaggeration,
                    chatterbox_turbo_temperature,
                    chatterbox_turbo_cfg_weight,
                    chatterbox_turbo_repetition_penalty,
                    chatterbox_turbo_min_p,
                    chatterbox_turbo_top_p,
                    chatterbox_turbo_seed,
                    # Kokoro parameters
                    kokoro_voice,
                    kokoro_speed,
                    # Fish Speech parameters
                    fish_ref_audio,
                    fish_ref_text,
                    fish_temperature,
                    fish_top_p,
                    fish_repetition_penalty,
                    fish_max_tokens,
                    fish_seed,
                    # IndexTTS parameters
                    indextts_ref_audio,
                    indextts_temperature,
                    indextts_seed,
                    # IndexTTS2 parameters
                    indextts2_ref_audio,
                    indextts2_emotion_mode,
                    indextts2_emotion_audio,
                    indextts2_emotion_description,
                    indextts2_emo_alpha,
                    indextts2_happy,
                    indextts2_angry,
                    indextts2_sad,
                    indextts2_afraid,
                    indextts2_disgusted,
                    indextts2_melancholic,
                    indextts2_surprised,
                    indextts2_calm,
                    indextts2_temperature,
                    indextts2_top_p,
                    indextts2_top_k,
                    indextts2_repetition_penalty,
                    indextts2_max_mel_tokens,
                    indextts2_seed,
                    indextts2_use_random,
                    # F5-TTS parameters
                    f5_ref_audio,
                    f5_ref_text,
                    f5_speed,
                    f5_cross_fade,
                    f5_remove_silence,
                    f5_seed,
                    # Higgs Audio parameters
                    higgs_ref_audio,
                    higgs_ref_text,
                    higgs_voice_preset,
                    higgs_system_prompt,
                    higgs_temperature,
                    higgs_top_p,
                    higgs_top_k,
                    higgs_max_tokens,
                    higgs_ras_win_len,
                    higgs_ras_win_max_num_repeat,
                    # KittenTTS parameters
                    kitten_voice,
                    # Qwen TTS parameters
                    qwen_ref_audio,
                    qwen_ref_text,
                    qwen_language,
                    qwen_xvector_only,
                    qwen_clone_model_size,
                    qwen_seed,
                    # VoxCPM parameters
                    voxcpm_ref_audio,
                    voxcpm_ref_text,
                    voxcpm_cfg_value,
                    voxcpm_inference_timesteps,
                    voxcpm_normalize,
                    voxcpm_denoise,
                    voxcpm_retry_badcase,
                    voxcpm_retry_badcase_max_times,
                    voxcpm_retry_badcase_ratio_threshold,
                    voxcpm_seed,
                    # Effects parameters
                    gain_db,
                    enable_eq,
                    eq_bass,
                    eq_mid,
                    eq_treble,
                    enable_reverb,
                    reverb_room,
                    reverb_damping,
                    reverb_wet,
                    enable_echo,
                    echo_delay,
                    echo_decay,
                    enable_pitch,
                    pitch_semitones,
                    # Advanced eBook settings
                    ebook_chunk_gap,
                    ebook_chapter_gap,
                ],
                outputs=[audiobook_output, audiobook_download, ebook_status],
            )

            # Connect eBook clear button
            clear_ebook_btn.click(
                fn=handle_clear_ebook,
                inputs=[],
                outputs=[
                    ebook_file,
                    ebook_info,
                    chapter_selection,
                    audiobook_output,
                    audiobook_download,
                    ebook_status,
                ],
            )

        # Custom voice upload event handlers (only if Kokoro is available)
        if KOKORO_AVAILABLE:
            # Upload custom voice
            upload_btn.click(
                fn=upload_and_refresh,
                inputs=[custom_voice_files, custom_voice_name],
                outputs=[
                    upload_status,
                    custom_voice_list,
                    custom_voice_name,
                    custom_voice_files,
                    kokoro_voice,  # Main voice selector
                    speaker_1_kokoro_voice,
                    speaker_2_kokoro_voice,
                    speaker_3_kokoro_voice,
                    speaker_4_kokoro_voice,
                    speaker_5_kokoro_voice,
                ],  # Conversation mode voice selectors
            )

            # Refresh voice list
            refresh_voices_btn.click(fn=refresh_kokoro_voice_list, outputs=[kokoro_voice])

            # Refresh all conversation mode voice selectors too
            refresh_voices_btn.click(
                fn=refresh_all_kokoro_voices,
                outputs=[
                    speaker_1_kokoro_voice,
                    speaker_2_kokoro_voice,
                    speaker_3_kokoro_voice,
                    speaker_4_kokoro_voice,
                    speaker_5_kokoro_voice,
                ],
            )

            # Refresh custom voice list
            refresh_voices_btn.click(fn=get_custom_voice_list, outputs=[custom_voice_list])

        # VibeVoice event handlers (only if VibeVoice is available)
        if VIBEVOICE_AVAILABLE:
            print("✅ VibeVoice is available, setting up event handlers...")

            # Update speaker dropdowns based on number of speakers
            def update_speaker_visibility(num_speakers):
                return [
                    gr.update(visible=True),  # Speaker 1 accordion always visible
                    gr.update(
                        visible=num_speakers >= 2
                    ),  # Speaker 2 accordion visible for 2+ speakers
                    gr.update(
                        visible=num_speakers >= 3
                    ),  # Speaker 3 accordion visible for 3+ speakers
                    gr.update(
                        visible=num_speakers >= 4
                    ),  # Speaker 4 accordion visible for 4 speakers
                ]

            vibevoice_num_speakers.change(
                fn=update_speaker_visibility,
                inputs=[vibevoice_num_speakers],
                outputs=[
                    vibevoice_speaker_1_accordion,
                    vibevoice_speaker_2_accordion,
                    vibevoice_speaker_3_accordion,
                    vibevoice_speaker_4_accordion,
                ],
            )

            # Model management
            def handle_vibevoice_load(selected_model_path, path, use_flash_attention):
                # Prefer radio selection; fall back to manual path
                effective_path = selected_model_path or path
                if not effective_path:
                    return "❌ No model path selected"
                success, message = init_vibevoice_model(
                    effective_path, use_flash_attention=use_flash_attention
                )
                return message

            def handle_vibevoice_unload():
                return unload_vibevoice_model()

            vibevoice_load_btn.click(
                fn=handle_vibevoice_load,
                inputs=[
                    vibevoice_downloaded_models,
                    vibevoice_model_path,
                    vibevoice_flash_attention,
                ],
                outputs=[vibevoice_model_status],
            )

            vibevoice_unload_btn.click(
                fn=handle_vibevoice_unload, inputs=[], outputs=[vibevoice_model_status]
            )

            # Model download
            def handle_vibevoice_download(model_name):
                if not VIBEVOICE_AVAILABLE:
                    yield "❌ VibeVoice not available"
                    return

                if not model_name:
                    yield "❌ No model selected"
                    return

                # Immediate feedback
                yield f"📥 Starting download for {model_name}... This may take a while."

                try:
                    handler = get_vibevoice_handler()
                    success, message = handler.download_model(model_name)
                    yield message
                except Exception as e:
                    yield f"❌ Error in download handler: {str(e)}"

            print("🔗 Connecting download button click handler...")
            vibevoice_download_btn.click(
                fn=handle_vibevoice_download,
                inputs=[vibevoice_model_select],
                outputs=[vibevoice_download_status],
                queue=True,
            )
            print("✅ Download button handler connected!")

            # Refresh voice dropdowns function
            def refresh_vibevoice_voices():
                if not VIBEVOICE_AVAILABLE:
                    return gr.update(), gr.update(), gr.update(), gr.update()

                handler = get_vibevoice_handler()
                voices = handler.get_available_voices()
                voice_choices = gr.update(choices=voices)

                return voice_choices, voice_choices, voice_choices, voice_choices

            # Refresh downloaded models list
            def refresh_vibevoice_model_list():
                if not VIBEVOICE_AVAILABLE:
                    return gr.update(choices=[], value=None)
                models = scan_vibevoice_models()
                if not models:
                    models = []
                default_value = models[0] if len(models) > 0 else None
                return gr.update(choices=models, value=default_value)

            # Refresh voices button handler
            vibevoice_refresh_voices_btn.click(
                fn=refresh_vibevoice_voices,
                inputs=[],
                outputs=[
                    vibevoice_speaker_1,
                    vibevoice_speaker_2,
                    vibevoice_speaker_3,
                    vibevoice_speaker_4,
                ],
            )

            # Refresh models button handler
            vibevoice_refresh_models_btn.click(
                fn=refresh_vibevoice_model_list, inputs=[], outputs=[vibevoice_downloaded_models]
            )

            # Add custom voice
            def handle_add_custom_voice(audio_file, voice_name):
                if not VIBEVOICE_AVAILABLE:
                    return (
                        "❌ VibeVoice not available",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

                try:
                    handler = get_vibevoice_handler()
                    result = handler.add_custom_voice(audio_file, voice_name)

                    # Refresh voice lists
                    voices = handler.get_available_voices()
                    voice_choices = gr.update(choices=voices)

                    return result, voice_choices, voice_choices, voice_choices, voice_choices, ""
                except Exception as e:
                    return (
                        f"❌ Error in add voice handler: {str(e)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "",
                    )

            add_voice_btn.click(
                fn=handle_add_custom_voice,
                inputs=[custom_voice_file, vibevoice_custom_voice_name],
                outputs=[
                    add_voice_status,
                    vibevoice_speaker_1,
                    vibevoice_speaker_2,
                    vibevoice_speaker_3,
                    vibevoice_speaker_4,
                    vibevoice_custom_voice_name,
                ],
            )

            # Generate podcast
            def handle_vibevoice_generation(
                num_speakers,
                script,
                speaker_1,
                speaker_2,
                speaker_3,
                speaker_4,
                cfg_scale,
                seed,
                audio_format,
            ):
                if not VIBEVOICE_AVAILABLE:
                    return None, "❌ VibeVoice not available"

                # Collect speaker voices
                speaker_voices = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]

                # Generate podcast
                audio_result, status = generate_vibevoice_podcast(
                    num_speakers=num_speakers,
                    script=script,
                    speaker_voices=speaker_voices,
                    cfg_scale=cfg_scale,
                    seed=int(seed) if seed is not None else None,
                    output_folder=get_runtime_output_dir("outputs"),
                    audio_format=audio_format,
                )

                return audio_result, status

            vibevoice_generate_btn.click(
                fn=handle_vibevoice_generation,
                inputs=[
                    vibevoice_num_speakers,
                    vibevoice_script,
                    vibevoice_speaker_1,
                    vibevoice_speaker_2,
                    vibevoice_speaker_3,
                    vibevoice_speaker_4,
                    vibevoice_cfg_scale,
                    vibevoice_seed,
                    vibevoice_audio_format,
                ],
                outputs=[vibevoice_output, vibevoice_status],
            )

        # ── MCP security integration (Phase 4a WI-5) ─────────────────────
        def _extract_bearer_token(request: gr.Request) -> str:
            """Extract bearer token from Gradio request headers."""
            if request is None:
                return ""
            try:
                auth_header = request.headers.get("authorization", "")
                if auth_header.lower().startswith("bearer "):
                    return auth_header[7:].strip()
            except (AttributeError, TypeError):
                pass
            return ""

        # ── MCP read-only tools (Phase 4a WI-3) ─────────────────────────
        def mcp_list_engines(request: gr.Request) -> list[dict[str, object]]:
            """List all available TTS engines with their capabilities and status.

            Returns:
                A list of engine objects with name, capabilities, voice mode, and service status.
            """
            from mcp_security import get_security
            from tts_service import list_engines

            get_security().guard("list_engines", _extract_bearer_token(request))
            return list_engines()

        def mcp_get_engine_info(engine_name: str, request: gr.Request) -> dict[str, object]:
            """Get detailed information about a specific TTS engine.

            Args:
                engine_name: The engine identifier (e.g., "F5-TTS", "ChatterboxTTS", "Kokoro TTS").

            Returns:
                Engine details including capabilities, supported cues, voice mode, and parameter schema.
            """
            from mcp_security import get_security
            from tts_service import get_engine_info

            get_security().guard("get_engine_info", _extract_bearer_token(request))
            return get_engine_info(engine_name)

        def mcp_list_voices(
            engine_name: str = "", request: gr.Request | None = None
        ) -> list[dict[str, object]]:
            """List available voices, optionally filtered by engine.

            Args:
                engine_name: If provided, filter voices for this engine only. Leave empty to list all.

            Returns:
                A list of voice objects with id, name, engine, and type fields.
            """
            from mcp_security import get_security
            from tts_service import list_voices

            get_security().guard("list_voices", _extract_bearer_token(request))
            return list_voices(engine_name if engine_name else None)

        def mcp_list_outputs(request: gr.Request) -> list[dict[str, object]]:
            """List generated audio output files sorted by most recent first.

            Returns:
                A list of output file objects with path, filename, size_bytes, and modified_at.
            """
            from mcp_security import get_security
            from tts_service import list_outputs

            get_security().guard("list_outputs", _extract_bearer_token(request))
            return list_outputs()

        def mcp_get_app_version(request: gr.Request) -> dict[str, str]:
            """Get the current application version and MCP server status.

            Returns:
                A dictionary with app_name, mcp_status, mcp_version, and tool_count fields.
            """
            from mcp_security import get_security

            get_security().guard("get_app_version", _extract_bearer_token(request))
            return {
                "app_name": "Ultimate TTS Studio",
                "mcp_status": "active",
                "mcp_version": "0.8.0",
                "tool_count": "14",
            }

        def register_mcp_api(fn: Any, api_name: str) -> None:
            getattr(gr, "api")(fn, api_name=api_name)

        register_mcp_api(mcp_list_engines, api_name="list_engines")
        register_mcp_api(mcp_get_engine_info, api_name="get_engine_info")
        register_mcp_api(mcp_list_voices, api_name="list_voices")
        register_mcp_api(mcp_list_outputs, api_name="list_outputs")
        register_mcp_api(mcp_get_app_version, api_name="get_app_version")

        # ── MCP stateless transform tools (Phase 4a WI-4) ────────────────
        def mcp_normalize_text(text: str, request: gr.Request) -> str:
            """Apply deterministic text normalization without LLM processing.

            Expands abbreviations, converts phone numbers, currencies, dates, times,
            and URLs into TTS-friendly spoken forms.

            Args:
                text: The raw text to normalize for TTS consumption.

            Returns:
                The normalized text with expansions applied.
            """
            from mcp_security import get_security
            from narration_transform import deterministic_normalize

            get_security().guard("normalize_text", _extract_bearer_token(request))
            return deterministic_normalize(text)

        def mcp_transform_text(
            text: str,
            provider_name: str = "Ollama",
            base_url: str = "",
            api_key: str = "",
            model_id: str = "",
            mode: str = "minimal",
            locale: str = "en-US",
            style: str = "conversational",
            engine_name: str = "",
            engine: str = "",
            temperature: str = "0.2",
            top_p: str = "0.9",
            max_tokens: str = "1024",
            timeout_seconds: str = "60",
            request: gr.Request | None = None,
        ) -> dict[str, str]:
            """Transform text using LLM narration processing for TTS optimization.

            Applies an LLM-powered narration transform to text, adding prosody cues,
            adjusting pacing, and polishing for natural speech. Falls back to
            deterministic normalization if the LLM is unavailable.

            Args:
                text: The source text to transform.
                provider_name: LLM provider name (e.g., "Ollama", "LM Studio", "Google Gemini").
                base_url: Provider API base URL. Leave empty to use the provider default.
                api_key: API key for the provider. Leave empty to resolve from environment.
                model_id: Model identifier. Leave empty to use the provider default.
                mode: Transform intensity — "minimal", "polish", or "vivid".
                locale: Target locale (e.g., "en-US", "en-GB").
                style: Speech style (e.g., "conversational", "formal", "narrative").
                engine_name: Target TTS engine name for engine-aware transform tuning.
                engine: Backward-compatible alias for engine_name.
                temperature: LLM sampling temperature (0.0 to 2.0).
                top_p: LLM nucleus sampling threshold (0.0 to 1.0).
                max_tokens: Maximum tokens in the LLM response.
                timeout_seconds: LLM request timeout in seconds.

            Returns:
                A dictionary with "transformed_text" and "status" fields.
            """
            from mcp_security import get_security
            from narration_transform import LLM_PROVIDER_CONFIGS, apply_llm_narration_transform

            get_security().guard("transform_text", _extract_bearer_token(request))
            if not base_url and provider_name in LLM_PROVIDER_CONFIGS:
                base_url = LLM_PROVIDER_CONFIGS[provider_name].get("base_url", "")
            if not model_id and provider_name in LLM_PROVIDER_CONFIGS:
                model_id = LLM_PROVIDER_CONFIGS[provider_name].get("default_model", "")

            transformed, status = apply_llm_narration_transform(
                source_text=text,
                enabled=True,
                provider_name=provider_name,
                base_url=base_url,
                api_key=api_key,
                model_id=model_id,
                mode=mode,
                locale=locale,
                style=style,
                max_tag_density=0.15,
                system_prompt="",
                timeout_seconds=int(timeout_seconds),
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
                allow_local_fallback=True,
                engine_name=engine_name,
                engine=engine,
            )
            return {"transformed_text": transformed, "status": status}

        def mcp_get_engine_script_profile(
            engine_name: str,
            request: gr.Request | None = None,
        ) -> dict[str, object]:
            """Return the full optimization profile for a TTS engine."""
            from engine_script_profiles import (
                get_engine_max_chunk_chars,
                get_engine_prompt_addendum,
                get_engine_script_rules,
            )
            from mcp_security import get_security

            get_security().guard("get_engine_script_profile", _extract_bearer_token(request))
            return {
                "engine_name": engine_name,
                "prompt_addendum": get_engine_prompt_addendum(engine_name),
                "script_rules": get_engine_script_rules(engine_name),
                "max_chunk_chars": get_engine_max_chunk_chars(engine_name),
            }

        def mcp_structure_conversation(
            script_text: str,
            request: gr.Request | None = None,
        ) -> dict[str, object]:
            """Parse a conversation script into structured speaker and line data.

            Accepts text in "Speaker: dialogue" format (one speaker per line) and
            returns structured data with speaker names and their lines.

            Args:
                script_text: The conversation script with lines in "Speaker: text" format.

            Returns:
                A dictionary with "speakers" (sorted unique names), "lines" (list of
                speaker/text dicts), and "line_count". Returns an "error" field if
                parsing fails.
            """
            from mcp_security import get_security
            from conversation_logic import get_speaker_names_from_script, parse_conversation_script

            get_security().guard("structure_conversation", _extract_bearer_token(request))
            lines, error = parse_conversation_script(script_text)
            if error:
                return {"error": error, "speakers": [], "lines": [], "line_count": 0}

            speakers = get_speaker_names_from_script(script_text)
            return {
                "speakers": speakers,
                "lines": lines,
                "line_count": len(lines),
            }

        def mcp_list_llm_providers(request: gr.Request) -> list[dict[str, str]]:
            """List available LLM providers and their default configuration.

            Returns provider names, base URLs, default models, and whether an API key
            is required. Useful for discovering valid provider_name values for the
            transform_text tool.

            Returns:
                A list of provider objects with name, base_url, default_model,
                requires_api_key, and kind fields.
            """
            from mcp_security import get_security
            from narration_transform import LLM_PROVIDER_CONFIGS

            get_security().guard("list_llm_providers", _extract_bearer_token(request))
            return [
                {
                    "name": name,
                    "base_url": cfg.get("base_url", ""),
                    "default_model": cfg.get("default_model", ""),
                    "requires_api_key": str(cfg.get("requires_api_key", False)),
                    "kind": cfg.get("kind", "custom"),
                }
                for name, cfg in LLM_PROVIDER_CONFIGS.items()
            ]

        register_mcp_api(mcp_normalize_text, api_name="normalize_text")
        register_mcp_api(mcp_transform_text, api_name="transform_text")
        register_mcp_api(mcp_get_engine_script_profile, api_name="get_engine_script_profile")
        register_mcp_api(mcp_structure_conversation, api_name="structure_conversation")
        register_mcp_api(mcp_list_llm_providers, api_name="list_llm_providers")

        # ── MCP synthesis tools (Phase 4a WI-6) ──────────────────────────
        def mcp_synthesize(
            text: str,
            engine: str = "Kokoro TTS",
            audio_format: str = "wav",
            voice: str = "",
            speed: str = "1.0",
            temperature: str = "0.7",
            ref_audio: str = "",
            request: gr.Request | None = None,
        ) -> dict[str, str]:
            """Synthesize speech from text using a TTS engine.

            Generates audio from the provided text using the specified engine.
            Use list_engines to discover available engines and their capabilities.
            Use list_voices to find valid voice identifiers for an engine.

            Args:
                text: The text to synthesize into speech.
                engine: TTS engine to use (e.g., "Kokoro TTS", "F5-TTS", "ChatterboxTTS").
                audio_format: Output audio format - "wav" or "mp3".
                voice: Voice identifier. For Kokoro: voice ID like "af_heart". For reference audio
                    engines: file path to reference audio. Leave empty to use engine default.
                speed: Speech speed multiplier (0.5 to 2.0). Not all engines support this.
                temperature: Generation temperature (0.0 to 1.5). Not all engines support this.
                ref_audio: Path to reference audio file for voice cloning engines. Alternative to the
                    voice parameter.
                request: Gradio request object (auto-injected, not visible to MCP clients).

            Returns:
                A dictionary with "status", and optionally "output_path" or "audio_format" and
                "sample_rate" fields.
            """
            from mcp_security import get_security

            get_security().guard("synthesize", _extract_bearer_token(request))

            from tts_service import TtsRequest, generate_tts

            engine_params: dict[str, object] = {}
            if voice:
                engine_params["voice"] = voice
            if ref_audio:
                engine_params["ref_audio"] = ref_audio
            try:
                engine_params["speed"] = float(speed)
            except (ValueError, TypeError):
                pass
            try:
                engine_params["temperature"] = float(temperature)
            except (ValueError, TypeError):
                pass

            tts_request = TtsRequest(
                text=text,
                engine=engine,
                audio_format=audio_format,
                engine_params=engine_params,
            )

            result = generate_tts(tts_request)

            response: dict[str, str] = {"status": result.status}
            if result.output_path:
                response["output_path"] = result.output_path
            if result.audio is not None:
                sample_rate, _audio_data = result.audio
                response["audio_format"] = audio_format
                response["sample_rate"] = str(sample_rate)
                response["note"] = (
                    "Audio generated in memory. Use the Gradio API client to retrieve binary data."
                )
            return response

        register_mcp_api(mcp_synthesize, api_name="synthesize")

        # ── MCP job tools (Phase 4a WI-7) ────────────────────────────
        def mcp_submit_synthesis_job(
            text: str,
            engine: str = "Kokoro TTS",
            audio_format: str = "wav",
            voice: str = "",
            speed: str = "1.0",
            temperature: str = "0.7",
            ref_audio: str = "",
            request: gr.Request | None = None,
        ) -> dict[str, str]:
            """Submit a background TTS synthesis job for asynchronous processing.

            Starts synthesis in an isolated subprocess. The job runs in the
            background and does not block the caller. Use get_job_status to poll
            for completion, and cancel_job to abort.

            Args:
                text: The text to synthesize into speech.
                engine: TTS engine to use (e.g., "Kokoro TTS", "F5-TTS", "ChatterboxTTS").
                audio_format: Output audio format - "wav" or "mp3".
                voice: Voice identifier or reference audio path.
                speed: Speech speed multiplier (0.5 to 2.0).
                temperature: Generation temperature (0.0 to 1.5).
                ref_audio: Path to reference audio file for voice cloning engines.
                request: Gradio request object (auto-injected, not visible to MCP clients).

            Returns:
                A dictionary with "job_id" and "status" fields.
            """
            from mcp_security import get_security

            get_security().guard("submit_synthesis_job", _extract_bearer_token(request))

            from job_manager import JobRequest, get_job_manager

            engine_params: dict[str, object] = {}
            if voice:
                engine_params["voice"] = voice
            if ref_audio:
                engine_params["ref_audio"] = ref_audio
            try:
                engine_params["speed"] = float(speed)
            except (ValueError, TypeError):
                pass
            try:
                engine_params["temperature"] = float(temperature)
            except (ValueError, TypeError):
                pass

            job_request = JobRequest(
                text=text,
                engine=engine,
                audio_format=audio_format,
                engine_params=engine_params,
            )
            job_id = get_job_manager().submit(job_request)
            return {"job_id": job_id, "status": "pending"}

        def mcp_get_job_status(
            job_id: str,
            request: gr.Request | None = None,
        ) -> dict[str, str]:
            """Get the current status of a background synthesis job.

            Poll this endpoint to check whether a submitted job has completed.
            Terminal states: "completed", "failed", "cancelled".

            Args:
                job_id: The job identifier returned by submit_synthesis_job.
                request: Gradio request object (auto-injected, not visible to MCP clients).

            Returns:
                A dictionary with "job_id", "status", and optionally "output_path",
                "error", or timing fields.
            """
            from mcp_security import get_security

            get_security().guard("get_job_status", _extract_bearer_token(request))

            from job_manager import get_job_manager

            try:
                info = get_job_manager().get_status(job_id)
            except KeyError:
                return {
                    "job_id": job_id,
                    "status": "not_found",
                    "error": f"Unknown job: {job_id}",
                }

            response: dict[str, str] = {
                "job_id": info.id,
                "status": info.status,
            }
            if info.error:
                response["error"] = info.error
            if info.result:
                if info.result.get("output_path"):
                    response["output_path"] = str(info.result["output_path"])
                if info.result.get("status"):
                    response["synthesis_status"] = str(info.result["status"])
            if info.created_at:
                response["created_at"] = str(info.created_at)
            if info.completed_at:
                response["completed_at"] = str(info.completed_at)
            return response

        def mcp_cancel_job(
            job_id: str,
            request: gr.Request | None = None,
        ) -> dict[str, str]:
            """Cancel a pending or running synthesis job.

            Terminates the background worker process if the job is still active.
            Jobs that have already completed, failed, or been cancelled cannot be
            re-cancelled.

            Args:
                job_id: The job identifier returned by submit_synthesis_job.
                request: Gradio request object (auto-injected, not visible to MCP clients).

            Returns:
                A dictionary with "job_id", "cancelled" (true/false), and "status" fields.
            """
            from mcp_security import get_security

            get_security().guard("cancel_job", _extract_bearer_token(request))

            from job_manager import get_job_manager

            try:
                cancelled = get_job_manager().cancel(job_id)
            except KeyError:
                return {"job_id": job_id, "cancelled": "false", "status": "not_found"}

            return {
                "job_id": job_id,
                "cancelled": str(cancelled).lower(),
                "status": "cancelled" if cancelled else "already_terminal",
            }

        register_mcp_api(mcp_submit_synthesis_job, api_name="submit_synthesis_job")
        register_mcp_api(mcp_get_job_status, api_name="get_job_status")
        register_mcp_api(mcp_cancel_job, api_name="cancel_job")

    return demo


# ===== MAIN EXECUTION =====
def _is_mcp_runtime_available() -> bool:
    """Return True only when Gradio MCP runtime dependencies are installed."""
    return importlib_util.find_spec("mcp") is not None


def _parse_semver_triplet(version: str) -> tuple[int, int, int] | None:
    """Extract major/minor/patch integers from a version-like string."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _warn_on_shared_env_mcp_f5_pydantic_conflict(mcp_server_enabled: bool) -> None:
    """Warn if shared env likely has MCP + F5 with an unsupported pydantic version."""
    if not (mcp_server_enabled and F5_TTS_AVAILABLE):
        return

    try:
        pydantic_version = importlib_metadata.version("pydantic")
    except importlib_metadata.PackageNotFoundError:
        return
    except Exception as error:
        print(f"ℹ️ Could not inspect pydantic version for MCP/F5 compatibility check: {error}")
        return

    parsed_version = _parse_semver_triplet(pydantic_version)
    if parsed_version is None:
        print(
            "ℹ️ Could not parse installed pydantic version "
            f"'{pydantic_version}' for MCP/F5 compatibility check."
        )
        return

    max_supported_for_f5 = (2, 10, 6)
    if parsed_version <= max_supported_for_f5:
        return

    print(
        "⚠️ Detected shared-env risk: MCP runtime is installed and F5-TTS is available, "
        f"but pydantic {pydantic_version} is newer than the F5-TTS tested limit (<=2.10.6)."
    )
    print(
        "ℹ️ Recommended: keep this main app environment focused on TTS and run MCP through "
        "the isolated launcher/MCP path instead of installing gradio[mcp] into the main app env."
    )


if __name__ == "__main__":
    print("🚀 Starting Unified TTS Pro...")

    # Create and launch the interface
    with suppress_specific_warnings():
        demo = create_gradio_interface()
        from mcp_security import initialize_security

        _mcp_token = initialize_security(
            token_path=Path(".mcp_token"),
            log_dir=Path("logs") / "mcp",
            auth_enabled=True,
        )
        print("MCP security initialized. Token file: .mcp_token")
        mcp_server_enabled = _is_mcp_runtime_available()
        if not mcp_server_enabled:
            print("⚠️ MCP runtime not found. Launching without Gradio MCP server support.")
            print('ℹ️ Install with: uv pip install "gradio[mcp]" (or pip install "gradio[mcp]")')
        else:
            print("🔌 MCP server enabled. Endpoint: http://127.0.0.1:<port>/gradio_api/mcp/sse")
            print("📋 Use .vscode/mcp.json for VS Code Copilot integration.")
            print(f"🔑 Bearer token: {_mcp_token[:8]}... (full token in .mcp_token)")

        _warn_on_shared_env_mcp_f5_pydantic_conflict(mcp_server_enabled)

        demo.launch(
            server_name="127.0.0.1",
            share=False,
            show_error=True,
            mcp_server=mcp_server_enabled,
        )
