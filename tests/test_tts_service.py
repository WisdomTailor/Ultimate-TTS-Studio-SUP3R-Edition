from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import tts_service
from engine_registry import ENGINE_EXPRESSIVENESS
from tts_service import TtsRequest, TtsResult


def test_module_has_zero_gradio_imports() -> None:
    source = (APP_DIR / "tts_service.py").read_text(encoding="utf-8")

    assert "import gradio" not in source
    assert "from gradio" not in source


def test_list_engines_returns_all_registry_engines() -> None:
    engines = tts_service.list_engines()

    assert len(engines) == len(ENGINE_EXPRESSIVENESS)
    assert {engine["name"] for engine in engines} == set(ENGINE_EXPRESSIVENESS)


@pytest.mark.parametrize(
    "engine_name",
    ["F5-TTS", "Kokoro TTS", "Qwen Custom Voice", "VoxCPM"],
)
def test_get_engine_info_returns_structured_metadata(engine_name: str) -> None:
    info = tts_service.get_engine_info(engine_name)

    assert info["name"] == engine_name
    assert info["display_name"] == engine_name
    assert isinstance(info["capabilities"], dict)
    assert isinstance(info["parameter_schema"], list)
    assert "voice_mode" in info
    assert "service_layer_status" in info


def test_get_engine_info_invalid_engine_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown engine"):
        tts_service.get_engine_info("NotARealEngine")


def test_get_engine_info_maps_qwen_custom_voice_parameter_aliases() -> None:
    info = tts_service.get_engine_info("Qwen Custom Voice")

    parameter_names = {entry["parameter"] for entry in info["parameter_schema"]}
    assert "speaker" in parameter_names
    assert "instruct" in parameter_names


def test_get_engine_info_maps_f5_cross_fade_alias() -> None:
    info = tts_service.get_engine_info("F5-TTS")

    parameter_names = {entry["parameter"] for entry in info["parameter_schema"]}
    assert "cross_fade_duration" in parameter_names


def test_list_voices_returns_kokoro_built_in_voices() -> None:
    voices = tts_service.list_voices("Kokoro TTS")

    assert any(voice["id"] == "af_heart" for voice in voices)
    assert all(voice["engine"] == "Kokoro TTS" for voice in voices)


def test_list_voices_returns_kitten_named_voices() -> None:
    voices = tts_service.list_voices("KittenTTS")

    assert any(voice["id"] == "expr-voice-2-f" for voice in voices)
    assert all(voice["type"] == "built_in" for voice in voices)


def test_list_voices_returns_qwen_custom_speakers() -> None:
    voices = tts_service.list_voices("Qwen Custom Voice")

    assert any(voice["id"] == "Ryan" for voice in voices)
    assert all(voice["type"] == "speaker_profile" for voice in voices)


def test_list_voices_scans_custom_kokoro_voices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    custom_dir = tmp_path / "custom_voices"
    custom_dir.mkdir()
    (custom_dir / "alpha.pt").write_bytes(b"alpha")
    (custom_dir / "beta.pt").write_bytes(b"beta")
    (custom_dir / "ignore.txt").write_text("nope", encoding="utf-8")

    voices = tts_service.list_voices("Kokoro TTS")
    voice_ids = {voice["id"] for voice in voices}

    assert "custom_alpha" in voice_ids
    assert "custom_beta" in voice_ids
    assert "ignore" not in voice_ids


@pytest.mark.parametrize(
    ("engine_name", "expected_type"),
    [
        ("F5-TTS", "reference_audio"),
        ("Qwen Voice Clone", "reference_audio"),
        ("Qwen Voice Design", "voice_description"),
        ("VibeVoice", "multi_speaker"),
    ],
)
def test_list_voices_returns_indicator_for_non_named_voice_engines(
    engine_name: str,
    expected_type: str,
) -> None:
    voices = tts_service.list_voices(engine_name)

    assert len(voices) == 1
    assert voices[0]["engine"] == engine_name
    assert voices[0]["type"] == expected_type


def test_list_voices_with_no_filter_returns_named_voice_sets_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    custom_dir = tmp_path / "custom_voices"
    custom_dir.mkdir()
    (custom_dir / "delta.pt").write_bytes(b"delta")

    voices = tts_service.list_voices()
    engines = {voice["engine"] for voice in voices}

    assert "Kokoro TTS" in engines
    assert "KittenTTS" in engines
    assert "Qwen Custom Voice" in engines
    assert "F5-TTS" not in engines


def test_list_voices_invalid_engine_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown engine"):
        tts_service.list_voices("Ghost Engine")


def test_get_output_dir_creates_outputs_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    output_dir = tts_service.get_output_dir()

    assert output_dir == tmp_path / "outputs"
    assert output_dir.exists()


def test_get_output_dir_creates_audiobooks_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    output_dir = tts_service.get_output_dir("audiobooks")

    assert output_dir == tmp_path / "audiobooks"
    assert output_dir.exists()


def test_get_output_dir_invalid_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported output directory kind"):
        tts_service.get_output_dir("archive")


def test_list_outputs_filters_audio_files_and_returns_metadata(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    old_file = outputs_dir / "one.wav"
    new_file = outputs_dir / "two.mp3"
    nested_dir = outputs_dir / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "three.flac"
    ignored_file = outputs_dir / "notes.txt"

    old_file.write_bytes(b"1")
    new_file.write_bytes(b"22")
    nested_file.write_bytes(b"333")
    ignored_file.write_text("ignore", encoding="utf-8")
    old_file.touch()
    new_file.touch()
    nested_file.touch()

    listed = tts_service.list_outputs(outputs_dir)
    filenames = [entry["filename"] for entry in listed]

    assert set(filenames) == {"one.wav", "two.mp3", "three.flac"}
    assert all(entry["size_bytes"] > 0 for entry in listed)
    assert all(Path(entry["path"]).is_absolute() for entry in listed)


def test_list_outputs_returns_empty_list_for_missing_directory(tmp_path: Path) -> None:
    listed = tts_service.list_outputs(tmp_path / "missing")

    assert listed == []


def test_tts_request_defaults() -> None:
    request = TtsRequest(text="Hello", engine="F5-TTS")

    assert request.audio_format == "wav"
    assert request.engine_params == {}
    assert request.effects == {}


def test_tts_result_defaults() -> None:
    result = TtsResult()

    assert result.audio is None
    assert result.status == ""
    assert result.output_path is None


def test_generate_tts_returns_unknown_engine_status() -> None:
    result = tts_service.generate_tts(TtsRequest(text="Hello", engine="Ghost Engine"))

    assert result.audio is None
    assert "Unknown engine" in result.status


def test_generate_tts_returns_not_yet_extracted_status() -> None:
    result = tts_service.generate_tts(TtsRequest(text="Hello", engine="Kokoro TTS"))

    assert result.audio is None
    assert result.status == "Engine 'Kokoro TTS' not yet available via service layer"


def test_generate_tts_returns_not_yet_extracted_status_for_registry_engine() -> None:
    unavailable_engines = [
        engine_name
        for engine_name in ENGINE_EXPRESSIVENESS
        if engine_name not in tts_service.SERVICE_AVAILABLE_ENGINES
    ]

    assert unavailable_engines, "Expected at least one registry engine outside the service layer"

    result = tts_service.generate_tts(TtsRequest(text="Hello", engine=unavailable_engines[0]))

    assert result.audio is None
    assert "not yet available via service layer" in result.status


def test_generate_tts_rejects_blank_text() -> None:
    result = tts_service.generate_tts(TtsRequest(text="   ", engine="F5-TTS"))

    assert result.status == "No text provided for synthesis"


def test_generate_tts_dispatches_to_function_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate(
        text: str, skip_file_saving: bool, audio_format: str
    ) -> tuple[tuple[int, np.ndarray], str]:
        captured["text"] = text
        captured["skip_file_saving"] = skip_file_saving
        captured["audio_format"] = audio_format
        return (24000, np.array([0.1, 0.2], dtype=np.float32)), "ok"

    fake_module = ModuleType("chatterbox_turbo_handler")
    fake_module.generate_chatterbox_turbo_tts = fake_generate
    monkeypatch.setattr(tts_service, "_import_module", lambda module_name: fake_module)

    request = TtsRequest(text="Hello", engine="Chatterbox Turbo")
    result = tts_service.generate_tts(request)

    assert result.status == "ok"
    assert result.output_path is None
    assert result.audio is not None
    assert result.audio[0] == 24000
    assert np.allclose(result.audio[1], np.array([0.1, 0.2], dtype=np.float32))
    assert captured == {"text": "Hello", "skip_file_saving": True, "audio_format": "wav"}


def test_generate_tts_passes_synthesis_parameters_to_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate(
        text: str,
        audio_format: str,
        voice: str,
        speed: float,
        temperature: float,
        ref_audio: str,
        skip_file_saving: bool,
    ) -> tuple[tuple[int, np.ndarray], str]:
        captured["text"] = text
        captured["audio_format"] = audio_format
        captured["voice"] = voice
        captured["speed"] = speed
        captured["temperature"] = temperature
        captured["ref_audio"] = ref_audio
        captured["skip_file_saving"] = skip_file_saving
        return (32000, np.array([0.25], dtype=np.float32)), "ok"

    fake_module = ModuleType("chatterbox_turbo_handler")
    fake_module.generate_chatterbox_turbo_tts = fake_generate
    monkeypatch.setattr(tts_service, "_import_module", lambda module_name: fake_module)

    result = tts_service.generate_tts(
        TtsRequest(
            text="Hello",
            engine="Chatterbox Turbo",
            audio_format="mp3",
            engine_params={
                "voice": "af_heart",
                "speed": 1.1,
                "temperature": 0.8,
                "ref_audio": "voice.wav",
            },
        )
    )

    assert result.status == "ok"
    assert result.output_path is None
    assert result.audio is not None
    assert result.audio[0] == 32000
    assert captured == {
        "text": "Hello",
        "audio_format": "mp3",
        "voice": "af_heart",
        "speed": 1.1,
        "temperature": 0.8,
        "ref_audio": "voice.wav",
        "skip_file_saving": True,
    }


def test_generate_tts_dispatches_to_method_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeHandler:
        def generate_speech(
            self, text: str, ref_audio_path: str, cross_fade_duration: float
        ) -> tuple[tuple[int, np.ndarray], str]:
            captured["text"] = text
            captured["ref_audio_path"] = ref_audio_path
            captured["cross_fade_duration"] = cross_fade_duration
            return (22050, np.array([0.5], dtype=np.float32)), "f5-ok"

    fake_module = SimpleNamespace(get_f5_tts_handler=lambda: FakeHandler())
    monkeypatch.setattr(tts_service, "_import_module", lambda module_name: fake_module)

    request = TtsRequest(
        text="Hello",
        engine="F5-TTS",
        engine_params={"reference_audio": "sample.wav", "cross_fade": 0.3},
    )
    result = tts_service.generate_tts(request)

    assert result.status == "f5-ok"
    assert result.audio is not None
    assert result.audio[0] == 22050
    assert captured == {
        "text": "Hello",
        "ref_audio_path": "sample.wav",
        "cross_fade_duration": 0.3,
    }


def test_generate_tts_returns_output_path_when_handler_saves_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_generate(text: str) -> tuple[str, str]:
        return ("outputs/final.wav", "saved")

    fake_module = ModuleType("voxcpm_handler")
    fake_module.generate_voxcpm_tts = fake_generate
    monkeypatch.setattr(tts_service, "_import_module", lambda module_name: fake_module)

    result = tts_service.generate_tts(TtsRequest(text="Hello", engine="VoxCPM"))

    assert result.audio is None
    assert result.output_path == "outputs/final.wav"
    assert result.status == "saved"


def test_generate_tts_returns_unavailable_status_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_import_error(module_name: str) -> ModuleType:
        raise ImportError("missing dependency")

    monkeypatch.setattr(tts_service, "_import_module", raise_import_error)

    result = tts_service.generate_tts(TtsRequest(text="Hello", engine="Qwen Voice Clone"))

    assert result.audio is None
    assert "unavailable" in result.status
    assert "missing dependency" in result.status


def test_generate_tts_returns_failure_status_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_generate(text: str) -> tuple[None, str]:
        raise RuntimeError("boom")

    fake_module = ModuleType("kitten_tts_handler")
    fake_module.generate_kitten_tts = fake_generate
    monkeypatch.setattr(tts_service, "_import_module", lambda module_name: fake_module)

    result = tts_service.generate_tts(TtsRequest(text="Hello", engine="KittenTTS"))

    assert result.audio is None
    assert result.output_path is None
    assert result.status == "Engine 'KittenTTS' failed: boom"


def test_generate_tts_drops_unsupported_kwargs_from_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate(text: str, speaker: str) -> tuple[tuple[int, np.ndarray], str]:
        captured["text"] = text
        captured["speaker"] = speaker
        return (16000, np.array([1.0], dtype=np.float32)), "ok"

    fake_module = ModuleType("qwen_tts_handler")
    fake_module.generate_qwen_custom_voice_tts = fake_generate
    monkeypatch.setattr(tts_service, "_import_module", lambda module_name: fake_module)

    request = TtsRequest(
        text="Hello",
        engine="Qwen Custom Voice",
        engine_params={"speaker_profile": "Ryan", "unused": "ignored"},
    )
    result = tts_service.generate_tts(request)

    assert result.status == "ok"
    assert captured == {"text": "Hello", "speaker": "Ryan"}


def test_engine_handler_registry_covers_multiple_extractable_engines() -> None:
    assert len(tts_service.ENGINE_HANDLERS) >= 3
    assert "Chatterbox Turbo" in tts_service.ENGINE_HANDLERS
    assert "VoxCPM" in tts_service.ENGINE_HANDLERS
