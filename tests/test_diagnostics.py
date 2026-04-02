from __future__ import annotations

import sys
import types
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import diagnostics


def test_module_has_zero_gradio_imports() -> None:
    source = (APP_DIR / "diagnostics.py").read_text(encoding="utf-8")

    assert "import gradio" not in source
    assert "from gradio" not in source


def test_check_engine_all_available() -> None:
    results = diagnostics.check_engine_availability(
        {
            "Chatterbox": True,
            "Fish Speech": True,
            "IndexTTS2": True,
        }
    )

    assert len(results) == 3
    assert all(result.status == "ok" for result in results)
    assert results[0].category == "engine"


def test_check_engine_some_unavailable() -> None:
    results = diagnostics.check_engine_availability(
        {
            "Chatterbox": True,
            "Fish Speech": False,
        }
    )

    assert results[0].status == "ok"
    assert results[1].status == "warning"
    assert "Reset Dependencies" in results[1].suggestion


def test_check_port_reachable_success() -> None:
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = None

    with patch("diagnostics.socket.create_connection", return_value=connection) as create_connection:
        result = diagnostics.check_port_reachable("127.0.0.1", 1234)

    assert result.status == "ok"
    assert "reachable" in result.message
    create_connection.assert_called_once_with(("127.0.0.1", 1234), timeout=3.0)


def test_check_port_reachable_failure() -> None:
    with patch(
        "diagnostics.socket.create_connection",
        side_effect=ConnectionRefusedError("Connection refused"),
    ):
        result = diagnostics.check_port_reachable("127.0.0.1", 9999)

    assert result.status == "error"
    assert "Cannot reach 127.0.0.1:9999" in result.message
    assert "firewall" in result.suggestion.lower()


def test_check_url_reachable_success() -> None:
    response = MagicMock()
    response.status = 200
    response.__enter__.return_value = response
    response.__exit__.return_value = None

    with patch("diagnostics.urllib.request.urlopen", return_value=response) as urlopen:
        result = diagnostics.check_url_reachable("http://localhost:1234/v1")

    assert result.status == "ok"
    assert "HTTP 200" in result.message
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://localhost:1234/v1"
    assert request.get_method() == "HEAD"


def test_check_url_reachable_failure() -> None:
    with patch(
        "diagnostics.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
        result = diagnostics.check_url_reachable("http://localhost:1234/v1")

    assert result.status == "error"
    assert "URL unreachable" in result.message
    assert "server is running" in result.suggestion


def test_check_llm_config_valid() -> None:
    results = diagnostics.check_llm_provider_config(
        provider_name="LM Studio OpenAI Server",
        base_url="http://localhost:1234/v1",
        model_id="qwen/test-model",
        api_key_present=True,
        requires_api_key=True,
    )

    assert [result.status for result in results] == ["ok", "ok", "ok"]
    assert any("qwen/test-model" in result.message for result in results)


def test_check_llm_config_missing_url() -> None:
    results = diagnostics.check_llm_provider_config(
        provider_name="LM Studio OpenAI Server",
        base_url="   ",
        model_id="qwen/test-model",
        api_key_present=False,
        requires_api_key=False,
    )

    assert results[0].status == "error"
    assert results[0].message == "Base URL is empty"


def test_check_llm_config_missing_key() -> None:
    results = diagnostics.check_llm_provider_config(
        provider_name="GitHub Models (OpenAI-compatible)",
        base_url="https://models.inference.ai.azure.com",
        model_id="gpt-4o-mini",
        api_key_present=False,
        requires_api_key=True,
    )

    assert results[-1].status == "error"
    assert "API key required" in results[-1].message


def test_interpret_error_known_pattern() -> None:
    result = diagnostics.interpret_error("Connection refused while contacting provider")

    assert result.category == "network"
    assert result.status == "error"
    assert "Start the service first" in result.suggestion


def test_interpret_error_unknown() -> None:
    result = diagnostics.interpret_error("Something odd happened")

    assert result.category == "unknown"
    assert "logs folder" in result.suggestion


def test_check_disk_space() -> None:
    free_bytes = 5 * 1024**3

    with patch("shutil.disk_usage", return_value=(10 * 1024**3, 5 * 1024**3, free_bytes)):
        result = diagnostics.check_disk_space(".")

    assert result.status == "ok"
    assert "5.0 GB free" in result.message


def test_check_gpu_availability_cuda_available() -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Test GPU",
            get_device_properties=lambda _index: types.SimpleNamespace(total_memory=8 * 1024**3),
        )
    )

    with patch.dict(sys.modules, {"torch": fake_torch}):
        result = diagnostics.check_gpu_availability()

    assert result.status == "ok"
    assert "Test GPU" in result.message
    assert "8.0 GB VRAM" in result.message


def test_format_diagnostic_report() -> None:
    report = diagnostics.format_diagnostic_report(
        [
            diagnostics.DiagnosticResult(
                category="engine",
                name="Chatterbox",
                status="ok",
                message="Loaded",
            ),
            diagnostics.DiagnosticResult(
                category="llm",
                name="LM Studio base URL",
                status="error",
                message="Base URL is empty",
                suggestion="Set the base URL.",
            ),
        ]
    )

    assert "## Diagnostic Report" in report
    assert "### TTS Engines" in report
    assert "### LLM Providers" in report
    assert "**Summary:** 1 OK, 0 warnings, 1 errors" in report


def test_run_full_diagnostics() -> None:
    disk_result = diagnostics.DiagnosticResult(
        category="system",
        name="Disk space",
        status="ok",
        message="10.0 GB free",
    )
    gpu_result = diagnostics.DiagnosticResult(
        category="system",
        name="GPU",
        status="warning",
        message="CPU only",
        suggestion="Install GPU drivers.",
    )

    with patch("diagnostics.check_disk_space", return_value=disk_result) as check_disk_space:
        with patch("diagnostics.check_gpu_availability", return_value=gpu_result):
            results = diagnostics.run_full_diagnostics(
                engine_flags={"Chatterbox": True, "Fish Speech": False},
                llm_configs=[
                    {
                        "provider_name": "LM Studio OpenAI Server",
                        "base_url": "http://localhost:1234/v1",
                        "model_id": "qwen/test-model",
                        "api_key_present": False,
                        "requires_api_key": False,
                    }
                ],
                app_path="./app",
            )

    assert len(results) == 6
    assert any(result.category == "engine" for result in results)
    assert any(result.category == "llm" for result in results)
    assert any(result.name == "Disk space" for result in results)
    assert any(result.name == "GPU" for result in results)
    check_disk_space.assert_called_once_with("./app")