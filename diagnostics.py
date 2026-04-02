"""Diagnostic utilities for Ultimate TTS Studio.

Provides connection testing, error interpretation, and configuration
suggestions for TTS engines and LLM providers. No Gradio imports.
"""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DiagnosticResult:
    """Result of a diagnostic check.

    Attributes:
        category: High-level diagnostic category.
        name: Human-readable check name.
        status: Result severity, one of "ok", "warning", or "error".
        message: Detailed result message.
        suggestion: Optional actionable recommendation.
    """

    category: str
    name: str
    status: str
    message: str
    suggestion: str = ""


_ERROR_PATTERNS: list[tuple[str, str, str]] = [
    (
        "Connection refused",
        "network",
        "The target server is not running or is blocking connections. Start the service first.",
    ),
    (
        "timed out",
        "network",
        "Request timed out. The server may be overloaded or unreachable. Try increasing the timeout.",
    ),
    ("401", "auth", "Authentication failed. Check your API key."),
    (
        "403",
        "auth",
        "Access forbidden. Your API key may lack the required permissions.",
    ),
    ("404", "config", "Endpoint not found. Check the base URL and model ID."),
    ("429", "rate_limit", "Rate limited. Wait before retrying or reduce request frequency."),
    ("500", "server", "Server internal error. The LLM provider is experiencing issues."),
    ("502", "server", "Bad gateway. The upstream server may be down."),
    (
        "503",
        "server",
        "Service unavailable. The server is overloaded or under maintenance.",
    ),
    (
        "CUDA out of memory",
        "gpu",
        "GPU memory exhausted. Try a smaller model or unload other models first.",
    ),
    (
        "No module named",
        "install",
        "A Python dependency is missing. Run the install script or reset dependencies.",
    ),
    (
        "model not found",
        "config",
        "The specified model is not available on this provider. Check the model ID.",
    ),
]


def check_engine_availability(engine_flags: Mapping[str, bool]) -> list[DiagnosticResult]:
    """Check which TTS engines are available.

    Args:
        engine_flags: Mapping of engine names to availability flags.

    Returns:
        One diagnostic result per engine.
    """

    results: list[DiagnosticResult] = []
    for engine_name, available in engine_flags.items():
        if available:
            results.append(
                DiagnosticResult(
                    category="engine",
                    name=engine_name,
                    status="ok",
                    message=f"{engine_name} handler loaded successfully",
                )
            )
            continue

        results.append(
            DiagnosticResult(
                category="engine",
                name=engine_name,
                status="warning",
                message=f"{engine_name} handler not available",
                suggestion=(
                    f"Run install script or check {engine_name} dependencies. "
                    "Use Reset Dependencies if needed."
                ),
            )
        )
    return results


def check_port_reachable(host: str, port: int, timeout: float = 3.0) -> DiagnosticResult:
    """Check if a TCP port is reachable.

    Args:
        host: Target hostname or IP address.
        port: Target TCP port.
        timeout: Connection timeout in seconds.

    Returns:
        A network diagnostic result.
    """

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return DiagnosticResult(
                category="network",
                name=f"{host}:{port}",
                status="ok",
                message=f"Port {port} on {host} is reachable",
            )
    except (socket.timeout, ConnectionRefusedError, OSError) as exc:
        return DiagnosticResult(
            category="network",
            name=f"{host}:{port}",
            status="error",
            message=f"Cannot reach {host}:{port}: {exc}",
            suggestion=f"Ensure the service is running on {host}:{port}. Check firewall settings.",
        )


def check_url_reachable(url: str, timeout: float = 5.0) -> DiagnosticResult:
    """Check if an HTTP or HTTPS URL is reachable.

    Args:
        url: Target URL.
        timeout: Request timeout in seconds.

    Returns:
        A network diagnostic result.
    """

    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return DiagnosticResult(
                category="network",
                name=url,
                status="ok",
                message=f"URL reachable (HTTP {response.status})",
            )
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404, 405):
            return DiagnosticResult(
                category="network",
                name=url,
                status="ok",
                message=f"URL reachable (HTTP {exc.code} - expected for HEAD probe)",
            )
        return DiagnosticResult(
            category="network",
            name=url,
            status="error",
            message=f"HTTP error {exc.code}: {exc.reason}",
            suggestion="Check the URL and API key configuration.",
        )
    except urllib.error.URLError as exc:
        return DiagnosticResult(
            category="network",
            name=url,
            status="error",
            message=f"URL unreachable: {exc.reason}",
            suggestion="Check if the server is running and the URL is correct.",
        )
    except Exception as exc:
        return DiagnosticResult(
            category="network",
            name=url,
            status="error",
            message=f"Connection error: {exc}",
            suggestion="Verify network connectivity and URL format.",
        )


def check_llm_provider_config(
    provider_name: str,
    base_url: str,
    model_id: str,
    api_key_present: bool,
    requires_api_key: bool,
) -> list[DiagnosticResult]:
    """Validate LLM provider configuration without making API calls.

    Args:
        provider_name: Display name for the provider.
        base_url: Configured endpoint base URL.
        model_id: Configured model identifier.
        api_key_present: Whether an API key is currently available.
        requires_api_key: Whether the provider requires an API key.

    Returns:
        Validation results for URL, model, and API key state.
    """

    results: list[DiagnosticResult] = []

    normalized_url = base_url.strip()
    normalized_model_id = model_id.strip()

    if not normalized_url:
        results.append(
            DiagnosticResult(
                category="llm",
                name=f"{provider_name} base URL",
                status="error",
                message="Base URL is empty",
                suggestion=f"Set the base URL for {provider_name} in LLM settings.",
            )
        )
    elif not normalized_url.startswith(("http://", "https://")):
        results.append(
            DiagnosticResult(
                category="llm",
                name=f"{provider_name} base URL",
                status="error",
                message=f"Invalid URL scheme: {normalized_url}",
                suggestion="URL must start with http:// or https://",
            )
        )
    else:
        results.append(
            DiagnosticResult(
                category="llm",
                name=f"{provider_name} base URL",
                status="ok",
                message=f"Base URL configured: {normalized_url}",
            )
        )

    if not normalized_model_id:
        results.append(
            DiagnosticResult(
                category="llm",
                name=f"{provider_name} model",
                status="error",
                message="No model ID configured",
                suggestion=f"Select or enter a model ID for {provider_name}.",
            )
        )
    else:
        results.append(
            DiagnosticResult(
                category="llm",
                name=f"{provider_name} model",
                status="ok",
                message=f"Model configured: {normalized_model_id}",
            )
        )

    if requires_api_key and not api_key_present:
        results.append(
            DiagnosticResult(
                category="llm",
                name=f"{provider_name} API key",
                status="error",
                message="API key required but not provided",
                suggestion=(
                    f"Set the API key for {provider_name} via environment variable or the UI."
                ),
            )
        )
    elif requires_api_key and api_key_present:
        results.append(
            DiagnosticResult(
                category="llm",
                name=f"{provider_name} API key",
                status="ok",
                message="API key provided",
            )
        )

    return results


def interpret_error(error_message: str) -> DiagnosticResult:
    """Interpret an error message and provide an actionable suggestion.

    Args:
        error_message: Raw error string.

    Returns:
        A best-effort interpreted diagnostic result.
    """

    trimmed_message = (error_message or "")[:500]
    lowered_message = trimmed_message.lower()

    for pattern, category, suggestion in _ERROR_PATTERNS:
        if pattern.lower() in lowered_message:
            return DiagnosticResult(
                category=category,
                name="Error interpretation",
                status="error",
                message=trimmed_message,
                suggestion=suggestion,
            )

    return DiagnosticResult(
        category="unknown",
        name="Error interpretation",
        status="error",
        message=trimmed_message,
        suggestion="Check the logs folder for detailed error information.",
    )


def check_disk_space(path: str, min_gb: float = 1.0) -> DiagnosticResult:
    """Check available disk space for a given path.

    Args:
        path: Filesystem path to inspect.
        min_gb: Minimum recommended free space in gigabytes.

    Returns:
        A system diagnostic result.
    """

    try:
        import shutil

        path_obj = Path(path)
        _total, _used, free = shutil.disk_usage(path_obj)
        free_gb = free / (1024**3)
        if free_gb < min_gb:
            return DiagnosticResult(
                category="system",
                name="Disk space",
                status="warning",
                message=f"Low disk space: {free_gb:.1f} GB free at {path_obj}",
                suggestion=f"Free up disk space. At least {min_gb} GB recommended.",
            )

        return DiagnosticResult(
            category="system",
            name="Disk space",
            status="ok",
            message=f"{free_gb:.1f} GB free at {path_obj}",
        )
    except Exception as exc:
        return DiagnosticResult(
            category="system",
            name="Disk space",
            status="error",
            message=f"Cannot check disk space: {exc}",
        )


def check_gpu_availability() -> DiagnosticResult:
    """Check whether CUDA acceleration is available through PyTorch.

    Returns:
        A system diagnostic result.
    """

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            vram_gb = properties.total_memory / (1024**3)
            return DiagnosticResult(
                category="system",
                name="GPU",
                status="ok",
                message=f"CUDA available: {gpu_name} ({vram_gb:.1f} GB VRAM)",
            )

        return DiagnosticResult(
            category="system",
            name="GPU",
            status="warning",
            message="No CUDA GPU detected. TTS will use CPU (slower).",
            suggestion="Install CUDA-compatible GPU drivers for faster generation.",
        )
    except ImportError:
        return DiagnosticResult(
            category="system",
            name="GPU",
            status="warning",
            message="PyTorch not available - cannot check GPU",
            suggestion="Install PyTorch to enable GPU acceleration.",
        )
    except Exception as exc:
        return DiagnosticResult(
            category="system",
            name="GPU",
            status="error",
            message=f"Cannot check GPU availability: {exc}",
            suggestion="Verify the PyTorch and CUDA installation.",
        )


def run_full_diagnostics(
    engine_flags: Mapping[str, bool],
    llm_configs: Sequence[Mapping[str, object]],
    app_path: str = "",
) -> list[DiagnosticResult]:
    """Run a comprehensive diagnostic sweep.

    Args:
        engine_flags: Mapping of engine names to availability booleans.
        llm_configs: Provider config dicts with keys such as provider_name,
            base_url, model_id, api_key_present, and requires_api_key.
        app_path: Filesystem path used for disk-space checks.

    Returns:
        Combined diagnostics for engine, LLM, and system checks.
    """

    results: list[DiagnosticResult] = []
    results.extend(check_engine_availability(engine_flags))

    for config in llm_configs:
        results.extend(
            check_llm_provider_config(
                provider_name=str(config.get("provider_name", "Unknown")),
                base_url=str(config.get("base_url", "")),
                model_id=str(config.get("model_id", "")),
                api_key_present=bool(config.get("api_key_present", False)),
                requires_api_key=bool(config.get("requires_api_key", False)),
            )
        )

    results.append(check_disk_space(app_path or os.getcwd()))
    results.append(check_gpu_availability())
    return results


def format_diagnostic_report(results: Sequence[DiagnosticResult]) -> str:
    """Format diagnostic results into a readable markdown report.

    Args:
        results: Diagnostic results to format.

    Returns:
        Markdown output grouped by category.
    """

    if not results:
        return "No diagnostics to report."

    status_labels = {"ok": "[OK]", "warning": "[WARN]", "error": "[ERROR]"}
    category_labels = {
        "engine": "TTS Engines",
        "llm": "LLM Providers",
        "network": "Network",
        "system": "System",
        "auth": "Authentication",
        "config": "Configuration",
        "gpu": "GPU",
        "install": "Installation",
        "rate_limit": "Rate Limiting",
        "server": "Server",
        "unknown": "Other",
    }

    grouped_results: dict[str, list[DiagnosticResult]] = {}
    for result in results:
        grouped_results.setdefault(result.category, []).append(result)

    lines = ["## Diagnostic Report", ""]
    for category, category_results in grouped_results.items():
        lines.append(f"### {category_labels.get(category, category.title())}")
        lines.append("")
        for result in category_results:
            status_label = status_labels.get(result.status, "[?]")
            lines.append(f"- {status_label} **{result.name}**: {result.message}")
            if result.suggestion:
                lines.append(f"  - Suggestion: {result.suggestion}")
        lines.append("")

    ok_count = sum(1 for result in results if result.status == "ok")
    warning_count = sum(1 for result in results if result.status == "warning")
    error_count = sum(1 for result in results if result.status == "error")
    lines.append(f"**Summary:** {ok_count} OK, {warning_count} warnings, {error_count} errors")
    return "\n".join(lines)