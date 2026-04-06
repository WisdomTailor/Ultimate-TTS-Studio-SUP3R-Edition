from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from mcp.server.fastmcp import Context, FastMCP
import uvicorn

from conversation_logic import get_speaker_names_from_script, parse_conversation_script
from job_manager import JobRequest, get_job_manager
from mcp_security import get_security, initialize_security
from narration_transform import (
    LLM_PROVIDER_CONFIGS,
    apply_llm_narration_transform,
    deterministic_normalize,
)
from tts_service import (
    TtsRequest,
    generate_tts,
    get_engine_info,
    list_engines,
    list_outputs,
    list_voices,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_MCP_MOUNT_PATH = "/gradio_api/mcp"


def _extract_bearer_token(authorization_header: str | None) -> str:
    """Extract a bearer token from an Authorization header value."""
    if not isinstance(authorization_header, str):
        return ""
    prefix = "bearer "
    if authorization_header.lower().startswith(prefix):
        return authorization_header[len(prefix) :].strip()
    return ""


def _extract_request_token(request: Request | None) -> str:
    """Extract MCP token from Authorization header or query parameters."""
    if request is None:
        return ""

    header_token = _extract_bearer_token(request.headers.get("authorization", ""))
    if header_token:
        return header_token

    # Browser-opened links cannot set Authorization headers, so allow explicit
    # query token forwarding for authenticated SSE diagnostics.
    query_token = request.query_params.get("token") or request.query_params.get("access_token")
    if isinstance(query_token, str):
        return query_token.strip()
    return ""


def _extract_token_from_context(ctx: Context | None) -> str:
    """Extract bearer token from FastMCP context when available."""
    if ctx is None:
        return ""

    try:
        request = ctx.request_context.request
        return _extract_request_token(request)
    except Exception:
        return ""


def _guard_tool(tool_name: str, ctx: Context | None = None) -> None:
    """Apply bearer-token auth and tool rate limiting via MCP security."""
    token = _extract_token_from_context(ctx)
    identity = ""
    if ctx is not None:
        try:
            identity = str(ctx.client_id or "")
        except Exception:
            identity = ""
    get_security().guard(tool_name, token=token, identity=identity)


def _resolve_provider(provider_name: str, base_url: str, model_id: str) -> tuple[str, str, str]:
    """Resolve provider aliases and fill default base URL/model when omitted."""
    aliases = {
        "Ollama": "Ollama (OpenAI-compatible)",
        "LM Studio": "LM Studio OpenAI Server",
        "Google Gemini": "Google Gemini API (OpenAI-compatible)",
        "GitHub Models": "GitHub Models (OpenAI-compatible)",
        "Microsoft Foundry": "Microsoft Foundry (OpenAI-compatible)",
    }
    resolved_provider = aliases.get(provider_name, provider_name)
    cfg = LLM_PROVIDER_CONFIGS.get(resolved_provider, {})

    resolved_base_url = base_url.strip() if isinstance(base_url, str) else ""
    resolved_model_id = model_id.strip() if isinstance(model_id, str) else ""

    if not resolved_base_url:
        resolved_base_url = str(cfg.get("base_url", ""))
    if not resolved_model_id:
        resolved_model_id = str(cfg.get("default_model", ""))

    return resolved_provider, resolved_base_url, resolved_model_id


def _app_version_payload() -> dict[str, str]:
    """Return MCP sidecar version metadata for discovery calls."""
    try:
        mcp_version = importlib_metadata.version("mcp")
    except importlib_metadata.PackageNotFoundError:
        mcp_version = "unknown"

    return {
        "app_name": "Ultimate TTS Studio",
        "mcp_status": "active",
        "mcp_version": mcp_version,
        "tool_count": "13",
    }


def create_mcp_server() -> FastMCP:
    """Create and configure the standalone FastMCP server."""
    server = FastMCP("Ultimate TTS Studio MCP Sidecar")

    @server.tool(name="list_engines")
    def mcp_list_engines(ctx: Context) -> list[dict[str, Any]]:
        _guard_tool("list_engines", ctx)
        return list_engines()

    @server.tool(name="get_engine_info")
    def mcp_get_engine_info(engine_name: str, ctx: Context) -> dict[str, Any]:
        _guard_tool("get_engine_info", ctx)
        return get_engine_info(engine_name)

    @server.tool(name="list_voices")
    def mcp_list_voices(engine_name: str = "", ctx: Context | None = None) -> list[dict[str, Any]]:
        _guard_tool("list_voices", ctx)
        return list_voices(engine_name if engine_name else None)

    @server.tool(name="list_outputs")
    def mcp_list_outputs(ctx: Context) -> list[dict[str, Any]]:
        _guard_tool("list_outputs", ctx)
        return list_outputs()

    @server.tool(name="get_app_version")
    def mcp_get_app_version(ctx: Context) -> dict[str, str]:
        _guard_tool("get_app_version", ctx)
        return _app_version_payload()

    @server.tool(name="normalize_text")
    def mcp_normalize_text(text: str, ctx: Context) -> str:
        _guard_tool("normalize_text", ctx)
        return deterministic_normalize(text)

    @server.tool(name="list_llm_providers")
    def mcp_list_llm_providers(ctx: Context) -> list[dict[str, str]]:
        _guard_tool("list_llm_providers", ctx)
        return [
            {
                "name": name,
                "base_url": str(cfg.get("base_url", "")),
                "default_model": str(cfg.get("default_model", "")),
                "requires_api_key": str(bool(cfg.get("requires_api_key", False))).lower(),
                "kind": str(cfg.get("kind", "custom")),
            }
            for name, cfg in LLM_PROVIDER_CONFIGS.items()
        ]

    @server.tool(name="transform_text")
    def mcp_transform_text(
        text: str,
        provider_name: str = "Ollama",
        base_url: str = "",
        api_key: str = "",
        model_id: str = "",
        mode: str = "minimal",
        locale: str = "en-US",
        style: str = "conversational",
        engine: str = "",
        temperature: str = "0.2",
        top_p: str = "0.9",
        max_tokens: str = "1024",
        timeout_seconds: str = "60",
        ctx: Context | None = None,
    ) -> dict[str, str]:
        _guard_tool("transform_text", ctx)
        resolved_provider, resolved_base_url, resolved_model_id = _resolve_provider(
            provider_name=provider_name,
            base_url=base_url,
            model_id=model_id,
        )
        transformed_text, status = apply_llm_narration_transform(
            source_text=text,
            enabled=True,
            provider_name=resolved_provider,
            base_url=resolved_base_url,
            api_key=api_key,
            model_id=resolved_model_id,
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
            engine=engine,
        )
        return {
            "transformed_text": transformed_text,
            "status": status,
        }

    @server.tool(name="structure_conversation")
    def mcp_structure_conversation(
        script_text: str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        _guard_tool("structure_conversation", ctx)
        lines, error = parse_conversation_script(script_text)
        if error:
            return {"error": error, "speakers": [], "lines": [], "line_count": 0}

        speakers = get_speaker_names_from_script(script_text)
        return {
            "speakers": speakers,
            "lines": lines,
            "line_count": len(lines),
        }

    @server.tool(name="synthesize")
    def mcp_synthesize(
        text: str,
        engine: str = "Kokoro TTS",
        audio_format: str = "wav",
        voice: str = "",
        speed: str = "1.0",
        temperature: str = "0.7",
        ref_audio: str = "",
        ctx: Context | None = None,
    ) -> dict[str, str]:
        _guard_tool("synthesize", ctx)
        engine_params: dict[str, object] = {}
        if voice:
            engine_params["voice"] = voice
        if ref_audio:
            engine_params["ref_audio"] = ref_audio
        try:
            engine_params["speed"] = float(speed)
        except (TypeError, ValueError):
            pass
        try:
            engine_params["temperature"] = float(temperature)
        except (TypeError, ValueError):
            pass

        request_payload = TtsRequest(
            text=text,
            engine=engine,
            audio_format=audio_format,
            engine_params=engine_params,
        )
        result = generate_tts(request_payload)

        response: dict[str, str] = {"status": result.status}
        if result.output_path:
            response["output_path"] = result.output_path
        if result.audio is not None:
            sample_rate, _audio_data = result.audio
            response["audio_format"] = audio_format
            response["sample_rate"] = str(sample_rate)
            response["note"] = "Audio generated in memory. Use service output_path when available."
        return response

    @server.tool(name="submit_synthesis_job")
    def mcp_submit_synthesis_job(
        text: str,
        engine: str = "Kokoro TTS",
        audio_format: str = "wav",
        voice: str = "",
        speed: str = "1.0",
        temperature: str = "0.7",
        ref_audio: str = "",
        ctx: Context | None = None,
    ) -> dict[str, str]:
        _guard_tool("submit_synthesis_job", ctx)
        engine_params: dict[str, object] = {}
        if voice:
            engine_params["voice"] = voice
        if ref_audio:
            engine_params["ref_audio"] = ref_audio
        try:
            engine_params["speed"] = float(speed)
        except (TypeError, ValueError):
            pass
        try:
            engine_params["temperature"] = float(temperature)
        except (TypeError, ValueError):
            pass

        job_request = JobRequest(
            text=text,
            engine=engine,
            audio_format=audio_format,
            engine_params=engine_params,
        )
        job_id = get_job_manager().submit(job_request)
        return {"job_id": job_id, "status": "pending"}

    @server.tool(name="get_job_status")
    def mcp_get_job_status(job_id: str, ctx: Context | None = None) -> dict[str, str]:
        _guard_tool("get_job_status", ctx)
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

    @server.tool(name="cancel_job")
    def mcp_cancel_job(job_id: str, ctx: Context | None = None) -> dict[str, str]:
        _guard_tool("cancel_job", ctx)
        try:
            cancelled = get_job_manager().cancel(job_id)
        except KeyError:
            return {"job_id": job_id, "cancelled": "false", "status": "not_found"}

        return {
            "job_id": job_id,
            "cancelled": str(cancelled).lower(),
            "status": "cancelled" if cancelled else "already_terminal",
        }

    return server


def create_http_app(mcp_server: FastMCP) -> FastAPI:
    """Create HTTP host app with root/status pages and mounted MCP SSE endpoint."""
    app = FastAPI(title="Ultimate TTS Studio MCP Sidecar", version="2.0")

    @app.middleware("http")
    async def mcp_auth_middleware(request: Request, call_next):
        if request.url.path.startswith(DEFAULT_MCP_MOUNT_PATH):
            token = _extract_request_token(request)
            if not get_security().tokens.validate(token):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "message": "Missing or invalid bearer token.",
                        "token_file": str(get_security().tokens.token_path),
                    },
                )
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return (
            "<html><body>"
            "<h1>Ultimate TTS Studio MCP Sidecar</h1>"
            f"<p>SSE endpoint: {DEFAULT_MCP_MOUNT_PATH}/sse</p>"
            "<p>Status endpoint: /status</p>"
            "</body></html>"
        )

    @app.get("/status")
    async def status() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "ultimate-tts-mcp-sidecar",
            "mcp_mount_path": DEFAULT_MCP_MOUNT_PATH,
            "sse_endpoint": f"{DEFAULT_MCP_MOUNT_PATH}/sse",
            "messages_endpoint": f"{DEFAULT_MCP_MOUNT_PATH}/messages",
            "token_file": str(get_security().tokens.token_path),
            "version": _app_version_payload(),
        }

    app.mount(DEFAULT_MCP_MOUNT_PATH, mcp_server.sse_app())
    return app


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MCP sidecar."""
    parser = argparse.ArgumentParser(
        description="Launch the optional Ultimate TTS Studio MCP sidecar.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host interface to bind the MCP sidecar to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="TCP port for the MCP sidecar.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch a dedicated standalone MCP sidecar in the isolated environment."""
    args = parse_args()
    if importlib_util.find_spec("mcp") is None:
        raise RuntimeError("MCP runtime not found in this environment. Run mcp_install.js first.")

    print("Starting Ultimate TTS Studio MCP sidecar...")
    token = initialize_security(
        token_path=Path(".mcp_token"),
        log_dir=Path("logs") / "mcp",
        auth_enabled=True,
    )
    mcp_server = create_mcp_server()
    app = create_http_app(mcp_server)

    print("MCP security initialized. Token file: .mcp_token")
    print(f"MCP SSE endpoint suffix: {DEFAULT_MCP_MOUNT_PATH}/sse")
    print("Use .vscode/mcp.json for VS Code Copilot integration.")
    print(f"Bearer token: {token[:8]}... (full token in .mcp_token)")
    print(f"Sidecar URL: http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
