from __future__ import annotations

import argparse
from importlib import util as importlib_util
from pathlib import Path

from launch import create_gradio_interface, suppress_specific_warnings
from mcp_security import initialize_security


DEFAULT_HOST = "127.0.0.1"


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
    """Launch a dedicated Gradio MCP sidecar in the isolated environment."""
    args = parse_args()
    if importlib_util.find_spec("mcp") is None:
        raise RuntimeError(
            "MCP runtime not found in this environment. Run mcp_install.js first."
        )

    print("Starting Ultimate TTS Studio MCP sidecar...")
    with suppress_specific_warnings():
        demo = create_gradio_interface()
        token = initialize_security(
            token_path=Path(".mcp_token"),
            log_dir=Path("logs") / "mcp",
            auth_enabled=True,
        )
        print("MCP security initialized. Token file: .mcp_token")
        print("MCP SSE endpoint suffix: /gradio_api/mcp/sse")
        print("Use .vscode/mcp.json for VS Code Copilot integration.")
        print(f"Bearer token: {token[:8]}... (full token in .mcp_token)")
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            show_error=True,
            mcp_server=True,
        )


if __name__ == "__main__":
    main()