from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_BASE_URL = "http://127.0.0.1:7860"
DEFAULT_TIMEOUT_SECONDS = 8


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_base_url(raw_url: str) -> str:
    value = (raw_url or "").strip()
    if not value:
        value = DEFAULT_BASE_URL
    return value.rstrip("/")


def _probe_http_json_or_text(
    url: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "url": url,
        "ok": False,
        "http_status": None,
        "content_type": "",
    }
    try:
        req = urllib_request.Request(url=url, headers=headers or {}, method="GET")
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            status_code = int(response.status)
            content_type = response.headers.get("Content-Type", "")
            payload = response.read(4096)
            text_payload = payload.decode("utf-8", errors="replace")

            result["http_status"] = status_code
            result["content_type"] = content_type

            parsed_body: Any
            try:
                parsed_body = json.loads(text_payload)
            except json.JSONDecodeError:
                parsed_body = text_payload.strip()

            result["body"] = parsed_body
            result["ok"] = status_code < 400
    except urllib_error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body_text = ""
        result["http_status"] = int(exc.code)
        result["error"] = f"HTTPError: {exc.reason}"
        result["body"] = body_text[:1000]
    except urllib_error.URLError as exc:
        result["error"] = f"URLError: {exc.reason}"
    except TimeoutError:
        result["error"] = "TimeoutError"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _probe_sse(
    url: str,
    token: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    headers = {"Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    result = _probe_http_json_or_text(url=url, headers=headers, timeout_seconds=timeout_seconds)
    content_type = str(result.get("content_type", ""))

    # SSE endpoints should return 200 with text/event-stream content type.
    result["ok"] = bool(result.get("http_status") == 200 and "text/event-stream" in content_type)
    return result


def _read_token(token_file: Path) -> tuple[str, dict[str, Any]]:
    token_info: dict[str, Any] = {
        "path": str(token_file),
        "exists": token_file.exists(),
        "read_ok": False,
        "error": "",
    }
    if not token_file.exists():
        token_info["error"] = "Token file does not exist"
        return "", token_info

    try:
        token = token_file.read_text(encoding="utf-8").strip()
        token_info["read_ok"] = True
        token_info["length"] = len(token)
        return token, token_info
    except Exception as exc:
        token_info["error"] = f"{type(exc).__name__}: {exc}"
        return "", token_info


def run_verification(base_url: str, token_file: Path, summary_path: Path) -> dict[str, Any]:
    normalized_base = _normalize_base_url(base_url)
    status_url = f"{normalized_base}/status"
    sse_url = f"{normalized_base}/gradio_api/mcp/sse"

    token, token_info = _read_token(token_file)
    status_result = _probe_http_json_or_text(status_url)
    sse_result = _probe_sse(sse_url, token)

    overall_pass = bool(
        status_result.get("ok") and sse_result.get("ok") and token_info.get("read_ok")
    )

    summary: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "sidecar_base_url": normalized_base,
        "status_probe": status_result,
        "sse_probe": sse_result,
        "token_file": token_info,
        "overall_pass": overall_pass,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify MCP endpoints and write durable summary JSON"
    )
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="MCP sidecar base URL")
    parser.add_argument("--token-file", default=".mcp_token", help="Path to bearer token file")
    parser.add_argument(
        "--summary-path",
        default="../app_state/mcp_verify_summary.json",
        help="Path to write verification summary JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token_file = Path(args.token_file).resolve()
    summary_path = Path(args.summary_path).resolve()

    summary: dict[str, Any]
    try:
        summary = run_verification(
            base_url=str(args.url), token_file=token_file, summary_path=summary_path
        )
    except Exception as exc:
        # Ensure we still emit a durable summary even for unexpected verifier failures.
        fallback_summary = {
            "timestamp_utc": _utc_now_iso(),
            "sidecar_base_url": _normalize_base_url(str(args.url)),
            "status_probe": {"ok": False, "error": "Verification did not run"},
            "sse_probe": {"ok": False, "error": "Verification did not run"},
            "token_file": {
                "path": str(token_file),
                "exists": token_file.exists(),
                "read_ok": False,
            },
            "overall_pass": False,
            "verifier_error": f"{type(exc).__name__}: {exc}",
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(fallback_summary, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        summary = fallback_summary

    print(f"MCP_VERIFY_SUMMARY_PATH={summary_path}")
    print(f"MCP_VERIFY_OVERALL_PASS={summary.get('overall_pass')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
