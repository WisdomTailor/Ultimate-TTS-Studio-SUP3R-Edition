from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from collections import defaultdict
from pathlib import Path


RATE_TIERS: dict[str, dict[str, int]] = {
    "read": {"max_requests": 60, "window_seconds": 60},
    "transform": {"max_requests": 20, "window_seconds": 60},
    "synthesize": {"max_requests": 2, "window_seconds": 60},
}


TOOL_TIERS: dict[str, str] = {
    "list_engines": "read",
    "get_engine_info": "read",
    "list_voices": "read",
    "list_outputs": "read",
    "get_app_version": "read",
    "normalize_text": "read",
    "list_llm_providers": "read",
    "transform_text": "transform",
    "structure_conversation": "transform",
    "synthesize": "synthesize",
}


class McpSecurityError(Exception):
    """Base exception for MCP security failures."""


class McpAuthError(McpSecurityError):
    """Bearer token missing or invalid."""


class McpRateLimitError(McpSecurityError):
    """Rate limit exceeded."""


class TokenManager:
    """Generate, persist, and validate bearer tokens using constant-time comparison."""

    def __init__(self, token_path: Path | None = None) -> None:
        self._token_path = token_path or Path(".mcp_token")
        self._token_hash = ""
        self._lock = threading.Lock()

    def initialize(self) -> str:
        """Generate a new token, save it to disk, and return the plaintext token."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            self._token_path.write_text(token, encoding="utf-8")
            try:
                self._token_path.chmod(0o600)
            except OSError:
                pass
        return token

    def validate(self, token: str) -> bool:
        """Validate a bearer token using constant-time comparison."""
        if not token or not self._token_hash:
            return False

        candidate = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return secrets.compare_digest(candidate, self._token_hash)

    @property
    def token_path(self) -> Path:
        """Return the on-disk token file path."""
        return self._token_path


class RateLimiter:
    """Thread-safe sliding-window rate limiter keyed by identity and tool."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check_and_record(self, identity: str, tool_name: str) -> tuple[bool, int, int]:
        """Check the limit for a tool call and record it if allowed.

        Args:
            identity: Stable caller identity, typically token-derived.
            tool_name: MCP tool name used for tier lookup.

        Returns:
            A tuple of `(allowed, current_count, max_allowed)`.
        """
        tier_name = TOOL_TIERS.get(tool_name, "transform")
        tier = RATE_TIERS[tier_name]
        key = f"{identity}:{tool_name}"
        now = time.monotonic()
        cutoff = now - tier["window_seconds"]
        max_requests = tier["max_requests"]

        with self._lock:
            entries = self._windows[key]
            entries[:] = [entry for entry in entries if entry > cutoff]
            used = len(entries)
            if used < max_requests:
                entries.append(now)
                return True, used + 1, max_requests
            return False, used, max_requests


class AuditLogger:
    """Structured audit logger for MCP tool invocations."""

    def __init__(self, log_dir: Path | None = None) -> None:
        resolved_log_dir = (log_dir or Path("logs") / "mcp").resolve()
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = resolved_log_dir / "audit.log"
        logger_key = hashlib.sha1(str(self._log_path).encode("utf-8")).hexdigest()[:12]
        self._logger = logging.getLogger(f"mcp.audit.{logger_key}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == self._log_path
            for handler in self._logger.handlers
        ):
            handler = logging.FileHandler(self._log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
            self._logger.addHandler(handler)

    @property
    def log_path(self) -> Path:
        """Return the audit log file path."""
        return self._log_path

    def log(self, tool_name: str, identity: str, allowed: bool, reason: str = "") -> None:
        """Record an MCP tool invocation to the audit log.

        Args:
            tool_name: The invoked MCP tool.
            identity: Caller identity used for rate limiting.
            allowed: Whether the invocation was allowed.
            reason: Optional deny reason or additional context.
        """
        status = "ALLOW" if allowed else "DENY"
        message = f"tool={tool_name} identity={identity} status={status}"
        if reason:
            message += f" reason={reason}"
        self._logger.info(message)


class McpSecurity:
    """Facade for MCP token auth, rate limiting, and audit logging."""

    def __init__(
        self,
        token_path: Path | None = None,
        log_dir: Path | None = None,
        auth_enabled: bool = True,
    ) -> None:
        self.tokens = TokenManager(token_path)
        self.rate_limiter = RateLimiter()
        self.audit = AuditLogger(log_dir)
        self._auth_enabled = auth_enabled

    def initialize(self) -> str:
        """Initialize the security layer and return the bearer token."""
        return self.tokens.initialize()

    def guard(self, tool_name: str, token: str = "", identity: str = "") -> None:
        """Authorize and rate-limit a tool call.

        Args:
            tool_name: The MCP tool being invoked.
            token: Bearer token extracted from the request.
            identity: Optional caller identity override.

        Raises:
            McpAuthError: If auth is enabled and the token is invalid.
            McpRateLimitError: If the caller exceeds the tool rate limit.
        """
        if self._auth_enabled:
            if not self.tokens.validate(token):
                denied_identity = identity or "anonymous"
                self.audit.log(tool_name, denied_identity, False, "auth_failed")
                raise McpAuthError(
                    "Authentication required. Provide a valid bearer token. "
                    f"Token file: {self.tokens.token_path}"
                )
            if not identity and token:
                identity = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]

        identity = identity or "anonymous"

        allowed, used, max_requests = self.rate_limiter.check_and_record(identity, tool_name)
        if not allowed:
            self.audit.log(tool_name, identity, False, f"rate_limit({used}/{max_requests})")
            raise McpRateLimitError(
                f"Rate limit exceeded for {tool_name}: {used}/{max_requests} requests per minute."
            )

        self.audit.log(tool_name, identity, True)


_security: McpSecurity | None = None
_lock = threading.Lock()


def get_security() -> McpSecurity:
    """Return the lazy-initialized module-level MCP security singleton."""
    global _security

    if _security is None:
        with _lock:
            if _security is None:
                _security = McpSecurity()
    return _security


def initialize_security(
    token_path: Path | None = None,
    log_dir: Path | None = None,
    auth_enabled: bool = True,
) -> str:
    """Initialize the global security singleton and return the bearer token."""
    global _security

    with _lock:
        _security = McpSecurity(
            token_path=token_path,
            log_dir=log_dir,
            auth_enabled=auth_enabled,
        )
    return _security.initialize()


__all__ = [
    "McpSecurity",
    "McpSecurityError",
    "McpAuthError",
    "McpRateLimitError",
    "TokenManager",
    "RateLimiter",
    "AuditLogger",
    "get_security",
    "initialize_security",
    "RATE_TIERS",
    "TOOL_TIERS",
]