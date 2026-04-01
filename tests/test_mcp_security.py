from __future__ import annotations

import hashlib
import importlib
import logging
import re
import sys
from pathlib import Path

import pytest


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


import mcp_security


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    original = mcp_security._security
    mcp_security._security = None
    try:
        yield
    finally:
        mcp_security._security = original


@pytest.fixture(autouse=True)
def close_audit_handlers() -> None:
    yield
    manager = logging.Logger.manager
    for name, logger in list(manager.loggerDict.items()):
        if not isinstance(logger, logging.Logger) or not name.startswith("mcp.audit."):
            continue
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


class TestTokenManager:
    def test_initialize_writes_token_file_and_validates(self, tmp_path: Path) -> None:
        manager = mcp_security.TokenManager(tmp_path / ".mcp_token")

        token = manager.initialize()

        assert token
        assert manager.token_path.read_text(encoding="utf-8") == token
        assert manager.validate(token) is True

    def test_validate_rejects_wrong_token(self, tmp_path: Path) -> None:
        manager = mcp_security.TokenManager(tmp_path / ".mcp_token")
        manager.initialize()

        assert manager.validate("wrong-token") is False

    def test_validate_rejects_empty_when_uninitialized(self, tmp_path: Path) -> None:
        manager = mcp_security.TokenManager(tmp_path / ".mcp_token")

        assert manager.validate("") is False
        assert manager.validate("anything") is False

    def test_initialize_creates_parent_directory(self, tmp_path: Path) -> None:
        token_path = tmp_path / "nested" / "secrets" / ".mcp_token"
        manager = mcp_security.TokenManager(token_path)

        token = manager.initialize()

        assert token_path.exists()
        assert token_path.read_text(encoding="utf-8") == token

    def test_validate_uses_compare_digest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = mcp_security.TokenManager(tmp_path / ".mcp_token")
        token = manager.initialize()
        called: list[tuple[str, str]] = []

        def fake_compare_digest(left: str, right: str) -> bool:
            called.append((left, right))
            return left == right

        monkeypatch.setattr(mcp_security.secrets, "compare_digest", fake_compare_digest)

        assert manager.validate(token) is True
        assert len(called) == 1
        assert called[0][0] == hashlib.sha256(token.encode("utf-8")).hexdigest()

    def test_initialize_generates_new_token_each_time(self, tmp_path: Path) -> None:
        manager = mcp_security.TokenManager(tmp_path / ".mcp_token")

        first = manager.initialize()
        second = manager.initialize()

        assert first != second
        assert manager.validate(second) is True
        assert manager.validate(first) is False


class TestRateLimiter:
    def test_allows_requests_within_read_limit(self) -> None:
        limiter = mcp_security.RateLimiter()

        allowed, used, max_allowed = limiter.check_and_record("token-a", "list_engines")

        assert allowed is True
        assert used == 1
        assert max_allowed == 60

    def test_denies_request_over_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limiter = mcp_security.RateLimiter()
        now = {"value": 100.0}

        monkeypatch.setattr(mcp_security.time, "monotonic", lambda: now["value"])
        for _ in range(2):
            allowed, _, _ = limiter.check_and_record("token-a", "synthesize")
            assert allowed is True

        allowed, used, max_allowed = limiter.check_and_record("token-a", "synthesize")

        assert allowed is False
        assert used == 2
        assert max_allowed == 2

    def test_window_expiry_allows_again_after_cutoff(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        limiter = mcp_security.RateLimiter()
        now = {"value": 100.0}

        monkeypatch.setattr(mcp_security.time, "monotonic", lambda: now["value"])
        for _ in range(2):
            limiter.check_and_record("token-a", "synthesize")

        now["value"] = 161.0
        allowed, used, max_allowed = limiter.check_and_record("token-a", "synthesize")

        assert allowed is True
        assert used == 1
        assert max_allowed == 2

    def test_different_tools_use_different_tiers(self) -> None:
        limiter = mcp_security.RateLimiter()

        _, _, read_max = limiter.check_and_record("token-a", "list_engines")
        _, _, transform_max = limiter.check_and_record("token-a", "transform_text")
        _, _, synthesize_max = limiter.check_and_record("token-a", "synthesize")

        assert read_max == 60
        assert transform_max == 20
        assert synthesize_max == 2

    def test_unknown_tools_default_to_transform(self) -> None:
        limiter = mcp_security.RateLimiter()

        _, _, max_allowed = limiter.check_and_record("token-a", "unknown_tool")

        assert max_allowed == 20

    def test_identity_isolated_per_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limiter = mcp_security.RateLimiter()
        monkeypatch.setattr(mcp_security.time, "monotonic", lambda: 100.0)

        for _ in range(2):
            assert limiter.check_and_record("token-a", "synthesize")[0] is True

        assert limiter.check_and_record("token-a", "synthesize")[0] is False
        assert limiter.check_and_record("token-b", "synthesize")[0] is True

    def test_tool_isolated_per_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limiter = mcp_security.RateLimiter()
        monkeypatch.setattr(mcp_security.time, "monotonic", lambda: 100.0)

        for _ in range(20):
            assert limiter.check_and_record("token-a", "transform_text")[0] is True

        assert limiter.check_and_record("token-a", "transform_text")[0] is False
        assert limiter.check_and_record("token-a", "structure_conversation")[0] is True

    def test_used_count_increments_on_allow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limiter = mcp_security.RateLimiter()
        monkeypatch.setattr(mcp_security.time, "monotonic", lambda: 100.0)

        first = limiter.check_and_record("token-a", "transform_text")
        second = limiter.check_and_record("token-a", "transform_text")

        assert first == (True, 1, 20)
        assert second == (True, 2, 20)


class TestAuditLogger:
    def test_log_writes_to_audit_file(self, tmp_path: Path) -> None:
        logger = mcp_security.AuditLogger(tmp_path / "logs" / "mcp")

        logger.log("list_engines", "token-a", True)
        for handler in logger._logger.handlers:
            handler.flush()

        content = logger.log_path.read_text(encoding="utf-8")
        assert "tool=list_engines identity=token-a status=ALLOW" in content

    def test_log_includes_reason_when_provided(self, tmp_path: Path) -> None:
        logger = mcp_security.AuditLogger(tmp_path / "logs" / "mcp")

        logger.log("transform_text", "token-a", False, "auth_failed")
        for handler in logger._logger.handlers:
            handler.flush()

        content = logger.log_path.read_text(encoding="utf-8")
        assert "reason=auth_failed" in content

    def test_log_line_has_timestamp_prefix(self, tmp_path: Path) -> None:
        logger = mcp_security.AuditLogger(tmp_path / "logs" / "mcp")

        logger.log("list_outputs", "token-a", True)
        for handler in logger._logger.handlers:
            handler.flush()

        line = logger.log_path.read_text(encoding="utf-8").strip().splitlines()[0]
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} ", line)


class TestMcpSecurity:
    def test_guard_allows_valid_token(self, tmp_path: Path) -> None:
        security = mcp_security.McpSecurity(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
        )
        token = security.initialize()

        security.guard("list_engines", token)

        content = security.audit.log_path.read_text(encoding="utf-8")
        assert "status=ALLOW" in content

    def test_guard_raises_auth_error_for_invalid_token(self, tmp_path: Path) -> None:
        security = mcp_security.McpSecurity(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
        )
        security.initialize()

        with pytest.raises(mcp_security.McpAuthError):
            security.guard("list_engines", "invalid-token")

    def test_guard_raises_rate_limit_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        security = mcp_security.McpSecurity(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
        )
        token = security.initialize()
        monkeypatch.setattr(mcp_security.time, "monotonic", lambda: 100.0)

        security.guard("synthesize", token)
        security.guard("synthesize", token)

        with pytest.raises(mcp_security.McpRateLimitError):
            security.guard("synthesize", token)

    def test_guard_auth_disabled_skips_token_validation(self, tmp_path: Path) -> None:
        security = mcp_security.McpSecurity(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
            auth_enabled=False,
        )

        security.guard("list_engines")

        content = security.audit.log_path.read_text(encoding="utf-8")
        assert "identity=anonymous" in content
        assert "status=ALLOW" in content

    def test_guard_derives_identity_from_token_hash(self, tmp_path: Path) -> None:
        security = mcp_security.McpSecurity(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
        )
        token = security.initialize()

        security.guard("list_engines", token)

        derived_identity = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
        content = security.audit.log_path.read_text(encoding="utf-8")
        assert f"identity={derived_identity}" in content

    def test_guard_keeps_explicit_identity(self, tmp_path: Path) -> None:
        security = mcp_security.McpSecurity(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
        )
        token = security.initialize()

        security.guard("list_engines", token, identity="session-123")

        content = security.audit.log_path.read_text(encoding="utf-8")
        assert "identity=session-123" in content

    def test_guard_logs_auth_failure_reason(self, tmp_path: Path) -> None:
        security = mcp_security.McpSecurity(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
        )
        security.initialize()

        with pytest.raises(mcp_security.McpAuthError):
            security.guard("list_engines", "bad-token")

        content = security.audit.log_path.read_text(encoding="utf-8")
        assert "status=DENY" in content
        assert "reason=auth_failed" in content

    def test_guard_logs_rate_limit_failure_reason(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        security = mcp_security.McpSecurity(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
        )
        token = security.initialize()
        monkeypatch.setattr(mcp_security.time, "monotonic", lambda: 100.0)

        security.guard("synthesize", token)
        security.guard("synthesize", token)
        with pytest.raises(mcp_security.McpRateLimitError):
            security.guard("synthesize", token)

        content = security.audit.log_path.read_text(encoding="utf-8")
        assert "reason=rate_limit(2/2)" in content


class TestModuleFunctions:
    def test_get_security_returns_singleton(self) -> None:
        first = mcp_security.get_security()
        second = mcp_security.get_security()

        assert first is second

    def test_initialize_security_returns_token_and_replaces_singleton(self, tmp_path: Path) -> None:
        token = mcp_security.initialize_security(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
        )

        security = mcp_security.get_security()
        assert token
        assert security.tokens.token_path.read_text(encoding="utf-8") == token

    def test_initialize_security_respects_auth_flag(self, tmp_path: Path) -> None:
        mcp_security.initialize_security(
            token_path=tmp_path / ".mcp_token",
            log_dir=tmp_path / "logs" / "mcp",
            auth_enabled=False,
        )

        security = mcp_security.get_security()
        security.guard("list_engines")

    def test_module_can_be_reloaded_cleanly(self) -> None:
        reloaded = importlib.reload(mcp_security)

        assert hasattr(reloaded, "McpSecurity")
        assert hasattr(reloaded, "initialize_security")