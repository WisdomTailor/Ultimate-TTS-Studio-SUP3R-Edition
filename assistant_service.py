"""In-app assistant service for Ultimate TTS Studio.

Stateless conversational interface that answers user questions about TTS engines,
voice configuration, troubleshooting, and workflow guidance. Reuses the existing
LLM provider infrastructure from narration_transform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from narration_transform import (
    _get_provider_config,
    call_openai_compatible_chat,
    resolve_llm_api_key,
)


DEFAULT_ASSISTANT_SYSTEM_PROMPT = (
    "You are the Ultimate TTS Studio assistant. "
    "Help users with text-to-speech tasks including engine selection, "
    "voice configuration, audio format settings, narration transforms, "
    "eBook conversion, and troubleshooting. "
    "Be concise and practical. When suggesting settings, include specific "
    "parameter values. If you don't know something, say so."
)


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a conversation.

    Attributes:
        role: Message role such as "system", "user", or "assistant".
        content: Message body.
    """

    role: str
    content: str


@dataclass(frozen=True)
class AssistantRequest:
    """Request payload for the assistant service.

    Attributes:
        user_message: The newest user message to answer.
        conversation_history: Prior messages supplied by the caller.
        provider_name: Configured LLM provider display name.
        base_url: Optional endpoint override.
        api_key: Optional API key override.
        model_id: Optional model override.
        system_prompt: Optional assistant system prompt override.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_tokens: Maximum completion tokens.
        timeout_seconds: Request timeout in seconds.
    """

    user_message: str
    conversation_history: Sequence[ChatMessage] = ()
    provider_name: str = "LM Studio OpenAI Server"
    base_url: str = ""
    api_key: str = ""
    model_id: str = ""
    system_prompt: str = ""
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 1024
    timeout_seconds: int = 30


@dataclass(frozen=True)
class AssistantResponse:
    """Response payload from the assistant service.

    Attributes:
        content: Assistant response text.
        provider_name: Provider used for the request.
        model_id: Model used for the request.
        elapsed_seconds: Request duration in seconds.
        error: Non-empty error message when the call failed.
    """

    content: str
    provider_name: str
    model_id: str
    elapsed_seconds: float
    error: str = ""


def build_messages(request: AssistantRequest) -> list[dict[str, str]]:
    """Build the logical message list for an assistant request.

    Args:
        request: Assistant request parameters.

    Returns:
        The ordered message list beginning with the effective system prompt.
    """

    system_prompt = request.system_prompt.strip() or DEFAULT_ASSISTANT_SYSTEM_PROMPT
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for message in request.conversation_history:
        messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": request.user_message})
    return messages


def chat(request: AssistantRequest) -> AssistantResponse:
    """Send a request to the assistant and return a structured response.

    Args:
        request: Assistant request parameters.

    Returns:
        The assistant response. Errors are returned in the response instead of
        being raised.
    """

    if not request.user_message or not request.user_message.strip():
        return AssistantResponse(
            content="",
            provider_name=request.provider_name,
            model_id=request.model_id.strip(),
            elapsed_seconds=0.0,
            error="Empty message",
        )

    provider_config = _get_provider_config(request.provider_name)
    resolved_api_key, _key_source = resolve_llm_api_key(request.provider_name, request.api_key)

    if provider_config["requires_api_key"] and not resolved_api_key:
        return AssistantResponse(
            content="",
            provider_name=request.provider_name,
            model_id=request.model_id.strip() or provider_config["default_model"],
            elapsed_seconds=0.0,
            error=f"API key required for {request.provider_name}",
        )

    base_url = request.base_url.strip() or provider_config["base_url"]
    model_id = request.model_id.strip() or provider_config["default_model"]
    system_prompt = request.system_prompt.strip() or DEFAULT_ASSISTANT_SYSTEM_PROMPT
    user_prompt = _flatten_history_to_user_prompt(
        request.conversation_history,
        request.user_message,
    )

    start = time.monotonic()
    try:
        content = call_openai_compatible_chat(
            base_url=base_url,
            api_key=resolved_api_key,
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=request.timeout_seconds,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            extra_headers=provider_config["headers"],
            auth_style=provider_config["auth_style"],
        )
    except Exception as error:
        elapsed = time.monotonic() - start
        return AssistantResponse(
            content="",
            provider_name=request.provider_name,
            model_id=model_id,
            elapsed_seconds=round(elapsed, 2),
            error=str(error),
        )

    elapsed = time.monotonic() - start
    return AssistantResponse(
        content=content,
        provider_name=request.provider_name,
        model_id=model_id,
        elapsed_seconds=round(elapsed, 2),
    )


def test_assistant_connection(
    provider_name: str,
    base_url: str,
    api_key: str,
    model_id: str,
) -> str:
    """Test assistant connectivity using the shared assistant chat path.

    Args:
        provider_name: Provider display name.
        base_url: Optional endpoint override.
        api_key: Optional API key override.
        model_id: Optional model override.

    Returns:
        A formatted success or failure status message.
    """

    provider_config = _get_provider_config(provider_name)
    effective_base_url = base_url.strip() or provider_config["base_url"]

    response = chat(
        AssistantRequest(
            user_message="Reply with OK only.",
            provider_name=provider_name,
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            system_prompt="Return exactly: OK",
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
            timeout_seconds=15,
        )
    )

    if response.error:
        return f"❌ Assistant connection failed: {response.error}"

    return (
        "✅ Assistant connection successful\n"
        f"Provider: {provider_name}\n"
        f"URL: {effective_base_url}\n"
        f"Model: {response.model_id}\n"
        f"Response: {response.content[:120]}\n"
        f"Latency: {response.elapsed_seconds}s"
    )


def _flatten_history_to_user_prompt(history: Sequence[ChatMessage], new_message: str) -> str:
    """Flatten history into a single user prompt for the shared LLM call path.

    Args:
        history: Prior conversation messages.
        new_message: The newest user message.

    Returns:
        A prompt string suitable for providers that accept only one user turn.
    """

    if not history:
        return new_message

    role_labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    parts: list[str] = []
    for message in history:
        normalized_role = role_labels.get(message.role.lower(), message.role.strip().title())
        parts.append(f"[{normalized_role}]: {message.content}")
    parts.append(f"[User]: {new_message}")
    return "\n\n".join(parts)


__all__ = [
    "AssistantRequest",
    "AssistantResponse",
    "ChatMessage",
    "DEFAULT_ASSISTANT_SYSTEM_PROMPT",
    "build_messages",
    "chat",
    "test_assistant_connection",
]