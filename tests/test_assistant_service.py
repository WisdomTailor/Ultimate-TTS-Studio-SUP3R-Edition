from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import assistant_service
from assistant_service import (
    AssistantRequest,
    AssistantResponse,
    ChatMessage,
    DEFAULT_ASSISTANT_SYSTEM_PROMPT,
)


def test_module_has_zero_gradio_imports() -> None:
    source = (APP_DIR / "assistant_service.py").read_text(encoding="utf-8")

    assert "import gradio" not in source
    assert "from gradio" not in source


def test_chat_empty_message() -> None:
    request = AssistantRequest(user_message="   ")

    with patch("assistant_service.call_openai_compatible_chat") as chat_mock:
        response = assistant_service.chat(request)

    assert response.error == "Empty message"
    assert response.content == ""
    chat_mock.assert_not_called()


def test_chat_missing_api_key() -> None:
    request = AssistantRequest(
        user_message="Help me choose a model",
        provider_name="GitHub Models (OpenAI-compatible)",
        api_key="",
    )

    with patch("assistant_service.resolve_llm_api_key", return_value=("", "missing")):
        with patch("assistant_service.call_openai_compatible_chat") as chat_mock:
            response = assistant_service.chat(request)

    assert response.error == "API key required for GitHub Models (OpenAI-compatible)"
    assert response.content == ""
    chat_mock.assert_not_called()


def test_chat_success() -> None:
    request = AssistantRequest(
        user_message="What engine should I use for narration?",
        provider_name="LM Studio OpenAI Server",
        base_url="http://localhost:1234/v1",
        model_id="qwen/test-model",
        temperature=0.25,
        top_p=0.85,
        max_tokens=256,
        timeout_seconds=12,
    )

    with patch(
        "assistant_service.call_openai_compatible_chat", return_value="Use F5-TTS."
    ) as chat_mock:
        response = assistant_service.chat(request)

    assert response.error == ""
    assert response.content == "Use F5-TTS."
    assert response.provider_name == "LM Studio OpenAI Server"
    assert response.model_id == "qwen/test-model"

    kwargs = chat_mock.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:1234/v1"
    assert kwargs["model_id"] == "qwen/test-model"
    assert kwargs["system_prompt"] == DEFAULT_ASSISTANT_SYSTEM_PROMPT
    assert kwargs["user_prompt"] == "What engine should I use for narration?"
    assert kwargs["temperature"] == 0.25
    assert kwargs["top_p"] == 0.85
    assert kwargs["max_tokens"] == 256
    assert kwargs["timeout_seconds"] == 12
    assert kwargs["extra_headers"] == {}
    assert kwargs["auth_style"] == "bearer"


def test_chat_with_history() -> None:
    history = [
        ChatMessage(role="user", content="I need expressive dialogue."),
        ChatMessage(role="assistant", content="Use a more expressive engine."),
    ]
    request = AssistantRequest(
        user_message="What settings should I try?",
        conversation_history=history,
    )

    with patch(
        "assistant_service.call_openai_compatible_chat", return_value="Try temperature 0.5."
    ) as chat_mock:
        response = assistant_service.chat(request)

    assert response.error == ""
    assert (
        chat_mock.call_args.kwargs["user_prompt"] == "[User]: I need expressive dialogue.\n\n"
        "[Assistant]: Use a more expressive engine.\n\n"
        "[User]: What settings should I try?"
    )


def test_chat_llm_error() -> None:
    request = AssistantRequest(user_message="Why did my synthesis fail?")

    with patch("assistant_service.call_openai_compatible_chat", side_effect=RuntimeError("boom")):
        response = assistant_service.chat(request)

    assert response.content == ""
    assert response.error == "boom"
    assert response.elapsed_seconds >= 0.0


def test_build_messages_no_history() -> None:
    request = AssistantRequest(user_message="How do I clone a voice?")

    messages = assistant_service.build_messages(request)

    assert messages == [
        {"role": "system", "content": DEFAULT_ASSISTANT_SYSTEM_PROMPT},
        {"role": "user", "content": "How do I clone a voice?"},
    ]


def test_build_messages_with_history() -> None:
    request = AssistantRequest(
        user_message="What next?",
        system_prompt="Custom assistant prompt",
        conversation_history=[
            ChatMessage(role="user", content="I loaded an ebook."),
            ChatMessage(role="assistant", content="Now convert it to a script."),
        ],
    )

    messages = assistant_service.build_messages(request)

    assert messages == [
        {"role": "system", "content": "Custom assistant prompt"},
        {"role": "user", "content": "I loaded an ebook."},
        {"role": "assistant", "content": "Now convert it to a script."},
        {"role": "user", "content": "What next?"},
    ]


def test_flatten_history_empty() -> None:
    result = assistant_service._flatten_history_to_user_prompt([], "New question")

    assert result == "New question"


def test_flatten_history_with_messages() -> None:
    history = [
        ChatMessage(role="system", content="Stay concise."),
        ChatMessage(role="user", content="Need help with voice settings."),
        ChatMessage(role="assistant", content="Which engine are you using?"),
    ]

    result = assistant_service._flatten_history_to_user_prompt(history, "F5-TTS")

    assert result == (
        "[System]: Stay concise.\n\n"
        "[User]: Need help with voice settings.\n\n"
        "[Assistant]: Which engine are you using?\n\n"
        "[User]: F5-TTS"
    )


def test_default_system_prompt() -> None:
    request = AssistantRequest(user_message="Help", system_prompt="   ")

    messages = assistant_service.build_messages(request)

    assert messages[0]["content"] == DEFAULT_ASSISTANT_SYSTEM_PROMPT


def test_test_assistant_connection_success() -> None:
    mocked_response = AssistantResponse(
        content="OK",
        provider_name="LM Studio OpenAI Server",
        model_id="qwen/test-model",
        elapsed_seconds=0.42,
    )

    with patch("assistant_service.chat", return_value=mocked_response) as chat_mock:
        result = assistant_service.test_assistant_connection(
            provider_name="LM Studio OpenAI Server",
            base_url="http://localhost:1234/v1",
            api_key="",
            model_id="qwen/test-model",
        )

    assert "✅ Assistant connection successful" in result
    assert "Provider: LM Studio OpenAI Server" in result
    assert "URL: http://localhost:1234/v1" in result
    assert "Model: qwen/test-model" in result
    assert "Response: OK" in result
    assert "Latency: 0.42s" in result

    request = chat_mock.call_args.args[0]
    assert isinstance(request, AssistantRequest)
    assert request.user_message == "Reply with OK only."
    assert request.system_prompt == "Return exactly: OK"
    assert request.temperature == 0.0
    assert request.top_p == 1.0
    assert request.max_tokens == 8
    assert request.timeout_seconds == 15


def test_test_assistant_connection_error() -> None:
    mocked_response = AssistantResponse(
        content="",
        provider_name="LM Studio OpenAI Server",
        model_id="qwen/test-model",
        elapsed_seconds=0.12,
        error="connection refused",
    )

    with patch("assistant_service.chat", return_value=mocked_response):
        result = assistant_service.test_assistant_connection(
            provider_name="LM Studio OpenAI Server",
            base_url="http://localhost:1234/v1",
            api_key="",
            model_id="qwen/test-model",
        )

    assert result == "❌ Assistant connection failed: connection refused"
