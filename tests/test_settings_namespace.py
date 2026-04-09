from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import ModuleType

import pytest


APP_DIR = Path(__file__).resolve().parents[1]
LAUNCH_PATH = APP_DIR / "launch.py"

REQUIRED_ASSIGNMENTS = {
    "APP_STATE_DIR",
    "APP_STATE_SETTINGS_FILE",
    "APP_STATE_VOICES_DIR",
    "APP_STATE_OUTPUTS_DIR",
    "DEFAULT_AUTOSAVE_SETTINGS",
    "LEGACY_LLM_SETTINGS_KEY_MAP",
    "LEGACY_DEFAULT_FILENAME_TEMPLATE",
    "PREVIOUS_DEFAULT_FILENAME_TEMPLATE",
    "PRESET_ONLY_FILENAME_TEMPLATE",
}

REQUIRED_FUNCTIONS = {
    "ensure_app_state_dirs",
    "load_app_state_settings",
    "save_app_state_settings",
    "normalize_llm_content_type",
    "normalize_llm_outcome_preset",
    "get_llm_outcome_preset_values",
    "_get_llm_settings_key",
    "_get_default_llm_setting",
    "_get_initial_namespaced_llm_settings",
    "get_initial_llm_panel_settings",
    "get_initial_assistant_llm_settings",
    "_save_namespaced_llm_settings",
    "save_llm_panel_settings",
    "save_assistant_llm_settings",
}


def _assignment_targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _extract_settings_module() -> ModuleType:
    source = LAUNCH_PATH.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(LAUNCH_PATH))
    selected_nodes: list[ast.stmt] = []

    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in REQUIRED_FUNCTIONS:
            selected_nodes.append(node)
            continue

        if _assignment_targets(node) & REQUIRED_ASSIGNMENTS:
            selected_nodes.append(node)

    module = ModuleType("launch_settings_subset")
    content_type_presets = {
        "General (Default)": {
            "system_prompt": None,
            "description": "General-purpose cleanup.",
        },
        "Audiobook Narration": {
            "system_prompt": "Audiobook system prompt",
            "description": "Audiobook narration cleanup.",
        },
    }
    module.__dict__.update(
        {
            "__builtins__": __builtins__,
            "json": json,
            "os": os,
            "CONTENT_TYPE_PRESETS": content_type_presets,
            "DEFAULT_CONTENT_TYPE_PRESET": "General (Default)",
            "DEFAULT_LLM_NARRATION_SYSTEM_PROMPT": "Narration system prompt",
            "DEFAULT_LLM_OUTCOME_PRESET": "Balanced",
            "LLM_OUTCOME_PRESETS": {
                "Balanced": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512},
                "Precise": {"temperature": 0.2, "top_p": 0.8, "max_tokens": 256},
            },
            "LLM_PROVIDER_CONFIGS": {
                "LM Studio OpenAI Server": {
                    "base_url": "http://127.0.0.1:1234/v1",
                    "default_model": "lmstudio-default",
                },
                "OpenAI": {
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-test",
                },
            },
            "LLM_PROVIDER_MODEL_SUGGESTIONS": {
                "LM Studio OpenAI Server": ["lmstudio-default", "other-local-model"],
                "OpenAI": ["gpt-test", "gpt-alt"],
            },
            "_DEFAULT_PROVIDER_CONFIG": {"base_url": "", "default_model": ""},
        }
    )
    module.__dict__["_get_provider_config"] = lambda provider_name: module.LLM_PROVIDER_CONFIGS.get(
        provider_name, module._DEFAULT_PROVIDER_CONFIG
    )
    module.__dict__["get_content_type_preset_names"] = lambda: list(content_type_presets.keys())
    module.__dict__["get_content_type_system_prompt"] = lambda preset_name: (
        content_type_presets.get(preset_name, {}).get("system_prompt")
        or module.DEFAULT_LLM_NARRATION_SYSTEM_PROMPT
    )

    extracted = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(extracted)
    exec(compile(extracted, str(LAUNCH_PATH), "exec"), module.__dict__)
    return module


@pytest.fixture
def settings_module(tmp_path: Path) -> ModuleType:
    module = _extract_settings_module()
    app_state_dir = tmp_path / "app_state"
    module.__dict__.update(
        {
            "APP_STATE_DIR": str(app_state_dir),
            "APP_STATE_SETTINGS_FILE": str(app_state_dir / "settings.json"),
            "APP_STATE_VOICES_DIR": str(app_state_dir / "voices"),
            "APP_STATE_OUTPUTS_DIR": str(app_state_dir / "outputs"),
        }
    )
    return module


class TestSettingsNamespace:
    def test_migrates_legacy_llm_keys_into_narration_namespace(
        self, settings_module: ModuleType
    ) -> None:
        legacy_payload = {
            "filename_template": "legacy_template",
            "llm_provider": "OpenAI",
            "llm_preset": "Precise",
            "llm_base_url": "https://legacy.example/v1",
            "llm_model_id": "legacy-model",
            "llm_system_prompt": "Legacy narration prompt",
        }

        settings_path = Path(settings_module.APP_STATE_SETTINGS_FILE)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        loaded = settings_module.load_app_state_settings()
        persisted = json.loads(settings_path.read_text(encoding="utf-8"))

        assert loaded["narration_llm_provider"] == "OpenAI"
        assert loaded["narration_llm_preset"] == "Precise"
        assert loaded["narration_llm_base_url"] == "https://legacy.example/v1"
        assert loaded["narration_llm_model_id"] == "legacy-model"
        assert loaded["narration_llm_system_prompt"] == "Legacy narration prompt"
        assert "llm_provider" not in loaded
        assert "llm_provider" not in persisted
        assert persisted["narration_llm_provider"] == "OpenAI"
        assert persisted["assistant_llm_provider"] == "LM Studio OpenAI Server"

    def test_fresh_install_defaults_include_both_namespaces(
        self, settings_module: ModuleType
    ) -> None:
        loaded = settings_module.load_app_state_settings()

        assert loaded == settings_module.DEFAULT_AUTOSAVE_SETTINGS
        assert loaded["narration_llm_provider"] == "LM Studio OpenAI Server"
        assert loaded["assistant_llm_provider"] == "LM Studio OpenAI Server"
        assert loaded["narration_llm_content_type"] == "General (Default)"
        assert loaded["narration_llm_system_prompt"] == ""
        assert loaded["assistant_llm_system_prompt"] == ""

    def test_narration_and_assistant_save_load_roundtrip(self, settings_module: ModuleType) -> None:
        settings_module.save_llm_panel_settings(
            provider_name="OpenAI",
            base_url="https://narration.example/v1",
            model_id="narration-model",
            api_key="unused-session-key",
            content_type_name="Audiobook Narration",
            system_prompt="Narration custom prompt",
            preset_name="Precise",
        )
        settings_module.save_assistant_llm_settings(
            provider_name="LM Studio OpenAI Server",
            base_url="http://127.0.0.1:9999/v1",
            model_id="assistant-model",
            api_key="unused-session-key",
            system_prompt="Assistant custom prompt",
            preset_name="Balanced",
        )

        loaded = settings_module.load_app_state_settings()
        narration = settings_module.get_initial_llm_panel_settings(loaded)
        assistant = settings_module.get_initial_assistant_llm_settings(loaded)

        assert loaded["narration_llm_provider"] == "OpenAI"
        assert loaded["assistant_llm_provider"] == "LM Studio OpenAI Server"
        assert narration["provider"] == "OpenAI"
        assert narration["preset"] == "Precise"
        assert narration["base_url"] == "https://narration.example/v1"
        assert narration["model_id"] == "narration-model"
        assert narration["content_type"] == "Audiobook Narration"
        assert narration["system_prompt"] == "Narration custom prompt"
        assert assistant["provider"] == "LM Studio OpenAI Server"
        assert assistant["base_url"] == "http://127.0.0.1:9999/v1"
        assert assistant["model_id"] == "assistant-model"
        assert assistant["system_prompt"] == "Assistant custom prompt"

    def test_assistant_updates_do_not_clobber_narration_namespace(
        self, settings_module: ModuleType
    ) -> None:
        settings_module.save_llm_panel_settings(
            provider_name="OpenAI",
            base_url="https://narration.example/v1",
            model_id="narration-model",
            api_key="unused-session-key",
            content_type_name="Audiobook Narration",
            system_prompt="Narration custom prompt",
            preset_name="Precise",
        )

        settings_module.save_assistant_llm_settings(
            provider_name="LM Studio OpenAI Server",
            base_url="http://127.0.0.1:7777/v1",
            model_id="assistant-model",
            api_key="unused-session-key",
            system_prompt="",
            preset_name="Balanced",
        )

        loaded = settings_module.load_app_state_settings()

        assert loaded["narration_llm_provider"] == "OpenAI"
        assert loaded["narration_llm_base_url"] == "https://narration.example/v1"
        assert loaded["narration_llm_content_type"] == "Audiobook Narration"
        assert loaded["narration_llm_model_id"] == "narration-model"
        assert loaded["assistant_llm_provider"] == "LM Studio OpenAI Server"
        assert loaded["assistant_llm_base_url"] == "http://127.0.0.1:7777/v1"
        assert loaded["assistant_llm_model_id"] == "assistant-model"

    def test_initial_assistant_settings_use_empty_default_system_prompt(
        self, settings_module: ModuleType
    ) -> None:
        loaded = settings_module.load_app_state_settings()

        narration = settings_module.get_initial_llm_panel_settings(loaded)
        assistant = settings_module.get_initial_assistant_llm_settings(loaded)

        assert narration["content_type"] == "General (Default)"
        assert narration["system_prompt"] == settings_module.DEFAULT_LLM_NARRATION_SYSTEM_PROMPT
        assert assistant["system_prompt"] == ""
        assert set(assistant) == {
            "provider",
            "preset",
            "temperature",
            "top_p",
            "max_tokens",
            "base_url",
            "api_key",
            "model_id",
            "model_choices",
            "system_prompt",
        }

    def test_narration_content_type_uses_preset_prompt_when_system_prompt_is_blank(
        self, settings_module: ModuleType
    ) -> None:
        settings_module.save_llm_panel_settings(
            provider_name="OpenAI",
            base_url="https://narration.example/v1",
            model_id="narration-model",
            api_key="unused-session-key",
            content_type_name="Audiobook Narration",
            system_prompt="Audiobook system prompt",
            preset_name="Precise",
        )

        loaded = settings_module.load_app_state_settings()
        narration = settings_module.get_initial_llm_panel_settings(loaded)

        assert loaded["narration_llm_content_type"] == "Audiobook Narration"
        assert loaded["narration_llm_system_prompt"] == ""
        assert narration["content_type"] == "Audiobook Narration"
        assert narration["system_prompt"] == "Audiobook system prompt"
