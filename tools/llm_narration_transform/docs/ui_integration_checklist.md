# UI Integration Checklist (Ultimate TTS Studio)

This checklist maps the narration-transform feature into the existing app UI and generation pipeline.

## Scope

- Add optional pre-TTS text transform stage.
- Keep all current TTS model flows unchanged when feature is disabled.
- Support local OpenAI-compatible endpoints (Ollama/vLLM/LM Studio/llama.cpp server).

## 1) UI Controls in `app/launch.py`

- [ ] Add a new accordion in the main text workflow: `Narration Transform (LLM)`
- [ ] Add toggle: `Enable LLM narration transform`
- [ ] Add provider preset dropdown:
  - `Ollama (OpenAI-compatible)`
  - `vLLM OpenAI server`
  - `LM Studio OpenAI server`
  - `Custom OpenAI-compatible`
- [ ] Add endpoint fields:
  - `Base URL`
  - `API Key` (optional for localhost)
  - `Model ID`
- [ ] Add task controls:
  - `Mode`: `STRICT | NORMALIZE | EXPRESSIVE`
  - `Locale`
  - `Style`
  - `Max tag density`
- [ ] Add prompt controls:
  - `System prompt` (multiline)
  - `User template` (multiline)
  - `Use default Eleven-v3 template` button
- [ ] Add actions:
  - `Test Connection`
  - `Transform Text Preview`
  - `Apply Transform to Input`

## 2) Pipeline Hook

- [ ] Insert transform stage just before calling model-specific generation functions.
- [ ] Input text source:
  - single-text mode: `text` textbox
  - conversation mode: optional per-line transform
  - ebook mode: transform per chunk before synthesis
- [ ] On transform failure, log error and fall back to original text unless strict-fail is enabled.

## 3) Persistence

- [ ] Save feature settings to `app_state/settings.json`
- [ ] Save reusable prompt profiles to `app_state/presets.json`
- [ ] Include `enabled` flag and provider profile name in autosave state.

## 4) Safety/Rule Enforcement

- [ ] Add post-transform validation before synthesis:
  - reject/strip SSML tags (`<break>`, `<phoneme>`, `<prosody>`, etc.)
  - reject markdown/code fences
  - optional forbidden tag filter (`[music]`, `[standing]`, etc.)
- [ ] Cap tag density and punctuation amplification by configured thresholds.

## 5) Logging & Observability

- [ ] Add concise logs for:
  - endpoint/model selected
  - transform enabled + mode
  - token/character counts in/out
  - fallback usage
- [ ] Avoid logging raw API keys or sensitive headers.

## 6) API/Client Layer (recommended)

- [ ] Add isolated helper module, e.g. `app/tools/llm_narration_transform/client.py`
- [ ] Add method:
  - `transform_text(source_text, settings) -> transformed_text`
- [ ] Keep request/response schema strictly text-only.

## 7) Validation Gates Before Merge

- [ ] Manual smoke test with feature disabled (no behavior change).
- [ ] Manual smoke test with localhost endpoint and each mode.
- [ ] Ensure transformed output remains plain text only.
- [ ] Confirm no SSML is passed to TTS engines.
- [ ] Confirm settings persist across app restarts.

## Suggested Rollout

1. Ship behind a feature toggle (default off).
2. Collect logs/feedback from advanced users.
3. Promote to default-on once stable.
