# Ultimate TTS Studio

This directory contains the application runtime for Ultimate TTS Studio.

The main entry point is `launch.py`, which builds the Gradio UI and coordinates:

- multi-engine text-to-speech generation
- narration transform and LLM-assisted script polishing
- conversation formatting and multi-speaker synthesis
- audiobook conversion workflows
- history indexing and reload
- queued job execution
- optional assistant chat tools
- optional MCP sidecar integration

If you are using the Pinokio launcher workspace, start with the root `README.md` first.

## Current UI Surface

The current primary tabs in `launch.py` are:

- Text to Synthesize
- Conversation Mode
- eBook to Audiobook
- VibeVoice
- Assistant
- History
- Jobs

### Supported Engine Tabs

- ChatterboxTTS
- Chatterbox Multilingual
- Chatterbox Turbo
- Kokoro TTS
- Fish Speech
- IndexTTS
- IndexTTS2
- F5-TTS
- Higgs Audio
- VoxCPM
- KittenTTS
- Qwen TTS

Engine availability is dependency-driven. If an optional engine import fails, the app keeps running
and disables only that engine path.

## Major Runtime Features

### Single-Speaker Generation

- engine-specific synthesis controls
- optional narration transform before synthesis
- audio effects pipeline
- deterministic seed capture
- structured autosave metadata

### Conversation Mode

- script parsing and speaker extraction
- AI formatting tools for multi-speaker scripts
- per-speaker engine-specific assignment flows
- checkpointed resume support via `app_state/conversation_checkpoints/`

### eBook To Audiobook

- chapter-oriented audiobook generation flow
- shared output storage and autosave metadata

### Assistant

- separate LLM provider settings from narration transform
- saved provider, model, prompt, and generation defaults
- API key source indicators in the UI

### History

- indexing of structured generated artifacts
- record search and filtering
- reload of prior generation context back into the UI

### Jobs

- queue-aware orchestration for long-running generation work
- persisted job records under `app_state/jobs/`
- live progress checkpoints written by the worker subprocesses
- stale running jobs are marked failed with their last known progress after restart

## Persistence And Storage

`launch.py` currently uses these app-state locations:

- `app_state/settings.json`
- `app_state/presets.json`
- `app_state/voices/`
- `app_state/outputs/`
- `app_state/job_assets/`
- `app_state/conversation_checkpoints/`
- `app_state/conversation_draft.json`

Other runtime directories used by the app include:

- `outputs/`
- `custom_voices/`
- `audiobooks/`
- `cache/`

### LLM Settings Namespaces

The app persists separate settings for:

- narration transform
- conversation formatting
- assistant chat

These settings are intentionally namespaced so changing one workflow does not overwrite another.

## Optional MCP Sidecar

`mcp_sidecar.py` exposes an optional SSE-based MCP server in a separate environment. It is not
required for normal web UI use.

Current MCP tools exposed by the sidecar include:

- `list_engines`
- `get_engine_info`
- `list_voices`
- `list_outputs`
- `get_app_version`
- `normalize_text`
- `list_llm_providers`
- `transform_text`
- `structure_conversation`
- `synthesize`
- `submit_synthesis_job`
- `get_job_status`
- `cancel_job`

Related files:

- `mcp_sidecar.py`
- `mcp_security.py`
- `mcp_verify_summary.py`
- `requirements_mcp_sidecar.txt`

## Running The App Directly

The app can be launched directly from this directory with:

```bash
python launch.py
```

The Pinokio launcher currently runs the app inside a conda environment named `tts_env`.

For the closest supported setup, mirror the dependency flow from the root launcher scripts:

```bash
conda install -c conda-forge pynini==2.1.6 -y
conda install -y -c conda-forge portaudio
conda install -y -c conda-forge sox
uv pip install -r requirements.txt
uv pip install WeTextProcessing --no-deps
uv pip install --upgrade --force-reinstall --no-deps --no-cache-dir onnxruntime-gpu==1.22.0
uv pip install voxcpm openai-whisper --no-deps
```

Additional platform tooling may still be required:

- `espeak-ng` for best Kokoro behavior
- GPU-compatible PyTorch stack for accelerated inference
- model downloads required by specific engines

## Working On `launch.py`

Do not start editing `launch.py` by search alone.

Before reviewing or editing it, read:

- `../Docs/launch-py-index.md`

That index identifies the current structural map, high-risk coupling points, and the extracted
modules that now own logic previously embedded in the monolith.

Important extracted modules referenced by the index:

- `narration_transform.py`
- `conversation_logic.py`
- `engine_registry.py`
- `narration_script.py`
- `pronunciation.py`
- `tts_service.py`
- `job_manager.py`

## Directory Highlights

```text
app/
|- launch.py
|- narration_transform.py
|- conversation_logic.py
|- engine_registry.py
|- job_manager.py
|- mcp_sidecar.py
|- mcp_security.py
|- tests/
|- tools/
|- app_state/
`- README.md
```

## Operational Notes

- The app intentionally suppresses many startup warnings to keep the terminal readable.
- Models are loaded on demand rather than eagerly at startup.
- Output storage can target project folders or a custom base path.
- History is driven by structured autosave artifacts, not arbitrary loose files.
- Conversation generation paths differ by engine family; not all engines share one implementation.

## Related Docs

- `../README.md` for launcher behavior
- `../Docs/launch-py-index.md` for `launch.py` navigation
- `../Docs/LLM-Narration-Transform-Guide.md` for narration transform behavior and scope
