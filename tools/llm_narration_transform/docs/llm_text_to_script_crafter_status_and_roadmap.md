# LLM Text-to-Script Crafter — Status & Development Roadmap

Last updated: 2026-02-22 Scope: Ultimate TTS Studio integration for transforming source text into
Eleven v3-friendly narration scripts.

## 1) Executive Status

Current status: **Partially complete (core foundation implemented, production hardening pending).**

The feature is already integrated in the app and usable for single-text workflows, including:

- LLM transform controls in UI
- OpenAI-compatible endpoint configuration
- Connection test and prompt reset actions
- Apply-to-text transform action
- Generation-path integration
- Local deterministic fallback when no LLM endpoint/model is available
- Before/after text artifact persistence and sidecar metadata capture

Primary remaining work is to complete end-to-end parity for all workflows (especially
conversation/eBook transform application path), validation, and cloud-provider onboarding
UX/security.

---

## 2) What Is Already Implemented

### 2.1 Integration and UX (single-text flow)

- API key resolution supports shell environment variables (`GOOGLE_API_KEY` or `OPENAI_API_KEY`)
  when UI key field is empty.
- User can apply transform directly to textbox content before TTS generation.

### 2.2 Runtime behavior

- Generation wrapper can call transform before synthesis.
- If endpoint/model is missing or unavailable, optional local fallback transform can still normalize
  text for TTS readiness.
- Strict-mode behavior avoids destructive rewriting when fallback is used.

### 2.3 Data lineage and auditability

- Original and transformed script variants are saved with artifacts.
- Sidecar metadata includes hashes and file paths for traceability.
- Conversation/eBook metadata call sites were updated to pass original/transformed text fields where
  available.

### 2.4 Tooling assets added

Under `app/tools/llm_narration_transform/`:

- `README.md` (implementation guidance)
- `configs/qlora_qwen2_5_7b.yaml` (training template)
- `data/starter_dataset_template.jsonl` (starter supervision data)
- `scripts/eval_narration_transform.py` (rule-based evaluator)
- `docs/ui_integration_checklist.md` (integration checklist)

---

## 3) Known Gaps / Remaining Work

### 3.1 Functional parity gaps

1. **Conversation/eBook transform execution path**
   - Metadata capture is in place, but transform should be explicitly and consistently applied to
     input text in all conversation/eBook generation paths (not only captured as metadata fields).

2. **Cross-mode consistency**
   - Ensure `STRICT`, `NORMALIZE`, and `EXPRESSIVE` behave consistently across single-text,
     conversation, and eBook flows.

### 3.2 Validation gaps

1. End-to-end smoke tests are needed for:
   - Local endpoint success path
   - Endpoint failure + fallback enabled
   - No endpoint configured + fallback enabled
2. Artifact validation is needed for:
   - `.original.txt`, `.transformed.txt`, `.txt` script bundle integrity
   - Hash/path consistency in sidecar metadata
3. Regression checks are needed to ensure existing TTS engines are unaffected.

### 3.3 UX and configuration gaps

1. Clarify user-facing status messages for each transform outcome (remote success, fallback used,
   passthrough).
2. Add concise in-app help text for provider fields and API-key handling expectations.
3. Optional: side-by-side before/after preview panel before overwrite.

### 3.4 Security and reliability gaps

1. API key handling policy should be formalized (no accidental persistence/logging).
2. Timeout/retry behavior should be made explicit and consistent.
3. Error messaging should separate network/auth/model-format failures for easier troubleshooting.

---

## 4) Future Development Plan

## Phase A — Complete Runtime Coverage (High Priority)

- Apply transform uniformly in conversation/eBook generation entry points.
- Ensure fallback routing is identical across all text pipelines.
- Add targeted unit/helper tests for transform selection logic.

**Definition of done:**

- Every text-to-audio path has deterministic transform routing.
- Sidecar fields reflect actual executed transform path and outputs.

## Phase B — Validation & QA (High Priority)

- Build a smoke test matrix covering endpoint success/failure/no-endpoint scenarios.
- Validate metadata bundle outputs and hash integrity.
- Perform regression checks on supported TTS engines and modes.

**Definition of done:**

- Test checklist passes for all core flows and fallback paths.
- No new critical regressions in standard generation flows.

## Phase C — UX Hardening (Medium Priority)

- Improve status/error messages with actionable guidance.
- Add user-facing help tooltip copy for model/base URL/API key inputs.
- Optional review panel for before/after script diff.

**Definition of done:**

- Users can understand failure cause and recover without docs.
- Script-transform actions are transparent and predictable.

## Phase D — Cloud Provider Readiness (Medium Priority)

- Add provider templates for major cloud LLM endpoints.
- Establish API-key and endpoint profile management behavior.
- Verify cost/latency controls (timeouts, token limits, model defaults).

**Definition of done:**

- Users can connect at least one cloud provider and one local provider with minimal setup friction.

---

## 5) Cloud LLM Evaluation Note (Requested)

**Action item:** Evaluate and implement robust support for connecting to a cloud LLM provider using
user-supplied API endpoint + API key.

### Evaluation criteria

1. **Compatibility:** OpenAI-compatible API support and model naming conventions.
2. **Security:** Key handling (masked input, avoid logging, controlled persistence behavior).
3. **UX:** Simple provider selection + preset defaults + clear validation feedback.
4. **Reliability:** Timeout, retry/backoff, and explicit error categories.
5. **Cost controls:** Token limits, temperature defaults, and optional budget guardrails.
6. **Privacy:** Clear notice on external data transmission when cloud mode is enabled.

### Candidate integration approach

- Keep current OpenAI-compatible abstraction.
- Add cloud provider presets (base URL + auth header pattern + recommended defaults).
- Preserve a single generic “Custom OpenAI-Compatible” option for advanced users.
- Maintain local fallback path so script crafting still works when cloud access is unavailable.

### Service selection and availability (explicit)

Yes — this plan includes selecting which services are available in-product.

Recommended initial availability set:

1. **LM Studio (Local) — Required in MVP**
   - Type: local OpenAI-compatible endpoint
   - Why: best no-cloud path, fast iteration, aligns with privacy/local-first workflows
   - UX preset: base URL default + model dropdown/manual model entry

2. **Google Gemini API (Cloud) — Required in MVP**
   - Type: cloud provider preset
   - Why: strong model quality and broad user demand
   - UX preset: provider-specific base URL/auth guidance + API key validation

3. **OpenAI-Compatible Custom Endpoint (Local/Cloud) — Required in MVP**
   - Type: generic adapter
   - Why: future-proofing for OpenRouter, self-hosted gateways, and other compatible APIs

4. **Additional cloud presets (Phase 2)**
   - Candidates: OpenAI, Anthropic, Azure OpenAI
   - Delivery: after Google + LM Studio baseline passes validation gates

### Provider rollout tiers

- **Tier 1 (must ship):** LM Studio, Google Gemini API, Custom OpenAI-compatible
- **Tier 2 (next):** OpenAI, Anthropic
- **Tier 3 (enterprise/advanced):** Azure OpenAI and organization-specific gateways

### Cloud/local setup checklist

For each provider preset, implement and verify:

1. Provider entry in UI selector
2. Safe API key input (masked, not logged)
3. Base URL default + override support
4. Connection test with actionable errors (auth/network/model)
5. Recommended defaults (timeout, max tokens, temperature)
6. Privacy notice when remote provider is selected
7. Fallback behavior when provider unavailable
8. Metadata stamp of provider/model/fallback path used

---

## 6) Risks and Mitigations

1. **Risk:** Inconsistent outputs across providers/models.
   - **Mitigation:** Canonical system prompt, deterministic defaults, evaluator checks.
2. **Risk:** API key leakage in logs/settings.
   - **Mitigation:** Redaction, avoid plaintext persistence by default, explicit opt-in save.
3. **Risk:** Latency/cost spikes in cloud mode.
   - **Mitigation:** enforce max token/output limits and request timeouts.
4. **Risk:** Transform over-edits harming content intent.
   - **Mitigation:** `STRICT` mode and preview/apply workflow.

---

## 7) Recommended Next Sprint (Concrete)

1. Implement transform execution parity for conversation/eBook pipelines.
2. Create and run smoke-test matrix for local/cloud/fallback scenarios.
3. Add cloud provider preset UX with API-key safety guarantees.
4. Finalize user-facing docs for setup and troubleshooting.

Target outcome: production-ready “LLM Text-to-Script Crafter” with reliable local + cloud operation
and full audit traceability.
