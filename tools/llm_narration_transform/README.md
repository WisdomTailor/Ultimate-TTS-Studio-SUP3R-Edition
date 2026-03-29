# LLM Narration Transform Bundle (Eleven v3-Oriented)

This bundle provides a practical MVP kit to build a small specialist model adapter for:

- story/text chunk → narration-ready text
- Eleven v3-compatible prompting constraints
- strict output controls (no SSML, no meta commentary)

## Included

- `configs/qlora_qwen2_5_7b.yaml` — QLoRA training configuration template for RTX 4090 workflows.
- `data/starter_dataset_template.jsonl` — starter dataset structure with mixed mode examples.
- `scripts/eval_narration_transform.py` — rule + quality evaluation script for model outputs.
- `docs/ui_integration_checklist.md` — implementation checklist for integrating the feature into Ultimate TTS Studio.
- `docs/llm_text_to_script_crafter_status_and_roadmap.md` — current delivery status, remaining gaps, and development roadmap.

## Task Modes

Use one adapter for multiple modes:

- `STRICT`: preserve wording; only minimal punctuation adjustments.
- `NORMALIZE`: expand TTS-hostile text (dates, currency, phone numbers, URLs, units, abbreviations).
- `EXPRESSIVE`: controlled punctuation and sparse voice-auditory tags while preserving meaning.

## Training Data Format

Each record should follow this shape:

```json
{
  "id": "ex_0001",
  "mode": "NORMALIZE",
  "locale": "en-US",
  "style": "cinematic_audiobook",
  "constraints": {
    "no_ssml": true,
    "no_explanations": true,
    "max_tag_density": 0.35
  },
  "source_text": "Dr. Smith arrived at 14:30 with $1,234.56...",
  "target_text": "Doctor Smith arrived at two thirty PM with one thousand two hundred thirty-four dollars and fifty-six cents..."
}
```

## Prompting Template (for training samples)

System:

```text
You are a TTS narration text transformer for Eleven v3.
Rules: no SSML, no commentary, preserve meaning, output text only.
```

User:

```text
MODE={mode}
LOCALE={locale}
STYLE={style}
CONSTRAINTS={constraints}
TEXT:
{source_text}
```

Assistant target = `target_text` only.

## Suggested First-MVP Dataset Mix

- 80 `NORMALIZE`
- 60 `STRICT`
- 40 `EXPRESSIVE`
- 20 hard negatives (cases where model should avoid adding tags or changing meaning)

## Suggested Inference Defaults

- `temperature`: `0.15 - 0.25`
- `top_p`: `0.85 - 0.90`
- `repetition_penalty`: `1.05 - 1.12`

## Evaluate Outputs

Run:

```bash
python app/tools/llm_narration_transform/scripts/eval_narration_transform.py \
  --input app/tools/llm_narration_transform/data/starter_dataset_template.jsonl \
  --output report.json
```

If your JSONL contains a `prediction` field per row, the evaluator compares it against `target_text`.
