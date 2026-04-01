# Golden Multi-Speaker Scripts

Test fixtures for evaluating speaker attribution accuracy in the AI conversation formatter.

## Purpose

These scripts provide ground-truth annotations for:

- Speaker detection accuracy (precision/recall)
- Line type classification (dialogue/narration/stage_direction)
- Ambiguity detection (flagging uncertain attributions)
- Confidence calibration

## Schema

Each JSON file follows the NarrationScript model (v1.0):

- `source_text`: Raw input as a user would provide it
- `expected_output`: Annotated NarrationScript with speaker labels, line types, and confidence
  scores

## Usage

```python
import json
from pathlib import Path

golden_dir = Path("tests/golden_scripts")
for script_file in golden_dir.glob("*.json"):
    with open(script_file) as f:
        golden = json.load(f)
    # Feed golden["source_text"] to the conversation formatter
    # Compare output against golden["expected_output"]
```

## Difficulty Levels

- **easy**: Clear speaker markers, no ambiguity
- **medium**: Some narrator lines, stage directions, contextual attribution needed
- **hard**: Ambiguous lines, rapid speaker changes, missing attribution markers
