import argparse
import json
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Dict, List


FORBIDDEN_TAGS = {
    "standing",
    "grinning",
    "pacing",
    "music",
}


@dataclass
class RowResult:
    id: str
    mode: str
    has_prediction: bool
    no_ssml: bool
    no_commentary: bool
    forbidden_tags_absent: bool
    tag_density_ok: bool
    similarity_ratio: float
    violations: List[str]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def has_ssml(text: str) -> bool:
    return bool(re.search(r"<\s*(speak|break|phoneme|prosody|lexicon|voice)\b", text, flags=re.IGNORECASE))


def has_commentary(text: str) -> bool:
    patterns = [
        r"^here('?s| is) (the|your)",
        r"^output:",
        r"^explanation:",
        r"^note:",
        r"```",
    ]
    joined = text.strip().lower()
    return any(re.search(p, joined) for p in patterns)


def extract_tags(text: str) -> List[str]:
    tags = re.findall(r"\[([^\]]+)\]", text)
    return [t.strip().lower() for t in tags if t.strip()]


def tag_density(text: str) -> float:
    tags = extract_tags(text)
    if not text.strip():
        return 0.0
    sentence_count = max(1, len(re.split(r"[.!?]+", text.strip())) - 1)
    return len(tags) / sentence_count


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def evaluate_row(row: Dict) -> RowResult:
    rid = row.get("id", "")
    mode = row.get("mode", "UNKNOWN")
    target = row.get("target_text", "")
    prediction = row.get("prediction")
    text = prediction if prediction is not None else target

    violations: List[str] = []

    no_ssml = not has_ssml(text)
    if not no_ssml:
        violations.append("contains_ssml")

    no_commentary = not has_commentary(text)
    if not no_commentary:
        violations.append("contains_commentary")

    tags = extract_tags(text)
    forbidden_tags_absent = all(tag not in FORBIDDEN_TAGS for tag in tags)
    if not forbidden_tags_absent:
        violations.append("contains_forbidden_tags")

    max_density = row.get("constraints", {}).get("max_tag_density", 0.35)
    density = tag_density(text)
    tag_density_ok = density <= max_density + 1e-9
    if not tag_density_ok:
        violations.append(f"tag_density_exceeded:{density:.3f}>{max_density:.3f}")

    sim = similarity(row.get("source_text", ""), text)

    return RowResult(
        id=rid,
        mode=mode,
        has_prediction=prediction is not None,
        no_ssml=no_ssml,
        no_commentary=no_commentary,
        forbidden_tags_absent=forbidden_tags_absent,
        tag_density_ok=tag_density_ok,
        similarity_ratio=sim,
        violations=violations,
    )


def summarize(results: List[RowResult]) -> Dict:
    n = len(results)
    if n == 0:
        return {"count": 0}

    def rate(attr: str) -> float:
        return sum(1 for r in results if getattr(r, attr)) / n

    summary = {
        "count": n,
        "has_prediction_count": sum(1 for r in results if r.has_prediction),
        "no_ssml_rate": rate("no_ssml"),
        "no_commentary_rate": rate("no_commentary"),
        "forbidden_tags_absent_rate": rate("forbidden_tags_absent"),
        "tag_density_ok_rate": rate("tag_density_ok"),
        "avg_similarity_to_source": mean(r.similarity_ratio for r in results),
        "total_violations": sum(len(r.violations) for r in results),
    }

    by_mode: Dict[str, Dict[str, float]] = {}
    modes = sorted(set(r.mode for r in results))
    for mode in modes:
        items = [r for r in results if r.mode == mode]
        m = len(items)
        by_mode[mode] = {
            "count": m,
            "no_ssml_rate": sum(1 for r in items if r.no_ssml) / m,
            "no_commentary_rate": sum(1 for r in items if r.no_commentary) / m,
            "tag_density_ok_rate": sum(1 for r in items if r.tag_density_ok) / m,
            "avg_similarity_to_source": mean(r.similarity_ratio for r in items),
        }

    summary["by_mode"] = by_mode
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate narration-transform outputs against Eleven-v3-oriented rules.")
    parser.add_argument("--input", required=True, help="Path to JSONL input with source_text, target_text, and optional prediction.")
    parser.add_argument("--output", default="", help="Optional path to write full JSON report.")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = load_jsonl(input_path)
    results = [evaluate_row(row) for row in rows]
    summary = summarize(results)

    print("=== Narration Transform Evaluation ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
        report = {
            "summary": summary,
            "rows": [asdict(r) for r in results],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
