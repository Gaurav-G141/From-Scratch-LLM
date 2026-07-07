#!/usr/bin/env python3
"""Export latex transcription scenarios as JSONL for batch testing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GOAL = ROOT / "goals" / "latex_transcription.yaml"
PROMPT = ROOT / "prompts" / "latex_transcription_v0.txt"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "prompts" / "latex_transcription_batch.jsonl"
    goal = yaml.safe_load(GOAL.read_text())
    system = PROMPT.read_text().strip()

    rows = []
    for split in ("dev", "heldout"):
        rel = goal["dev_scenarios_path" if split == "dev" else "heldout_scenarios_path"]
        path = (GOAL.parent / rel).resolve()
        scenarios = yaml.safe_load(path.read_text())
        for item in scenarios:
            rows.append(
                {
                    "id": item["id"],
                    "split": split,
                    "system": system,
                    "user": item["input"].strip(),
                    "grading_notes": item.get("context", ""),
                    "tags": item.get("tags", []),
                }
            )

    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts to {out}")


if __name__ == "__main__":
    main()
