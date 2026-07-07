#!/usr/bin/env python3
"""Export blind scenario inputs for Opus translator agent (no reference outputs).

Usage:
  python scripts/export_blind_eval_inputs.py --bank ultra_hard
  python scripts/export_blind_eval_inputs.py --bank unseen
  python scripts/export_blind_eval_inputs.py --bank both
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

BANKS = {
    "ultra_hard": ROOT / "scenarios" / "byzantine_transcription_ultra_hard.yaml",
    "unseen": ROOT / "scenarios" / "byzantine_transcription_unseen.yaml",
}


def export_bank(path: Path) -> list[dict]:
    scenarios = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        {
            "id": s["id"],
            "direction": s.get("direction", "byz_to_west"),
            "echos": s.get("echos", ""),
            "tags": s.get("tags", []),
            "input": s["input"].strip(),
        }
        for s in scenarios
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank",
        choices=["ultra_hard", "unseen", "both"],
        default="both",
        help="Which scenario bank to export (default: both = 33 cases)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: runs/blind_eval_inputs_<bank>.json)",
    )
    args = parser.parse_args()

    if args.bank == "both":
        items = export_bank(BANKS["ultra_hard"]) + export_bank(BANKS["unseen"])
        bank_label = "ultra_hard+unseen"
    else:
        items = export_bank(BANKS[args.bank])
        bank_label = args.bank

    payload = {
        "task": "byzantine_transcription_blind",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bank": bank_label,
        "n_scenarios": len(items),
        "prompt_file": "prompts/byzantine_transcription_v2.txt",
        "goal_file": "goals/byzantine_transcription.yaml",
        "instructions_file": "docs/byzantine_opus_blind_eval.md",
        "submission_template": "runs/byzantine_opus_blind_submission.template.json",
        "scenarios": items,
    }

    out = Path(args.out) if args.out else ROOT / "runs" / f"blind_eval_inputs_{bank_label.replace('+', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(items)} blind inputs → {out}")


if __name__ == "__main__":
    main()
