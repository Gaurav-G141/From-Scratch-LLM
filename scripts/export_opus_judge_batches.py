#!/usr/bin/env python3
"""Export judge batches for Opus Cursor-agent grading of sweep outputs.

Usage:
  python scripts/export_opus_judge_batches.py
  python scripts/export_opus_judge_batches.py --model claude-opus-4-20250514 --prompt v2 --suite final_dev
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUITES: dict[str, str] = {
    "dev": "scenarios/byzantine_transcription_dev.yaml",
    "heldout": "scenarios/byzantine_transcription_heldout.yaml",
    "final_dev": "scenarios/byzantine_transcription_final_dev.yaml",
    "break_dev": "scenarios/byzantine_transcription_break_dev.yaml",
    "ultra_hard": "scenarios/byzantine_transcription_ultra_hard.yaml",
    "unseen": "scenarios/byzantine_transcription_unseen.yaml",
}

PROMPTS = ("v0", "v2")


def _slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-4-20250514")
    parser.add_argument("--prompt", choices=["v0", "v2", "both"], default="both")
    parser.add_argument("--suite", choices=[*SUITES.keys(), "all"], default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "opus_judge_inputs"))
    args = parser.parse_args()

    from eval_harness.config import load_goal

    goal = load_goal(ROOT / "goals/byzantine_transcription.yaml")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_keys = list(PROMPTS) if args.prompt == "both" else [args.prompt]
    suite_keys = list(SUITES.keys()) if args.suite == "all" else [args.suite]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "translator_model": args.model,
        "judge": "claude-opus-4 (Cursor agent)",
        "goal_file": "goals/byzantine_transcription.yaml",
        "pass_thresholds": goal.pass_thresholds,
        "dimensions": goal.dimensions,
        "dimension_guidance": goal.dimension_guidance,
        "batches": {},
    }

    for prompt_key in prompt_keys:
        for suite_key in suite_keys:
            outputs_path = ROOT / "runs" / f"byzantine_{_slug(args.model)}_{prompt_key}_{suite_key}_outputs.json"
            if not outputs_path.exists():
                raise SystemExit(f"Missing outputs: {outputs_path}")

            rows = json.loads(outputs_path.read_text(encoding="utf-8"))
            batch_id = f"{prompt_key}_{suite_key}"
            payload = {
                "batch_id": batch_id,
                "translator_model": args.model,
                "prompt_file": f"prompts/byzantine_transcription_{prompt_key}.txt",
                "scenarios_file": SUITES[suite_key],
                "outputs_file": str(outputs_path.relative_to(ROOT)),
                "behavior_spec": goal.description,
                "dimensions": goal.dimensions,
                "dimension_guidance": goal.dimension_guidance,
                "pass_thresholds": goal.pass_thresholds,
                "scenarios": [
                    {
                        "id": r["id"],
                        "direction": r.get("direction"),
                        "input": r.get("input"),
                        "reference_output": r.get("reference_output"),
                        "context": r.get("context", ""),
                        "model_output": r.get("model_output"),
                        "rule_failures": r.get("rule_failures") or [],
                    }
                    for r in rows
                ],
            }
            out_path = out_dir / f"{batch_id}.json"
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest["batches"][batch_id] = {
                "inputs": str(out_path.relative_to(ROOT)),
                "submission": f"runs/opus_judge_submissions/{batch_id}.json",
                "n_scenarios": len(rows),
            }
            print(f"Wrote {len(rows)} judge tasks → {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    main()
