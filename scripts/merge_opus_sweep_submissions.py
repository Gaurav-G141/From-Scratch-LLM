#!/usr/bin/env python3
"""Merge Opus Cursor-agent sweep submissions into harness output JSON files.

Usage:
  python scripts/merge_opus_sweep_submissions.py
  python scripts/merge_opus_sweep_submissions.py --suite final_dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_SLUG = "claude-opus-4-20250514"
SUITES: dict[str, str] = {
    "dev": "scenarios/byzantine_transcription_dev.yaml",
    "heldout": "scenarios/byzantine_transcription_heldout.yaml",
    "final_dev": "scenarios/byzantine_transcription_final_dev.yaml",
    "break_dev": "scenarios/byzantine_transcription_break_dev.yaml",
    "ultra_hard": "scenarios/byzantine_transcription_ultra_hard.yaml",
    "unseen": "scenarios/byzantine_transcription_unseen.yaml",
}


def merge_one(*, suite_key: str, submission_path: Path) -> list[Path]:
    from eval_harness.config import load_goal
    from eval_harness.judge.byzantine_checks import check_byzantine_transcription

    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    scenarios = yaml.safe_load((ROOT / SUITES[suite_key]).read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in scenarios}
    goal = load_goal(ROOT / "goals/byzantine_transcription.yaml")

    written: list[Path] = []
    for prompt_key, rows in submission.get("results", {}).items():
        answers = {r["id"]: r["model_output"] for r in rows}
        expected = [s["id"] for s in scenarios]
        missing = [i for i in expected if i not in answers]
        if missing:
            raise SystemExit(
                f"{submission_path} prompt {prompt_key} missing ids: {', '.join(missing[:5])}"
            )

        results = []
        for sid in expected:
            s = by_id[sid]
            out = answers[sid]
            results.append(
                {
                    "id": sid,
                    "direction": s.get("direction"),
                    "echos": s.get("echos"),
                    "tags": s.get("tags"),
                    "input": s["input"].strip(),
                    "reference_output": s["reference_output"].strip(),
                    "context": s.get("context", "").strip(),
                    "model_output": out,
                    "rule_failures": check_byzantine_transcription(
                        model_output=out,
                        direction=s.get("direction", "byz_to_west"),
                        forbidden_extra=goal.forbidden_patterns,
                    ),
                    "translator_meta": {
                        "translator": submission.get("translator"),
                        "prompt_file": submission.get("prompt_files", {}).get(prompt_key),
                        "submitted_at": submission.get("submitted_at"),
                        "honor_code": submission.get("honor_code"),
                    },
                }
            )

        out_path = ROOT / "runs" / f"byzantine_{MODEL_SLUG}_{prompt_key}_{suite_key}_outputs.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(out_path)
        print(f"Wrote {len(results)} → {out_path.relative_to(ROOT)}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=[*SUITES.keys(), "all"], default="all")
    parser.add_argument("--submission-dir", default=str(ROOT / "runs" / "opus_sweep_submissions"))
    args = parser.parse_args()

    sub_dir = Path(args.submission_dir)
    suite_keys = list(SUITES.keys()) if args.suite == "all" else [args.suite]
    for key in suite_keys:
        path = sub_dir / f"{key}.json"
        if not path.exists():
            raise SystemExit(f"Missing submission: {path}")
        merge_one(suite_key=key, submission_path=path)


if __name__ == "__main__":
    main()
