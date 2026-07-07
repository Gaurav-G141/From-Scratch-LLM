#!/usr/bin/env python3
"""Merge blind Opus submission into full harness output JSON for grading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

BANK_PATHS = {
    "ultra_hard": ROOT / "scenarios" / "byzantine_transcription_ultra_hard.yaml",
    "unseen": ROOT / "scenarios" / "byzantine_transcription_unseen.yaml",
}


def load_scenarios_by_id() -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for path in BANK_PATHS.values():
        for s in yaml.safe_load(path.read_text(encoding="utf-8")):
            by_id[s["id"]] = s
    return by_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="Filled blind submission JSON")
    parser.add_argument("--inputs", required=True, help="Blind inputs JSON used for the run")
    parser.add_argument("--out", default="runs/byzantine_opus_blind_outputs.json")
    args = parser.parse_args()

    from eval_harness.config import load_goal
    from eval_harness.judge.byzantine_checks import check_byzantine_transcription

    submission = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    by_id = load_scenarios_by_id()
    goal = load_goal(ROOT / "goals/byzantine_transcription.yaml")

    answers = {r["id"]: r["model_output"] for r in submission.get("results", [])}
    expected_ids = [s["id"] for s in inputs["scenarios"]]

    missing = [i for i in expected_ids if i not in answers]
    if missing:
        raise SystemExit(f"Submission missing {len(missing)} ids: {', '.join(missing[:5])}...")

    results = []
    for sid in expected_ids:
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
                    "prompt_file": submission.get("prompt_file"),
                    "submitted_at": submission.get("submitted_at"),
                    "honor_code": submission.get("honor_code"),
                },
            }
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(results)} results → {out_path}")


if __name__ == "__main__":
    main()
