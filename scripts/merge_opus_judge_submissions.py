#!/usr/bin/env python3
"""Merge Opus Cursor-agent judge submissions into sweep summary JSON.

Usage:
  python scripts/merge_opus_judge_submissions.py
  python scripts/merge_opus_judge_submissions.py --summary-out runs/byzantine_claude-opus-4-20250514_sweep_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUITES = ("dev", "heldout", "final_dev", "break_dev", "ultra_hard", "unseen")
PROMPTS = ("v0", "v2")


def _slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def summarize_rules(results: list[dict]) -> dict:
    passed = [r for r in results if not r.get("rule_failures")]
    errors = [r for r in results if str(r.get("model_output", "")).startswith("ERROR:")]
    return {
        "n_scenarios": len(results),
        "rule_pass": len(passed),
        "rule_pass_rate": f"{len(passed)}/{len(results)}" if results else "0/0",
        "rule_fail_ids": [r["id"] for r in results if r.get("rule_failures")],
        "error_count": len(errors),
        "error_ids": [r["id"] for r in errors],
    }


def aggregate_judge(submission: dict, *, pass_thresholds: dict) -> dict:
    dims_order = submission.get("dimensions") or [
        "melodic_equivalence",
        "mode_fidelity",
        "notation_convention",
        "meaning_preservation",
    ]
    rows = submission.get("results") or []
    dim_sums = {d: 0.0 for d in dims_order}
    scenario_results = []
    for row in rows:
        scores = {s["dimension"]: int(s["score"]) for s in row.get("scores", [])}
        mean = sum(scores.get(d, 0) for d in dims_order) / len(dims_order) if dims_order else 0
        rule_ok = not row.get("rule_failures")
        strict_ok = rule_ok and all(
            scores.get(dim, 0) >= threshold for dim, threshold in pass_thresholds.items()
        )
        for d in dims_order:
            dim_sums[d] += scores.get(d, 0)
        scenario_results.append(
            {
                "id": row["id"],
                "strict_pass": strict_ok,
                "mean": mean,
                "scores": scores,
            }
        )
    n = len(rows) or 1
    passed = sum(1 for r in scenario_results if r["strict_pass"])
    return {
        "overall_mean": sum(r["mean"] for r in scenario_results) / n if rows else 0,
        "dimensions": {d: dim_sums[d] / n for d in dims_order},
        "strict_pass": passed,
        "strict_pass_rate": f"{passed}/{len(rows)}" if rows else "0/0",
        "strict_fail_ids": [r["id"] for r in scenario_results if not r["strict_pass"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-4-20250514")
    parser.add_argument("--submission-dir", default=str(ROOT / "runs" / "opus_judge_submissions"))
    parser.add_argument(
        "--summary-out",
        default="",
        help="Output summary path (default: runs/byzantine_{model}_sweep_summary.json)",
    )
    parser.add_argument(
        "--preserve-generations",
        action="store_true",
        help="Merge judge into existing summary if present",
    )
    args = parser.parse_args()

    from eval_harness.config import load_goal

    goal = load_goal(ROOT / "goals/byzantine_transcription.yaml")
    sub_dir = Path(args.submission_dir)
    summary_path = Path(args.summary_out) if args.summary_out else (
        ROOT / "runs" / f"byzantine_{_slug(args.model)}_sweep_summary.json"
    )

    if args.preserve_generations and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "model": args.model,
            "provider": "anthropic",
            "prompts": list(PROMPTS),
            "suites": {},
            "total_generations": 0,
        }

    summary["judge_model"] = "claude-opus-4-20250514"
    summary["judge_backend"] = "anthropic"
    summary["judge_method"] = "cursor_agent"
    summary["config"] = "config/byzantine.yaml"

    for prompt_key in PROMPTS:
        for suite_key in SUITES:
            batch_id = f"{prompt_key}_{suite_key}"
            sub_path = sub_dir / f"{batch_id}.json"
            if not sub_path.exists():
                raise SystemExit(f"Missing judge submission: {sub_path}")

            submission = json.loads(sub_path.read_text(encoding="utf-8"))
            outputs_path = ROOT / "runs" / f"byzantine_{_slug(args.model)}_{prompt_key}_{suite_key}_outputs.json"
            results = json.loads(outputs_path.read_text(encoding="utf-8"))

            entry = summary.get("suites", {}).get(prompt_key, {}).get(suite_key, {})
            entry.update({
                "scenarios_file": f"scenarios/byzantine_transcription_{suite_key}.yaml",
                "output_file": str(outputs_path.relative_to(ROOT)),
                "prompt_file": f"prompts/byzantine_transcription_{prompt_key}.txt",
                **summarize_rules(results),
            })
            entry["judge"] = aggregate_judge(
                submission,
                pass_thresholds=goal.pass_thresholds,
            )
            summary.setdefault("suites", {}).setdefault(prompt_key, {})[suite_key] = entry
            print(
                f"{batch_id}: strict {entry['judge']['strict_pass_rate']} "
                f"overall {entry['judge']['overall_mean']:.2f}"
            )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
