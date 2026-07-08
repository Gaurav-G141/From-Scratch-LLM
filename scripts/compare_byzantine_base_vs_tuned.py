#!/usr/bin/env python3
"""Compare base Qwen3 vs smoke-tuned adapter on Byzantine held-out scenarios.

Writes a JSON summary for Day 2 base-vs-tuned table.

Usage:
  python scripts/compare_byzantine_base_vs_tuned.py \\
    --adapter-path models/byzantine_sft_smoke \\
    --out runs/byzantine_smoke_base_vs_tuned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _scores_dict(goal, round_scores) -> dict:
    dims = {dim: round_scores.mean_for(dim) for dim in goal.dimensions}
    passed = 0
    if goal.pass_thresholds:
        for result in round_scores.scenario_results:
            ok = all(
                result.score_for(dim) >= threshold
                for dim, threshold in goal.pass_thresholds.items()
            ) and result.rule_check.passed
            if ok:
                passed += 1
    return {
        "overall_mean": round_scores.overall_mean(),
        "dimensions": dims,
        "strict_pass_rate": f"{passed}/{len(round_scores.scenario_results)}",
        "n_scenarios": len(round_scores.scenario_results),
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default=str(ROOT / "goals/byzantine_transcription.yaml"))
    parser.add_argument("--config", default=str(ROOT / "config/byzantine_eval.yaml"))
    parser.add_argument("--model", default="", help="Override config local_model (e.g. Qwen/Qwen3-1.7B)")
    parser.add_argument("--prompt-file", default=str(ROOT / "prompts/byzantine_transcription_v0.txt"))
    parser.add_argument("--split", choices=["dev", "heldout"], default="heldout")
    parser.add_argument("--adapter-path", default=str(ROOT / "models/byzantine_sft_smoke"))
    parser.add_argument("--out", default=str(ROOT / "runs/byzantine_smoke_base_vs_tuned.json"))
    args = parser.parse_args()

    from eval_harness.cli import _goal_paths, _run_local_eval
    from eval_harness.config import load_config, load_goal
    from eval_harness.scenarios.loader import load_scenarios

    goal_path = Path(args.goal).resolve()
    goal = load_goal(goal_path)
    config = load_config(args.config)
    if args.model:
        config.local_model = args.model
    system_prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    dev_path, heldout_path = _goal_paths(goal_path, goal)
    scenarios = load_scenarios(heldout_path if args.split == "heldout" else dev_path)

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise SystemExit(f"Adapter not found: {adapter_path}. Run train_byzantine_sft.py first.")

    print("Evaluating base model...", file=sys.stderr)
    base_scores = _run_local_eval(
        goal=goal,
        config=config,
        scenarios=scenarios,
        system_prompt=system_prompt,
        adapter_path=None,
        label="base",
    )

    print("Evaluating tuned model...", file=sys.stderr)
    tuned_scores = _run_local_eval(
        goal=goal,
        config=config,
        scenarios=scenarios,
        system_prompt=system_prompt,
        adapter_path=str(adapter_path),
        label="tuned",
    )

    summary = {
        "goal": goal.name,
        "split": args.split,
        "model": config.local_model,
        "adapter_path": str(adapter_path),
        "prompt_file": args.prompt_file,
        "base": _scores_dict(goal, base_scores),
        "tuned": _scores_dict(goal, tuned_scores),
        "delta_overall": tuned_scores.overall_mean() - base_scores.overall_mean(),
        "delta_dimensions": {
            dim: tuned_scores.mean_for(dim) - base_scores.mean_for(dim)
            for dim in goal.dimensions
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote → {out_path}")


if __name__ == "__main__":
    main()
