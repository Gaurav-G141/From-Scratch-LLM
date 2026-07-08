#!/usr/bin/env python3
"""Compare base vs LoRA-tuned local model across multiple scenario banks.

Loads each model once and evaluates all suites (faster than per-suite reload).

Usage:
  python scripts/compare_local_sft_all_suites.py \\
    --model Qwen/Qwen3-1.7B \\
    --adapter-path models/byzantine_sft_v1_1.7b \\
    --prompt prompts/byzantine_transcription_v2.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

SUITES: dict[str, str] = {
    "dev": "scenarios/byzantine_transcription_dev.yaml",
    "heldout": "scenarios/byzantine_transcription_heldout.yaml",
    "unseen": "scenarios/byzantine_transcription_unseen.yaml",
    "ultra_hard": "scenarios/byzantine_transcription_ultra_hard.yaml",
}


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
        "strict_pass": passed,
        "strict_pass_rate": f"{passed}/{len(round_scores.scenario_results)}",
        "n_scenarios": len(round_scores.scenario_results),
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default=str(ROOT / "goals/byzantine_transcription.yaml"))
    parser.add_argument("--config", default=str(ROOT / "config/byzantine_eval.yaml"))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--adapter-path",
        default=str(ROOT / "models/byzantine_sft_v1_1.7b"),
    )
    parser.add_argument(
        "--prompt",
        default=str(ROOT / "prompts/byzantine_transcription_v2.txt"),
    )
    parser.add_argument(
        "--suites",
        default="dev,heldout,unseen,ultra_hard",
        help="Comma-separated suite keys",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "runs/byzantine_local_sft_all_suites.json"),
    )
    args = parser.parse_args()

    from eval_harness.backends.local_hf import LocalHFBackend
    from eval_harness.cli import _make_judge
    from eval_harness.config import load_config, load_goal
    from eval_harness.judge.runner import evaluate_scenarios
    from eval_harness.scenarios.loader import load_scenarios

    goal = load_goal(Path(args.goal))
    config = load_config(args.config)
    config.local_model = args.model
    system_prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise SystemExit(f"Adapter not found: {adapter_path}")

    suite_keys = [s.strip() for s in args.suites.split(",") if s.strip()]
    for key in suite_keys:
        if key not in SUITES:
            raise SystemExit(f"Unknown suite {key!r}; choose from {list(SUITES)}")

    suite_data = {
        key: load_scenarios(ROOT / SUITES[key]) for key in suite_keys
    }

    judge = _make_judge(config)
    summary: dict = {
        "model": args.model,
        "adapter_path": str(adapter_path),
        "prompt_file": args.prompt,
        "judge_model": config.openai_judge_model,
        "suites": {},
        "totals": {"base": {}, "tuned": {}, "delta_overall": 0.0},
    }

    for label, adapter in [("base", None), ("tuned", str(adapter_path))]:
        print(f"\n=== Loading {label} ({args.model}) ===", flush=True)
        backend = LocalHFBackend(model_name=args.model, adapter_path=adapter)
        arm: dict = {}
        total_n = 0
        total_strict = 0
        weighted_sum = 0.0
        for key in suite_keys:
            scenarios = suite_data[key]
            print(f"  {label} | {key} | n={len(scenarios)}", flush=True)
            round_scores = evaluate_scenarios(
                goal=goal,
                scenarios=scenarios,
                system_prompt=system_prompt,
                backend=backend,
                judge=judge,
                round_index=0,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            arm[key] = _scores_dict(goal, round_scores)
            n = arm[key]["n_scenarios"]
            total_n += n
            total_strict += arm[key]["strict_pass"]
            weighted_sum += arm[key]["overall_mean"] * n
            print(
                f"    overall={arm[key]['overall_mean']:.2f} "
                f"strict={arm[key]['strict_pass_rate']} "
                f"melodic={arm[key]['dimensions']['melodic_equivalence']:.2f}",
                flush=True,
            )
        summary["suites"][label] = arm
        summary["totals"][label] = {
            "n_scenarios": total_n,
            "strict_pass": total_strict,
            "strict_pass_rate": f"{total_strict}/{total_n}",
            "overall_mean": weighted_sum / total_n if total_n else 0.0,
        }

    base_mean = summary["totals"]["base"]["overall_mean"]
    tuned_mean = summary["totals"]["tuned"]["overall_mean"]
    summary["totals"]["delta_overall"] = tuned_mean - base_mean
    summary["totals"]["delta_strict"] = (
        summary["totals"]["tuned"]["strict_pass"] - summary["totals"]["base"]["strict_pass"]
    )

    # Per-suite deltas
    summary["deltas"] = {}
    for key in suite_keys:
        b = summary["suites"]["base"][key]
        t = summary["suites"]["tuned"][key]
        summary["deltas"][key] = {
            "overall_mean": t["overall_mean"] - b["overall_mean"],
            "strict_pass": t["strict_pass"] - b["strict_pass"],
            "dimensions": {
                dim: t["dimensions"][dim] - b["dimensions"][dim]
                for dim in goal.dimensions
            },
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote → {out_path}")
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
