#!/usr/bin/env python3
"""Run GPT-4o (or other) translator across all Byzantine scenario banks.

Usage:
  python scripts/run_byzantine_full_sweep.py --model gpt-4o
  python scripts/run_byzantine_full_sweep.py --model gpt-4o --judge
  python scripts/run_byzantine_full_sweep.py --model gpt-4o --suite heldout --prompt v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

SUITES: dict[str, str] = {
    "dev": "scenarios/byzantine_transcription_dev.yaml",
    "heldout": "scenarios/byzantine_transcription_heldout.yaml",
    "final_dev": "scenarios/byzantine_transcription_final_dev.yaml",
    "break_dev": "scenarios/byzantine_transcription_break_dev.yaml",
    "ultra_hard": "scenarios/byzantine_transcription_ultra_hard.yaml",
    "unseen": "scenarios/byzantine_transcription_unseen.yaml",
}

PROMPTS: dict[str, str] = {
    "v0": "prompts/byzantine_transcription_v0.txt",
    "v2": "prompts/byzantine_transcription_v2.txt",
}


def _model_slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def _output_path(*, model: str, prompt_key: str, suite_key: str) -> Path:
    return ROOT / "runs" / f"byzantine_{_model_slug(model)}_{prompt_key}_{suite_key}_outputs.json"


def run_suite(
    *,
    backend,
    goal,
    prompt_text: str,
    scenarios_path: Path,
    temperature: float,
    max_tokens: int,
) -> list[dict]:
    from eval_harness.judge.byzantine_checks import check_byzantine_transcription
    from eval_harness.scenarios.loader import load_scenarios

    scenarios = load_scenarios(scenarios_path)
    results: list[dict] = []
    for i, s in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {s.id}", flush=True)
        try:
            out = backend.generate(
                prompt_text,
                s.input,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            out = f"ERROR: {exc}"
        rule_failures = check_byzantine_transcription(
            model_output=out,
            direction=s.direction,
            forbidden_extra=goal.forbidden_patterns,
        )
        results.append(
            {
                "id": s.id,
                "direction": s.direction,
                "echos": s.echos,
                "tags": s.tags,
                "input": s.input,
                "reference_output": s.reference_output,
                "context": s.context,
                "model_output": out,
                "rule_failures": rule_failures,
            }
        )
    return results


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


def judge_results(
    *,
    goal,
    judge,
    prompt_text: str,
    scenarios_path: Path,
    results: list[dict],
) -> dict:
    from eval_harness.judge.rule_checks import run_rule_checks
    from eval_harness.scenarios.loader import load_scenarios
    from eval_harness.types import DimensionScore, RoundScores, RuleCheckResult, ScenarioResult

    by_id = {s.id: s for s in load_scenarios(scenarios_path)}
    scenario_results: list[ScenarioResult] = []
    for row in results:
        scenario = by_id.get(row["id"])
        if not scenario:
            continue
        model_output = str(row.get("model_output") or "")
        rule_check = run_rule_checks(goal, model_output)
        from eval_harness.judge.byzantine_checks import check_byzantine_transcription

        byz_failures = check_byzantine_transcription(
            model_output=model_output,
            direction=scenario.direction,
            forbidden_extra=goal.forbidden_patterns,
        )
        if byz_failures:
            rule_check = RuleCheckResult(
                passed=False,
                failures=rule_check.failures + byz_failures,
            )
        raw_scores = judge.score(
            behavior_spec=goal.description,
            dimensions=goal.dimensions,
            user_input=scenario.input,
            model_output=model_output,
            rule_failures=rule_check.failures,
            dimension_guidance=goal.dimension_guidance,
            context=scenario.context,
            reference_output=scenario.reference_output,
        )
        dimension_scores = [
            DimensionScore(
                dimension=str(item["dimension"]),
                score=int(item["score"]),
                rationale=str(item["rationale"]),
            )
            for item in raw_scores
        ]
        scenario_results.append(
            ScenarioResult(
                scenario_id=scenario.id,
                user_input=scenario.input,
                model_output=model_output,
                rule_check=rule_check,
                dimension_scores=dimension_scores,
                tags=scenario.tags,
            )
        )

    round_scores = RoundScores(
        round_index=0,
        system_prompt=prompt_text,
        scenario_results=scenario_results,
    )
    passed = 0
    if goal.pass_thresholds:
        for result in scenario_results:
            ok = all(
                result.score_for(dim) >= threshold
                for dim, threshold in goal.pass_thresholds.items()
            ) and result.rule_check.passed
            if ok:
                passed += 1
    return {
        "overall_mean": round_scores.overall_mean(),
        "dimensions": {dim: round_scores.mean_for(dim) for dim in goal.dimensions},
        "strict_pass": passed,
        "strict_pass_rate": f"{passed}/{len(scenario_results)}" if scenario_results else "0/0",
        "strict_fail_ids": [
            r.scenario_id
            for r in scenario_results
            if not (
                all(
                    r.score_for(dim) >= threshold
                    for dim, threshold in (goal.pass_thresholds or {}).items()
                )
                and r.rule_check.passed
            )
        ],
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Full Byzantine translator sweep")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--prompt", choices=["v0", "v2", "both"], default="both")
    parser.add_argument(
        "--suite",
        choices=[*SUITES.keys(), "all"],
        default="all",
        help="Run one suite or all six banks",
    )
    parser.add_argument("--judge", action="store_true", help="Judge-rescore outputs (no re-generation)")
    parser.add_argument("--judge-only", action="store_true", help="Skip generation; judge existing outputs")
    parser.add_argument("--config", default=None, help="Harness config YAML (default: byzantine.yaml for anthropic, byzantine_eval.yaml for openai)")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--summary-out",
        default="",
        help="Summary JSON path (default: runs/byzantine_{model}_sweep_summary.json)",
    )
    args = parser.parse_args()

    from eval_harness.backends.anthropic_api import AnthropicBackend, AnthropicJudge
    from eval_harness.backends.openai_api import OpenAIBackend, OpenAIJudge
    from eval_harness.config import load_config, load_goal

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
    elif args.provider == "anthropic":
        config_path = ROOT / "config/byzantine.yaml"
    else:
        config_path = ROOT / "config/byzantine_eval.yaml"

    goal = load_goal(ROOT / "goals/byzantine_transcription.yaml")
    config = load_config(config_path)

    if args.provider == "anthropic":
        model = args.model or "claude-opus-4-20250514"
        if not args.judge_only and not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY not set. Add it to .env, or use the Cursor Opus agent workflow:\n"
                "  python scripts/export_opus_sweep_inputs.py\n"
                "  (Opus agents translate → runs/opus_sweep_submissions/*.json)\n"
                "  python scripts/merge_opus_sweep_submissions.py\n"
                "  python scripts/export_opus_judge_batches.py\n"
                "  # Opus agents → runs/opus_judge_submissions/\n"
                "  python scripts/merge_opus_judge_submissions.py"
            )
        backend = AnthropicBackend(model=model)
    else:
        model = args.model or "gpt-4o"
        backend = OpenAIBackend(model=model)

    if args.judge or args.judge_only:
        if config.judge_backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "ANTHROPIC_API_KEY missing; using GPT-4o judge from byzantine_eval.yaml",
                file=sys.stderr,
            )
            config = load_config(ROOT / "config/byzantine_eval.yaml")
        if config.judge_backend == "anthropic":
            judge = AnthropicJudge(model=config.anthropic_judge_model)
            judge_model_name = config.anthropic_judge_model
        else:
            judge = OpenAIJudge(model=config.openai_judge_model)
            judge_model_name = config.openai_judge_model
    else:
        judge = None
        judge_model_name = None

    suite_keys = list(SUITES.keys()) if args.suite == "all" else [args.suite]
    prompt_keys = list(PROMPTS.keys()) if args.prompt == "both" else [args.prompt]

    summary: dict = {
        "model": model,
        "provider": args.provider,
        "prompts": prompt_keys,
        "suites": {},
        "total_generations": 0,
        "judge_model": judge_model_name,
        "judge_backend": config.judge_backend if judge else None,
        "config": str(config_path.relative_to(ROOT)),
    }

    for prompt_key in prompt_keys:
        prompt_path = ROOT / PROMPTS[prompt_key]
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        for suite_key in suite_keys:
            scenarios_rel = SUITES[suite_key]
            scenarios_path = ROOT / scenarios_rel
            out_path = _output_path(model=model, prompt_key=prompt_key, suite_key=suite_key)
            label = f"{model} | {prompt_key} | {suite_key}"
            print(f"\n=== {label} ===", flush=True)

            if args.judge_only:
                if not out_path.exists():
                    print(f"  SKIP missing {out_path}", file=sys.stderr)
                    continue
                results = json.loads(out_path.read_text(encoding="utf-8"))
            else:
                print(
                    f"Model: {backend.name} | Prompt: {PROMPTS[prompt_key]} | N=?",
                    flush=True,
                )
                results = run_suite(
                    backend=backend,
                    goal=goal,
                    prompt_text=prompt_text,
                    scenarios_path=scenarios_path,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"Wrote {out_path}", flush=True)
                summary["total_generations"] += len(results)

            entry = {
                "scenarios_file": scenarios_rel,
                "output_file": str(out_path.relative_to(ROOT)),
                "prompt_file": PROMPTS[prompt_key],
                **summarize_rules(results),
            }
            if judge:
                print(f"  Judging {len(results)} outputs...", flush=True)
                entry["judge"] = judge_results(
                    goal=goal,
                    judge=judge,
                    prompt_text=prompt_text,
                    scenarios_path=scenarios_path,
                    results=results,
                )

            summary["suites"].setdefault(prompt_key, {})[suite_key] = entry

    summary_path = Path(args.summary_out) if args.summary_out else (
        ROOT / "runs" / f"byzantine_{_model_slug(model)}_sweep_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
