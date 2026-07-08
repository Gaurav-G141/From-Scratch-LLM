from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from eval_harness.backends.anthropic_api import AnthropicJudge
from eval_harness.backends.openai_api import OpenAIBackend, OpenAIJudge, OpenAIPromptEditor
from eval_harness.config import load_config, load_goal
from eval_harness.judge.runner import evaluate_scenarios
from eval_harness.optimizer.loop import run_litmus_loop
from eval_harness.report.writer import make_run_dir, write_report
from eval_harness.scenarios.generator import generate_scenarios, write_scenarios
from eval_harness.scenarios.loader import load_scenarios, resolve_scenario_path


def _make_judge(config):
    if config.judge_backend == "anthropic":
        return AnthropicJudge(model=config.anthropic_judge_model)
    return OpenAIJudge(model=config.openai_judge_model)


def _goal_paths(goal_path: Path, goal) -> tuple[Path, Path]:
    dev = resolve_scenario_path(goal_path, goal.dev_scenarios_path)
    heldout = resolve_scenario_path(goal_path, goal.heldout_scenarios_path)
    return dev, heldout


def cmd_litmus(args: argparse.Namespace) -> None:
    load_dotenv()
    goal_path = Path(args.goal).resolve()
    goal = load_goal(goal_path)
    config = load_config(args.config)

    dev_path, heldout_path = _goal_paths(goal_path, goal)
    dev_scenarios = load_scenarios(dev_path)
    heldout_scenarios = load_scenarios(heldout_path)

    backend = OpenAIBackend(model=config.openai_model)
    judge = _make_judge(config)
    editor = OpenAIPromptEditor(model=config.openai_editor_model)

    report = run_litmus_loop(
        goal=goal,
        config=config,
        dev_scenarios=dev_scenarios,
        heldout_scenarios=heldout_scenarios,
        backend=backend,
        judge=judge,
        editor=editor,
    )

    run_dir = make_run_dir(goal.name)
    write_report(report, run_dir)
    print(f"Verdict: {report.verdict.outcome}")
    print(report.verdict.reason)
    print(f"Report written to {run_dir}")


def cmd_eval(args: argparse.Namespace) -> None:
    load_dotenv()
    goal_path = Path(args.goal).resolve()
    goal = load_goal(goal_path)
    config = load_config(args.config)

    system_prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    dev_path, _ = _goal_paths(goal_path, goal)
    scenarios = load_scenarios(dev_path)

    if args.backend == "local":
        from eval_harness.backends.local_hf import LocalHFBackend

        backend = LocalHFBackend(model_name=config.local_model)
    else:
        backend = OpenAIBackend(model=config.openai_model)

    judge = _make_judge(config)
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

    print(f"Overall mean: {round_scores.overall_mean():.2f}")
    for dim in goal.dimensions:
        print(f"  {dim}: {round_scores.mean_for(dim):.2f}")

    if len(goal.dimensions) == 2:
        d0, d1 = goal.dimensions
        print(f"Dual rubric avg: {d0}={round_scores.mean_for(d0):.2f}, {d1}={round_scores.mean_for(d1):.2f}")

    if goal.pass_thresholds:
        passed = 0
        for result in round_scores.scenario_results:
            ok = all(
                result.score_for(dim) >= threshold
                for dim, threshold in goal.pass_thresholds.items()
            ) and result.rule_check.passed
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            dims = ", ".join(
                f"{dim}={result.score_for(dim):.0f}"
                for dim in goal.pass_thresholds
            )
            line = f"  [{status}] {result.scenario_id}: {dims}"
            print(line)
            if args.verbose or not ok:
                print(f"      in:  {result.user_input[:100]}")
                print(f"      out: {result.model_output[:160]}")
        print(f"Strict pass rate: {passed}/{len(round_scores.scenario_results)} (requires {goal.pass_thresholds}, no rule failures)")

    scored = [r for r in round_scores.scenario_results if r.groups_correct is not None]
    if scored:
        avg_groups = sum(r.groups_correct or 0 for r in scored) / len(scored)
        solved = sum(1 for r in scored if r.puzzle_solved)
        print(f"Groups correct: {avg_groups:.2f}/4.0 avg")
        print(f"Puzzles solved: {solved}/{len(scored)}")


def _run_local_eval(
    *,
    goal,
    config,
    scenarios,
    system_prompt: str,
    adapter_path: str | None = None,
    label: str = "local",
):
    from eval_harness.backends.local_hf import LocalHFBackend

    backend = LocalHFBackend(
        model_name=config.local_model,
        adapter_path=adapter_path,
    )
    judge = _make_judge(config)
    return evaluate_scenarios(
        goal=goal,
        scenarios=scenarios,
        system_prompt=system_prompt,
        backend=backend,
        judge=judge,
        round_index=0,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )


def _print_round_scores(goal, round_scores, label: str, split: str) -> None:
    print(f"\n{label} on {split} set")
    print(f"Overall mean: {round_scores.overall_mean():.2f}")
    for dim in goal.dimensions:
        print(f"  {dim}: {round_scores.mean_for(dim):.2f}")


def cmd_compare(args: argparse.Namespace) -> None:
    load_dotenv()
    goal_path = Path(args.goal).resolve()
    goal = load_goal(goal_path)
    config = load_config(args.config)

    system_prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    dev_path, heldout_path = _goal_paths(goal_path, goal)
    scenarios = load_scenarios(heldout_path if args.split == "heldout" else dev_path)

    adapter_path = getattr(args, "adapter_path", None) or None
    label = config.local_model
    if adapter_path:
        label = f"{config.local_model} (adapter: {adapter_path})"

    round_scores = _run_local_eval(
        goal=goal,
        config=config,
        scenarios=scenarios,
        system_prompt=system_prompt,
        adapter_path=adapter_path,
        label=label,
    )
    _print_round_scores(goal, round_scores, label, args.split)

    if getattr(args, "adapter_path", None) and getattr(args, "compare_base", False):
        base_scores = _run_local_eval(
            goal=goal,
            config=config,
            scenarios=scenarios,
            system_prompt=system_prompt,
            adapter_path=None,
            label=config.local_model,
        )
        _print_round_scores(goal, base_scores, f"{config.local_model} (base)", args.split)
        print("\nDelta (tuned - base) overall:", f"{round_scores.overall_mean() - base_scores.overall_mean():+.2f}")


def cmd_generate_scenarios(args: argparse.Namespace) -> None:
    load_dotenv()
    goal_path = Path(args.goal).resolve()
    goal = load_goal(goal_path)
    config = load_config(args.config)

    scenarios = generate_scenarios(
        goal=goal,
        judge_model=config.openai_judge_model,
        count=args.count,
        split=args.split,
    )

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = goal_path.parent.parent / "scenarios" / f"{goal.name}_{args.split}.yaml"

    write_scenarios(out_path, scenarios)
    print(f"Wrote {len(scenarios)} scenarios to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt optimization and litmus test harness")
    parser.add_argument("--config", default=None, help="Path to harness config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    litmus = sub.add_parser("litmus", help="Optimize prompt on dev set and run litmus verdict")
    litmus.add_argument("--goal", required=True, help="Path to behavior goal YAML")
    litmus.set_defaults(func=cmd_litmus)

    eval_cmd = sub.add_parser("eval", help="Run a single eval round without prompt editing")
    eval_cmd.add_argument("--goal", required=True)
    eval_cmd.add_argument("--prompt-file", required=True)
    eval_cmd.add_argument("--backend", choices=["openai", "local"], default="openai")
    eval_cmd.add_argument("--verbose", action="store_true", help="Print all model outputs")
    eval_cmd.set_defaults(func=cmd_eval)

    compare = sub.add_parser("compare", help="Evaluate local base model with a prompt file")
    compare.add_argument("--goal", required=True)
    compare.add_argument("--prompt-file", required=True)
    compare.add_argument("--split", choices=["dev", "heldout"], default="heldout")
    compare.add_argument("--adapter-path", default=None, help="Path to PEFT/LoRA adapter for tuned model")
    compare.add_argument(
        "--compare-base",
        action="store_true",
        help="When --adapter-path is set, also eval base model and print delta",
    )
    compare.set_defaults(func=cmd_compare)

    gen = sub.add_parser("generate-scenarios", help="Generate scenarios from a behavior spec")
    gen.add_argument("--goal", required=True)
    gen.add_argument("--count", type=int, default=10)
    gen.add_argument("--split", choices=["dev", "heldout"], default="dev")
    gen.add_argument("--output", default=None)
    gen.set_defaults(func=cmd_generate_scenarios)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
