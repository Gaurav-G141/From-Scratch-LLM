from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval_harness.config import load_config, load_goal
from eval_harness.types import BehaviorGoal, HarnessConfig, RunReport


def _runs_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "runs"


def make_run_dir(goal_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _runs_dir() / f"{goal_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "rounds").mkdir()
    return run_dir


def write_report(report: RunReport, run_dir: Path) -> None:
    snapshot = {
        "goal": {
            "name": report.goal.name,
            "description": report.goal.description,
            "dimensions": report.goal.dimensions,
            "dev_scenarios_path": report.goal.dev_scenarios_path,
            "heldout_scenarios_path": report.goal.heldout_scenarios_path,
        },
        "config": report.config.__dict__,
    }
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)

    with open(run_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    with open(run_dir / "best_prompt.txt", "w", encoding="utf-8") as f:
        f.write(report.best_prompt)

    if report.heldout:
        heldout_payload = {
            "overall_mean": report.heldout.overall_mean(),
            "dimension_means": {
                dim: report.heldout.mean_for(dim) for dim in report.goal.dimensions
            },
            "scenario_results": [
                {
                    "scenario_id": sr.scenario_id,
                    "user_input": sr.user_input,
                    "model_output": sr.model_output,
                    "mean_score": sr.mean_score,
                    "groups_correct": sr.groups_correct,
                    "puzzle_solved": sr.puzzle_solved,
                    "rule_check": sr.rule_check.__dict__,
                }
                for sr in report.heldout.scenario_results
            ],
        }
        with open(run_dir / "heldout_eval.json", "w", encoding="utf-8") as f:
            json.dump(heldout_payload, f, indent=2)

    for rnd in report.rounds:
        prefix = f"{rnd.round_index:02d}"
        with open(run_dir / "rounds" / f"{prefix}_prompt.txt", "w", encoding="utf-8") as f:
            f.write(rnd.system_prompt)
        round_json = {
            "round_index": rnd.round_index,
            "overall_mean": rnd.overall_mean(),
            "dimension_means": {
                dim: rnd.mean_for(dim) for dim in report.goal.dimensions
            },
            "edit_rationale": rnd.edit_rationale,
            "scenario_results": [
                {
                    "scenario_id": sr.scenario_id,
                    "mean_score": sr.mean_score,
                    "model_output": sr.model_output,
                    "rule_failures": sr.rule_check.failures,
                }
                for sr in rnd.scenario_results
            ],
        }
        with open(run_dir / "rounds" / f"{prefix}_scores.json", "w", encoding="utf-8") as f:
            json.dump(round_json, f, indent=2)

    summary = _render_summary(report)
    with open(run_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary)


def _render_summary(report: RunReport) -> str:
    primary = report.goal.primary_dimension
    secondary = (
        report.goal.dimensions[1]
        if len(report.goal.dimensions) > 1
        else primary
    )
    primary_label = primary.replace("_", " ").title()
    secondary_label = secondary.replace("_", " ").title()

    lines = [
        f"# Litmus Report: {report.goal.name}",
        "",
        "## Behavior spec",
        report.goal.description,
        "",
        f"Primary litmus dimension: `{primary}`",
        "",
        "## Round scores (dev set)",
        "",
        f"| Round | {primary_label} | {secondary_label} | Overall |",
        "|-------|----------|------------|---------|",
    ]
    for rnd in report.rounds:
        lines.append(
            f"| {rnd.round_index} | {rnd.mean_for(primary):.2f} | "
            f"{rnd.mean_for(secondary):.2f} | {rnd.overall_mean():.2f} |"
        )

    if report.heldout:
        lines.extend(
            [
                "",
                "## Held-out evaluation (best prompt)",
                "",
                f"- {primary_label}: {report.heldout.mean_for(primary):.2f}",
                f"- {secondary_label}: {report.heldout.mean_for(secondary):.2f}",
                f"- Overall: {report.heldout.overall_mean():.2f}",
            ]
        )
        scored = [r for r in report.heldout.scenario_results if r.groups_correct is not None]
        if scored:
            avg_groups = sum(r.groups_correct or 0 for r in scored) / len(scored)
            solved = sum(1 for r in scored if r.puzzle_solved)
            lines.extend(
                [
                    f"- Groups correct: {avg_groups:.2f}/4.0 avg",
                    f"- Puzzles solved: {solved}/{len(scored)}",
                ]
            )

    lines.extend(
        [
            "",
            f"## Verdict: **{report.verdict.outcome}**",
            "",
            report.verdict.reason,
            "",
            f"Best prompt saved from round {report.best_round_index}.",
        ]
    )
    return "\n".join(lines) + "\n"
