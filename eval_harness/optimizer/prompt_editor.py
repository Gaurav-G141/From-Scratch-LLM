from __future__ import annotations

from eval_harness.backends.openai_api import OpenAIPromptEditor
from eval_harness.types import RoundScores, ScenarioResult


def _judge_notes(result: ScenarioResult) -> str:
    parts = [f"{d.dimension}={d.score}: {d.rationale}" for d in result.dimension_scores]
    return "; ".join(parts)


def select_editor_examples(
    round_scores: RoundScores,
    *,
    max_failures: int,
    max_passes: int,
) -> tuple[list[dict], list[dict]]:
    ranked = sorted(round_scores.scenario_results, key=lambda r: r.mean_score)
    failures = []
    for result in ranked:
        if result.mean_score >= 1.8 and result.rule_check.passed:
            continue
        failures.append(
            {
                "scenario_id": result.scenario_id,
                "user_input": result.user_input,
                "model_output": result.model_output,
                "rule_failures": result.rule_check.failures,
                "judge_notes": _judge_notes(result),
            }
        )
        if len(failures) >= max_failures:
            break

    successes = []
    for result in reversed(ranked):
        if result.mean_score < 1.5:
            continue
        successes.append(
            {
                "scenario_id": result.scenario_id,
                "user_input": result.user_input,
                "model_output": result.model_output,
            }
        )
        if len(successes) >= max_passes:
            break

    return failures, successes


def revise_prompt(
    *,
    editor: OpenAIPromptEditor,
    behavior_spec: str,
    current_prompt: str,
    round_scores: RoundScores,
    max_failures: int,
    max_passes: int,
) -> tuple[str, str]:
    failures, successes = select_editor_examples(
        round_scores,
        max_failures=max_failures,
        max_passes=max_passes,
    )
    return editor.revise(
        behavior_spec=behavior_spec,
        current_prompt=current_prompt,
        failures=failures,
        successes=successes,
    )
