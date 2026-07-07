from __future__ import annotations

from eval_harness.backends.openai_api import OpenAIJudge
from eval_harness.judge.byzantine_checks import check_byzantine_transcription
from eval_harness.judge.connections import grouping_accuracy_score, score_connections
from eval_harness.judge.rule_checks import run_rule_checks
from eval_harness.judge.sanitization_checks import check_passive_aggressive
from eval_harness.types import (
    BehaviorGoal,
    DimensionScore,
    RoundScores,
    RuleCheckResult,
    Scenario,
    ScenarioResult,
)
from eval_harness.backends.base import ModelBackend


def evaluate_scenario(
    *,
    goal: BehaviorGoal,
    scenario: Scenario,
    system_prompt: str,
    backend: ModelBackend,
    judge: OpenAIJudge,
    max_new_tokens: int,
    temperature: float,
) -> ScenarioResult:
    model_output = backend.generate(
        system_prompt,
        scenario.input,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    rule_check = run_rule_checks(goal, model_output)

    if goal.name == "sanitization":
        pa_failures = check_passive_aggressive(model_output)
        if pa_failures:
            rule_check = RuleCheckResult(
                passed=False,
                failures=rule_check.failures + pa_failures,
            )

    if goal.name == "byzantine_transcription":
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

    groups_correct: int | None = None
    puzzle_solved = False
    if scenario.expected_groups and scenario.words:
        groups_correct, puzzle_solved, connection_failures = score_connections(
            input_words=scenario.words,
            expected_groups=scenario.expected_groups,
            model_output=model_output,
        )
        if connection_failures:
            rule_check = RuleCheckResult(
                passed=False,
                failures=rule_check.failures + connection_failures,
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

    hidden_rudeness_note = ""
    if goal.name == "sanitization":
        detected, pa_rationale = judge.detect_hidden_rudeness(
            original=scenario.input,
            sanitized=model_output,
            context=scenario.context,
        )
        if detected:
            hidden_rudeness_note = pa_rationale
            dimension_scores = [
                DimensionScore(
                    dimension=d.dimension,
                    score=0 if d.dimension == "cleanliness" else d.score,
                    rationale=(
                        f"Failed harsh gate: hidden rudeness — {pa_rationale}"
                        if d.dimension == "cleanliness"
                        else d.rationale
                    ),
                )
                for d in dimension_scores
            ]
            rule_check = RuleCheckResult(
                passed=False,
                failures=rule_check.failures + [f"Hidden rudeness detected: {pa_rationale}"],
            )

    if groups_correct is not None:
        objective = grouping_accuracy_score(groups_correct)
        dimension_scores = [
            DimensionScore(
                dimension=d.dimension,
                score=objective if d.dimension == "grouping_accuracy" else d.score,
                rationale=(
                    f"{groups_correct}/4 groups correct (objective)."
                    if d.dimension == "grouping_accuracy"
                    else d.rationale
                ),
            )
            for d in dimension_scores
        ]

    return ScenarioResult(
        scenario_id=scenario.id,
        user_input=scenario.input,
        model_output=model_output,
        rule_check=rule_check,
        dimension_scores=dimension_scores,
        tags=scenario.tags,
        groups_correct=groups_correct,
        puzzle_solved=puzzle_solved,
    )


def evaluate_scenarios(
    *,
    goal: BehaviorGoal,
    scenarios: list[Scenario],
    system_prompt: str,
    backend: ModelBackend,
    judge: OpenAIJudge,
    round_index: int,
    max_new_tokens: int,
    temperature: float,
    edit_rationale: str = "",
) -> RoundScores:
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        results.append(
            evaluate_scenario(
                goal=goal,
                scenario=scenario,
                system_prompt=system_prompt,
                backend=backend,
                judge=judge,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        )
    return RoundScores(
        round_index=round_index,
        system_prompt=system_prompt,
        scenario_results=results,
        edit_rationale=edit_rationale,
    )
