from __future__ import annotations

from eval_harness.backends.openai_api import OpenAIJudge
from eval_harness.judge.byzantine_checks import check_byzantine_transcription
from eval_harness.judge.rule_checks import run_rule_checks
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

    return ScenarioResult(
        scenario_id=scenario.id,
        user_input=scenario.input,
        model_output=model_output,
        rule_check=rule_check,
        dimension_scores=dimension_scores,
        tags=scenario.tags,
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
