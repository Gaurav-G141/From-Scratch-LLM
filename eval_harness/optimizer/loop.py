from __future__ import annotations

from eval_harness.backends.openai_api import OpenAIBackend, OpenAIJudge, OpenAIPromptEditor
from eval_harness.judge.runner import evaluate_scenarios
from eval_harness.optimizer.prompt_editor import revise_prompt
from eval_harness.types import BehaviorGoal, HarnessConfig, LitmusVerdict, RoundScores, RunReport, Scenario
from eval_harness.backends.base import ModelBackend


def compute_verdict(
    *,
    config: HarnessConfig,
    primary_dimension: str,
    heldout_primary: float,
    heldout_secondary: float,
    best_dev_primary: float,
    rounds_run: int,
    avg_groups_correct: float | None = None,
    puzzles_solved: int | None = None,
    puzzles_total: int | None = None,
) -> LitmusVerdict:
    dim_label = primary_dimension.replace("_", " ")

    if heldout_primary >= config.success_threshold:
        extra = ""
        if avg_groups_correct is not None:
            extra = f" Objective score: {avg_groups_correct:.2f}/4.0 groups correct avg."
            if puzzles_solved is not None and puzzles_total:
                extra += f" {puzzles_solved}/{puzzles_total} puzzles fully solved."
        return LitmusVerdict(
            outcome="FAIL",
            reason=(
                f"Held-out {dim_label} {heldout_primary:.2f} reached success threshold "
                f"{config.success_threshold:.2f}. A well-prompted frontier model can do this "
                f"reliably — pick a harder behavior.{extra}"
            ),
            heldout_spec_adherence=heldout_primary,
            heldout_robustness=heldout_secondary,
            best_dev_spec_adherence=best_dev_primary,
            rounds_run=rounds_run,
        )

    if heldout_primary <= config.train_threshold:
        return LitmusVerdict(
            outcome="PASS",
            reason=(
                f"Held-out {dim_label} plateaued at {heldout_primary:.2f}, below train threshold "
                f"{config.train_threshold:.2f}. Prompt optimization could not make the behavior "
                "reliable — good candidate for fine-tuning."
            ),
            heldout_spec_adherence=heldout_primary,
            heldout_robustness=heldout_secondary,
            best_dev_spec_adherence=best_dev_primary,
            rounds_run=rounds_run,
        )

    if config.borderline_low <= heldout_primary < config.borderline_high:
        return LitmusVerdict(
            outcome="BORDERLINE",
            reason=(
                f"Held-out {dim_label} {heldout_primary:.2f} is in the borderline band "
                f"({config.borderline_low:.2f}-{config.borderline_high:.2f}). Tighten the spec "
                "or add adversarial scenarios before committing."
            ),
            heldout_spec_adherence=heldout_primary,
            heldout_robustness=heldout_secondary,
            best_dev_spec_adherence=best_dev_primary,
            rounds_run=rounds_run,
        )

    if heldout_primary < config.borderline_low:
        return LitmusVerdict(
            outcome="PASS",
            reason=(
                f"Held-out {dim_label} {heldout_primary:.2f} stayed well below reliable prompting "
                f"levels after {rounds_run} optimization rounds."
            ),
            heldout_spec_adherence=heldout_primary,
            heldout_robustness=heldout_secondary,
            best_dev_spec_adherence=best_dev_primary,
            rounds_run=rounds_run,
        )

    return LitmusVerdict(
        outcome="BORDERLINE",
        reason=(
            f"Held-out {dim_label} {heldout_primary:.2f} improved but did not clearly pass or fail "
            "the litmus thresholds."
        ),
        heldout_spec_adherence=heldout_primary,
        heldout_robustness=heldout_secondary,
        best_dev_spec_adherence=best_dev_primary,
        rounds_run=rounds_run,
    )


def _objective_stats(round_scores: RoundScores) -> tuple[float | None, int | None, int | None]:
    scored = [r for r in round_scores.scenario_results if r.groups_correct is not None]
    if not scored:
        return None, None, None
    avg_groups = sum(r.groups_correct or 0 for r in scored) / len(scored)
    solved = sum(1 for r in scored if r.puzzle_solved)
    return avg_groups, solved, len(scored)


def run_litmus_loop(
    *,
    goal: BehaviorGoal,
    config: HarnessConfig,
    dev_scenarios: list[Scenario],
    heldout_scenarios: list[Scenario],
    backend: ModelBackend,
    judge: OpenAIJudge,
    editor: OpenAIPromptEditor,
) -> RunReport:
    primary = goal.primary_dimension
    secondary = goal.dimensions[1] if len(goal.dimensions) > 1 else primary

    rounds: list[RoundScores] = []
    current_prompt = goal.initial_system_prompt
    best_prompt = current_prompt
    best_round_index = 0
    best_dev_primary = 0.0

    no_improve_streak = 0
    prev_dev_primary = -1.0

    for round_index in range(config.max_iterations):
        round_scores = evaluate_scenarios(
            goal=goal,
            scenarios=dev_scenarios,
            system_prompt=current_prompt,
            backend=backend,
            judge=judge,
            round_index=round_index,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        rounds.append(round_scores)

        dev_primary = round_scores.mean_for(primary)
        if dev_primary > best_dev_primary:
            best_dev_primary = dev_primary
            best_prompt = current_prompt
            best_round_index = round_index

        if round_index == config.max_iterations - 1:
            break

        if prev_dev_primary >= 0 and (dev_primary - prev_dev_primary) < config.min_delta:
            no_improve_streak += 1
        else:
            no_improve_streak = 0
        prev_dev_primary = dev_primary

        if no_improve_streak >= config.patience:
            break

        new_prompt, rationale = revise_prompt(
            editor=editor,
            behavior_spec=goal.description,
            current_prompt=current_prompt,
            round_scores=round_scores,
            max_failures=config.max_failures_for_editor,
            max_passes=config.max_passes_for_editor,
        )
        if new_prompt == current_prompt:
            break
        current_prompt = new_prompt
        rounds[-1].edit_rationale = rationale

    heldout = evaluate_scenarios(
        goal=goal,
        scenarios=heldout_scenarios,
        system_prompt=best_prompt,
        backend=backend,
        judge=judge,
        round_index=len(rounds),
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )

    avg_groups, puzzles_solved, puzzles_total = _objective_stats(heldout)
    verdict = compute_verdict(
        config=config,
        primary_dimension=primary,
        heldout_primary=heldout.mean_for(primary),
        heldout_secondary=heldout.mean_for(secondary),
        best_dev_primary=best_dev_primary,
        rounds_run=len(rounds),
        avg_groups_correct=avg_groups,
        puzzles_solved=puzzles_solved,
        puzzles_total=puzzles_total,
    )

    return RunReport(
        goal=goal,
        config=config,
        rounds=rounds,
        best_prompt=best_prompt,
        best_round_index=best_round_index,
        heldout=heldout,
        verdict=verdict,
    )
