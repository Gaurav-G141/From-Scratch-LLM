from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BehaviorGoal:
    name: str
    description: str
    initial_system_prompt: str
    forbidden_patterns: list[str] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=lambda: [
        "spec_adherence",
        "robustness",
        "task_quality",
        "consistency",
    ])
    scenario_generation_hint: str = ""
    dev_scenarios_path: str = ""
    heldout_scenarios_path: str = ""
    json_schema: dict[str, Any] | None = None
    litmus_dimension: str = ""
    dimension_guidance: dict[str, str] = field(default_factory=dict)
    pass_thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def primary_dimension(self) -> str:
        if self.litmus_dimension:
            return self.litmus_dimension
        if self.dimensions:
            return self.dimensions[0]
        return "spec_adherence"


@dataclass
class Scenario:
    id: str
    input: str
    tags: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    expected_groups: list[list[str]] = field(default_factory=list)
    context: str = ""
    direction: str = "byz_to_west"  # byz_to_west | west_to_byz
    reference_output: str = ""
    echos: str = ""
    source_url: str = ""


@dataclass
class RuleCheckResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class DimensionScore:
    dimension: str
    score: int
    rationale: str


@dataclass
class ScenarioResult:
    scenario_id: str
    user_input: str
    model_output: str
    rule_check: RuleCheckResult
    dimension_scores: list[DimensionScore]
    tags: list[str] = field(default_factory=list)
    groups_correct: int | None = None
    groups_total: int = 4
    puzzle_solved: bool = False

    @property
    def mean_score(self) -> float:
        if not self.dimension_scores:
            return 0.0
        return sum(d.score for d in self.dimension_scores) / len(self.dimension_scores)

    def score_for(self, dimension: str) -> float:
        for d in self.dimension_scores:
            if d.dimension == dimension:
                return float(d.score)
        return 0.0


@dataclass
class RoundScores:
    round_index: int
    system_prompt: str
    scenario_results: list[ScenarioResult]
    edit_rationale: str = ""

    def mean_for(self, dimension: str) -> float:
        if not self.scenario_results:
            return 0.0
        return sum(r.score_for(dimension) for r in self.scenario_results) / len(
            self.scenario_results
        )

    def overall_mean(self) -> float:
        if not self.scenario_results:
            return 0.0
        return sum(r.mean_score for r in self.scenario_results) / len(
            self.scenario_results
        )


@dataclass
class HarnessConfig:
    max_iterations: int = 8
    patience: int = 2
    min_delta: float = 0.02
    success_threshold: float = 1.85
    train_threshold: float = 1.2
    borderline_low: float = 1.4
    borderline_high: float = 1.75
    openai_model: str = "gpt-4o"
    openai_judge_model: str = "gpt-4o"
    openai_editor_model: str = "gpt-4o"
    judge_backend: str = "openai"  # openai | anthropic
    anthropic_judge_model: str = "claude-opus-4-20250514"
    local_model: str = "Qwen/Qwen3-0.6B"
    max_new_tokens: int = 512
    temperature: float = 0.7
    consistency_samples: int = 1
    max_failures_for_editor: int = 5
    max_passes_for_editor: int = 2
    inject_adversarial: bool = False
    adversarial_per_round: int = 2


@dataclass
class LitmusVerdict:
    outcome: str  # PASS, FAIL, BORDERLINE
    reason: str
    heldout_spec_adherence: float
    heldout_robustness: float
    best_dev_spec_adherence: float
    rounds_run: int


@dataclass
class RunReport:
    goal: BehaviorGoal
    config: HarnessConfig
    rounds: list[RoundScores]
    best_prompt: str
    best_round_index: int
    heldout: RoundScores | None
    verdict: LitmusVerdict

    def to_dict(self) -> dict[str, Any]:
        def round_to_dict(r: RoundScores) -> dict[str, Any]:
            return {
                "round_index": r.round_index,
                "system_prompt": r.system_prompt,
                "edit_rationale": r.edit_rationale,
                "overall_mean": r.overall_mean(),
                "dimension_means": {
                    dim: r.mean_for(dim) for dim in (self.goal.dimensions or [])
                },
                "scenario_results": [
                    {
                        "scenario_id": sr.scenario_id,
                        "user_input": sr.user_input,
                        "model_output": sr.model_output,
                        "tags": sr.tags,
                        "mean_score": sr.mean_score,
                        "rule_check": asdict(sr.rule_check),
                        "dimension_scores": [asdict(d) for d in sr.dimension_scores],
                    }
                    for sr in r.scenario_results
                ],
            }

        return {
            "goal_name": self.goal.name,
            "rounds": [round_to_dict(r) for r in self.rounds],
            "best_prompt": self.best_prompt,
            "best_round_index": self.best_round_index,
            "heldout": round_to_dict(self.heldout) if self.heldout else None,
            "verdict": asdict(self.verdict),
        }
