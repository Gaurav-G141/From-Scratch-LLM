from __future__ import annotations

from pathlib import Path

import yaml

from eval_harness.types import Scenario


def load_scenarios(path: Path | str) -> list[Scenario]:
    scenario_path = Path(path)
    with open(scenario_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(f"Scenario file {scenario_path} must contain a list")

    scenarios: list[Scenario] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario entry {idx} in {scenario_path} must be a mapping")
        scenario_id = str(item.get("id") or f"scenario_{idx}")
        user_input = item.get("input")
        if not user_input:
            raise ValueError(f"Scenario {scenario_id} missing input")
        expected_groups = item.get("expected_groups") or []
        words = list(item.get("words") or [])
        if not words and expected_groups:
            words = [word for group in expected_groups for word in group]
        scenarios.append(
            Scenario(
                id=scenario_id,
                input=str(user_input).strip(),
                tags=list(item.get("tags") or []),
                words=[str(w) for w in words],
                expected_groups=[
                    [str(w) for w in group]
                    for group in expected_groups
                    if isinstance(group, list)
                ],
                context=str(item.get("context") or "").strip(),
                direction=str(item.get("direction") or "byz_to_west").strip(),
                reference_output=str(item.get("reference_output") or "").strip(),
                echos=str(item.get("echos") or "").strip(),
                source_url=str(item.get("source_url") or "").strip(),
            )
        )
    return scenarios


def resolve_scenario_path(goal_path: Path, relative: str) -> Path:
    if not relative:
        raise ValueError("Scenario path is empty")
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    return (goal_path.parent / candidate).resolve()
