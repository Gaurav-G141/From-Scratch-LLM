from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eval_harness.types import BehaviorGoal, HarnessConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_config(path: Path | str | None = None) -> HarnessConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return HarnessConfig()

    raw = load_yaml(config_path)
    from dataclasses import fields

    known = {f.name for f in fields(HarnessConfig)}
    filtered = {k: v for k, v in raw.items() if k in known}
    return HarnessConfig(**filtered)


def load_goal(path: Path | str) -> BehaviorGoal:
    raw = load_yaml(path)
    required = ("name", "description", "initial_system_prompt")
    for key in required:
        if key not in raw:
            raise ValueError(f"Goal file {path} missing required field: {key}")

    return BehaviorGoal(
        name=raw["name"],
        description=raw["description"].strip(),
        initial_system_prompt=raw["initial_system_prompt"].strip(),
        forbidden_patterns=list(raw.get("forbidden_patterns") or []),
        required_patterns=list(raw.get("required_patterns") or []),
        dimensions=list(raw.get("dimensions") or BehaviorGoal().dimensions),
        scenario_generation_hint=str(raw.get("scenario_generation_hint") or "").strip(),
        dev_scenarios_path=str(raw.get("dev_scenarios_path") or ""),
        heldout_scenarios_path=str(raw.get("heldout_scenarios_path") or ""),
        json_schema=raw.get("json_schema"),
        litmus_dimension=str(raw.get("litmus_dimension") or "").strip(),
        dimension_guidance={
            str(k): str(v).strip()
            for k, v in (raw.get("dimension_guidance") or {}).items()
        },
        pass_thresholds={
            str(k): float(v) for k, v in (raw.get("pass_thresholds") or {}).items()
        },
    )
