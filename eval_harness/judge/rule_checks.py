from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from eval_harness.types import BehaviorGoal, RuleCheckResult


def run_rule_checks(
    goal: BehaviorGoal,
    model_output: str,
) -> RuleCheckResult:
    failures: list[str] = []
    text = model_output.strip()

    for pattern in goal.forbidden_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            failures.append(f"Matched forbidden pattern: {pattern}")

    for pattern in goal.required_patterns:
        if not re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            failures.append(f"Missing required pattern: {pattern}")

    if goal.json_schema:
        parsed, parse_error = _extract_json_strict(text)
        if parsed is None:
            failures.append(parse_error or "Output is not valid JSON")
        else:
            try:
                jsonschema.validate(instance=parsed, schema=goal.json_schema)
            except jsonschema.ValidationError as exc:
                failures.append(f"JSON schema validation failed: {exc.message}")

    return RuleCheckResult(passed=len(failures) == 0, failures=failures)


def _extract_json_strict(text: str) -> tuple[Any | None, str | None]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed, None
    except json.JSONDecodeError:
        pass

    relaxed = _extract_json(text)
    if relaxed is not None:
        return relaxed, None

    return None, "Output is not valid JSON (must be a single JSON object with no surrounding text)"


def _extract_json(text: str) -> Any | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            return None

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
