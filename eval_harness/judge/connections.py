from __future__ import annotations

import json
import re
from typing import Any


def normalize_word(word: str) -> str:
    return word.strip().upper()


def normalize_group(words: list[str]) -> frozenset[str]:
    return frozenset(normalize_word(w) for w in words)


def parse_connections_output(text: str) -> list[list[str]] | None:
    parsed = _extract_json(text)
    if not isinstance(parsed, dict):
        return None

    groups_raw = parsed.get("groups")
    if not isinstance(groups_raw, list):
        return None

    groups: list[list[str]] = []
    for item in groups_raw:
        if isinstance(item, dict) and isinstance(item.get("words"), list):
            words = [str(w) for w in item["words"]]
        elif isinstance(item, list):
            words = [str(w) for w in item]
        else:
            return None
        if len(words) != 4:
            return None
        groups.append(words)

    if len(groups) != 4:
        return None
    return groups


def count_correct_groups(
    expected: list[list[str]],
    actual: list[list[str]],
) -> int:
    expected_sets = [normalize_group(g) for g in expected]
    actual_sets = [normalize_group(g) for g in actual]
    used: set[int] = set()
    correct = 0
    for expected_set in expected_sets:
        for idx, actual_set in enumerate(actual_sets):
            if idx not in used and expected_set == actual_set:
                correct += 1
                used.add(idx)
                break
    return correct


def validate_word_partition(
    input_words: list[str],
    groups: list[list[str]],
) -> list[str]:
    failures: list[str] = []
    expected = {normalize_word(w) for w in input_words}
    found: list[str] = []
    for group in groups:
        found.extend(normalize_word(w) for w in group)

    found_set = set(found)
    if len(found) != 16:
        failures.append(f"Expected 16 words in groups, found {len(found)}")
    if len(found_set) != len(found):
        failures.append("Duplicate words appear in output groups")
    missing = expected - found_set
    extra = found_set - expected
    if missing:
        failures.append(f"Missing input words: {', '.join(sorted(missing))}")
    if extra:
        failures.append(f"Output contains words not in input: {', '.join(sorted(extra))}")
    return failures


def score_connections(
    *,
    input_words: list[str],
    expected_groups: list[list[str]],
    model_output: str,
) -> tuple[int, bool, list[str]]:
    """Return (groups_correct, puzzle_solved, failures)."""
    failures: list[str] = []
    groups = parse_connections_output(model_output)
    if groups is None:
        return 0, False, ["Could not parse 4 groups of 4 words from JSON output"]

    failures.extend(validate_word_partition(input_words, groups))
    groups_correct = count_correct_groups(expected_groups, groups)
    puzzle_solved = groups_correct == 4 and not failures
    if groups_correct < 4:
        failures.append(f"Only {groups_correct}/4 groups match the solution")
    return groups_correct, puzzle_solved, failures


def grouping_accuracy_score(groups_correct: int) -> int:
    """Map 0-4 correct groups to rubric score 0-2."""
    if groups_correct >= 4:
        return 2
    if groups_correct >= 2:
        return 1
    return 0


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
