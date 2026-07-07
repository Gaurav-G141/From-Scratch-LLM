from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


def _normalize_judge_scores(parsed: Any, dimensions: list[str]) -> list[dict[str, Any]]:
    """Accept list, wrapped list, per-dimension map, or single score object."""
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    if not isinstance(parsed, dict):
        return []

    for key in ("scores", "dimensions", "results", "evaluations"):
        if key in parsed and isinstance(parsed[key], list):
            return [item for item in parsed[key] if isinstance(item, dict)]

    if "dimension" in parsed and "score" in parsed:
        return [parsed]

    items: list[dict[str, Any]] = []
    for dim in dimensions:
        if dim not in parsed:
            continue
        value = parsed[dim]
        if isinstance(value, dict):
            items.append({"dimension": dim, **value})
        elif isinstance(value, (int, float)):
            items.append({"dimension": dim, "score": int(value), "rationale": ""})
    if items:
        return items

    # Last resort: any nested dicts that look like score rows
    for value in parsed.values():
        if isinstance(value, dict) and "score" in value:
            item = dict(value)
            if "dimension" not in item:
                item["dimension"] = next(iter(parsed.keys()), "unknown")
            items.append(item)
    return items


class OpenAIBackend:
    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.name = f"openai:{model}"
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        }
        # o-series / gpt-5 models use max_completion_tokens; omit temperature if unsupported
        if self.model.startswith(("o", "gpt-5")):
            kwargs["max_completion_tokens"] = max_new_tokens
        else:
            kwargs["max_tokens"] = max_new_tokens
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()


class OpenAIJudge:
    def __init__(self, model: str = "gpt-4o", *, api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def score(
        self,
        *,
        behavior_spec: str,
        dimensions: list[str],
        user_input: str,
        model_output: str,
        rule_failures: list[str],
        dimension_guidance: dict[str, str] | None = None,
        context: str = "",
        reference_output: str = "",
    ) -> list[dict[str, Any]]:
        guidance = dimension_guidance or {}
        dim_lines = "\n".join(
            f"- {d}: score 0-2. {guidance.get(d, '0=violates, 1=partial, 2=fully meets this criterion.')}"
            for d in dimensions
        )
        rule_section = ""
        if rule_failures:
            rule_section = (
                "\nDeterministic rule checks failed:\n"
                + "\n".join(f"- {f}" for f in rule_failures)
            )
        context_section = ""
        if context:
            context_section = f"\nAdditional context for judging:\n{context}\n"
        reference_section = ""
        if reference_output:
            reference_section = (
                "\nGold reference transcription (for comparison — model did NOT see this):\n"
                f"{reference_output}\n"
            )

        strict_section = ""
        if any("HARSH GATE" in v for v in guidance.values()):
            strict_section = """
IMPORTANT: Apply a harsh gate on cleanliness. Politeness that still belittles,
blames, or snipes is a 0 or 1 — never a 2. Passive-aggression is as bad as open rudeness.
"""

        prompt = f"""You are an evaluation judge for a behavioral spec.
{strict_section}

Behavior spec:
{behavior_spec}

Score the model output on each dimension (integer 0, 1, or 2 only):
{dim_lines}
{context_section}{reference_section}
Original text to transform:
{user_input}

Model output (sanitized/transformed text):
{model_output}
{rule_section}

Return ONLY valid JSON object with this shape:
{{
  "scores": [
    {{"dimension": "<name>", "score": 0, "rationale": "<short reason>"}}
  ]
}}
Include one object per dimension listed above."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        score_rows = _normalize_judge_scores(parsed, dimensions)

        if not score_rows:
            raise ValueError(f"Judge returned unexpected JSON: {content}")

        by_dim = {str(item.get("dimension")): item for item in score_rows}
        results: list[dict[str, Any]] = []
        for dim in dimensions:
            item = by_dim.get(dim, {})
            score = int(item.get("score", 0))
            score = max(0, min(2, score))
            results.append(
                {
                    "dimension": dim,
                    "score": score,
                    "rationale": str(item.get("rationale") or "No rationale provided."),
                }
            )
        return results

    def detect_hidden_rudeness(self, *, original: str, sanitized: str, context: str = "") -> tuple[bool, str]:
        context_line = f"\nContext: {context}" if context else ""
        prompt = f"""Does this sanitized rewrite still contain passive-aggression, backhanded comments,
condescension, sarcasm, veiled insults, dismissive tone, or "politely cruel" phrasing?
Answer harshly — if there is ANY snide subtext, hidden blame, or talking down to someone, answer yes.
{context_line}

Original:
{original}

Sanitized:
{sanitized}

Return ONLY JSON: {{"detected": true/false, "rationale": "<one sentence>"}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        return bool(parsed.get("detected")), str(parsed.get("rationale") or "")


class OpenAIPromptEditor:
    def __init__(self, model: str = "gpt-4o", *, api_key: str | None = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def revise(
        self,
        *,
        behavior_spec: str,
        current_prompt: str,
        failures: list[dict[str, Any]],
        successes: list[dict[str, Any]],
    ) -> tuple[str, str]:
        failure_text = "\n\n".join(
            f"Scenario: {f['scenario_id']}\n"
            f"User: {f['user_input']}\n"
            f"Model output: {f['model_output']}\n"
            f"Rule failures: {', '.join(f.get('rule_failures') or []) or 'none'}\n"
            f"Judge notes: {f.get('judge_notes') or 'none'}"
            for f in failures
        )
        success_text = "\n\n".join(
            f"Scenario: {s['scenario_id']}\n"
            f"User: {s['user_input']}\n"
            f"Model output: {s['model_output']}"
            for s in successes
        )

        prompt = f"""You improve system prompts for a model that must follow a behavioral spec.

Behavior spec:
{behavior_spec}

Current system prompt:
{current_prompt}

Failures to fix:
{failure_text or 'None'}

Examples that already work (preserve these behaviors):
{success_text or 'None'}

Write a revised system prompt that targets the failure modes without unnecessary verbosity.
Do not add few-shot examples. Return ONLY valid JSON:
{{"system_prompt": "<revised prompt>", "rationale": "<one paragraph on what changed and why>"}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        new_prompt = str(parsed.get("system_prompt") or current_prompt).strip()
        rationale = str(parsed.get("rationale") or "").strip()
        return new_prompt, rationale
