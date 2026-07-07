from __future__ import annotations

import json
import os
from typing import Any

from eval_harness.backends.openai_api import _normalize_judge_scores


class AnthropicBackend:
    """Generation backend via Anthropic API (Claude Opus / Sonnet)."""

    def __init__(
        self,
        model: str = "claude-opus-4-20250514",
        *,
        api_key: str | None = None,
    ):
        self.model = model
        self.name = f"anthropic:{model}"
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts).strip()


class AnthropicJudge:
    """LLM-as-judge via Anthropic API (e.g. Claude Opus for high-quality music eval)."""

    def __init__(
        self,
        model: str = "claude-opus-4-20250514",
        *,
        api_key: str | None = None,
    ):
        self.model = model
        self.name = f"anthropic:{model}"
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package required for AnthropicJudge. "
                "Install with: pip install anthropic"
            ) from exc
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

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

        prompt = f"""You are an expert evaluation judge for music notation transcription tasks.
You have deep knowledge of Byzantine (Chrysanthine) and Western staff notation.

Behavior spec:
{behavior_spec}

Score the model output on each dimension (integer 0, 1, or 2 only):
{dim_lines}
{context_section}{reference_section}
Source notation (input to model):
{user_input}

Model transcription output:
{model_output}
{rule_section}

Return ONLY valid JSON object with this shape:
{{
  "scores": [
    {{"dimension": "<name>", "score": 0, "rationale": "<short reason>"}}
  ]
}}
Include one object per dimension listed above."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

        parsed = json.loads(content)
        score_rows = _normalize_judge_scores(parsed, dimensions)
        if not score_rows:
            raise ValueError(f"Judge returned unexpected JSON: {content}")

        by_dim = {str(item.get("dimension")): item for item in score_rows}
        results: list[dict[str, Any]] = []
        for dim in dimensions:
            item = by_dim.get(dim, {})
            score = max(0, min(2, int(item.get("score", 0))))
            results.append(
                {
                    "dimension": dim,
                    "score": score,
                    "rationale": str(item.get("rationale") or "No rationale provided."),
                }
            )
        return results
