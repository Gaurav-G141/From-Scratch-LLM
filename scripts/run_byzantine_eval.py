#!/usr/bin/env python3
"""Run Byzantine transcription eval and write JSON outputs for Opus grading.

Usage:
  python scripts/run_byzantine_eval.py --provider openai --model gpt-4.1 \\
    --prompt prompts/byzantine_transcription_v2.txt \\
    --scenarios scenarios/byzantine_transcription_final_dev.yaml

  # Re-run only GPT-4.1 failures from summary JSON (prompt iteration)
  python scripts/run_byzantine_eval.py --provider openai --model gpt-4.1 \\
    --prompt prompts/byzantine_transcription_v3.txt \\
    --from-summary runs/byzantine_final_opus_summary.json \\
    --summary-key gpt-4.1_failures

  python scripts/run_byzantine_eval.py --provider anthropic --model claude-opus-4-20250514 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt", default="prompts/byzantine_transcription_v2.txt")
    parser.add_argument("--scenarios", default="scenarios/byzantine_transcription_final_dev.yaml")
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated scenario ids to run (subset of --scenarios file)",
    )
    parser.add_argument(
        "--from-summary",
        default="",
        help="Load scenario ids from a summary JSON key, e.g. runs/byzantine_final_opus_summary.json",
    )
    parser.add_argument(
        "--summary-key",
        default="gpt-4.1_failures",
        help="JSON key for id list when using --from-summary (default: gpt-4.1_failures)",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    from eval_harness.backends.anthropic_api import AnthropicBackend
    from eval_harness.backends.openai_api import OpenAIBackend
    from eval_harness.config import load_goal
    from eval_harness.judge.byzantine_checks import check_byzantine_transcription
    from eval_harness.scenarios.loader import load_scenarios, resolve_scenario_path

    goal = load_goal(ROOT / "goals/byzantine_transcription.yaml")
    prompt = (ROOT / args.prompt).read_text(encoding="utf-8").strip()
    scenarios = load_scenarios(ROOT / args.scenarios)

    filter_ids: list[str] = []
    if args.from_summary:
        summary = json.loads((ROOT / args.from_summary).read_text(encoding="utf-8"))
        raw = summary.get(args.summary_key)
        if not isinstance(raw, list):
            raise SystemExit(f"Summary key {args.summary_key!r} missing or not a list in {args.from_summary}")
        filter_ids = [str(x) for x in raw]
    elif args.ids.strip():
        filter_ids = [x.strip() for x in args.ids.split(",") if x.strip()]

    if filter_ids:
        by_id = {s.id: s for s in scenarios}
        missing = [sid for sid in filter_ids if sid not in by_id]
        if missing:
            raise SystemExit(f"Unknown scenario ids in {args.scenarios}: {', '.join(missing)}")
        scenarios = [by_id[sid] for sid in filter_ids]

    if args.provider == "anthropic":
        model = args.model or "claude-opus-4-20250514"
        backend = AnthropicBackend(model=model)
    else:
        model = args.model or "gpt-4.1"
        backend = OpenAIBackend(model=model)

    print(f"Model: {backend.name} | Prompt: {args.prompt} | N={len(scenarios)}", flush=True)

    results = []
    for i, s in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {s.id}", flush=True)
        try:
            out = backend.generate(
                prompt, s.input, max_new_tokens=args.max_tokens, temperature=args.temperature
            )
        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            out = f"ERROR: {exc}"
        rule_failures = check_byzantine_transcription(
            model_output=out, direction=s.direction, forbidden_extra=goal.forbidden_patterns
        )
        results.append(
            {
                "id": s.id,
                "direction": s.direction,
                "echos": s.echos,
                "tags": s.tags,
                "input": s.input,
                "reference_output": s.reference_output,
                "context": s.context,
                "model_output": out,
                "rule_failures": rule_failures,
            }
        )

    slug = model.replace("/", "-").replace(":", "-")
    if args.out:
        out_path = Path(args.out)
    elif filter_ids:
        out_path = ROOT / "runs" / f"byzantine_{slug}_failures_outputs.json"
    else:
        out_path = ROOT / "runs" / f"byzantine_{slug}_final_outputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
