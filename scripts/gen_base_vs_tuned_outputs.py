#!/usr/bin/env python3
"""Generate base + tuned local-model outputs for Byzantine eval banks (no judge).

Judge-free half of the base-vs-tuned comparison: loads the base model once and
the LoRA-tuned model once, generates on each scenario, and dumps input +
reference_output + both outputs to JSON for offline (agent) grading on the
0-2 rubric. Used when no API judge is available.

Usage:
  python scripts/gen_base_vs_tuned_outputs.py \
    --model Qwen/Qwen3-1.7B \
    --adapter-path models/byzantine_sft_translation_1.7b \
    --prompt prompts/byzantine_transcription_v2.txt \
    --suites heldout,unseen,ultra_hard \
    --out runs/byzantine_translation_1.7b_outputs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUITES = {
    "dev": "scenarios/byzantine_transcription_dev.yaml",
    "heldout": "scenarios/byzantine_transcription_heldout.yaml",
    "unseen": "scenarios/byzantine_transcription_unseen.yaml",
    "ultra_hard": "scenarios/byzantine_transcription_ultra_hard.yaml",
}


def main() -> None:
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-path", default=str(ROOT / "models/byzantine_sft_translation_1.7b"))
    parser.add_argument("--prompt", default=str(ROOT / "prompts/byzantine_transcription_v2.txt"))
    parser.add_argument("--suites", default="heldout,unseen,ultra_hard")
    parser.add_argument("--out", default=str(ROOT / "runs/byzantine_translation_1.7b_outputs.json"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    from eval_harness.backends.local_hf import LocalHFBackend
    from eval_harness.scenarios.loader import load_scenarios

    system_prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    suite_keys = [s.strip() for s in args.suites.split(",") if s.strip()]
    for k in suite_keys:
        if k not in SUITES:
            raise SystemExit(f"Unknown suite {k!r}; choose from {list(SUITES)}")
    suite_data = {k: load_scenarios(ROOT / SUITES[k]) for k in suite_keys}

    result: dict = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "prompt_file": args.prompt,
        "temperature": args.temperature,
        "suites": {},
    }

    for label, adapter in [("base", None), ("tuned", args.adapter_path)]:
        print(f"\n=== Loading {label} ({args.model}) ===", flush=True)
        backend = LocalHFBackend(model_name=args.model, adapter_path=adapter)
        for key in suite_keys:
            scenarios = suite_data[key]
            print(f"  {label} | {key} | n={len(scenarios)}", flush=True)
            rows = result["suites"].setdefault(key, {})
            for i, sc in enumerate(scenarios):
                out = backend.generate(
                    system_prompt,
                    sc.input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                entry = rows.setdefault(
                    sc.id,
                    {
                        "id": sc.id,
                        "direction": sc.direction,
                        "tags": sc.tags,
                        "input": sc.input,
                        "reference_output": sc.reference_output,
                    },
                )
                entry[f"{label}_output"] = out
                print(f"    [{label}] {key} {i+1}/{len(scenarios)} {sc.id}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(len(v) for v in result["suites"].values())
    print(f"\nWrote {total} scenarios across {len(suite_keys)} suites -> {out_path}")


if __name__ == "__main__":
    main()
