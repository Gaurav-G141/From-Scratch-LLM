#!/usr/bin/env python3
"""Export blind scenario inputs for Opus Cursor-agent full sweep (all six banks).

Usage:
  python scripts/export_opus_sweep_inputs.py
  python scripts/export_opus_sweep_inputs.py --suite final_dev
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

SUITES: dict[str, str] = {
    "dev": "scenarios/byzantine_transcription_dev.yaml",
    "heldout": "scenarios/byzantine_transcription_heldout.yaml",
    "final_dev": "scenarios/byzantine_transcription_final_dev.yaml",
    "break_dev": "scenarios/byzantine_transcription_break_dev.yaml",
    "ultra_hard": "scenarios/byzantine_transcription_ultra_hard.yaml",
    "unseen": "scenarios/byzantine_transcription_unseen.yaml",
}

PROMPTS = {
    "v0": "prompts/byzantine_transcription_v0.txt",
    "v2": "prompts/byzantine_transcription_v2.txt",
}


def export_suite(suite_key: str, scenarios_rel: str) -> dict:
    scenarios = yaml.safe_load((ROOT / scenarios_rel).read_text(encoding="utf-8"))
    return {
        "task": "byzantine_transcription_opus_sweep",
        "suite": suite_key,
        "scenarios_file": scenarios_rel,
        "n_scenarios": len(scenarios),
        "prompts": PROMPTS,
        "instructions": (
            "Translate every scenario twice: once with v0 prompt, once with v2 prompt. "
            "Blind only — do NOT read reference_output or context from scenario YAML."
        ),
        "scenarios": [
            {
                "id": s["id"],
                "direction": s.get("direction", "byz_to_west"),
                "echos": s.get("echos", ""),
                "tags": s.get("tags", []),
                "input": s["input"].strip(),
            }
            for s in scenarios
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=[*SUITES.keys(), "all"], default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "opus_sweep_inputs"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_keys = list(SUITES.keys()) if args.suite == "all" else [args.suite]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suites": {},
        "prompts": PROMPTS,
        "submission_dir": "runs/opus_sweep_submissions",
    }

    for key in suite_keys:
        payload = export_suite(key, SUITES[key])
        out_path = out_dir / f"{key}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["suites"][key] = {
            "inputs": str(out_path.relative_to(ROOT)),
            "n_scenarios": payload["n_scenarios"],
            "submission": f"runs/opus_sweep_submissions/{key}.json",
        }
        print(f"Wrote {payload['n_scenarios']} blind inputs → {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
