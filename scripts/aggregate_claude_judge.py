#!/usr/bin/env python3
"""Aggregate the blind Claude-subagent judge results into per-run 4-dim means.

The judging was blind: subagents scored opaquely-named batches (batch_NN.json)
with no idea which model or that a comparison was happening. tmp/judge_batches/
_map.json maps each opaque batch back to its run tag (v3 / ngram8). This joins
results to runs and prints per-run dimension means + strict-pass, comparable to
the Opus scenario-bank sweep and the deterministic proxy.

Dimensions (0-2): melodic_equivalence, mode_fidelity, notation_convention,
meaning_preservation. Strict pass = melodic>=1.5 AND meaning>=1.5 (per-row).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

BDIR = Path("/Users/gaurav/From-Scratch-LLM/tmp/judge_batches")
DIMS = ["melodic_equivalence", "mode_fidelity", "notation_convention", "meaning_preservation"]


def main() -> None:
    mapping = json.loads((BDIR / "_map.json").read_text())  # batch_NN.json -> "<run>_<idx>"
    # run -> list of per-row score dicts
    by_run: dict[str, list[dict]] = defaultdict(list)
    missing = []
    for batch_file, tag in mapping.items():
        run = re.sub(r"_\d+$", "", tag)
        res = BDIR / f"result_{batch_file}"
        if not res.exists():
            missing.append(res.name)
            continue
        data = json.loads(res.read_text())
        by_run[run].extend(data.get("scores", []))

    if missing:
        print(f"WARNING: {len(missing)} result files missing: {', '.join(missing)}\n")

    print(f"{'run':<10} {'n':>4} {'melodic':>8} {'mode':>6} {'notat':>6} {'meaning':>8} {'strict':>8}")
    summary = {}
    for run in sorted(by_run):
        rows = by_run[run]
        n = len(rows)
        means = {d: sum(r.get(d, 0) for r in rows) / n for d in DIMS}
        strict = sum(1 for r in rows
                     if r.get("melodic_equivalence", 0) >= 1.5 and r.get("meaning_preservation", 0) >= 1.5)
        summary[run] = {"n": n, "dimensions": {d: round(means[d], 3) for d in DIMS},
                        "strict_pass": f"{strict}/{n}", "strict_rate": round(strict / n, 4)}
        print(f"{run:<10} {n:>4} {means['melodic_equivalence']:>8.2f} {means['mode_fidelity']:>6.2f} "
              f"{means['notation_convention']:>6.2f} {means['meaning_preservation']:>8.2f} {strict:>4}/{n:<3}")

    # delta if both present
    if "v3" in summary and "ngram8" in summary:
        print("\ndelta (ngram8 - v3):")
        for d in DIMS:
            dv = summary["ngram8"]["dimensions"][d] - summary["v3"]["dimensions"][d]
            print(f"  {d:<22} {dv:+.2f}")

    out = BDIR.parent.parent / "runs" / "claude_judge_v3_vs_ngram8.json"
    out.write_text(json.dumps({
        "judge": "claude-opus (blind subagents, 4-dim rubric from goals/byzantine_transcription.yaml)",
        "eval": "data/byzantine/sft_aligned_n2w_heldout.jsonl (100-row paired sample, n2w)",
        "caveat": "different eval set than Opus scenario-bank sweep; judge is Claude not Opus-API. Directional but real rubric judgment.",
        "runs": summary,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
