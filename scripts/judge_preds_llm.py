#!/usr/bin/env python3
"""Grade a predictions JSONL on the 4 LLM-judge rubric dimensions Opus was scored on.

Reuses the harness judge (eval_harness.backends.openai_api.OpenAIJudge) and the
goal's dimensions + guidance, so the rubric matches the original Opus/GPT-4o
sweeps EXACTLY: melodic_equivalence, mode_fidelity, notation_convention,
meaning_preservation (each 0-2). Strict pass = melodic>=1.5 AND meaning>=1.5.

This judges PRE-EXISTING predictions (it does not generate), pairing each
prediction with the gold assistant turn from the eval JSONL as the reference the
judge compares against.

IMPORTANT caveats (see docs): Opus's published numbers were on the hand-crafted
scenario banks judged by Opus. This grades the DTW-aligned REAL heldout with
gpt-4o. Different eval set AND different judge -> read the delta as DIRECTIONAL,
not apples-to-apples. gpt-4o vs Opus judges scored near-identically in the prior
sweeps (docs/byzantine_opus_sweep.md), which is why gpt-4o is an acceptable proxy.

Usage:
  python3 scripts/judge_preds_llm.py \
    --eval data/byzantine/sft_aligned_n2w_heldout.jsonl \
    --pred runs/v3b_n2w_ngram8_preds.jsonl \
    --out runs/v3b_ngram8_n2w_llmjudge.json \
    [--limit N] [--model gpt-4o]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_harness.config import load_goal            # noqa: E402
from eval_harness.backends.openai_api import OpenAIJudge  # noqa: E402

DIMS = ["melodic_equivalence", "mode_fidelity", "notation_convention", "meaning_preservation"]


def load_gold(path: Path) -> dict[str, dict]:
    """id -> {user_input, reference_output} from an SFT eval JSONL."""
    gold: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row["messages"]
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            ref = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            gold[row["id"]] = {"user_input": user, "reference_output": ref}
    return gold


def load_preds(path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            preds[row["id"]] = row.get("prediction", "")
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="SFT eval JSONL with gold assistant turns")
    ap.add_argument("--pred", required=True, help="predictions JSONL {id, prediction}")
    ap.add_argument("--out", required=True, help="write full per-row + summary JSON here")
    ap.add_argument("--goal", default=str(ROOT / "goals/byzantine_transcription.yaml"))
    ap.add_argument("--model", default="gpt-4o", help="OpenAI judge model")
    ap.add_argument("--limit", type=int, default=0, help="judge only first N matched rows (0=all)")
    args = ap.parse_args()

    goal = load_goal(args.goal)
    gold = load_gold(Path(args.eval))
    preds = load_preds(Path(args.pred))

    ids = [i for i in gold if i in preds]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        ap.error("no ids shared between --eval and --pred")

    judge = OpenAIJudge(model=args.model)
    print(f"judging {len(ids)} rows with {args.model} on dims: {', '.join(DIMS)}", file=sys.stderr)

    per_row = []
    sums = {d: 0.0 for d in DIMS}
    strict = 0
    for n, rid in enumerate(ids, 1):
        g = gold[rid]
        try:
            raw = judge.score(
                behavior_spec=goal.description,
                dimensions=DIMS,
                user_input=g["user_input"],
                model_output=preds[rid],
                rule_failures=[],
                dimension_guidance=goal.dimension_guidance,
                reference_output=g["reference_output"],
            )
        except Exception as e:  # keep going; one bad call shouldn't sink the run
            print(f"  [{n}/{len(ids)}] {rid}: judge error {e}", file=sys.stderr)
            continue
        row_scores = {str(item["dimension"]): int(item["score"]) for item in raw}
        for d in DIMS:
            sums[d] += row_scores.get(d, 0)
        mel = row_scores.get("melodic_equivalence", 0)
        mean = row_scores.get("meaning_preservation", 0)
        passed = mel >= 1.5 and mean >= 1.5
        strict += int(passed)
        per_row.append({"id": rid, "scores": row_scores, "strict_pass": passed})
        if n % 25 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)}", file=sys.stderr, flush=True)

    n_judged = len(per_row)
    dim_means = {d: round(sums[d] / n_judged, 3) for d in DIMS} if n_judged else {d: 0.0 for d in DIMS}
    report = {
        "eval_file": args.eval,
        "pred_file": args.pred,
        "judge_model": args.model,
        "n_judged": n_judged,
        "dimensions": dim_means,
        "strict_pass": f"{strict}/{n_judged}",
        "strict_pass_rate": round(strict / n_judged, 4) if n_judged else 0.0,
        "note": "gpt-4o judge on DTW-real heldout; directional delta vs Opus scenario-bank sweep, not apples-to-apples.",
        "per_row": per_row,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "per_row"}, indent=2))
    print(f"\nFull per-row report -> {args.out}")


if __name__ == "__main__":
    main()
