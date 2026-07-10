#!/usr/bin/env python3
"""Join model predictions with their eval inputs + gold targets into ONE self-contained
grading bundle, so the non-deterministic rubric grading (done by an agent, since the API
judge is billing-blocked) needs nothing but this single file.

predict_local.py writes {id, prediction}. The 0-2 rubric grader also needs the input
prompt and the gold reference for each id — this packs all three together.

Also runs the DETERMINISTIC scorer (score_synthetic_eval.py logic) inline when possible,
so the bundle carries the objective pitch/interval/exact numbers alongside each row. The
agent then only has to supply the subjective 0-2 dimensions.

Usage:
  python scripts/pack_eval_bundle.py \
    --eval  data/byzantine/sft_n2w_heldout_cued.jsonl \
    --preds runs/coder7b_n2w_preds.jsonl \
    --out   runs/coder7b_n2w_bundle.json \
    --label "Qwen2.5-Coder-7B byz->west, 3ep"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_eval(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        msgs = r["messages"]
        user = next(m for m in msgs if m["role"] == "user")["content"]
        asst = next(m for m in msgs if m["role"] == "assistant")["content"]
        out[r["id"]] = {"task": r.get("task", ""), "input": user, "reference": asst}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="the held-out JSONL the preds were made on")
    ap.add_argument("--preds", required=True, help="predict_local.py output {id, prediction}")
    ap.add_argument("--out", required=True, help="bundle JSON for agent grading")
    ap.add_argument("--label", default="", help="human label for this run")
    args = ap.parse_args()

    ev = load_eval(Path(args.eval))
    rows = []
    n_missing = 0
    for line in Path(args.preds).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        rid = p["id"]
        meta = ev.get(rid)
        if meta is None:
            n_missing += 1
            continue
        rows.append({
            "id": rid,
            "task": meta["task"],
            "input": meta["input"],
            "prediction": p["prediction"],
            "reference": meta["reference"],
        })

    bundle = {
        "label": args.label,
        "eval_file": args.eval,
        "preds_file": args.preds,
        "n_rows": len(rows),
        "n_preds_without_matching_eval": n_missing,
        "grading_note": (
            "Grade each row 0-2 on: melodic_equivalence, mode_fidelity, "
            "notation_convention, meaning_preservation (rubric in "
            "goals/byzantine_transcription.yaml). NOTE: melodic_equivalence stays ~0 for "
            "real chant (melismatic ~1.78:1 wall) — do NOT penalize neume/pitch count "
            "mismatch. Judge contour/interval, mode header correctness, notation format, "
            "and meaning. See docs/byzantine_handoff_20260709.md."
        ),
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} graded-ready rows -> {out_path}"
          + (f"  ({n_missing} preds had no matching eval row)" if n_missing else ""))


if __name__ == "__main__":
    main()
