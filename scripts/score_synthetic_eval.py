#!/usr/bin/env python3
"""Deterministic pitch-accuracy scorer for the SYNTHETIC musicality eval slice.

WHY THIS EXISTS
---------------
The main eval harness scores melodic_equivalence with an LLM judge (Opus/gpt-4o) — see
eval_harness/judge/. That is noisy, costs API calls, and was even hand-graded when
billing was down (docs/byzantine_day3_results_20260708.md). But our synthetic data
(scripts/build_synthetic_musicality.py) is CORRECT BY CONSTRUCTION: every target pitch
is a pure interval walk from the ison anchor. So for the synthetic slice we can score
melodic accuracy EXACTLY, deterministically, with zero LLM variance and zero API cost.

This is the deterministic "did it learn the ladder?" instrument that pairs with the
held-out slice (data/byzantine/sft_synthetic_musicality_heldout.jsonl).

SCOPE / SAFETY
--------------
- STANDALONE. Does NOT import anything from eval_harness/ and does NOT load a model, so
  running it can never interfere with a training/eval job in progress. It scores a JSONL
  of {id, prediction} records that you produce however you like (see --help for the
  expected input), against the gold targets in the eval file.
- Only meaningful for the synthetic slice, whose gold is exact. Do NOT use it on the
  real melismatic corpus (neumes:pitches ~1.78:1 — exact match is not expected there;
  that is what the LLM judge is for).

METRICS (per neume_to_west row; west_to_neume handled analogously on neume tokens)
  exact_match         1.0 if predicted pitch sequence == gold, else 0.0
  pitch_accuracy      fraction of positions correct (len-normalised, position-aligned)
  interval_accuracy   fraction of consecutive intervals correct (contour/relative motion)
  norm_edit_distance  Levenshtein(pred, gold) / max(len) — 0.0 is perfect
  melodic_equivalence_0_2   a 0/1/2 score mirroring goals/byzantine_transcription.yaml
                            so it is comparable to the judge's dimension:
                              2 if exact or pitch_accuracy >= 0.95
                              1 if pitch_accuracy >= 0.6 or interval_accuracy >= 0.8
                              0 otherwise

Usage:
  # Score a predictions file against the held-out eval targets:
  python scripts/score_synthetic_eval.py \
      --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
      --pred runs/my_predictions.jsonl

  # predictions JSONL: one object per line, {"id": "<row id>", "prediction": "<raw model text>"}
  # ids must match the eval file's row ids (e.g. synth_010000000_t0_n2w).

  # Self-test the scorer (no files, no model):
  python scripts/score_synthetic_eval.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"

# A Western staff pitch token, e.g. G4, F#4, Bb3, or a held note with a beat count
# G4:2. We keep the accidental if present (a model emitting sharps on synthetic data is
# scored wrong, not normalised) AND the :beats suffix (duration is part of the answer).
PITCH_TOKEN = re.compile(r"\b[A-G][#b]?\d(?::\d+)?\b")
# Neume tokens are lowercase words with underscores (our vocab): oligon, apostrophos, ...
NEUME_TOKEN = re.compile(r"\b[a-z][a-z_]+\b")


def extract_pitches(text: str) -> list[str]:
    """Pull the predicted pitch sequence from raw model text, robust to think-blocks,
    prose, and the Mode/Ison header. Strategy: drop any <think>…</think>, then take the
    LAST non-empty line that is dominated by pitch tokens (the answer line), else fall
    back to all pitch tokens in the text."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # prefer a line that is mostly pitches and is NOT the ison header
    best: list[str] = []
    for ln in lines:
        if ln.lower().startswith("ison"):
            continue  # "Ison: G4" is the anchor, not the melody
        toks = ln.split()
        pitches = PITCH_TOKEN.findall(ln)
        if toks and len(pitches) >= max(2, len(toks) // 2):
            best = pitches  # keep updating -> ends on the LAST qualifying line
    if best:
        return best
    return PITCH_TOKEN.findall(text)


def extract_neumes(text: str, vocab: set[str]) -> list[str]:
    """Pull the predicted neume sequence for west_to_neume rows. Restrict to known vocab
    so header words ('mode', 'ison') and prose don't count as neumes."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    best: list[str] = []
    for ln in lines:
        if ln.lower().startswith(("mode", "(ison", "ison")):
            continue
        toks = [t for t in NEUME_TOKEN.findall(ln) if t in vocab]
        if len(toks) >= 2:
            best = toks
    if best:
        return best
    return [t for t in NEUME_TOKEN.findall(text) if t in vocab]


def gold_pitches(row: dict) -> list[str]:
    """Exact gold pitch line from a neume_to_west eval row (assistant target line 3)."""
    return row["messages"][2]["content"].split("\n")[2].split()


def gold_neumes(row: dict) -> list[str]:
    """Exact gold neume line from a west_to_neume eval row (assistant target line 3)."""
    return row["messages"][2]["content"].split("\n")[2].split()


def levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def intervals(pitches: list[str]) -> list[int]:
    """Signed diatonic-degree intervals between consecutive pitches (accidentals ignored
    for the interval view; letter+octave define the degree). Non-parseable -> skip."""
    NAMES = "CDEFGAB"

    def deg(p):
        p = p.split(":")[0]  # drop any :beats duration suffix; degree is pitch-only
        m = re.match(r"^([A-G])[#b]?(\d)$", p)
        return None if not m else NAMES.index(m.group(1)) + 7 * int(m.group(2))

    ds = [deg(p) for p in pitches]
    out = []
    for i in range(1, len(ds)):
        if ds[i] is None or ds[i - 1] is None:
            out.append(None)
        else:
            out.append(ds[i] - ds[i - 1])
    return out


def score_seq(pred: list[str], gold: list[str]) -> dict:
    """Core deterministic comparison of two token sequences (pitches or neumes)."""
    exact = 1.0 if pred == gold else 0.0
    n = max(len(pred), len(gold))
    pos_correct = sum(1 for i in range(min(len(pred), len(gold))) if pred[i] == gold[i])
    pitch_acc = pos_correct / n if n else 1.0
    gi, pi = intervals(gold), intervals(pred)
    m = max(len(gi), len(pi))
    iv_correct = sum(1 for i in range(min(len(gi), len(pi))) if gi[i] == pi[i] and gi[i] is not None)
    interval_acc = iv_correct / m if m else 1.0
    ned = levenshtein(pred, gold) / n if n else 0.0

    if exact or pitch_acc >= 0.95:
        mel = 2
    elif pitch_acc >= 0.6 or interval_acc >= 0.8:
        mel = 1
    else:
        mel = 0
    return {
        "exact_match": exact,
        "pitch_accuracy": round(pitch_acc, 4),
        "interval_accuracy": round(interval_acc, 4),
        "norm_edit_distance": round(ned, 4),
        "melodic_equivalence_0_2": mel,
        "len_pred": len(pred),
        "len_gold": len(gold),
    }


def load_eval(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def build_neume_vocab(eval_rows: dict[str, dict]) -> set[str]:
    vocab: set[str] = set()
    for r in eval_rows.values():
        if r.get("task") == "west_to_neume":
            vocab.update(gold_neumes(r))
    return vocab


def score_file(eval_path: Path, pred_path: Path) -> dict:
    eval_rows = load_eval(eval_path)
    vocab = build_neume_vocab(eval_rows)
    preds = {}
    for line in pred_path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            p = json.loads(line)
            preds[p["id"]] = p.get("prediction", "")

    per_row = []
    missing = 0
    for rid, row in eval_rows.items():
        if rid not in preds:
            missing += 1
            continue
        task = row.get("task")
        if task == "neume_to_west":
            s = score_seq(extract_pitches(preds[rid]), gold_pitches(row))
        elif task == "west_to_neume":
            s = score_seq(extract_neumes(preds[rid], vocab), gold_neumes(row))
        else:
            continue
        s["id"] = rid
        s["task"] = task
        per_row.append(s)

    def agg(rows):
        if not rows:
            return {}
        keys = ["exact_match", "pitch_accuracy", "interval_accuracy",
                "norm_edit_distance", "melodic_equivalence_0_2"]
        out = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
        out["n"] = len(rows)
        out["strict_pass_rate"] = round(
            sum(1 for r in rows if r["melodic_equivalence_0_2"] >= 2) / len(rows), 4)
        return out

    n2w = [r for r in per_row if r["task"] == "neume_to_west"]
    w2n = [r for r in per_row if r["task"] == "west_to_neume"]
    return {
        "eval_file": str(eval_path),
        "pred_file": str(pred_path),
        "n_scored": len(per_row),
        "n_missing_predictions": missing,
        "overall": agg(per_row),
        "neume_to_west": agg(n2w),
        "west_to_neume": agg(w2n),
        "per_row": per_row,
    }


def self_test() -> int:
    """Validate the scorer with NO files and NO model: gold-vs-gold must be perfect, and
    known corruptions must be penalised monotonically."""
    gold = "Mode 1\nIson: D4\nC4 D4 C4 D4 E4 C4"
    gold_pitch = gold.split("\n")[2].split()

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    # 1) perfect: identical, and robust to think-block + header noise
    s = score_seq(extract_pitches(gold), gold_pitch)
    check("gold vs gold -> exact_match 1.0, mel 2", s["exact_match"] == 1.0 and s["melodic_equivalence_0_2"] == 2)

    noisy = "<think>let me reason C5 wrong A9</think>\nMode 1\nIson: D4\nC4 D4 C4 D4 E4 C4"
    s = score_seq(extract_pitches(noisy), gold_pitch)
    check("think-block + header stripped -> still exact", s["exact_match"] == 1.0)

    # ison header must NOT be mistaken for the melody
    s = score_seq(extract_pitches("Ison: D4\nC4 D4 C4 D4 E4 C4"), gold_pitch)
    check("ison header ignored", s["exact_match"] == 1.0)

    # 2) one wrong note -> not exact, high but <1 pitch accuracy
    s = score_seq(extract_pitches("C4 D4 C4 D4 E4 G4"), gold_pitch)
    check("one wrong note -> exact 0, pitch_acc ~5/6", s["exact_match"] == 0.0 and abs(s["pitch_accuracy"] - 5/6) < 1e-3)

    # 3) transposed by a third (all wrong absolute, but intervals identical) -> interval_acc 1.0
    s = score_seq(["E4", "F4", "E4", "F4", "G4", "E4"], gold_pitch)
    check("transposed -> pitch_acc 0 but interval_acc 1.0", s["pitch_accuracy"] == 0.0 and s["interval_accuracy"] == 1.0)

    # 4) totally wrong -> mel 0
    s = score_seq(["G4", "G4", "G4"], gold_pitch)
    check("garbage -> mel 0", s["melodic_equivalence_0_2"] == 0)

    # 5) neume-direction scoring with a vocab
    vocab = {"oligon", "apostrophos", "ison", "elaphron", "petaste"}
    gneu = ["apostrophos", "oligon", "apostrophos", "oligon", "oligon", "elaphron"]
    s = score_seq(extract_neumes("Mode 1\n(Ison D4)\n" + " ".join(gneu), vocab), gneu)
    check("neume gold vs gold -> exact", s["exact_match"] == 1.0)

    # 6) edit distance sanity
    check("levenshtein basic", levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1)

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=str(DATA / "sft_synthetic_musicality_heldout.jsonl"),
                    help="eval JSONL with exact gold targets (the synthetic held-out slice)")
    ap.add_argument("--pred", help="predictions JSONL: {id, prediction} per line")
    ap.add_argument("--out", help="write full per-row JSON report here")
    ap.add_argument("--self-test", action="store_true", help="run built-in checks, no files")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.pred:
        ap.error("--pred is required (or use --self-test)")

    report = score_file(Path(args.eval), Path(args.pred))
    summary = {k: v for k, v in report.items() if k != "per_row"}
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull per-row report -> {args.out}")


if __name__ == "__main__":
    main()
