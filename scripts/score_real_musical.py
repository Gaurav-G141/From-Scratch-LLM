#!/usr/bin/env python3
"""Deterministic REAL-data musical-property scorer (contour / distribution / shape).

WHY THIS EXISTS
---------------
On the REAL melismatic corpus, neumes:pitches align ~1.78:1, so exact per-position pitch
is NOT recoverable from the input — `score_synthetic_eval.py` correctly refuses to be used
there, and every exact-match experiment on real data scored ~0 (see
docs/byzantine_coder7b_results_20260710.md: the 7B adapters collapsed to a `G4 G4 G4…`
drone). That zero is real, but it is also uninformative: it cannot tell a model that
learned genuine musical structure (right scale, right range, right melodic shape) from one
that learned nothing.

This scorer grades the properties that ARE recoverable on real data and that a drone
CANNOT fake:
  - does the output use the right PITCHES (scale/mode membership), regardless of order?
  - does it move in the right CONTOUR (up/down/same), alignment-robustly (via DTW)?
  - does it span the right RANGE (ambitus)?
  - does it reproduce local melodic SHAPE (pitch/neume n-grams)?
  - is it the right LENGTH (melisma awareness), and NON-degenerate (variety)?
  - exact MODE and ISON header lines (the conventional, learnable part).

It is the instrument for the synthetic->real curriculum: it turns "works on real data"
into a number you can move, instead of a wall of zeros. A modal drone scores near 0 on
every distributional metric here; a model that captures real chant structure scores high
even when it cannot pin exact positions.

SCOPE / SAFETY
--------------
- Standalone except for reusing the parse/util helpers in `score_synthetic_eval.py`
  (same scripts/ dir). Loads NO model, touches NO eval_harness, so it cannot interfere
  with a training/eval job.
- Complements, does not replace, the LLM judge: these are objective structural metrics,
  not holistic musical judgement.

METRICS (per row; n2w on pitches, w2n on neumes)
  mode_correct        1.0 if the "Mode …" line matches gold exactly
  ison_correct        1.0 if the "Ison: …" line matches gold exactly (n2w only; else n/a)
  length_ratio        min(len)/max(len) of the body sequence — 1.0 same length, ->0 degenerate
  variety             distinct/total tokens in the prediction — a drone -> ~0
  set_f1              F1 of the DISTINCT-token sets (right vocabulary/scale, order-free)
  hist_sim            1 - 0.5*TV distance of token-frequency histograms (usage profile)
  contour_sim         DTW-aligned agreement of up/down/same motion (shape, order-robust)
  interval_hist_sim   1 - 0.5*TV of signed-interval histograms (relative-motion profile)
  ambitus_match       range (max-min) closeness: 1 - min(1, |Rp-Rg|/max(Rg,1))
  ngram_f1            multiset F1 of consecutive bigrams (local melodic shape)
  real_musicality_0_2 composite mapped to 0/1/2 (see compose_score)

Usage:
  python scripts/score_real_musical.py \
      --eval data/byzantine/sft_n2w_heldout.jsonl \
      --pred runs/coder7b_n2w_preds.jsonl \
      --out runs/coder7b_n2w_real_score.json

  python scripts/score_real_musical.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the vetted parsing/util helpers rather than duplicating them.
from score_synthetic_eval import (  # noqa: E402
    extract_pitches,
    extract_neumes,
    gold_pitches,
    gold_neumes,
    intervals,
    build_neume_vocab,
    load_eval,
)

NAMES = "CDEFGAB"


def _degrees(pitches: list[str]) -> list[int]:
    """Diatonic degree per pitch (letter+octave), unparseable dropped. For range/ambitus."""
    out = []
    for p in pitches:
        m = re.match(r"^([A-G])[#b]?(\d)$", p)
        if m:
            out.append(NAMES.index(m.group(1)) + 7 * int(m.group(2)))
    return out


def _contour(seq_intervals: list[int | None]) -> list[int]:
    """Map intervals to motion signs: +1 up, -1 down, 0 same. Drop None (unparseable)."""
    return [(1 if d > 0 else -1 if d < 0 else 0) for d in seq_intervals if d is not None]


def _tv_sim(a: list, b: list) -> float:
    """Histogram similarity = 1 - 0.5 * total-variation distance of the two token
    frequency distributions. 1.0 identical usage, 0.0 disjoint. Empty vs empty -> 1.0,
    one empty -> 0.0."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    na, nb = sum(ca.values()), sum(cb.values())
    keys = set(ca) | set(cb)
    tv = 0.5 * sum(abs(ca.get(k, 0) / na - cb.get(k, 0) / nb) for k in keys)
    return round(1.0 - tv, 4)


def _set_f1(a: list, b: list) -> float:
    """F1 over DISTINCT token sets (order-free vocabulary agreement)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(sa), inter / len(sb)
    return round(2 * prec * rec / (prec + rec), 4)


def _multiset_f1(a: list, b: list) -> float:
    """F1 over multisets (counts matter). Used for bigram overlap."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    if inter == 0:
        return 0.0
    prec, rec = inter / sum(ca.values()), inter / sum(cb.values())
    return round(2 * prec * rec / (prec + rec), 4)


def _bigrams(seq: list) -> list[tuple]:
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def _dtw_agreement(a: list[int], b: list[int]) -> float:
    """Alignment-robust agreement of two contour (motion-sign) sequences via DTW with a
    0/1 substitution cost. Returns 1 - normalized_path_cost, so identical shape -> 1.0,
    unrelated -> ~0.0. Length differences and shifts are absorbed by the warping path.
    This is what gives credit for the right melodic shape without demanding the walled
    per-position alignment."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    INF = float("inf")
    prev = [INF] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    # normalize by the longest monotonic path length (max of the two seq lengths is a
    # tight-enough lower bound on path length for a 0/1 cost)
    return round(1.0 - prev[m] / max(n, m), 4)


def compose_score(sub: dict) -> int:
    """Map the alignment-robust subscores to a 0/1/2 real-musicality grade. Deliberately
    excludes exact-position metrics (walled on real data). A modal drone fails variety,
    contour_sim, set_f1, and hist_sim simultaneously, so it lands at 0."""
    core = sum(sub[k] for k in
               ("set_f1", "hist_sim", "contour_sim", "ngram_f1", "length_ratio")) / 5.0
    # anti-drone gate: if the output has almost no variety it cannot be "real music"
    if sub["variety"] < 0.15:
        return 0
    # length gate: a transcription off by >2x in length is wrong by construction (the
    # model ran on or truncated), regardless of how similar its interval/note *distribution*
    # looks. length_ratio is symmetric min/max, so <0.5 == >2x mismatch either direction.
    # Without this, a 3x-too-long run-on with the right note-bag scored a high composite
    # while a blind LLM judge (correctly) scored its melodic_equivalence 0. See
    # runs/claude_judge_v3_vs_ngram8.json.
    if sub.get("length_ratio", 1.0) < 0.5:
        return 0
    if core >= 0.70:
        return 2
    if core >= 0.45:
        return 1
    return 0


def _header_line(text: str, prefix: str) -> str | None:
    for ln in text.splitlines():
        s = ln.strip()
        if s.lower().startswith(prefix):
            return s
    return None


def _gold_lines(row: dict) -> list[str]:
    return [ln.strip() for ln in row["messages"][2]["content"].splitlines() if ln.strip()]


def score_row(pred_text: str, row: dict, vocab: set[str]) -> dict:
    task = row.get("task")
    if task == "neume_to_west":
        pred, gold = extract_pitches(pred_text), gold_pitches(row)
        pv = _degrees(pred)
        gv = _degrees(gold)
        p_iv = intervals(pred)
        g_iv = intervals(gold)
    elif task == "west_to_neume":
        pred, gold = extract_neumes(pred_text, vocab), gold_neumes(row)
        pv = gv = []          # no numeric degrees for neumes
        p_iv = g_iv = []      # interval/ambitus not defined on neume tokens
    else:
        return {}

    gl = _gold_lines(row)
    gmode = next((x for x in gl if x.lower().startswith("mode")), None)
    gison = next((x for x in gl if x.lower().startswith("ison")), None)
    pmode = _header_line(pred_text, "mode")
    pison = _header_line(pred_text, "ison")

    lp, lg = len(pred), len(gold)
    sub = {
        "mode_correct": 1.0 if (gmode and pmode == gmode) else 0.0,
        "ison_correct": (1.0 if (gison and pison == gison) else 0.0) if gison else None,
        "length_ratio": round(min(lp, lg) / max(lp, lg), 4) if max(lp, lg) else 1.0,
        "variety": round(len(set(pred)) / len(pred), 4) if pred else 0.0,
        "set_f1": _set_f1(pred, gold),
        "hist_sim": _tv_sim(pred, gold),
        "contour_sim": _dtw_agreement(_contour(p_iv), _contour(g_iv)) if task == "neume_to_west" else None,
        "interval_hist_sim": _tv_sim([d for d in p_iv if d is not None],
                                     [d for d in g_iv if d is not None]) if task == "neume_to_west" else None,
        "ambitus_match": None,
        "ngram_f1": _multiset_f1(_bigrams(pred), _bigrams(gold)),
        "len_pred": lp,
        "len_gold": lg,
    }
    if task == "neume_to_west" and gv:
        Rg = (max(gv) - min(gv)) if gv else 0
        Rp = (max(pv) - min(pv)) if pv else 0
        sub["ambitus_match"] = round(1.0 - min(1.0, abs(Rp - Rg) / max(Rg, 1)), 4)

    # composite needs contour_sim; for w2n (no contour) substitute ngram_f1 as the shape term
    comp_sub = dict(sub)
    if comp_sub.get("contour_sim") is None:
        comp_sub["contour_sim"] = sub["ngram_f1"]
    sub["real_musicality_0_2"] = compose_score(comp_sub)
    sub["id"] = row["id"]
    sub["task"] = task
    return sub


def score_file(eval_path: Path, pred_path: Path) -> dict:
    eval_rows = load_eval(eval_path)
    vocab = build_neume_vocab(eval_rows)
    preds = {}
    for line in pred_path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            p = json.loads(line)
            preds[p["id"]] = p.get("prediction", "")

    per_row, missing = [], 0
    for rid, row in eval_rows.items():
        if rid not in preds:
            missing += 1
            continue
        s = score_row(preds[rid], row, vocab)
        if s:
            per_row.append(s)

    def agg(rows):
        if not rows:
            return {}
        keys = ["mode_correct", "ison_correct", "length_ratio", "variety", "set_f1",
                "hist_sim", "contour_sim", "interval_hist_sim", "ambitus_match",
                "ngram_f1", "real_musicality_0_2"]
        out = {}
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                out[k] = round(sum(vals) / len(vals), 4)
        out["n"] = len(rows)
        out["good_rate"] = round(sum(1 for r in rows if r["real_musicality_0_2"] >= 2) / len(rows), 4)
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


def _fake_row(task: str, body: list[str], mode="Mode 1", ison="Ison: G4") -> dict:
    """Build a minimal eval row with a gold assistant turn (line[2] = header/ison/body)."""
    if task == "neume_to_west":
        gold = f"{mode}\n{ison}\n" + " ".join(body)
    else:
        gold = f"{mode}\n{ison}\n" + " ".join(body)
    return {"id": f"t_{task}", "task": task,
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u"},
                         {"role": "assistant", "content": gold}]}


def self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    gold_body = "G4 A4 B4 A4 G4 F4 G4 A4 B4 C5 B4 A4 G4".split()
    row = _fake_row("neume_to_west", gold_body)
    vocab: set[str] = set()

    # 1) gold vs gold -> everything ~perfect, composite 2
    perfect = f"Mode 1\nIson: G4\n" + " ".join(gold_body)
    s = score_row(perfect, row, vocab)
    check("n2w gold-vs-gold set_f1=1", s["set_f1"] == 1.0)
    check("n2w gold-vs-gold contour_sim=1", s["contour_sim"] == 1.0)
    check("n2w gold-vs-gold ngram_f1=1", s["ngram_f1"] == 1.0)
    check("n2w gold-vs-gold real_musicality=2", s["real_musicality_0_2"] == 2)
    check("n2w gold-vs-gold mode+ison correct", s["mode_correct"] == 1.0 and s["ison_correct"] == 1.0)

    # 2) drone -> distributional metrics collapse, composite 0 (the key discriminator)
    drone = "Mode 1\nIson: G4\n" + " ".join(["G4"] * len(gold_body))
    s = score_row(drone, row, vocab)
    check("drone variety ~0", s["variety"] < 0.15)
    check("drone contour_sim low", s["contour_sim"] < 0.4)
    check("drone real_musicality=0", s["real_musicality_0_2"] == 0)

    # 3) right notes, wrong order (shuffle) -> set/hist high, contour/ngram lower -> partial
    shuf = "Mode 1\nIson: G4\nA4 G4 C5 B4 A4 G4 F4 B4 A4 G4 B4 A4 G4"
    s = score_row(shuf, row, vocab)
    check("shuffle set_f1 high", s["set_f1"] >= 0.8)
    check("shuffle hist_sim high", s["hist_sim"] >= 0.7)
    check("shuffle not scored as perfect (ngram<1)", s["ngram_f1"] < 1.0)

    # 4) wrong scale entirely -> set_f1 low, composite <=1
    wrong = "Mode 1\nIson: G4\nC3 C#3 D3 D#3 E3 F3 F#3 G3 G#3 A3 A#3 B3 C4"
    s = score_row(wrong, row, vocab)
    check("wrong-scale set_f1 low", s["set_f1"] < 0.4)
    check("wrong-scale real_musicality <=1", s["real_musicality_0_2"] <= 1)

    # 4b) LENGTH GATE: right notes/intervals but ~3x too long (run-on) -> composite 0.
    # This is the exact v3b failure a blind LLM judge caught: interval-histogram similarity
    # stays high on a run-on, but the transcription is wrong. Must gate to 0.
    runon = "Mode 1\nIson: G4\n" + " ".join(gold_body * 3)
    s = score_row(runon, row, vocab)
    check("run-on length_ratio < 0.5", s["length_ratio"] < 0.5)
    check("run-on keeps high set_f1 (why it fooled the composite)", s["set_f1"] >= 0.8)
    check("run-on real_musicality=0 (length gate)", s["real_musicality_0_2"] == 0)
    # and truncation the other way (gold 3x longer than pred) also gates
    trunc_gold = _fake_row("neume_to_west", gold_body * 3)
    s = score_row(perfect, trunc_gold, vocab)
    check("truncated (pred<<gold) length_ratio < 0.5", s["length_ratio"] < 0.5)
    check("truncated real_musicality=0 (length gate)", s["real_musicality_0_2"] == 0)

    # 5) w2n path: gold-vs-gold on neumes
    gneu = "oligon apostrophos ison oligon oligon apostrophos elaphron oligon".split()
    wrow = _fake_row("west_to_neume", gneu)
    wvocab = set(gneu)
    wperfect = "Mode 1\nIson: G4\n" + " ".join(gneu)
    s = score_row(wperfect, wrow, wvocab)
    check("w2n gold-vs-gold set_f1=1", s["set_f1"] == 1.0)
    check("w2n gold-vs-gold ngram_f1=1", s["ngram_f1"] == 1.0)
    check("w2n gold-vs-gold real_musicality=2", s["real_musicality_0_2"] == 2)
    check("w2n contour n/a (None)", s["contour_sim"] is None)

    # 6) w2n drone -> 0
    wdrone = "Mode 1\nIson: G4\n" + " ".join(["oligon"] * len(gneu))
    s = score_row(wdrone, wrow, wvocab)
    check("w2n drone real_musicality=0", s["real_musicality_0_2"] == 0)

    # 7) helper sanity
    check("tv_sim identical=1", _tv_sim(["a", "a", "b"], ["a", "a", "b"]) == 1.0)
    check("tv_sim disjoint=0", _tv_sim(["a"], ["b"]) == 0.0)
    check("dtw identical=1", _dtw_agreement([1, -1, 0, 1], [1, -1, 0, 1]) == 1.0)
    check("dtw shifted still high", _dtw_agreement([1, 1, -1, -1], [1, -1, -1]) >= 0.5)

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=str(DATA / "sft_n2w_heldout.jsonl"),
                    help="real eval JSONL (e.g. sft_n2w_heldout.jsonl / sft_w2n_heldout.jsonl)")
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
