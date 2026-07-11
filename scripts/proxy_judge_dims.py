#!/usr/bin/env python3
"""Approximate the 4 LLM-judge rubric dimensions from DETERMINISTIC metrics.

NOT the real judge. The original Opus/GPT-4o sweeps scored 4 dimensions 0-2 with
an LLM: melodic_equivalence, mode_fidelity, notation_convention,
meaning_preservation. The OpenAI account is inactive (billing) and no Anthropic
key is set, so this maps those 4 dims from the deterministic metrics already in a
score_real_musical.py report, to give a DIRECTIONAL delta without any API spend.

Mapping (each clamped to 0-2), chosen to mirror the rubric wording in
goals/byzantine_transcription.yaml:
  melodic_equivalence  <- relative motion: mean(interval_hist_sim, contour_sim) * 2
  mode_fidelity        <- mean(mode_correct, ison_correct) * 2
  notation_convention  <- fraction of prediction tokens that are valid gold-vocab
                          tokens (clean formatting) * 2
  meaning_preservation <- semantic fidelity: mean(set_f1, hist_sim, ambitus_match) * 2

Read the numbers as a PROXY. Two independent caveats vs the Opus sweep:
  1. Different judge (formula, not an LLM).
  2. Different eval set (DTW real heldout, not the hand-crafted scenario banks).
So this is a rough delta, explicitly labeled, not apples-to-apples.

Usage:
  python3 scripts/proxy_judge_dims.py \
    --realscore runs/v3b_ngram8_n2w_realscore.json \
    [--eval data/byzantine/sft_aligned_n2w_heldout.jsonl] \
    [--pred runs/v3b_n2w_ngram8_preds.jsonl] \
    [--out runs/v3b_ngram8_n2w_proxydims.json]
  # batch table:
  python3 scripts/proxy_judge_dims.py --table runs/v3_n2w_realscore.json runs/v3b_ngram8_n2w_realscore.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
DIMS = ["melodic_equivalence", "mode_fidelity", "notation_convention", "meaning_preservation"]

# a pitch token like G4, A#3, B-4, C#5, D-5 (matches the gold vocab shape)
PITCH_RE = re.compile(r"^[A-G][#-]?\d$")


def clamp2(x: float) -> float:
    return max(0.0, min(2.0, x))


def gold_vocab(eval_path: Path) -> set[str]:
    vocab: set[str] = set()
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            a = next((m["content"] for m in row["messages"] if m["role"] == "assistant"), "")
            for tok in a.split():
                vocab.add(tok)
    return vocab


def strip_pred(text: str) -> str:
    """Drop <think> blocks and Mode/Ison header lines, keep the note/neume body."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith(("mode", "ison", "(ison")):
            continue
        lines.append(s)
    return " ".join(lines) if lines else text


def token_validity(pred_path: Path, vocab: set[str]) -> float:
    """Mean fraction of prediction body tokens that are in the gold vocab."""
    fracs = []
    with pred_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            toks = strip_pred(row.get("prediction", "")).split()
            if not toks:
                fracs.append(0.0)
                continue
            good = sum(1 for t in toks if t in vocab)
            fracs.append(good / len(toks))
    return mean(fracs) if fracs else 0.0


def block(report: dict) -> dict:
    for k in ("neume_to_west", "west_to_neume"):
        if report.get(k):
            return report[k]
    return report.get("overall", {})


def proxy_dims(report: dict, notation: float | None) -> dict[str, float]:
    b = block(report)
    g = lambda k: float(b.get(k, 0.0))  # noqa: E731
    melodic = clamp2(mean([g("interval_hist_sim"), g("contour_sim")]) * 2)
    mode = clamp2(mean([g("mode_correct"), g("ison_correct")]) * 2)
    if notation is None:
        # fall back to (1 - drone-ish); without pred file we can't measure validity
        notation = g("variety")  # weak proxy; prefer passing --pred
    notation_dim = clamp2(notation * 2)
    meaning = clamp2(mean([g("set_f1"), g("hist_sim"), g("ambitus_match")]) * 2)
    return {
        "melodic_equivalence": round(melodic, 3),
        "mode_fidelity": round(mode, 3),
        "notation_convention": round(notation_dim, 3),
        "meaning_preservation": round(meaning, 3),
    }


def infer_pred_eval(report: dict) -> tuple[Path | None, Path | None]:
    pred = report.get("pred_file")
    ev = report.get("eval_file")
    pp = Path(pred) if pred else None
    if pp and not pp.is_absolute():
        pp = ROOT / pp
    ep = Path(ev) if ev else None
    if ep and not ep.is_absolute():
        ep = ROOT / ep
    return (pp if pp and pp.exists() else None, ep if ep and ep.exists() else None)


def dims_for(realscore_path: Path, eval_path: Path | None, pred_path: Path | None) -> dict:
    report = json.loads(realscore_path.read_text())
    if eval_path is None or pred_path is None:
        ip, ie = infer_pred_eval(report)
        pred_path = pred_path or ip
        eval_path = eval_path or ie
    notation = None
    if eval_path and pred_path:
        notation = token_validity(pred_path, gold_vocab(eval_path))
    return proxy_dims(report, notation)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realscore", help="a score_real_musical.py report JSON")
    ap.add_argument("--eval", help="gold eval JSONL (else inferred from realscore)")
    ap.add_argument("--pred", help="predictions JSONL (else inferred from realscore)")
    ap.add_argument("--out", help="write proxy-dims JSON here")
    ap.add_argument("--table", nargs="*", help="print a compact table for several realscore files")
    args = ap.parse_args()

    if args.table:
        print(f"{'run':<16} {'melodic':>8} {'mode':>6} {'notat':>6} {'meaning':>8} {'strict~':>8}")
        for p in args.table:
            rp = Path(p) if Path(p).is_absolute() else ROOT / p
            d = dims_for(rp, None, None)
            strict = "PASS" if (d["melodic_equivalence"] >= 1.5 and d["meaning_preservation"] >= 1.5) else "-"
            tag = re.sub(r"_realscore\.json$", "", Path(p).name)
            print(f"{tag:<16} {d['melodic_equivalence']:>8.2f} {d['mode_fidelity']:>6.2f} "
                  f"{d['notation_convention']:>6.2f} {d['meaning_preservation']:>8.2f} {strict:>8}")
        print("\nPROXY from deterministic metrics — NOT the LLM judge. Directional only.")
        print("melodic<-interval_hist+contour  mode<-mode+ison  notat<-token-validity  meaning<-set_f1+hist+ambitus")
        return

    if not args.realscore:
        ap.error("--realscore required (or use --table)")
    rp = Path(args.realscore) if Path(args.realscore).is_absolute() else ROOT / args.realscore
    ep = (Path(args.eval) if args.eval else None)
    pp = (Path(args.pred) if args.pred else None)
    dims = dims_for(rp, ep, pp)
    strict = dims["melodic_equivalence"] >= 1.5 and dims["meaning_preservation"] >= 1.5
    out = {
        "realscore_file": str(args.realscore),
        "method": "deterministic proxy for the 4 LLM-judge dims — NOT the real judge",
        "dimensions": dims,
        "strict_pass_proxy": strict,
        "caveat": "different judge (formula) AND different eval set (DTW real heldout, not Opus scenario banks); directional delta only.",
    }
    print(json.dumps(out, indent=2))
    if args.out:
        op = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
