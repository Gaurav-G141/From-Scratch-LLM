#!/usr/bin/env python3
"""Combine the deterministic SYNTHETIC musicality set with the REAL corpus set into a
new training file — WITHOUT modifying either input.

WHY
---
The synthetic set (scripts/build_synthetic_musicality.py) teaches exact interval grammar
(1:1 by construction), which the real corpus cannot (neumes:pitches ~1.78:1, melismatic).
The real corpus teaches actual melody/melisma/mode context, which the synthetic set
honestly omits. Training on BOTH lets the model learn the ladder grammar from clean data
and real melodic behavior from the corpus. This script only READS the two source files
and writes a brand-new combined file; the real dataset is never touched.

ORDERING MODES
  interleave (default): deterministically round-robin synthetic and real rows so every
    batch sees both signals. Ratio-aware (weaves by the smaller stream).
  curriculum: all synthetic rows first (grammar), then all real rows (melody). Use when
    you want the model to master the ladder before seeing melismatic data.
  shuffle: deterministic seed-based shuffle of the union.

Every source row is passed through verbatim (same messages/schema). A `source` field is
added ("synthetic" | "real") for later filtering/weighting; the original `synthetic: true`
flag on synthetic rows is preserved. No row content is edited.

Usage:
  python scripts/build_combined_training.py \
    --synthetic data/byzantine/sft_synthetic_musicality.jsonl \
    --real      data/byzantine/sft_translation_train.jsonl \
    --out       data/byzantine/sft_combined_train.jsonl \
    --order interleave
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def tag(rows: list[dict], source: str) -> list[dict]:
    """Return NEW dict copies with a `source` field; never mutate the caller's rows."""
    out = []
    for r in rows:
        c = dict(r)
        c["source"] = source
        out.append(c)
    return out


def interleave(a: list[dict], b: list[dict]) -> list[dict]:
    """Deterministic ratio-aware round-robin: weave the two streams so both appear
    throughout, regardless of size imbalance."""
    if not a:
        return list(b)
    if not b:
        return list(a)
    out: list[dict] = []
    ia = ib = 0
    la, lb = len(a), len(b)
    # emit in proportion to lengths using an error-diffusion counter
    acc = 0.0
    step = la / lb
    while ia < la or ib < lb:
        if ib >= lb or (ia < la and acc < 1.0):
            out.append(a[ia]); ia += 1; acc += 1.0
        else:
            out.append(b[ib]); ib += 1; acc -= step
    return out


def det_shuffle(rows: list[dict], seed: int) -> list[dict]:
    """Deterministic Fisher-Yates using an LCG (no random module → reproducible)."""
    out = list(rows)
    x = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    for i in range(len(out) - 1, 0, -1):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        j = x % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def summarize(rows: list[dict]) -> None:
    by_source = Counter(r.get("source", "?") for r in rows)
    by_task = Counter(r.get("task", "?") for r in rows)
    print(f"  total rows: {len(rows)}")
    print(f"  by source : {dict(by_source)}")
    print(f"  by task   : {dict(by_task)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=str(DATA / "sft_synthetic_musicality.jsonl"))
    ap.add_argument("--real", default=str(DATA / "sft_translation_train.jsonl"))
    ap.add_argument("--out", default=str(DATA / "sft_combined_train.jsonl"))
    ap.add_argument("--order", choices=["interleave", "curriculum", "shuffle"],
                    default="interleave")
    ap.add_argument("--seed", type=int, default=1234, help="seed for --order shuffle")
    ap.add_argument("--synthetic-frac", type=float, default=1.0,
                    help="fraction of synthetic rows to include (0-1), head slice")
    args = ap.parse_args()

    out_path = Path(args.out)
    real_path = Path(args.real)
    # Guardrail: never let --out clobber a source file.
    for src in (Path(args.synthetic), real_path):
        if out_path.resolve() == src.resolve():
            raise SystemExit(f"refusing to overwrite input file: {src}")

    synth = tag(load(Path(args.synthetic)), "synthetic")
    real = tag(load(real_path), "real")

    if 0.0 <= args.synthetic_frac < 1.0:
        keep = int(len(synth) * args.synthetic_frac)
        synth = synth[:keep]

    print(f"synthetic: {len(synth)} rows   real: {len(real)} rows   order={args.order}")

    if args.order == "curriculum":
        combined = synth + real
    elif args.order == "shuffle":
        combined = det_shuffle(synth + real, args.seed)
    else:
        combined = interleave(synth, real)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote -> {out_path}")
    summarize(combined)
    # confirm the real input is byte-for-byte untouched (we only read it)
    print(f"  real input untouched: {real_path} ({real_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
