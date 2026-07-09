#!/usr/bin/env python3
"""Split the translation tasks (neume_to_west + west_to_neume) into train/heldout.

Splits by HYMN STEM, not by row: each hymn produces many windowed rows, so a naive
row split would leak windows of the same hymn across train and heldout. We hold out
whole hymns.

Reads data/byzantine/sft_neume.jsonl, writes:
  data/byzantine/sft_translation_train.jsonl
  data/byzantine/sft_translation_heldout.jsonl

Usage:
  python scripts/build_translation_split.py [--heldout-frac 0.1]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"
SRC = DATA / "sft_neume.jsonl"

TRANSLATION_TASKS = {"neume_to_west", "west_to_neume"}


def stem_of(rid: str) -> str:
    # ids look like <stem>_n2w_<wi> / <stem>_w2n_<wi>
    return re.sub(r"_(n2w|w2n)_\d+$", "", rid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    trans = [r for r in rows if r["task"] in TRANSLATION_TASKS]

    stems = sorted({stem_of(r["id"]) for r in trans})
    # deterministic held-out selection: every Nth stem
    n = max(1, round(1 / args.heldout_frac))
    heldout_stems = set(stems[::n])

    train = [r for r in trans if stem_of(r["id"]) not in heldout_stems]
    held = [r for r in trans if stem_of(r["id"]) in heldout_stems]

    with (DATA / "sft_translation_train.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "sft_translation_heldout.jsonl").open("w", encoding="utf-8") as f:
        for r in held:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # verify no stem leakage
    tr_stems = {stem_of(r["id"]) for r in train}
    he_stems = {stem_of(r["id"]) for r in held}
    leak = tr_stems & he_stems
    print(f"translation rows: {len(trans)}")
    print(f"train: {len(train)} rows / {len(tr_stems)} hymns")
    print(f"heldout: {len(held)} rows / {len(he_stems)} hymns")
    print(f"stem leakage: {len(leak)} (must be 0)")
    print(f"train by task: {dict(Counter(r['task'] for r in train))}")


if __name__ == "__main__":
    main()
