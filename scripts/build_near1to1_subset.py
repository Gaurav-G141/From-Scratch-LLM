#!/usr/bin/env python3
"""Build the near-1:1 aligned training subset (handoff idea #3).

The melodic_equivalence wall (docs/byzantine_handoff_20260709.md §2) is caused by the
~1.8:1 melisma ratio: neumes under-specify pitch, so whole-corpus seq2seq can't teach an
exact mapping. BUT ~12% of hymns are near-1:1 (pitch:pitch-bearing-neume ratio 0.9-1.1),
where the mapping is approximately valid. This script isolates those hymns into a small,
cleaner training + held-out set — the handoff's own recommended "controlled test of
whether *any* aligned real data moves the needle."

Approach (fully deterministic, READ-ONLY on all inputs):
  1. Recompute each parallel hymn's pitch:neume ratio from neumes_*.jsonl + omr/*.jsonl
     (same method as the handoff's finding), using ez_neume_map.json categories to count
     only pitch-bearing neumes.
  2. Select stems with ratio in [--lo, --hi] (default 0.9-1.1).
  3. Filter the existing windowed translation rows (sft_translation_train.jsonl +
     _heldout.jsonl) to those stems — we REUSE the already-built, already-vetted windows
     rather than re-deriving, so no alignment is asserted that wasn't already there.
  4. Split by hymn stem (no window leakage) into train/heldout.

Outputs (new files; nothing existing is modified):
  data/byzantine/sft_near1to1_train.jsonl
  data/byzantine/sft_near1to1_heldout.jsonl
  data/byzantine/near1to1_stems.json   (the selected stems + their ratios, for audit)

Usage:
  python scripts/build_near1to1_subset.py [--lo 0.9 --hi 1.1 --heldout-frac 0.15]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"

# neume categories that carry a pitch move (from the official EZ table). Everything else
# (time/breath/martyria/qualitative) is non-pitch and excluded from the neume count, so
# the ratio reflects pitch-bearing neumes vs pitches — the handoff's definition.
PITCH_CATS = {"ascending", "ascending_petaste", "descending", "support_combo"}


def load_categories() -> dict[str, str]:
    ez = json.load((DATA / "ez_neume_map.json").open())["map"]
    return {v["name"]: v["category"] for v in ez.values()}


def pitch_bearing_count(neumes: list[str], name2cat: dict[str, str]) -> int:
    return sum(1 for t in neumes if name2cat.get(t) in PITCH_CATS)


def load_neumes() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for src in ["goa-dcs", "new-byzantium", "st-anthonys"]:
        p = DATA / f"neumes_{src}.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("n_neumes", 0) > 0:
                stem = re.sub(r"_byz$", "", Path(r["path"]).stem)
                out[stem] = r["neumes"]
    return out


def load_omr_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for src in ["goa", "newbyz", "sam"]:
        p = DATA / "omr" / f"omr_{src}.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("n_notes", 0) > 0:
                stem = re.sub(r"_west$", "", Path(r["path"]).stem)
                out[stem] = sum(len(st) for st in r["staves"])
    return out


def stem_of(rid: str) -> str:
    return re.sub(r"_(n2w|w2n)_\d+$", "", rid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.9)
    ap.add_argument("--hi", type=float, default=1.1)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    args = ap.parse_args()

    name2cat = load_categories()
    neu = load_neumes()
    omr = load_omr_counts()

    # 1-2. select near-1:1 stems
    selected: dict[str, dict] = {}
    for stem in neu:
        if stem not in omr:
            continue
        pb = pitch_bearing_count(neu[stem], name2cat)
        if pb == 0:
            continue
        ratio = omr[stem] / pb
        if args.lo <= ratio <= args.hi:
            selected[stem] = {"pitch_bearing_neumes": pb, "pitches": omr[stem],
                              "ratio": round(ratio, 3)}
    print(f"near-1:1 hymns ({args.lo}-{args.hi}): {len(selected)}")

    # 3. filter existing windowed translation rows to those stems (reuse vetted windows)
    src_files = [DATA / "sft_translation_train.jsonl", DATA / "sft_translation_heldout.jsonl"]
    rows: list[dict] = []
    for sf in src_files:
        if not sf.exists():
            continue
        for line in sf.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task") in ("neume_to_west", "west_to_neume") and stem_of(r["id"]) in selected:
                rows.append(r)
    print(f"windowed rows matching those stems: {len(rows)}")
    if not rows:
        raise SystemExit("no matching rows — check that translation files exist/are built")

    # 4. split by stem (no window leakage) — deterministic every-Nth-stem
    stems = sorted({stem_of(r["id"]) for r in rows})
    n = max(1, round(1 / args.heldout_frac))
    held_stems = set(stems[::n])
    train = [r for r in rows if stem_of(r["id"]) not in held_stems]
    held = [r for r in rows if stem_of(r["id"]) in held_stems]

    (DATA / "sft_near1to1_train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
    (DATA / "sft_near1to1_heldout.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in held), encoding="utf-8")
    (DATA / "near1to1_stems.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")

    # audit
    tr_stems = {stem_of(r["id"]) for r in train}
    he_stems = {stem_of(r["id"]) for r in held}
    leak = tr_stems & he_stems
    print(f"train: {len(train)} rows / {len(tr_stems)} hymns")
    print(f"heldout: {len(held)} rows / {len(he_stems)} hymns")
    print(f"stem leakage: {len(leak)} (must be 0)")
    print(f"train by task: {dict(Counter(r['task'] for r in train))}")
    print(f"wrote sft_near1to1_{{train,heldout}}.jsonl + near1to1_stems.json")


if __name__ == "__main__":
    main()
