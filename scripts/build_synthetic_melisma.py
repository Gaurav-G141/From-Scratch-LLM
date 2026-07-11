#!/usr/bin/env python3
"""Correct-by-construction MELISMATIC synthetic data (one neume -> several pitches).

WHY
---
The existing synthetic set (build_synthetic_musicality.py) is strictly 1:1 — one neume
maps to exactly one pitch step. Real chant (and the DTW-aligned real data) is MELISMATIC:
one neume governs several pitches. That 1:1-vs-melisma gap is the wall the project keeps
hitting. This generator teaches melisma on CLEAN data — each melisma neume deterministically
expands to a FIXED multi-pitch figure from the running anchor — so the model learns
one-neume->many-pitches before it meets the noisy real version. Still exact: given the mode,
anchor, and neume sequence, the pitch sequence is uniquely determined, so n2w is fully
gradeable (w2n has the honest many-to-one where different figures share a contour).

DESIGN
------
- Plain step neumes (from the 1:1 grammar) advance the cursor by one step, emit one pitch.
- MELISMA neumes expand to a fixed sequence of degree-deltas RELATIVE to the running cursor,
  emitting several pitches and leaving the cursor at the figure's end. Figures are musical
  ornaments (passing tones, turns, cadential descents) drawn from diatonic motion only.
- Breath/barline no-ops emit nothing (n2w only), same as the base generator.
- Everything is degree-based on the same C3..C6 diatonic ladder → no microtones.

Vocabulary note: melisma token names are attested-plausible Byzantine ornament neumes
(petaste_qualitative, kentemata, oligon_kentemata, apli, psifiston) that already appear in
the real corpus, so the synthetic vocab stays a subset of the real one (good for transfer).

Usage:
  python scripts/build_synthetic_melisma.py --out data/byzantine/sft_synth_melisma.jsonl --n 2000
  python scripts/build_synthetic_melisma.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"

SYSTEM_PROMPT = (
    "You are a Byzantine chant notation assistant. You work with Byzantine neume "
    "sequences (ison, oligon, petaste, apostrophos, gorgon, martyria, ...) and their "
    "Western staff-notation transcriptions. Output the answer only — no commentary."
)

LADDER = [
    "C3", "D3", "E3", "F3", "G3", "A3", "B3",
    "C4", "D4", "E4", "F4", "G4", "A4", "B4",
    "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6",
]
LADDER_INDEX = {p: i for i, p in enumerate(LADDER)}

MODES = {"Mode 1": "D4", "Mode pl. 1": "A4", "Mode 4": "G4", "Mode pl. 4": "C4"}

# 1:1 step neumes: (name -> single degree delta). Cursor moves by the delta, emits 1 pitch.
STEP_NEUMES = {
    "ison": 0, "oligon": +1, "petaste": +1, "apostrophos": -1,
    "oligon_kentema": +3, "oligon_hypsili": +4, "elaphron": -2, "chamile": -4,
}

# MELISMA neumes: name -> tuple of degree deltas applied SEQUENTIALLY from the running
# cursor, each producing one pitch. e.g. (+1, -1) = up a step then back (upper neighbor);
# emits 2 pitches, net cursor change 0. All figures are diatonic and singable.
MELISMA_NEUMES = {
    "kentemata": (+1, -1),                 # upper-neighbor ornament (2 pitches, net 0)
    "petaste_qualitative": (+1, +1, -1),   # ascending passing turn (3 pitches, net +1)
    "oligon_kentemata": (+1, +1),          # two-step ascent (2 pitches, net +2)
    "psifiston": (-1, +1),                 # lower-neighbor ornament (2 pitches, net 0)
    "apli": (0, 0),                        # sustained/lengthened tone (2 pitches, net 0)
    "elaphron_apostrophos": (-1, -1),      # two-step cadential descent (2 pitches, net -2)
}

BREATH_NOOPS = ["breath_mark_m", "comma_breath", "measure_bar"]

# weighted pool: mostly steps, sprinkle melismas
_POOL_WEIGHTS = [
    ("oligon", 4), ("apostrophos", 4), ("ison", 2), ("petaste", 2), ("elaphron", 2),
    ("kentemata", 3), ("petaste_qualitative", 2), ("oligon_kentemata", 2),
    ("psifiston", 2), ("apli", 1), ("elaphron_apostrophos", 1),
]
POOL = [n for n, w in _POOL_WEIGHTS for _ in range(w)]
AMBIT = 9


def _lcg(x: int) -> int:
    return (x * 1103515245 + 12345) & 0x7FFFFFFF


def _deltas(name: str) -> tuple[int, ...]:
    """The degree-delta sequence a neume emits (1-tuple for step neumes, longer for melisma)."""
    if name in STEP_NEUMES:
        return (STEP_NEUMES[name],)
    return MELISMA_NEUMES[name]


def gen_neumes(seed: int, length: int) -> list[str]:
    """Deterministic mixed step+melisma walk (names only), ambit-clamped so the whole
    figure of each neume stays on-ladder relative to the running cursor."""
    x = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    out: list[str] = []
    cum = 0
    for _ in range(length):
        x = _lcg(x)
        name = POOL[x % len(POOL)]
        deltas = _deltas(name)
        # check every intermediate offset of the figure stays within ambit
        c = cum
        ok = True
        for d in deltas:
            c += d
            if c > AMBIT or c < -AMBIT:
                ok = False
                break
        if not ok:
            name = "apostrophos" if cum > 0 else "oligon"
            deltas = _deltas(name)
            c = cum + deltas[0]
        cum = c
        out.append(name)
    return out


def maybe_insert_breaths(neumes: list[str], seed: int) -> list[str]:
    x = _lcg((seed ^ 0x5DEECE66D) & 0xFFFFFFFF)
    if x % 5 < 2:
        return list(neumes)
    out: list[str] = []
    for i, n in enumerate(neumes):
        out.append(n)
        x = _lcg(x)
        if 0 < i < len(neumes) - 1 and x % 6 == 0:
            out.append(BREATH_NOOPS[x % len(BREATH_NOOPS)])
    return out


def pitches_from(anchor_idx: int, neumes: list[str]) -> list[str]:
    """Expand the neume line to pitches: each step neume emits 1, each melisma neume emits
    len(figure) pitches, breaths emit none. Cursor is shared and running."""
    idx = anchor_idx
    out: list[str] = []
    for n in neumes:
        if n in BREATH_NOOPS:
            continue
        for d in _deltas(n):
            idx += d
            if not (0 <= idx < len(LADDER)):
                raise ValueError("off-ladder")
            out.append(LADDER[idx])
    return out


def make_rows(mode: str, anchor_idx: int, neumes_plain: list[str],
              neumes_n2w: list[str], rid: str) -> list[dict]:
    ison = LADDER[anchor_idx]
    pitches = pitches_from(anchor_idx, neumes_plain)
    pitch_str = " ".join(pitches)
    west_block = f"{mode}\nIson: {ison}\n{pitch_str}"
    byz_block = f"{mode}\n(Ison {ison})\n" + " ".join(neumes_plain)
    n_neumes = len([n for n in neumes_n2w])
    n_pitch = len(pitches)
    rows = [
        {"id": f"{rid}_n2w", "task": "neume_to_west", "synthetic": True, "melisma": True,
         "messages": [
             {"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content":
              f"Transcribe this Byzantine neume sequence ({n_neumes} neumes) to Western staff pitches:\n"
              f"{mode}\nIson: {ison}\n" + " ".join(neumes_n2w)},
             {"role": "assistant", "content": west_block}]},
        {"id": f"{rid}_w2n", "task": "west_to_neume", "synthetic": True, "melisma": True,
         "messages": [
             {"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content":
              f"Transcribe these Western staff pitches ({n_pitch} pitches) to a Byzantine neume sequence:\n"
              f"{mode}\nIson: {ison}\n{pitch_str}"},
             {"role": "assistant", "content": byz_block}]},
    ]
    return rows


def build(out_path: Path, n_walks: int, seed_start: int) -> None:
    rows: list[dict] = []
    modes = list(MODES)
    for i in range(n_walks):
        seed = seed_start + i
        length = 6 + (seed % 13)  # 6..18 neumes
        mode = modes[seed % len(modes)]
        base_anchor = LADDER_INDEX[MODES[mode]]
        # transpose within a small band, deterministically
        anchor = base_anchor + ((seed // 7) % 5) - 2
        anchor = max(LADDER_INDEX["G3"], min(LADDER_INDEX["A5"], anchor))
        neumes_plain = gen_neumes(seed, length)
        # retry anchor downward if the figure runs off-ladder
        for _try in range(6):
            try:
                pitches_from(anchor, neumes_plain)
                break
            except ValueError:
                anchor -= 1
        else:
            continue
        neumes_n2w = maybe_insert_breaths(neumes_plain, seed)
        rows.extend(make_rows(mode, anchor, neumes_plain, neumes_n2w, f"melis_{seed:09d}"))
    rows.sort(key=lambda r: r["id"])
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n2w = sum(1 for r in rows if r["task"] == "neume_to_west")
    print(f"Wrote {len(rows)} rows ({n2w} n2w / {len(rows)-n2w} w2n) -> {out_path}")


def self_test() -> int:
    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    # melisma expands to multiple pitches
    a = LADDER_INDEX["G4"]
    # kentemata = (+1,-1): from G4 -> A4, G4
    p = pitches_from(a, ["kentemata"])
    check("kentemata -> 2 pitches A4 G4", p == ["A4", "G4"])
    # petaste_qualitative = (+1,+1,-1): G4 -> A4 B4 A4
    p = pitches_from(a, ["petaste_qualitative"])
    check("petaste_qualitative -> A4 B4 A4", p == ["A4", "B4", "A4"])
    # mixed: oligon then kentemata: G4->A4 (oligon), then A4->B4->A4 (kentemata)
    p = pitches_from(a, ["oligon", "kentemata"])
    check("oligon+kentemata -> A4 B4 A4", p == ["A4", "B4", "A4"])
    # breaths emit nothing
    p = pitches_from(a, ["oligon", "breath_mark_m", "apostrophos"])
    check("breath emits no pitch", p == ["A4", "G4"])

    # melisma ratio: a walk should emit MORE pitches than neumes (that's the point)
    neu = gen_neumes(42, 12)
    pit = pitches_from(LADDER_INDEX["G4"], neu)
    check("melisma walk has pitches > neumes", len(pit) > len(neu))

    # determinism
    check("deterministic gen", gen_neumes(7, 10) == gen_neumes(7, 10))

    # every generated walk is on-ladder for its chosen anchor (build-time invariant)
    good = 0
    for s in range(200):
        neu = gen_neumes(1000 + s, 10)
        for anchor in range(LADDER_INDEX["G3"], LADDER_INDEX["A5"]):
            try:
                pitches_from(anchor, neu); good += 1; break
            except ValueError:
                continue
    check("all 200 sample walks placeable on ladder", good == 200)

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "sft_synth_melisma.jsonl"))
    ap.add_argument("--n", type=int, default=2000, help="number of walks (each -> 2 rows)")
    ap.add_argument("--seed-start", type=int, default=30_000_000)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    build(Path(args.out), args.n, args.seed_start)


if __name__ == "__main__":
    main()
