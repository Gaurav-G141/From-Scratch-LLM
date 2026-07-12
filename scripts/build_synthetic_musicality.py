#!/usr/bin/env python3
"""Generate *correct-by-construction* synthetic Byzantine <-> Western musicality tasks.

WHY THIS EXISTS
---------------
Day-3 training hit a wall (docs/byzantine_day3_results_20260708.md): whole-hymn
seq2seq pairs from the real corpus teach contour/format but NOT exact pitch mapping,
because real neumes and OMR pitches align ~1.78:1 (melismatic) — there is no clean
per-neume alignment to learn from.

This generator sidesteps that wall entirely. Byzantine neumes are *intervallic*: the
authoritative guide (docs/byzantine_notation_for_slm.md, Nick Nicholas Unicode
Technical Note 1.1) defines each interval sign as a relative move on the modal ladder.
If we *generate* the melody as a walk over those intervals, the neume sequence and the
pitch sequence are EXACTLY 1:1 aligned BY CONSTRUCTION. Nothing to recover, nothing to
misread — the pairing is a mathematical identity.

  GROUND-TRUTH RULE (governs everything): a neume's pitch action is only ever a fixed
  integer degree-shift that the guide explicitly vouches for. We NEVER read pitch
  values from ez_neume_map.json's `interval` field — the repo's own `_note` refuted it
  against OMR data. Each token below cites the exact guide line.

WHAT THIS VERSION ADDS (deterministic extensions, see the /btw plan)
--------------------------------------------------------------------
1. Wider intervals: leaps up to a fifth in both directions (was: step + a third),
   using guide-vouched combination tokens that are ALSO attested in the real corpus.
2. Transposition augmentation: each walk is re-anchored at every fitting ladder
   position. Neumes are relative, so the sequence is unchanged and pitches shift —
   still exactly re-derivable. Teaches that absolute pitch depends on the ison anchor,
   and multiplies the set severalfold.
3. Structural variety: 4 diatonic modes x many anchors/registers, varied lengths, and
   optional breath/barline PASS-THROUGH tokens (n2w only) that are unambiguous no-ops
   on the pitch cursor — teaching the model to ignore non-pitch signs.

HARD BOUNDARIES (left to the real corpus, cannot be faked deterministically):
  - no microtones / fthora / diesis     (mode-specific comma tuning)
  - no chromatic / enharmonic modes      (II, pl. II, III, Varys)
  - no melisma                            (synthetic walks are 1:1 by construction)

Every emitted row is re-derived by an independent checker before writing, and the
companion scripts/verify_synthetic_musicality.py re-checks the file from scratch.

Usage:
  python scripts/build_synthetic_musicality.py --n 3000 --out data/byzantine/sft_synthetic_musicality.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"

# Match the exact system prompt used by the real neume tasks (scripts/build_neume_tasks.py)
# so synthetic rows mix cleanly into the same SFT run.
SYSTEM_PROMPT = (
    "You are a Byzantine chant notation assistant. You work with Byzantine neume "
    "sequences (ison, oligon, petaste, apostrophos, gorgon, martyria, ...) and their "
    "Western staff-notation transcriptions. Output the answer only — no commentary."
)

# ---------------------------------------------------------------------------
# Musical ground truth. Every value is a guide-vouched integer degree-shift.
# Musical interval -> diatonic degrees: 2nd=1, 3rd=2, 4th=3, 5th=4.
# Token names are all ATTESTED in the real corpus (checked), so the synthetic
# vocabulary is a subset of the real one — good for transfer.
# ---------------------------------------------------------------------------
INTERVAL_NEUMES = {
    # step / unison (guide L88-93)
    "ison": 0,                    # repeat previous pitch
    "oligon": +1,                 # up a second, unaccented
    "petaste": +1,                # up a second, accented (same pitch action)
    "apostrophos": -1,            # down a second
    # guide-vouched leaps (combination tokens, exact table values)
    "oligon_kentema": +3,         # L122 "Kentima over oligon = up fourth"
    "oligon_hypsili": +4,         # L124 "Ypsili right/middle of oligon = up fifth"
    "ypsili_left_oligon": +5,     # L126 "Ypsili at left of oligon = up sixth"
    "ypsili_kentima_oligon": +6,  # L128 "Ypsili next to kentima over oligon = up seventh"
    "ypsili_over_kentima_oligon": +7,  # L130 "Ypsili over kentima over oligon = up octave"
    "elaphron": -2,               # L95  "Elaphron = descending third"
    "elaphron_apostrophos": -3,   # L134 "Elaphron over apostrophos = down fourth"
    "chamile": -4,                # L96  "Hamili/Chamili = descending fifth"
}

# Deliberately EXCLUDED: bare `kentima`/`ypsili` (guide: "when validly combined" =
# context-dependent -> rule #2 leaves them out). This leaves a +2 gap ascending; we
# accept the honest asymmetry rather than invent a value.

# Breath / barline signs: unambiguous NO-OPS on the pitch cursor. They emit no pitch.
# Used ONLY in the neume->west direction (they are not recoverable from pitches, so
# putting them in a west->neume target would teach guessing). Verifier skips them.
BREATH_NOOPS = ["breath_mark_m", "comma_breath", "measure_bar"]

# Duration signs (guide "Duration Rules", L157-174): they extend the PRECEDING note's
# beat count; they do NOT move the pitch cursor. Default note = 1 beat. We use only a
# BIJECTIVE subset — each total-beat value maps to exactly ONE token — so the duration
# is exactly recoverable in BOTH directions (unlike breaths). apli=+1 (2 beats total),
# dipli=+2 (3), tetrapli=+4 (5). klasma (+1) and gorgon (fractional) are excluded to
# keep beats<->token a bijection. A duration sign attaches to the last pitch-bearing
# neume and renders inline as "<pitch>:<beats>" when beats>1 (plain "<pitch>" == 1 beat).
DURATION_BEATS = {"apli": 2, "dipli": 3, "tetrapli": 5}
BEATS_TO_DURATION = {v: k for k, v in DURATION_BEATS.items()}  # bijection for w2n

# Diatonic natural-note ladder, C3..C6 (real OMR corpus is >98% naturals, octaves 3-5;
# the wider span gives transposition headroom). No accidentals => no 12-TET microtone
# is ever asserted.
LADDER = [
    "C3", "D3", "E3", "F3", "G3", "A3", "B3",
    "C4", "D4", "E4", "F4", "G4", "A4", "B4",
    "C5", "D5", "E5", "F5", "G5", "A5", "B5",
    "C6",
]
LADDER_INDEX = {p: i for i, p in enumerate(LADDER)}

# Diatonic modes only. Value = the mode's canonical ison/base per the guide's
# "Mode snapshot" (docs/byzantine_notation_for_slm.md). Transposition (below) may
# re-anchor a walk to another ladder position; the emitted Ison line always states the
# true anchor, so every row is self-describing and correct regardless of anchor.
MODES = {
    "Mode 1": "D4",       # Pa, diatonic
    "Mode pl. 1": "A4",   # plagal I, diatonic, A-based
    "Mode 4": "G4",       # diatonic (Di)
    "Mode pl. 4": "C4",   # plagal IV, C/Ni finalis, diatonic
}

# Melodic ambit: keep each walk within an 11th of its own start so it stays singable and
# fits on the ladder at several anchors. Widened from 9 to 11 to give the up-to-octave
# ascending leaps (ypsili combos, +7) room without immediately clamping.
AMBIT = 11
# Transposition anchor band (ladder indices): G3..C5 — a realistic ison register.
ANCHOR_BAND = range(LADDER_INDEX["G3"], LADDER_INDEX["C5"] + 1)

# Weighted move pool (biased to stepwise motion, as real chant is). Leaps appear but
# are rarer. Expanded into a flat list for O(1) sampling.
_MOVE_WEIGHTS = [
    ("apostrophos", 5), ("oligon", 4), ("petaste", 3), ("ison", 2), ("elaphron", 2),
    ("oligon_kentema", 1), ("oligon_hypsili", 1), ("elaphron_apostrophos", 1),
    ("chamile", 1),
    # larger guide-vouched ascending leaps (rarer, as big leaps are in real chant)
    ("ypsili_left_oligon", 1), ("ypsili_kentima_oligon", 1), ("ypsili_over_kentima_oligon", 1),
]
MOVE_POOL = [name for name, w in _MOVE_WEIGHTS for _ in range(w)]
_ASC = [n for n in INTERVAL_NEUMES if INTERVAL_NEUMES[n] > 0]
_DESC = [n for n in INTERVAL_NEUMES if INTERVAL_NEUMES[n] < 0]


def _lcg(x: int) -> int:
    return (x * 1103515245 + 12345) & 0x7FFFFFFF


def gen_neumes(seed: int, length: int) -> tuple[list[str], int, int]:
    """Deterministically build a valid interval-neume walk (names only). Returns
    (neumes, min_offset, max_offset) where offsets are cumulative degrees relative to
    the (as-yet-unfixed) anchor. Ambit-clamped so the walk fits at multiple anchors."""
    x = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    neumes: list[str] = []
    cum = 0
    lo = hi = 0
    for _ in range(length):
        x = _lcg(x)
        name = MOVE_POOL[x % len(MOVE_POOL)]
        deg = INTERVAL_NEUMES[name]
        # ambit clamp: if this move would exceed the melodic ambit, pick a token that
        # moves the opposite way instead (never invent — choose from the real vocab).
        if cum + deg > AMBIT or cum + deg < -AMBIT:
            cand = _DESC if cum > 0 else _ASC
            name = cand[x % len(cand)]
            deg = INTERVAL_NEUMES[name]
            if cum + deg > AMBIT or cum + deg < -AMBIT:
                name, deg = "ison", 0
        cum += deg
        lo, hi = min(lo, cum), max(hi, cum)
        neumes.append(name)
    return neumes, lo, hi


def maybe_insert_durations(neumes: list[str], seed: int) -> list[str]:
    """Deterministically attach duration signs to some pitch-bearing neumes (BOTH
    directions — duration is exactly recoverable). A duration token follows the neume
    whose note it lengthens; never at the start, never two in a row. ~50% of walks get
    durations. Returns a NEW list."""
    x = _lcg((seed ^ 0x2545F4914F6CDD1D) & 0xFFFFFFFF)
    if x % 2 == 0:  # ~50% of walks stay plain (1 beat everywhere)
        return list(neumes)
    dur_tokens = list(DURATION_BEATS)
    out: list[str] = []
    for i, n in enumerate(neumes):
        out.append(n)
        x = _lcg(x)
        # attach a duration to ~1 in 4 interior pitch-bearing neumes
        if i > 0 and x % 4 == 0:
            out.append(dur_tokens[x % len(dur_tokens)])
    return out


def maybe_insert_breaths(neumes: list[str], seed: int) -> list[str]:
    """Deterministically sprinkle breath/barline no-ops into a neume line (n2w only).
    Returns a NEW list; pitch derivation must skip these tokens."""
    x = _lcg((seed ^ 0x5DEECE66D) & 0xFFFFFFFF)
    if x % 5 < 2:  # ~40% of walks get breaths
        return list(neumes)
    out: list[str] = []
    for i, n in enumerate(neumes):
        out.append(n)
        x = _lcg(x)
        # insert a breath after some interior tokens, never two in a row, never at end
        if 0 < i < len(neumes) - 1 and x % 6 == 0:
            out.append(BREATH_NOOPS[x % len(BREATH_NOOPS)])
    return out


def pitches_from(anchor_idx: int, neumes: list[str]) -> list[str]:
    """Emit the pitch for each PITCH-BEARING neume from a sequence that may contain
    breath no-ops (skipped) and duration signs (lengthen the PRECEDING note, no cursor
    move). A note held >1 beat renders as "<pitch>:<beats>"; 1-beat notes as "<pitch>".
    So a duration sign is exactly recoverable from the beats annotation in both dirs."""
    idx = anchor_idx
    out: list[str] = []
    for n in neumes:
        if n in BREATH_NOOPS:
            continue
        if n in DURATION_BEATS:
            if not out:
                raise ValueError("duration sign with no preceding note")
            # replace the last emitted token's beats with this duration's total
            base = out[-1].split(":")[0]
            out[-1] = f"{base}:{DURATION_BEATS[n]}"
            continue
        idx += INTERVAL_NEUMES[n]
        if not (0 <= idx < len(LADDER)):
            raise ValueError("off-ladder")
        out.append(LADDER[idx])
    return out


def make_rows(mode: str, anchor_idx: int, neumes_dur: list[str],
              neumes_n2w: list[str], rid: str) -> list[dict]:
    """Both directions for one anchored walk. `neumes_dur` (durations, no breaths) drives
    w2n and pitch derivation; `neumes_n2w` may also contain breath no-ops shown only in
    the n2w prompt. Duration signs appear in both directions (exactly recoverable via the
    <pitch>:<beats> annotation); breaths appear only n2w."""
    ison = LADDER[anchor_idx]
    pitches = pitches_from(anchor_idx, neumes_dur)
    pitch_str = " ".join(pitches)
    n2w_neume_str = " ".join(neumes_n2w)
    w2n_neume_str = " ".join(neumes_dur)

    west_block = f"{mode}\nIson: {ison}\n{pitch_str}"
    byz_block = f"{mode}\n(Ison {ison})\n{w2n_neume_str}"

    # Length cue: the honest INPUT count (neumes shown for n2w, pitches given for w2n),
    # embedded in the instruction line so downstream line indices are unchanged. Teaches
    # the model to scale output to input length instead of emitting a fixed window.
    n_in_neumes = len(neumes_n2w)
    n_in_pitches = len(pitches)

    return [
        {
            "id": f"{rid}_n2w", "task": "neume_to_west", "synthetic": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Transcribe this Byzantine neume sequence ({n_in_neumes} neumes) to Western staff pitches:\n"
                    # Ison anchor given: neumes are RELATIVE, so absolute pitch is only
                    # determined once the starting anchor is known (transposition re-anchors
                    # the same neumes to different isons). This is a starting reference, not
                    # the answer. Matches the real eval format ("Ni = Γα") and the w2n dir.
                    f"{mode}\nIson: {ison}\n{n2w_neume_str}"},
                {"role": "assistant", "content": west_block},
            ],
        },
        {
            "id": f"{rid}_w2n", "task": "west_to_neume", "synthetic": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Transcribe these Western staff pitches ({n_in_pitches} pitches) to a Byzantine neume sequence:\n"
                    f"{mode}\nIson: {ison}\n{pitch_str}"},
                {"role": "assistant", "content": byz_block},
            ],
        },
    ]


def verify_walk(mode: str, anchor_idx: int, neumes_dur: list[str],
                neumes_n2w: list[str]) -> None:
    """Raise if the anchored walk is not internally consistent. Runs on EVERY walk."""
    assert all(n in INTERVAL_NEUMES or n in DURATION_BEATS for n in neumes_dur), "unknown neume/duration"
    assert all(n in INTERVAL_NEUMES or n in DURATION_BEATS or n in BREATH_NOOPS
               for n in neumes_n2w), "bad n2w token"
    # n2w with breaths stripped must equal the duration-bearing sequence (same music)
    assert [n for n in neumes_n2w if n not in BREATH_NOOPS] == neumes_dur, "breath desync"
    # duration signs never lead and never repeat (each attaches to a real preceding note)
    for i, n in enumerate(neumes_dur):
        if n in DURATION_BEATS:
            assert i > 0 and neumes_dur[i - 1] not in DURATION_BEATS, "dangling duration"
    pitches = pitches_from(anchor_idx, neumes_dur)
    n_pitch_bearing = sum(1 for n in neumes_dur if n in INTERVAL_NEUMES)
    assert len(pitches) == n_pitch_bearing, "length mismatch"
    # each pitch token is "<ladder pitch>" or "<ladder pitch>:<beats>"; beats must be a
    # value that maps back to exactly one duration sign (bijection => reversible w2n)
    for p in pitches:
        parts = p.split(":")
        assert parts[0] in LADDER_INDEX, "off-ladder pitch"
        if len(parts) == 2:
            assert int(parts[1]) in BEATS_TO_DURATION, "unrecoverable beats"


def _signatures_from_file(path: Path) -> set[tuple]:
    """Reconstruct the (mode, neume-tuple, anchor_idx) signatures from an existing
    dataset so a new run can EXCLUDE them. Used to guarantee a held-out slice shares no
    walk with the training set, even if disjoint seeds coincidentally collide."""
    sigs: set[tuple] = set()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("task") != "west_to_neume":
            continue  # w2n assistant carries the clean neume line + anchor
        alines = r["messages"][2]["content"].split("\n")
        mode = alines[0]
        anchor = alines[1].removeprefix("(Ison ").rstrip(")")
        neumes = tuple(alines[2].split())
        if anchor in LADDER_INDEX:
            sigs.add((mode, neumes, LADDER_INDEX[anchor]))
    return sigs


def build(out_path: str, n_walks: int, min_len: int, max_len: int,
          max_transpose: int, seed_start: int = 0, exclude_path: str | None = None) -> None:
    rows: list[dict] = []
    seen: set[tuple] = set()          # (mode, neume-tuple, anchor_idx)
    if exclude_path:
        seen = _signatures_from_file(Path(exclude_path))
        print(f"Excluding {len(seen)} walk signatures from {exclude_path}")
    seed = seed_start
    kept_walks = 0
    kept_anchored = 0
    attempts = 0

    while kept_walks < n_walks:
        attempts += 1
        if attempts > n_walks * 200:
            print(f"WARN: stopping early at {kept_walks}/{n_walks} walks")
            break
        length = min_len + (seed % (max_len - min_len + 1))
        mode = list(MODES.keys())[seed % len(MODES)]
        neumes_plain, lo, hi = gen_neumes(seed, length)
        # duration signs go in BOTH directions (exactly recoverable); breaths only n2w.
        neumes_dur = maybe_insert_durations(neumes_plain, seed)
        neumes_n2w = maybe_insert_breaths(neumes_dur, seed)
        base_seed = seed
        seed += 1

        # candidate anchors: canonical first (if it fits), then the rest of the band.
        canon = LADDER_INDEX[MODES[mode]]
        candidates = [canon] + [i for i in ANCHOR_BAND if i != canon]
        fitting = [i for i in candidates if i + lo >= 0 and i + hi < len(LADDER)]
        if not fitting:
            continue
        chosen = fitting[:max_transpose]

        emitted_any = False
        for ti, anchor_idx in enumerate(chosen):
            sig = (mode, tuple(neumes_plain), anchor_idx)
            if sig in seen:
                continue
            try:
                verify_walk(mode, anchor_idx, neumes_dur, neumes_n2w)
            except (AssertionError, ValueError):
                continue
            seen.add(sig)
            rid = f"synth_{base_seed:06d}_t{ti}"
            rows.extend(make_rows(mode, anchor_idx, neumes_dur, neumes_n2w, rid))
            kept_anchored += 1
            emitted_any = True
        if emitted_any:
            kept_walks += 1

    rows.sort(key=lambda r: r["id"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_task = Counter(r["task"] for r in rows)
    by_mode = Counter(r["messages"][1]["content"].split("\n")[1] for r in rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    print(f"  base walks: {kept_walks}   anchored walks (x2 dirs): {kept_anchored}")
    print(f"  by task: {dict(by_task)}")
    print(f"  by mode: {dict(by_mode)}")
    print(f"All {kept_anchored} anchored walks passed independent re-derivation vetting.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "sft_synthetic_musicality.jsonl"))
    ap.add_argument("--n", type=int, default=3000, help="number of unique base walks")
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--max-transpose", type=int, default=4,
                    help="max anchors (transpositions) per base walk")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed (use a large offset for a disjoint held-out slice)")
    ap.add_argument("--exclude", default=None,
                    help="JSONL whose walk signatures to skip (e.g. the training set), "
                         "guaranteeing zero overlap for a held-out slice")
    args = ap.parse_args()
    build(args.out, args.n, args.min_len, args.max_len, args.max_transpose,
          seed_start=args.seed_start, exclude_path=args.exclude)


if __name__ == "__main__":
    main()
