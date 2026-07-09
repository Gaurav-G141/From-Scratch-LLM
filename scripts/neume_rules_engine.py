#!/usr/bin/env python3
"""Deterministic neume→pitch engine from byzantine_notation_for_slm.md rules.

Byzantine notation is RELATIVE: each quantitative neume moves the running pitch by a
number of modal degrees (ison=0, oligon=+1, apostrophos=-1, elaphron=-2, ...). Degrees
map to diatonic pitch letters (Ni=C, Pa=D, Vou=E, Ga=F, Di=G, Ke=A, Zo=B). Given a
starting degree (from martyria/Ni) we can compute the pitch sequence.

This is the theory engine the project has repeatedly said was the missing piece. If its
output matches the OMR pitches on real hymns, it can generate per-neume-aligned training
targets — the real fix for melodic_equivalence.

Scope note: this implements the QUANTITATIVE (interval) neumes only. Time/quality/martyria
neumes carry no pitch (step 0). Melisma (one neume held over several notes) and microtonal
genus detail are NOT modeled — so exact match to OMR is not expected; we measure how close.
"""

from __future__ import annotations

# Modal degree per neume NAME (as emitted by scripts/extract_neumes.py / ez_neume_map.json).
# Values are diatonic-degree steps from the previous pitch. None = non-pitch (skip).
STEP = {
    "ison": 0,
    "oligon": +1, "petaste": +1, "oxeia": +1, "koufisma": +1, "pelaston": +1,
    "kentemata": +1,          # ascending step (soft/tied)
    "kentema": +2,            # upward third-ish leap modifier (approx +2 degrees)
    "ypsili": +4,             # upward fifth-ish
    "apostrophos": -1,
    "hyporroi": -2,           # two descending steps
    "elaphron": -2,           # descending third-ish (-2 degrees)
    "chamile": -4,            # descending fifth-ish
}

# name-prefix families (extractor emits variants like apostrophos_2, oligon_kentema, etc.)
def base_step(name: str):
    n = name.lower()
    # composite oligon+kentema style names -> treat as oligon step (+1) unless kentema dominant
    for key in ("oligon_kentema", "kentemata_support", "oligon_with_kentema"):
        if n.startswith(key):
            return +1
    for key, val in STEP.items():
        if n.startswith(key):
            return val
    return None  # non-pitch (martyria, breath, gorgon, apli, psifiston, etc.)


NAMES = "CDEFGAB"
NI_DEGREE = {"Ni": 0, "Pa": 1, "Vou": 2, "Ga": 3, "Di": 4, "Ke": 5, "Zo": 6}
# martyria neume names in our vocab hint at a degree; default to Pa (D) if unknown.
DEFAULT_START_LETTER = "D"   # Pa, the most common Byzantine base


def _letter_to_index(letter: str, octave: int = 4) -> int:
    return NAMES.index(letter) + 7 * octave


def _index_to_pitch(idx: int) -> str:
    return f"{NAMES[idx % 7]}{idx // 7}"


def neumes_to_pitches(neumes: list[str], start_letter: str = DEFAULT_START_LETTER,
                      start_octave: int = 4) -> list[str]:
    """Compute the Western pitch sequence from a neume-name sequence."""
    cur = _letter_to_index(start_letter, start_octave)
    out: list[str] = []
    started = False
    for name in neumes:
        step = base_step(name)
        if step is None:
            continue  # non-pitch neume
        if not started:
            # The first pitch-bearing neume SOUNDS at the starting degree (the martyria/Ni
            # sets where you begin); its interval is relative to that start, applied as the
            # note itself. Convention: ison=start, a step neume = start+step for the FIRST
            # emitted note.
            started = True
            cur = cur + step
            out.append(_index_to_pitch(cur))
            continue
        cur += step
        out.append(_index_to_pitch(cur))
    return out


if __name__ == "__main__":
    # tiny smoke test
    seq = ["ison", "oligon", "oligon", "apostrophos", "kentemata", "elaphron"]
    print(seq)
    print(neumes_to_pitches(seq, "D"))
