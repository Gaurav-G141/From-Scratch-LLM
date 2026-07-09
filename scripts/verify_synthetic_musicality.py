#!/usr/bin/env python3
"""INDEPENDENT verifier for sft_synthetic_musicality.jsonl.

This deliberately re-implements the interval math from scratch (it does NOT import the
generator) so a shared typo cannot hide a bug. The STEP/LADDER tables below are typed
out separately on purpose. For each row it checks:

  1. schema: system/user/assistant present, task in {neume_to_west, west_to_neume}
  2. header format matches the real corpus: "Mode X" / "Ison: <p>" / "(Ison <p>)"
  3. neume vocab is the restricted, guide-backed set only (+ breath no-ops in n2w)
  4. pitches are on the diatonic natural-note ladder (no accidentals/microtones)
  5. the Ison anchor is a real ladder pitch, and the pitch sequence is EXACTLY the
     interval walk of the neumes from that anchor (breaths skipped) — re-derived here
  6. reversibility: the n2w row and its w2n twin describe the identical walk
     (after stripping n2w breath no-ops)

Exit 0 == every row correct by construction. Non-zero == at least one bad row.

Usage:
  python scripts/verify_synthetic_musicality.py data/byzantine/sft_synthetic_musicality.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Independently re-declared (NOT imported). Musical interval -> diatonic degrees.
STEP = {
    "ison": 0, "oligon": 1, "petaste": 1, "apostrophos": -1,
    "oligon_kentema": 3,        # up a fourth
    "oligon_hypsili": 4,        # up a fifth
    "elaphron": -2,             # down a third
    "elaphron_apostrophos": -3, # down a fourth
    "chamile": -4,              # down a fifth
}
BREATH_NOOPS = {"breath_mark_m", "comma_breath", "measure_bar"}

LADDER = [
    "C3", "D3", "E3", "F3", "G3", "A3", "B3",
    "C4", "D4", "E4", "F4", "G4", "A4", "B4",
    "C5", "D5", "E5", "F5", "G5", "A5", "B5",
    "C6",
]
LADDER_IX = {p: i for i, p in enumerate(LADDER)}
MODES = {"Mode 1", "Mode pl. 1", "Mode 4", "Mode pl. 4"}
PITCH_RE = re.compile(r"^[A-G][3-6]$")


def derive(anchor: str, neumes: list[str]) -> list[str]:
    """Pitches for pitch-bearing neumes; breaths are skipped no-ops."""
    ix = LADDER_IX[anchor]
    out = []
    for n in neumes:
        if n in BREATH_NOOPS:
            continue
        ix += STEP[n]
        if not (0 <= ix < len(LADDER)):
            raise ValueError("off-ladder")
        out.append(LADDER[ix])
    return out


def check_n2w(row: dict) -> None:
    # prompt lines: instruction / mode / "Ison: X" / neume_str
    ulines = row["messages"][1]["content"].split("\n")
    alines = row["messages"][2]["content"].split("\n")
    mode = ulines[1]
    um = re.match(r"^Ison: ([A-G][3-6])$", ulines[2])
    assert um, "prompt ison anchor missing/malformed"
    prompt_anchor = um.group(1)
    neumes = ulines[3].split()
    assert mode in MODES, f"unknown mode {mode!r}"
    assert all(n in STEP or n in BREATH_NOOPS for n in neumes), "bad token in user"
    assert alines[0] == mode, "mode header mismatch"
    m = re.match(r"^Ison: ([A-G][3-6])$", alines[1])
    assert m, "ison header malformed"
    anchor = m.group(1)
    assert anchor in LADDER_IX, "anchor off-ladder"
    assert anchor == prompt_anchor, "prompt anchor != target anchor"
    pitches = alines[2].split()
    assert all(PITCH_RE.match(p) and p in LADDER_IX for p in pitches), "off-ladder pitch"
    assert derive(anchor, neumes) == pitches, "pitches != interval walk"


def check_w2n(row: dict) -> None:
    ulines = row["messages"][1]["content"].split("\n")
    alines = row["messages"][2]["content"].split("\n")
    mode = ulines[1]
    assert mode in MODES, f"unknown mode {mode!r}"
    m = re.match(r"^Ison: ([A-G][3-6])$", ulines[2])
    assert m, "ison header malformed"
    anchor = m.group(1)
    pitches = ulines[3].split()
    assert alines[0] == mode, "mode header mismatch"
    assert alines[1] == f"(Ison {anchor})", "ison paren header mismatch"
    neumes = alines[2].split()
    # w2n targets must NOT contain breath no-ops (not recoverable from pitches)
    assert all(n in STEP for n in neumes), "non-interval token in w2n target"
    assert derive(anchor, neumes) == pitches, "neumes don't reproduce pitches"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/byzantine/sft_synthetic_musicality.jsonl"
    rows = [json.loads(l) for l in Path(path).open() if l.strip()]
    bad = 0
    walks: dict[str, dict] = defaultdict(dict)
    for r in rows:
        try:
            assert r["task"] in ("neume_to_west", "west_to_neume"), "bad task"
            assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
            if r["task"] == "neume_to_west":
                check_n2w(r)
                walks[r["id"][:-4]]["n2w"] = r
            else:
                check_w2n(r)
                walks[r["id"][:-4]]["w2n"] = r
        except (AssertionError, ValueError, KeyError, IndexError) as e:
            bad += 1
            if bad <= 10:
                print(f"BAD {r.get('id')}: {e}")

    rev_bad = 0
    for wid, pair in walks.items():
        if "n2w" not in pair or "w2n" not in pair:
            rev_bad += 1
            continue
        n2w_neumes = [t for t in pair["n2w"]["messages"][1]["content"].split("\n")[3].split()
                      if t not in BREATH_NOOPS]
        w2n_neumes = pair["w2n"]["messages"][2]["content"].split("\n")[2].split()
        if n2w_neumes != w2n_neumes:
            rev_bad += 1

    print(f"rows checked: {len(rows)}  walks: {len(walks)}")
    print(f"content errors: {bad}")
    print(f"reversibility errors: {rev_bad}")
    if bad == 0 and rev_bad == 0:
        print("PASS: 100% of rows are correct by construction.")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
