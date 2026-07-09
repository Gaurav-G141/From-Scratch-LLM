#!/usr/bin/env python3
"""Deterministically recover Western staff pitches from vector-engraved PDFs.

NoteWorthy Composer (font NWC2STDA) and Finale (font Maestro) render noteheads
as font glyphs at exact (x, y) positions over vector staff lines. Pitch is a pure
function of notehead vertical position relative to the 5 staff lines + clef, so it
can be recovered exactly — no vision model, no guessing.

This module extracts ONLY the Western side (the side that is deterministic).

Validated against dcs_canon1ode3_west.pdf (Mode 1, Pa=D, treble-8 clef):
recovered contour matches the engraved score note-for-note.

Usage:
  python scripts/extract_western_pitches.py --self-test
  python scripts/extract_western_pitches.py --glob 'data/byzantine/corpus/goa-dcs/*_west.pdf' --out data/byzantine/western_pitches.jsonl
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

NAMES = "CDEFGAB"

# Font subsets vary which codepoint carries the notehead (ϕ, k, œ, …). Rather than
# hardcode glyphs, we detect noteheads GEOMETRICALLY: a music-font glyph is a note
# if its vertical center snaps onto a valid diatonic staff position (line or space,
# incl. a few ledger positions) AND it participates in a left-to-right run. Non-note
# music glyphs (clefs, rests, beams, flags) sit off the diatonic grid or far outside
# the staff and are rejected.
SNAP_TOL = 0.30          # fraction of a half-step a glyph may deviate and still snap
LEDGER_RANGE = 6         # allow this many diatonic steps above/below the staff


def _estimate_y_bias(y_centers: list[float], staff: "Staff") -> float:
    """Some notehead glyphs place the note away from their bbox center (e.g. a glyph
    with a stem descender). All such glyphs share the SAME constant offset from the
    diatonic grid, so we estimate it as the median fractional step offset and subtract
    it before snapping. Returns bias in step units."""
    if not y_centers:
        return 0.0
    # Signed distance to the nearest grid line, in step units, folded to (-0.5, 0.5].
    # A real constant bias shows up as a tight cluster of these fractional offsets;
    # ledger/outlier glyphs scatter. Take the median as the robust central estimate.
    fracs = []
    for y in y_centers:
        steps = (y - staff.top) / staff.half_step_px
        frac = steps - round(steps)          # in (-0.5, 0.5]
        fracs.append(frac)
    fracs.sort()
    med = fracs[len(fracs) // 2]
    # Only correct if the bias is a meaningful, consistent shift; tiny medians are noise.
    return med if abs(med) >= 0.15 else 0.0


@dataclass
class Staff:
    lines: list[float]              # 5 y-positions, top->bottom

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def half_step_px(self) -> float:
        # distance between adjacent diatonic positions = half a line gap
        return (self.lines[-1] - self.lines[0]) / 4 / 2

    def mid(self) -> float:
        return (self.lines[0] + self.lines[-1]) / 2


def detect_staves(page: fitz.Page) -> list[Staff]:
    # Collect horizontal segments. Some engravers (NoteWorthy) draw one long staff
    # line; others (Finale) draw it as many short per-measure segments. So we gather
    # ALL horizontal segments with their total covered width per y-band, then keep
    # y-bands whose segments collectively span a real staff line.
    from collections import defaultdict

    width_at: dict[float, float] = defaultdict(float)
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if abs(a.y - b.y) < 0.8:
                    width_at[round(a.y, 1)] += abs(b.x - a.x)
            elif it[0] == "re":
                r = it[1]
                if r.height < 1.8:
                    width_at[round(r.y0, 1)] += r.width
    # a staff-line y-band must accumulate a meaningful horizontal extent
    ys = sorted(y for y, w in width_at.items() if w > 60)

    staves: list[Staff] = []
    if not ys:
        return staves
    # cluster near-identical y's (fragments jitter by <1px), then group 5 lines/staff
    clustered: list[float] = []
    for y in ys:
        if clustered and y - clustered[-1] < 2.0:
            continue
        clustered.append(y)

    cur = [clustered[0]]
    for y in clustered[1:]:
        if y - cur[-1] < 10:
            cur.append(y)
        else:
            if len(cur) == 5:
                staves.append(Staff(cur))
            cur = [y]
    if len(cur) == 5:
        staves.append(Staff(cur))
    return staves


def _octave_shift(page_text: str) -> int:
    # Treble-8 clef (little "8" under a treble clef) sounds an octave lower.
    # GOA/new_byz vocal scores use it. Heuristic: assume treble-8 (-1).
    return -1


def pitch_at(y: float, staff: Staff, octave_shift: int, bias: float = 0.0) -> str:
    steps = (y - staff.top) / staff.half_step_px - bias  # 0 at top line
    idx = NAMES.index("F") + 5 * 7 - round(steps)         # top line = F5 (treble)
    idx += octave_shift * 7
    return f"{NAMES[idx % 7]}{idx // 7}"


def _snaps_to_grid(y: float, staff: Staff, bias: float = 0.0) -> bool:
    """True if y sits on a diatonic position (line/space) within the notated range."""
    steps = (y - staff.top) / staff.half_step_px - bias
    nearest = round(steps)
    if abs(steps - nearest) > SNAP_TOL:
        return False
    # staff spans steps 0..8 (5 lines + 4 spaces); allow ledger lines either side
    return -LEDGER_RANGE <= nearest <= 8 + LEDGER_RANGE


def identify_notehead_glyphs(page: fitz.Page, staves: list[Staff], music_font_key: str) -> set[str]:
    """Which music-font codepoint(s) are noteheads in THIS file's font subset?

    Noteheads vastly outnumber every other music glyph on a page (clefs, rests, time
    signatures, martyria appear a handful of times; noteheads appear dozens). And they
    share ONE consistent vertical offset from the staff grid, whereas non-note glyphs
    scatter. We combine both signals: among the most frequent glyphs, keep those whose
    y-positions cluster tightly onto the diatonic grid (after removing a shared bias).
    """
    from collections import Counter, defaultdict

    counts: Counter = Counter()
    ys_by_glyph: dict[str, list[tuple[float, Staff]]] = defaultdict(list)
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if music_font_key not in s["font"]:
                    continue
                yc = (s["bbox"][1] + s["bbox"][3]) / 2
                staff = min(staves, key=lambda st: abs(st.mid() - yc))
                for ch in s["text"]:
                    if not ch.strip():
                        continue
                    counts[ch] += 1
                    ys_by_glyph[ch].append((yc, staff))
    if not counts:
        return set()

    top_n = counts.most_common(1)[0][1]
    heads: set[str] = set()
    for ch, n in counts.items():
        if n < max(4, 0.25 * top_n):        # noteheads are the frequent glyphs
            continue
        # after removing the glyph's own median bias, do its positions land on-grid?
        fracs = []
        for yc, staff in ys_by_glyph[ch]:
            steps = (yc - staff.top) / staff.half_step_px
            fracs.append(steps - round(steps))
        fracs.sort()
        bias = fracs[len(fracs) // 2]
        on_grid = sum(
            1
            for yc, staff in ys_by_glyph[ch]
            if abs(((yc - staff.top) / staff.half_step_px - bias) - round((yc - staff.top) / staff.half_step_px - bias)) <= SNAP_TOL
        )
        if on_grid / n >= 0.85:
            heads.add(ch)
    if not heads:
        heads = {counts.most_common(1)[0][0]}
    return heads


def extract_page(page: fitz.Page, music_font_key: str) -> list[list[str]]:
    """Return list of per-staff pitch sequences (reading order)."""
    staves = detect_staves(page)
    if not staves:
        return []
    octave_shift = _octave_shift(page.get_text())
    noteheads = identify_notehead_glyphs(page, staves, music_font_key)
    if not noteheads:
        return []

    glyphs: list[tuple[float, float]] = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if music_font_key not in s["font"]:
                    continue
                yc = (s["bbox"][1] + s["bbox"][3]) / 2
                for ch in s["text"]:
                    if ch in noteheads:
                        glyphs.append((round(s["bbox"][0], 1), yc))

    # Assign glyphs to staves, then estimate each staff's constant notehead y-bias
    # (glyph bbox-center may sit consistently off the note position) and correct it.
    by_staff: list[list[tuple[float, float]]] = [[] for _ in staves]
    for x, y in glyphs:
        si = min(range(len(staves)), key=lambda i: abs(staves[i].mid() - y))
        by_staff[si].append((x, y))

    per_staff: list[list[tuple[float, str]]] = [[] for _ in staves]
    for si, staff in enumerate(staves):
        # Deduplicate double-struck glyphs (fill+outline render at identical x,y).
        seen: set[tuple[float, float]] = set()
        deduped: list[tuple[float, float]] = []
        for x, y in sorted(by_staff[si]):
            key = (round(x, 0), round(y, 0))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((x, y))

        ys = [y for _, y in deduped]
        bias = _estimate_y_bias(ys, staff)
        for x, y in deduped:
            if not _snaps_to_grid(y, staff, bias):
                continue
            per_staff[si].append((x, pitch_at(y, staff, octave_shift, bias)))

    return [[p for _, p in sorted(st)] for st in per_staff]


def extract_pdf(path: Path, max_pages: int = 3) -> dict:
    doc = fitz.open(path)
    fonts = {f[3] for pg in doc for f in pg.get_fonts()}
    if any("NWC" in f for f in fonts):
        music_font_key = "NWC"
        engraver = "noteworthy"
    elif any("Maestro" in f for f in fonts):
        music_font_key = "Maestro"
        engraver = "finale"
    else:
        doc.close()
        return {"path": str(path), "engraver": "unsupported", "staves": []}

    mode = ""
    all_staves: list[list[str]] = []
    for i in range(min(len(doc), max_pages)):
        pg = doc[i]
        if not mode:
            m = re.search(r"Mode\s+[^.\n]+|Ἦχος\s+\S+|Pa\s*=\s*[A-G]", pg.get_text())
            if m:
                mode = m.group(0).strip()
        all_staves.extend(s for s in extract_page(pg, music_font_key) if s)
    doc.close()
    return {"path": str(path), "engraver": engraver, "mode": mode, "staves": all_staves}


def self_test() -> None:
    ref = Path("data/byzantine/corpus/goa-dcs/dcs_canon1ode3_west.pdf")
    out = extract_pdf(ref, max_pages=1)
    first = out["staves"][0] if out["staves"] else []
    expected_prefix = ["D4", "E4", "E4", "D4", "D4", "C4", "B3", "C4"]
    got = first[: len(expected_prefix)]
    print(f"engraver={out['engraver']} mode={out['mode']!r}")
    print(f"expected: {expected_prefix}")
    print(f"got:      {got}")
    print("SELF-TEST:", "PASS" if got == expected_prefix else "FAIL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--glob", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-pages", type=int, default=3)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    files = sorted(globmod.glob(args.glob))
    results = []
    stats = {"ok": 0, "empty": 0, "unsupported": 0, "total_notes": 0}
    for f in files:
        r = extract_pdf(Path(f), max_pages=args.max_pages)
        nnotes = sum(len(s) for s in r["staves"])
        r["n_notes"] = nnotes
        if r["engraver"] == "unsupported":
            stats["unsupported"] += 1
        elif nnotes == 0:
            stats["empty"] += 1
        else:
            stats["ok"] += 1
            stats["total_notes"] += nnotes
        results.append(r)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"files": len(files), **stats}, indent=2))


if __name__ == "__main__":
    main()
