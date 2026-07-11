#!/usr/bin/env python3
"""DTW contour-alignment of real Byzantine neumes to OMR Western pitches (Stage B).

WHY
---
The real translation SFT pairs in `build_neume_tasks.py` are built by PROPORTIONAL SLICING:
neumes and pitches are cut into the same NUMBER of chunks and paired by position. Because
neumes:pitches run ~1.78:1 (melismatic), chunk k of neumes does NOT correspond to chunk k
of pitches — the position-level labels are wrong, which is why every model trained on them
collapsed to a drone / hallucination (docs/byzantine_curriculum_v2_results_20260711.md).

Stage A (segment at shared landmarks) was ruled out: the OMR pitch side has NO structural
boundaries (pure pitches, no rests/bars), so there is nothing to align neume breath-marks
to. But a DTW feasibility probe on OBSERVED CONTOUR (neume-step SIGN vs pitch-motion SIGN,
letting the warp find the correspondence) scored median 0.77, 60/73 hymns >0.7 — strong.
Melisma is a STRETCHING problem, which is exactly what DTW solves.

WHAT THIS DOES
--------------
Per hymn, DTW-align the PITCH-BEARING neume contour to the pitch-motion contour, giving each
pitch position the neume that governs it. From that alignment we rebuild n2w/w2n windows
where the pitch window is the pitches actually governed by the neume window — a musically
correct pairing, not a proportional guess. Non-pitch neumes (breath/martyria/tempo/duration/
expression) are carried in the neume sequence but excluded from the contour used for warping.

OUTPUT
------
- Per-hymn alignment quality report (sign-agreement on the DTW path).
- NEW SFT files (never overwrites existing data):
    data/byzantine/sft_aligned_n2w.jsonl
    data/byzantine/sft_aligned_w2n.jsonl
  Same schema/prompt format as build_neume_tasks.py, so the training/eval pipeline is
  unchanged. A `dtw_quality` field is added per row for later filtering.

SCOPE / SAFETY
--------------
- Pure Python, no model, no network. Reuses the same JSONL sources as build_neume_tasks.py.
- Emits NEW files; refuses to touch the originals.
- --pilot restricts to the highest-quality hymns first (feasibility gate before a retrain).

Usage:
  python scripts/align_neume_pitch_dtw.py --self-test
  python scripts/align_neume_pitch_dtw.py --report-only          # quality distribution, no write
  python scripts/align_neume_pitch_dtw.py --min-quality 0.6      # build from hymns >=0.6
  python scripts/align_neume_pitch_dtw.py --pilot 60             # build from top-60 hymns only
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

SYSTEM_PROMPT = (
    "You are a Byzantine chant notation assistant. You work with Byzantine neume "
    "sequences (ison, oligon, petaste, apostrophos, gorgon, martyria, ...) and their "
    "Western staff-notation transcriptions. Output the answer only — no commentary."
)

# ---------------------------------------------------------------------------
# Neume -> pitch step. Only PITCH-BEARING families get a step; everything else
# (breath, martyria, tempo gorgon/argon, duration apli/dipli/klasma, expression
# psifiston/antikenoma/homalon/heteron/stavros/endofonon, bars, supports) is a
# NON-PITCH modifier -> contributes NO motion (step=None) and is skipped in the
# contour used for warping, but kept in the neume token sequence.
# Values are diatonic-degree steps (sign is what the DTW cost uses; magnitude
# helps break ties). Matches the synthetic grammar (build_synthetic_musicality.py).
# ---------------------------------------------------------------------------
_STEP_FAMILIES = [
    # (prefix, step)  — longest/most-specific prefixes FIRST
    ("oligon_double_hypsili", +6),
    ("oligon_hypsili", +4),
    ("oligon_kentemata_support", +1),
    ("oligon_kentemata", +1),
    ("oligon_kentema", +3),
    ("oligon_with_kentema_below", +1),
    ("oligon", +1),
    ("petaste_kentema", +3),
    ("petaste_qualitative", +1),
    ("petaste_support", +1),
    ("petaste", +1),
    ("elaphron_apostrophos", -3),
    ("elaphron", -2),
    ("chamile", -4),
    ("apostrophos_support", -1),
    ("apostrophos", -1),
    ("kentemata_support", +1),
    ("kentemata_combo", +1),
    ("kentemata", +1),
    ("kentema", +1),
    ("ison", 0),
    ("endofonon", 0),   # sustained/ison-like
    ("homalon", 0),
]

# Explicit non-pitch prefixes (documented; anything not matching _STEP_FAMILIES is
# treated as non-pitch anyway, but listing makes intent auditable).
_NONPITCH_PREFIXES = (
    "breath_mark", "comma_breath", "period_breath", "slash_breath",
    "martyria", "measure_bar",
    "gorgon", "digorgon", "argon",
    "apli", "dipli", "klasma",
    "psifiston", "antikenoma", "heteron", "stavros",
    "support_combo",
)


def neume_step(tok: str):
    """Return the diatonic step for a pitch-bearing neume, or None if non-pitch."""
    for prefix in _NONPITCH_PREFIXES:
        if tok.startswith(prefix):
            return None
    for prefix, step in _STEP_FAMILIES:
        if tok.startswith(prefix):
            return step
    return None  # unknown -> treat as non-pitch (safe: excluded from contour)


# ---------------------------------------------------------------------------
# Pitch parsing
# ---------------------------------------------------------------------------
_PITCH_DEG = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}


def pitch_degree(p: str):
    """Diatonic degree (letter+octave) for a Western pitch, or None. Accidentals do not
    change the degree (contour is diatonic)."""
    m = re.match(r"^([A-G])[#b-]?(\d)$", p)
    if not m:
        return None
    return _PITCH_DEG[m.group(1)] + 7 * int(m.group(2))


# ---------------------------------------------------------------------------
# Data loaders (mirror build_neume_tasks.py so we align the SAME corpus)
# ---------------------------------------------------------------------------
def load_neumes() -> dict:
    out = {}
    for src in ["goa-dcs", "new-byzantium", "st-anthonys"]:
        p = DATA / f"neumes_{src}.jsonl"
        if not p.exists():
            continue
        for r in (json.loads(l) for l in p.open() if l.strip()):
            if r["n_neumes"] == 0:
                continue
            stem = re.sub(r"_byz$", "", Path(r["path"]).stem)
            out[stem] = r
    vp = DATA / "neumes_vector.jsonl"
    if vp.exists():
        for r in (json.loads(l) for l in vp.open() if l.strip()):
            stem = re.sub(r"_byz$", "", Path(r["path"]).stem)
            if stem not in out and r["n_neumes"] > 0:
                out[stem] = r
    return out


def load_omr() -> dict:
    out = {}
    for src in ["goa", "newbyz", "sam"]:
        p = DATA / "omr" / f"omr_{src}.jsonl"
        if not p.exists():
            continue
        for r in (json.loads(l) for l in p.open() if l.strip()):
            if r.get("status") == "ok" and r.get("n_notes", 0) > 0:
                stem = re.sub(r"_west$", "", Path(r["path"]).stem)
                out[stem] = [p for st in r["staves"] for p in st]
    return out


def pdf_mode(path: str):
    try:
        import fitz
        doc = fitz.open(path)
        t = doc[0].get_text()
        doc.close()
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"mode\s+(pl\.?\s*)?([1-4]|first|second|third|fourth|grave)", t, re.I)
    return m.group(0).strip() if m else None


# ---------------------------------------------------------------------------
# DTW alignment
# ---------------------------------------------------------------------------
def dtw_align(nsteps: list[int], pmotion: list[int]):
    """DTW between the pitch-bearing neume step-signs and the pitch motion-signs.
    Cost 0 when signs agree (both up / both down / both flat), else 1. Returns
    (quality, path) where path is a list of (i, j) index pairs (neume-step i aligned to
    pitch-motion j) and quality = 1 - normalized_path_cost."""
    n, m = len(nsteps), len(pmotion)
    if n == 0 or m == 0:
        return 0.0, []
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    def sgn(x): return (x > 0) - (x < 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if sgn(nsteps[i - 1]) == sgn(pmotion[j - 1]) else 1
            best, move = D[i - 1][j - 1], (i - 1, j - 1)
            if D[i - 1][j] < best:
                best, move = D[i - 1][j], (i - 1, j)
            if D[i][j - 1] < best:
                best, move = D[i][j - 1], (i, j - 1)
            D[i][j] = c + best
            bt[i][j] = move
    # backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = bt[i][j]
    path.reverse()
    quality = 1.0 - D[n][m] / max(n, m)
    return quality, path


def align_hymn(neumes: list[str], pitches: list[str]):
    """DTW-align a hymn. Returns (quality, neume2pitch) where neume2pitch maps each
    FULL-sequence neume index -> sorted list of pitch indices it governs (only for
    pitch-bearing neumes that landed on the warp path). Non-pitch neumes are excluded
    from the contour warp and get no pitch span (they carry no note)."""
    nb_idx = [k for k, t in enumerate(neumes) if neume_step(t) is not None]
    nsteps = [neume_step(neumes[k]) for k in nb_idx]
    pdeg = [pitch_degree(p) for p in pitches]
    valid_p = [k for k, d in enumerate(pdeg) if d is not None]
    pmotion = [pdeg[valid_p[k]] - pdeg[valid_p[k - 1]] for k in range(1, len(valid_p))]
    if len(nsteps) < 3 or len(pmotion) < 3:
        return None, None
    # align neume-steps[1:] (each neume's motion vs its predecessor) to pitch-motions.
    quality, path = dtw_align(nsteps[1:], pmotion)
    # path (i, j): neume-step i  ->  pitch-motion j.
    #   neume-step i corresponds to full-neume index nb_idx[i + 1]
    #   pitch-motion j corresponds to pitch position valid_p[j + 1]
    neume2pitch: dict[int, list[int]] = {}
    # anchor: first pitch-bearing neume governs the first valid pitch
    neume2pitch[nb_idx[0]] = [valid_p[0]]
    for (i, j) in path:
        fn = nb_idx[i + 1]
        pit = valid_p[j + 1]
        neume2pitch.setdefault(fn, []).append(pit)
    for k in neume2pitch:
        neume2pitch[k] = sorted(set(neume2pitch[k]))
    return quality, neume2pitch


# ---------------------------------------------------------------------------
# Report + build
# ---------------------------------------------------------------------------
def compute_qualities():
    neumes, omr = load_neumes(), load_omr()
    shared = sorted(set(neumes) & set(omr))
    results = []
    for stem in shared:
        seq = [t for t in neumes[stem]["neumes"] if not t.startswith("unk_")]
        q, _ = align_hymn(seq, omr[stem])
        if q is not None:
            results.append((stem, q))
    return results, neumes, omr


def load_stem_modes() -> dict:
    """Recover the mode header per stem from the already-built sft_neume.jsonl (assistant
    line 0), so we need no PDF/fitz access. Falls back to 'Mode ?' when unknown."""
    modes = {}
    src = DATA / "sft_neume.jsonl"
    if not src.exists():
        return modes
    for l in src.open(encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("task") == "neume_to_west":
            stem = re.sub(r"_n2w_\d+$", "", r["id"])
            modes.setdefault(stem, r["messages"][2]["content"].split("\n")[0])
    return modes


WINDOW_NEUMES = 24
MIN_WINDOW = 6
MAX_TARGET = 120


def _collapse_repeats(tokens: list[str], max_run: int = 3) -> list[str]:
    out, run = [], 0
    for t in tokens:
        if out and t == out[-1]:
            run += 1
            if run >= max_run:
                continue
        else:
            run = 0
        out.append(t)
    return out


def _msg(user, assistant, task, rid, quality):
    return {
        "id": rid, "task": task, "dtw_quality": round(quality, 4),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def aligned_windows(neumes, pitches, neume2pitch):
    """Yield (neume_window, pitch_window) split at ALIGNED boundaries: cut the neume
    sequence every WINDOW_NEUMES, and take the pitch window as exactly the pitches the
    DTW mapped to those neumes (governed span) — a musically-correct pairing, not a
    proportional guess."""
    n = len(neumes)
    n_windows = max(1, round(n / WINDOW_NEUMES))
    for w in range(n_windows):
        na, nb = w * n // n_windows, (w + 1) * n // n_windows
        nw = neumes[na:nb]
        # pitches governed by any neume in [na, nb)
        pidx = sorted(p for k in range(na, nb) for p in neume2pitch.get(k, []))
        if not pidx:
            continue
        pw = pitches[pidx[0]:pidx[-1] + 1]  # contiguous governed span
        if len(nw) >= MIN_WINDOW and MIN_WINDOW <= len(pw) <= MAX_TARGET and len(nw) <= MAX_TARGET:
            yield nw, pw


def build(min_quality: float, pilot: int):
    results, neumes, omr = compute_qualities()
    results.sort(key=lambda x: -x[1])
    stem_modes = load_stem_modes()

    chosen = [(s, q) for s, q in results if q >= min_quality]
    if pilot > 0:
        chosen = chosen[:pilot]
    print(f"building from {len(chosen)} hymns (min_quality={min_quality}"
          f"{f', pilot top-{pilot}' if pilot else ''})")

    n2w_rows, w2n_rows = [], []
    for stem, q in chosen:
        seq = [t for t in neumes[stem]["neumes"] if not t.startswith("unk_")]
        _, neume2pitch = align_hymn(seq, omr[stem])
        if not neume2pitch:
            continue
        pitches = omr[stem]
        mode_hdr = stem_modes.get(stem, "Mode ?")
        ison = Counter(pitches).most_common(1)[0][0] if pitches else ""
        for wi, (nw, pw) in enumerate(aligned_windows(seq, pitches, neume2pitch)):
            neume_str, pitch_str = " ".join(nw), " ".join(pw)
            west_block = f"{mode_hdr}\nIson: {ison}\n{pitch_str}"
            byz_block = f"{mode_hdr}\n(Ison {ison})\n" + " | ".join(_collapse_repeats(nw))
            n2w_rows.append(_msg(
                f"Transcribe this Byzantine neume sequence ({len(nw)} neumes) to Western staff pitches:\n"
                f"{mode_hdr}\n{neume_str}",
                west_block, "neume_to_west", f"{stem}_n2w_{wi}", q))
            w2n_rows.append(_msg(
                f"Transcribe these Western staff pitches ({len(pw)} pitches) to a Byzantine neume sequence:\n"
                f"{mode_hdr}\nIson: {ison}\n{pitch_str}",
                byz_block, "west_to_neume", f"{stem}_w2n_{wi}", q))

    n2w_rows.sort(key=lambda r: r["id"])
    w2n_rows.sort(key=lambda r: r["id"])
    out_n2w = DATA / "sft_aligned_n2w.jsonl"
    out_w2n = DATA / "sft_aligned_w2n.jsonl"
    for path, rows in [(out_n2w, n2w_rows), (out_w2n, w2n_rows)]:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows -> {path}")
    return out_n2w, out_w2n


def self_test() -> int:
    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    # step map
    check("oligon = +1", neume_step("oligon") == 1)
    check("apostrophos = -1", neume_step("apostrophos") == -1)
    check("elaphron_apostrophos = -3 (specific beats elaphron)", neume_step("elaphron_apostrophos") == -3)
    check("oligon_hypsili = +4 (beats oligon)", neume_step("oligon_hypsili") == 4)
    check("oligon_kentema = +3", neume_step("oligon_kentema") == 3)
    check("ison = 0", neume_step("ison") == 0)
    check("breath_mark_m non-pitch", neume_step("breath_mark_m") is None)
    check("martyria_V non-pitch", neume_step("martyria_V") is None)
    check("gorgon non-pitch", neume_step("gorgon") is None)
    check("apli non-pitch", neume_step("apli") is None)
    check("petaste_qualitative = +1", neume_step("petaste_qualitative") == 1)

    # pitch degree
    check("G4 degree", pitch_degree("G4") == _PITCH_DEG["G"] + 28)
    check("B-4 accidental ignored for degree", pitch_degree("B-4") == pitch_degree("B4"))
    check("bad pitch -> None", pitch_degree("Isole") is None)

    # DTW: identical contour -> quality 1
    q, path = dtw_align([1, -1, 0, 1], [1, -1, 0, 1])
    check("dtw identical = 1.0", abs(q - 1.0) < 1e-9)
    # DTW: melisma stretch (neume up,down vs pitch up,up,down,down) -> should still align well
    q2, _ = dtw_align([1, -1], [1, 1, -1, -1])
    check("dtw absorbs melisma stretch (>=0.5)", q2 >= 0.5)
    # DTW: opposite -> low
    q3, _ = dtw_align([1, 1, 1], [-1, -1, -1])
    check("dtw opposite ~0", q3 < 0.4)

    # align_hymn end-to-end: neumes going up-up-up against pitches C4 D4 E4 F4 should map
    # each pitch-bearing neume to a pitch and preserve monotonic contour.
    neumes = ["ison", "oligon", "oligon", "oligon"]
    pitches = ["C4", "D4", "E4", "F4"]
    q, n2p = align_hymn(neumes, pitches)
    check("align_hymn returns a mapping", isinstance(n2p, dict) and len(n2p) >= 3)
    check("align_hymn high quality on clean ascending", q >= 0.9)
    # every governed pitch index is valid
    allp = sorted(p for v in n2p.values() for p in v)
    check("governed pitch indices in range", all(0 <= p < len(pitches) for p in allp))

    # non-pitch neumes get no pitch span
    neumes2 = ["ison", "breath_mark_m", "oligon", "martyria_V", "apostrophos", "oligon"]
    pitches2 = ["G4", "A4", "G4", "A4"]
    q2b, n2p2 = align_hymn(neumes2, pitches2)
    nonpitch_governed = any(k for k in (n2p2 or {}) if neume_step(neumes2[k]) is None)
    check("non-pitch neumes govern no pitches", not nonpitch_governed)

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true", help="print quality distribution, write nothing")
    ap.add_argument("--min-quality", type=float, default=0.6, help="only build from hymns with DTW quality >= this")
    ap.add_argument("--pilot", type=int, default=0, help="build from the top-N highest-quality hymns only (0=all qualifying)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    results, neumes, omr = compute_qualities()
    results.sort(key=lambda x: -x[1])
    qs = [q for _, q in results]
    import statistics as st
    print(f"hymns aligned: {len(results)}")
    if qs:
        print(f"quality: median={st.median(qs):.3f} mean={st.mean(qs):.3f} "
              f">=0.7: {sum(1 for q in qs if q>=0.7)}  >=0.6: {sum(1 for q in qs if q>=0.6)}")

    if args.report_only:
        print("\ntop 15 hymns by alignment quality:")
        for stem, q in results[:15]:
            print(f"  {q:.3f}  {stem}")
        return

    build(args.min_quality, args.pilot)


if __name__ == "__main__":
    main()
