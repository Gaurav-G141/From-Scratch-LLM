#!/usr/bin/env python3
"""Build Byzantine-neume SFT tasks from named-neume + OMR-pitch data.

Both sides are now REAL, deterministically-extracted data:
  - Byzantine side: named neume sequences (scripts/extract_neumes.py, official EZ table)
  - Western side:   exact pitches (scripts/omr_extract_western.py, Audiveris)

Because neumes and notes do NOT align 1:1 (Byzantine notation interleaves non-pitch
modifiers/martyria), we do NOT claim per-note transcription. Instead we build tasks that
are correct at the sequence/aggregate level:

  neume_read     given the raw neume codes, output the named neume sequence
  neume_count    given a neume sequence, count quantitative vs modifier neumes
  neume_to_west  given a full hymn's neume sequence, output the parallel Western pitch
                 sequence (sequence-to-sequence; the model learns the mapping, we don't
                 hand-align). Labels are the real OMR pitches for the SAME hymn.
  west_to_neume  the reverse direction (Western pitches -> neume sequence), same pair.
  mode_from_neumes  given a neume sequence, identify the mode (from PDF title)

Usage:
  python scripts/build_neume_tasks.py --out data/byzantine/sft_neume.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"

SYSTEM_PROMPT = (
    "You are a Byzantine chant notation assistant. You work with Byzantine neume "
    "sequences (ison, oligon, petaste, apostrophos, gorgon, martyria, ...) and their "
    "Western staff-notation transcriptions. Output the answer only — no commentary."
)

# Mode normalizer. Byzantine has 8 modes: authentic 1-4, plagal 1/2/4, and grave (varys).
# CRITICAL: plagal and grave must be matched BEFORE the plain authentic patterns, else
# "Plagal First Mode" wrongly collapses to "Mode 1" (this bug affected ~151 rows).
WORD_TO_NUM = {"first": 1, "second": 2, "third": 3, "fourth": 4,
               "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}
# Grave / Varys = the 7th mode (grave). Emitted as "Mode grave".
MODE_RE_GRAVE = re.compile(r"\b(grave|var[yi]s|βαρ[υύ]ς)\b", re.I)
# "Plagal First/Second/Fourth [Mode|Tone]" or "Plagal 1/2/4" or "Mode pl. N"
MODE_RE_PLAGAL_WORD = re.compile(
    r"\bplagal\s+(first|second|third|fourth|[1-4])\b", re.I)
MODE_RE_PLAGAL_ABBR = re.compile(r"\bMode\s+pl\.?\s*([1-4])\b", re.I)
# Authentic: "Mode N" (not pl.) or "First/…/Fourth Tone|Mode"
MODE_RE_NUM = re.compile(r"\bMode\s+([1-8])\b", re.I)
MODE_RE_WORD = re.compile(
    r"\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)\s+(Tone|Mode)", re.I)


def norm_mode(text: str) -> str | None:
    # 1) plagal (abbreviated then spelled-out) — must precede authentic checks
    m = MODE_RE_PLAGAL_ABBR.search(text)
    if m:
        return f"Mode pl. {m.group(1)}"
    m = MODE_RE_PLAGAL_WORD.search(text)
    if m:
        deg = m.group(1).lower()
        return f"Mode pl. {WORD_TO_NUM.get(deg, deg)}"
    # 2) grave / varys (7th mode)
    if MODE_RE_GRAVE.search(text):
        return "Mode grave"
    # 3) authentic numeric / spelled-out
    m = MODE_RE_NUM.search(text)
    if m:
        return f"Mode {m.group(1)}"
    m = MODE_RE_WORD.search(text)
    if m:
        return f"Mode {WORD_TO_NUM[m.group(1).lower()]}"
    return None


def pdf_mode(pdf_path: str) -> str | None:
    try:
        doc = fitz.open(pdf_path)
        t = doc[0].get_text()
        doc.close()
    except Exception:  # noqa: BLE001
        return None
    return norm_mode(t)


def load_neumes() -> dict:
    out = {}
    # Font-based extraction (high fidelity) takes precedence.
    for src in ["goa-dcs", "new-byzantium", "st-anthonys"]:
        p = DATA / f"neumes_{src}.jsonl"
        if not p.exists():
            continue
        for r in (json.loads(l) for l in p.open() if l.strip()):
            if r["n_neumes"] == 0:
                continue  # skip empty rows so vector extraction can fill these gaps
            stem = re.sub(r"_byz$", "", Path(r["path"]).stem)
            out[stem] = r
    # Vector-path extraction (lower fidelity, ~84% named) fills in files with no font
    # neumes. Only add if not already present from a font source.
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


QUANT = {"ison", "oligon", "petaste", "apostrophos", "elaphron", "chamile", "kentemata"}

# Neume/pitch windowing. Whole hymns are long (median 130 neumes / 173 pitches, up to
# 1800+) and neumes:pitches align ~1.78:1 (melismatic — no clean per-neume alignment).
# We therefore emit WHOLE-HYMN pairs split into proportional windows: a window of neumes
# is paired with the proportionally-corresponding window of pitches. This bounds sequence
# length, removes the old fixed-60 truncation artifact (which taught "always emit 60
# tokens"), and multiplies training signal — without asserting false per-neume alignment.
WINDOW_NEUMES = 24          # neumes per training window
MIN_WINDOW = 6              # skip trailing scraps shorter than this
MAX_TARGET = 120            # skip pathologically melismatic windows (pitch:neume blowout)


def _ison_of(pitches: list[str]) -> str:
    """Ison (drone) proxy = most common pitch in the hymn."""
    return Counter(pitches).most_common(1)[0][0] if pitches else ""


def _collapse_repeats(tokens: list[str], max_run: int = 3) -> list[str]:
    """Clamp any token repeated more than max_run times in a row (kills the
    degenerate `measure_bar measure_bar ...` / repeated-token loops in some
    west_to_neume targets)."""
    out: list[str] = []
    run = 0
    for t in tokens:
        if out and t == out[-1]:
            run += 1
            if run >= max_run:
                continue
        else:
            run = 0
        out.append(t)
    return out


def _windows(neumes: list[str], pitches: list[str]):
    """Yield (neume_window, pitch_window) pairs by proportional slicing."""
    n_n, n_p = len(neumes), len(pitches)
    if n_n < MIN_WINDOW or n_p < MIN_WINDOW:
        return
    n_windows = max(1, round(n_n / WINDOW_NEUMES))
    for i in range(n_windows):
        na, nb = i * n_n // n_windows, (i + 1) * n_n // n_windows
        pa, pb = i * n_p // n_windows, (i + 1) * n_p // n_windows
        nw, pw = neumes[na:nb], pitches[pa:pb]
        if len(nw) >= MIN_WINDOW and len(pw) >= MIN_WINDOW:
            # skip windows where the pitch span dwarfs the neume span (extreme melisma):
            # such pairs teach the model to over-generate and hurt length discipline.
            if len(pw) <= MAX_TARGET and len(nw) <= MAX_TARGET:
                yield nw, pw


def msg(user: str, assistant: str, task: str, rid: str) -> dict:
    return {
        "id": rid,
        "task": task,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def build(out_path: str) -> None:
    neumes = load_neumes()
    omr = load_omr()
    rows: list[dict] = []
    counts: Counter = Counter()

    for stem, nrec in neumes.items():
        seq = [t for t in nrec["neumes"] if not t.startswith("unk_")]
        if len(seq) < 8:
            continue
        base = stem

        # neume_read: raw codes -> named neume sequence (fully deterministic, exact)
        codes = nrec["codes"][:40]
        if len(codes) >= 8:
            names = " ".join(seq[:40])
            rows.append(msg(
                f"Decode this EZ Byzantine neume code string into named neumes:\n{codes}",
                names,
                "neume_read",
                f"{base}_read",
            ))
            counts["neume_read"] += 1

        # mode_from_neumes
        mode = pdf_mode(nrec["path"])
        if mode:
            rows.append(msg(
                "Identify the mode of this Byzantine chant from its neumes:\n"
                + " ".join(seq[:30]),
                mode,
                "mode_from_neumes",
                f"{base}_mode",
            ))
            counts["mode_from_neumes"] += 1

        # Bidirectional transcription between the same hymn's neume sequence and its
        # parallel OMR Western pitches, split into proportional windows. Targets carry the
        # Mode/Ison header (matching the eval reference format) so the model learns to emit
        # it. west_to_neume targets are repeat-collapsed to kill degeneration loops.
        if stem in omr and len(omr[stem]) >= MIN_WINDOW:
            pitches = omr[stem]
            ison = _ison_of(pitches)
            mode_hdr = mode or "Mode ?"
            for wi, (nw, pw) in enumerate(_windows(seq, pitches)):
                neume_str = " ".join(nw)
                pitch_str = " ".join(pw)
                # target western block: header + ison + pitch line (eval format)
                west_block = f"{mode_hdr}\nIson: {ison}\n{pitch_str}"
                # target byzantine block: header + neume chain (repeat-collapsed)
                byz_block = f"{mode_hdr}\n(Ison {ison})\n" + " | ".join(_collapse_repeats(nw))

                # neume -> west
                rows.append(msg(
                    "Transcribe this Byzantine neume sequence to Western staff pitches:\n"
                    f"{mode_hdr}\n{neume_str}",
                    west_block,
                    "neume_to_west",
                    f"{base}_n2w_{wi}",
                ))
                counts["neume_to_west"] += 1
                # west -> neume (reverse direction, same window)
                rows.append(msg(
                    "Transcribe these Western staff pitches to a Byzantine neume sequence:\n"
                    f"{mode_hdr}\nIson: {ison}\n{pitch_str}",
                    byz_block,
                    "west_to_neume",
                    f"{base}_w2n_{wi}",
                ))
                counts["west_to_neume"] += 1

    rows.sort(key=lambda r: r["id"])
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {out_path}")
    print("By task:", dict(counts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "sft_neume.jsonl"))
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
