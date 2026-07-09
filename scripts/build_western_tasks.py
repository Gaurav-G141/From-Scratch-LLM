#!/usr/bin/env python3
"""Build Western-music SFT tasks from OMR-extracted pitch data.

Input: data/byzantine/omr/omr_{goa,newbyz,sam}.jsonl — deterministically extracted
(Audiveris + music21) per-staff pitch sequences. These labels are exact, unlike the
discarded vision-extracted transcription data.

All tasks are transposition/octave-invariant OR anchor to metadata read from the PDF
title, so the constant clef offset (Audiveris reads plain treble; scores are treble-8)
does not affect correctness.

Task types:
  mode_id        given a melody, identify the Byzantine mode/tone (label from title)
  continuation   given the opening N notes, predict the next M notes
  contour        given a melody, output its up/down/same interval contour
  transpose      transpose a melody by a stated interval

Output: chat-format JSONL (system/user/assistant) ready for train_byzantine_sft.py.

Usage:
  python scripts/build_western_tasks.py --out data/byzantine/sft_western.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
OMR_DIR = ROOT / "data" / "byzantine" / "omr"

NAMES = "CDEFGAB"

SYSTEM_PROMPT = (
    "You are a Western music theory assistant working with Byzantine chant melodies "
    "written in staff notation. Answer using pitch names (C, D, E, ... with octave "
    "numbers) and standard interval/mode terminology. Output the answer only — no "
    "commentary."
)

# Byzantine has 8 modes: authentic 1-4, plagal 1/2/4, grave (varys, the 7th).
# Plagal and grave MUST be matched before the plain authentic patterns.
WORD_TO_NUM = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
}
MODE_RE_GRAVE = re.compile(r"\b(grave|var[yi]s|βαρ[υύ]ς)\b", re.I)
MODE_RE_PLAGAL_WORD = re.compile(r"\bplagal\s+(first|second|third|fourth|[1-4])\b", re.I)
MODE_RE_PLAGAL_ABBR = re.compile(r"\bMode\s+pl\.?\s*([1-4])\b", re.I)
MODE_RE_NUM = re.compile(r"\bMode\s+([1-8])\b", re.I)
MODE_RE_WORD = re.compile(
    r"\b(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)\s+(Tone|Mode)", re.I
)


def normalize_mode(text: str) -> str | None:
    """Return a canonical mode label like 'Mode 1' / 'Mode pl. 2' / 'Mode grave'."""
    m = MODE_RE_PLAGAL_ABBR.search(text)
    if m:
        return f"Mode pl. {m.group(1)}"
    m = MODE_RE_PLAGAL_WORD.search(text)
    if m:
        deg = m.group(1).lower()
        return f"Mode pl. {WORD_TO_NUM.get(deg, deg)}"
    if MODE_RE_GRAVE.search(text):
        return "Mode grave"
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
        text = doc[0].get_text()
        doc.close()
    except Exception:  # noqa: BLE001
        return None
    return normalize_mode(text)


def midi(p: str) -> int:
    m = re.match(r"([A-G])([#b]?)(\d)", p)
    if not m:
        return 0
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m.group(1)]
    acc = 1 if m.group(2) == "#" else (-1 if m.group(2) == "b" else 0)
    return base + acc + 12 * (int(m.group(3)) + 1)


def contour(seq: list[str]) -> list[str]:
    out = []
    for a, b in zip(seq, seq[1:]):
        d = midi(b) - midi(a)
        out.append("U" if d > 0 else ("D" if d < 0 else "S"))
    return out


def transpose(seq: list[str], semitones: int) -> list[str]:
    out = []
    for p in seq:
        m = midi(p) + semitones
        octave = m // 12 - 1
        pc = m % 12
        # prefer sharps
        name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][pc]
        out.append(f"{name}{octave}")
    return out


def row_msgs(user: str, assistant: str, task: str, meta: dict) -> dict:
    return {
        "id": meta["id"],
        "task": task,
        "source": meta["source"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def clean_staff(seq: list[str]) -> list[str]:
    # keep only well-formed pitch tokens
    return [p for p in seq if re.match(r"^[A-G][#b]?\d$", p)]


def build(args) -> None:
    rows: list[dict] = []
    counts: Counter = Counter()

    for src, tag in [("goa", "goa_dcs"), ("newbyz", "new_byzantium"), ("sam", "st_anthonys")]:
        path = OMR_DIR / f"omr_{src}.jsonl"
        if not path.exists():
            continue
        for rec in (json.loads(l) for l in path.open() if l.strip()):
            if rec.get("status") != "ok" or rec.get("n_notes", 0) < 8:
                continue
            base_id = Path(rec["path"]).stem
            mode = pdf_mode(rec["path"])
            # merge staves into one melody (reading order) + keep per-staff phrases
            staves = [clean_staff(s) for s in rec.get("staves", [])]
            staves = [s for s in staves if len(s) >= 6]
            if not staves:
                continue
            melody = [p for s in staves for p in s]

            # --- mode_id (whole melody -> mode) ---
            if mode and len(melody) >= 8:
                snippet = " ".join(melody[:24])
                rows.append(row_msgs(
                    f"Identify the Byzantine mode of this melody:\n{snippet}",
                    mode,
                    "mode_id",
                    {"id": f"{base_id}_modeid", "source": tag},
                ))
                counts["mode_id"] += 1

            # --- continuation (opening -> next notes), one per staff/phrase ---
            for i, s in enumerate(staves):
                if len(s) >= 10:
                    k = len(s) // 2
                    rows.append(row_msgs(
                        "Continue this chant melody with the next "
                        f"{min(6, len(s)-k)} notes:\n{' '.join(s[:k])}",
                        " ".join(s[k:k + 6]),
                        "continuation",
                        {"id": f"{base_id}_cont{i}", "source": tag},
                    ))
                    counts["continuation"] += 1

            # --- contour (melody -> U/D/S string) ---
            if len(melody) >= 8:
                seq = melody[:20]
                rows.append(row_msgs(
                    "Give the melodic contour of this phrase as a sequence of "
                    "U (up), D (down), S (same):\n" + " ".join(seq),
                    " ".join(contour(seq)),
                    "contour",
                    {"id": f"{base_id}_contour", "source": tag},
                ))
                counts["contour"] += 1

            # --- transpose (melody + interval -> transposed) ---
            if len(melody) >= 8:
                seq = melody[:12]
                semis = 2  # up a whole tone; deterministic, verifiable
                rows.append(row_msgs(
                    "Transpose this melody up by a major second (2 semitones):\n"
                    + " ".join(seq),
                    " ".join(transpose(seq, semis)),
                    "transpose",
                    {"id": f"{base_id}_transpose", "source": tag},
                ))
                counts["transpose"] += 1

    # shuffle deterministically by id hash-free: sort by id for reproducibility
    rows.sort(key=lambda r: r["id"])

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} task rows -> {out}")
    print("By task:", dict(counts))
    print("By source:", dict(Counter(r["source"] for r in rows)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/byzantine/sft_western.jsonl"))
    build(ap.parse_args())


if __name__ == "__main__":
    main()
