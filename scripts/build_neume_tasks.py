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

        # Bidirectional transcription (seq2seq) between the same hymn's neume sequence
        # and its parallel OMR Western pitches. Both directions from one paired source.
        if stem in omr and len(omr[stem]) >= 8:
            src_n = " ".join(seq[:60])
            tgt_p = " ".join(omr[stem][:60])
            # neume -> west
            rows.append(msg(
                "Transcribe this Byzantine neume sequence to Western staff pitches:\n"
                + src_n,
                tgt_p,
                "neume_to_west",
                f"{base}_n2w",
            ))
            counts["neume_to_west"] += 1
            # west -> neume (reverse direction, same pair)
            rows.append(msg(
                "Transcribe these Western staff pitches to a Byzantine neume sequence:\n"
                + tgt_p,
                src_n,
                "west_to_neume",
                f"{base}_w2n",
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
