#!/usr/bin/env python3
"""Extract Byzantine neume sequences from VECTOR-PATH byz PDFs (no neume font).

Some GOA byz PDFs render neumes as filled vector paths instead of font text, so
scripts/extract_neumes.py (which reads font glyphs) skips them. This script recovers
them geometrically:

  1. Collect black filled vector paths (drop the page-background rect and red martyria).
  2. Perceptual-hash each glyph (render -> normalize to ink bbox -> 16x16 -> bit signature)
     and cluster corpus-wide by Hamming distance. ~90 clusters emerge, matching the ~90
     known font glyphs.
  3. Name clusters via triangulated evidence (bitmap similarity to the EZ/ED font glyphs +
     frequency alignment against the known font-neume distribution + manual shape check).
     Only high-confidence clusters are named; the rest are emitted as unk_<id>.
  4. Order glyphs by (y-band, x) into a reading sequence.

Output row shape matches scripts/extract_neumes.py so it drops into build_neume_tasks.py.

NOTE: This is lower-fidelity than the font-based extractor. Naming is reliable for the
core high-frequency neumes (~80% of glyphs); rarer variants/composites are left unk_.
Validate output against the parallel OMR pitch side before trusting it.

Usage:
  python scripts/extract_neumes_vector.py --out data/byzantine/neumes_vector.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"

HASH_SIZE = 16
HAMMING_THRESH = 20  # bits (of 256) to be the same cluster


def is_black(d: dict) -> bool:
    f = d.get("fill")
    return bool(f) and f[0] < 0.3 and f[1] < 0.3 and f[2] < 0.3


def is_glyph_rect(r: fitz.Rect) -> bool:
    return 2 <= r.width <= 70 and 2 <= r.height <= 70


def glyph_hash(page: fitz.Page, r: fitz.Rect) -> int:
    clip = fitz.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1)
    zoom = max(1.0, HASH_SIZE / max(r.width, r.height, 1))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, colorspace=fitz.csGRAY)
    im = Image.frombytes("L", (pix.width, pix.height), pix.samples).resize((HASH_SIZE, HASH_SIZE))
    px = np.array(im) < 128
    bits = 0
    for v in px.flatten():
        bits = (bits << 1) | int(v)
    return bits


def popcount(n: int) -> int:
    return bin(n).count("1")


def build_vector_file_list() -> list[Path]:
    """byz PDFs that have OMR pitches, no neume font, and vector drawings."""
    west = set()
    for src in ["goa", "newbyz", "sam"]:
        p = DATA / "omr" / f"omr_{src}.jsonl"
        for r in (json.loads(l) for l in p.open() if l.strip()):
            if r.get("status") == "ok" and r.get("n_notes", 0) > 0:
                west.add(re.sub(r"_west$", "", Path(r["path"]).stem))
    byz_text = set()
    for src in ["goa-dcs", "new-byzantium", "st-anthonys"]:
        p = DATA / f"neumes_{src}.jsonl"
        if not p.exists():
            continue
        for r in (json.loads(l) for l in p.open() if l.strip()):
            if r["n_neumes"] > 0:
                byz_text.add(re.sub(r"_byz$", "", Path(r["path"]).stem))
    out = []
    for st in sorted(west - byz_text):
        for dd in ["goa-dcs", "new-byzantium", "st-anthonys"]:
            pdf = DATA / "corpus" / dd / f"{st}_byz.pdf"
            if pdf.exists():
                try:
                    doc = fitz.open(pdf)
                    pg = doc[0]
                    ok = (not pg.get_images()) and len(pg.get_drawings()) >= 20
                    doc.close()
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    out.append(pdf)
                break
    return out


def cluster_glyphs(files: list[Path]) -> tuple[list[int], dict]:
    """Return cluster representative hashes and per-file glyph->cluster assignment."""
    reps: list[int] = []
    for pdf in files:
        doc = fitz.open(pdf)
        pg = doc[0]
        for d in pg.get_drawings():
            if not is_black(d) or not is_glyph_rect(d["rect"]):
                continue
            h = glyph_hash(pg, d["rect"])
            best, bd = None, 999
            for i, rh in enumerate(reps):
                hd = popcount(h ^ rh)
                if hd < bd:
                    bd, best = hd, i
            if best is None or bd > HAMMING_THRESH:
                reps.append(h)
        doc.close()
    return reps, {}


def assign_cluster(h: int, reps: list[int]) -> int:
    best, bd = -1, 999
    for i, rh in enumerate(reps):
        hd = popcount(h ^ rh)
        if hd < bd:
            bd, best = hd, i
    return best if bd <= HAMMING_THRESH else -1


def name_by_reference(h: int, ref: list[tuple[int, str]]) -> str:
    """Name a glyph by nearest named-reference hash (canonical, ID-stable)."""
    best, bd = None, 999
    for rh, nm in ref:
        hd = popcount(h ^ rh)
        if hd < bd:
            bd, best = hd, nm
    return best if (best is not None and bd <= HAMMING_THRESH) else "unk"


def extract_file(pdf: Path, ref: list[tuple[int, str]]) -> dict:
    doc = fitz.open(pdf)
    pg = doc[0]
    glyphs = []  # (y, x, name)
    for d in pg.get_drawings():
        if not is_black(d) or not is_glyph_rect(d["rect"]):
            continue
        r = d["rect"]
        glyphs.append((r.y0, r.x0, name_by_reference(glyph_hash(pg, r), ref)))
    doc.close()
    glyphs.sort(key=lambda g: (round(g[0] / 8), g[1]))
    tokens = [nm if nm != "unk" else "unk_x" for _, _, nm in glyphs]
    return {
        "path": str(pdf),
        "n_neumes": len(tokens),
        "neumes": tokens,
        "codes": "",
        "source": "vector",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "neumes_vector.jsonl"))
    ap.add_argument("--names", default="/tmp/cluster_names.json",
                    help="JSON dict cluster_id -> neume name (built during analysis)")
    args = ap.parse_args()

    files = build_vector_file_list()
    print(f"vector-recoverable byz files: {len(files)}")
    ref_raw = json.load(open(args.names))  # [[hash_str, name], ...]
    ref = [(int(h), nm) for h, nm in ref_raw]
    print(f"named reference hashes: {len(ref)}")

    rows = []
    named = total = 0
    for pdf in files:
        r = extract_file(pdf, ref)
        rows.append(r)
        total += r["n_neumes"]
        named += sum(1 for t in r["neumes"] if not t.startswith("unk_"))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"files: {len(rows)}, glyphs: {total}, named: {named} ({100*named/total:.0f}%)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
