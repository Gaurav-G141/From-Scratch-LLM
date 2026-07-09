#!/usr/bin/env python3
"""Extract named Byzantine neume sequences from EZ-font PDFs.

The EZ Byzantine music fonts (EZ Psaltica etc.) render neumes as font glyphs keyed to
a stable ASCII layout, documented in EZ-CharacterTables.pdf. PDF content streams
reference glyphs by ASCII code (0x21-0x7E), sometimes offset into the PUA (+0xF000).
We normalize to ASCII and map each code to its documented neume name/category via
data/byzantine/ez_neume_map.json.

This yields a correct SYMBOLIC transcription (sequence of named neumes). It does NOT
resolve pitches — see the map's _note; pitch resolution needs a Byzantine theory engine.

Usage:
  python scripts/extract_neumes.py --glob 'data/byzantine/corpus/new-byzantium/*_byz.pdf' \
      --out data/byzantine/neumes_newbyz.jsonl
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import re
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = ROOT / "data" / "byzantine" / "ez_neume_map.json"

# The neume-carrying EZ sub-fonts (exclude EZ Omega = lyric text).
# GOA DCS uses the older "ED" fonts (Elie Daoun) that EZ was built from — verified to
# share the identical ASCII keyboard layout, so the same neume map applies. Names appear
# with hyphen/spacing variants across subsets, so we match by normalized prefix.
NEUME_FONTS = {"EZPsaltica", "EZFthora", "EZSpecial-I", "EZSpecial-II", "EZOxeia"}
NEUME_FONT_PREFIXES = ("EZPsaltica", "EZFthora", "EZSpecial", "EZOxeia",
                       "EDPsaltica", "EDIsson", "EDFthora", "EDOxeia", "EDSpecial",
                       "ED-Psaltica", "ED-Isson", "ED-Fthora")


def is_neume_font(fk: str) -> bool:
    fk2 = fk.replace("-", "").replace(" ", "")
    return any(fk2.startswith(p.replace("-", "")) for p in NEUME_FONT_PREFIXES)


def load_map() -> dict:
    return json.load(open(MAP_PATH))["map"]


def normalize_code(ch: str) -> int | None:
    o = ord(ch)
    if o >= 0xF000:
        o -= 0xF000
    return o if 33 <= o <= 126 else None


def font_key(font_name: str) -> str:
    return font_name.split("+")[-1].replace(" ", "")


def extract_neumes(pdf: Path, neume_map: dict) -> dict:
    """Return ordered neume tokens (names) + raw codes for one PDF."""
    doc = fitz.open(pdf)
    tokens: list[str] = []
    codes: list[str] = []
    unknown: Counter = Counter()
    for pg in doc:
        for b in pg.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    fk = font_key(s["font"])
                    if not is_neume_font(fk):
                        continue
                    for ch in s["text"]:
                        code = normalize_code(ch)
                        if code is None:
                            continue
                        key = chr(code)
                        codes.append(key)
                        entry = neume_map.get(key)
                        if entry:
                            tokens.append(entry["name"])
                        else:
                            tokens.append(f"unk_{code}")
                            unknown[key] += 1
    doc.close()
    return {
        "path": str(pdf),
        "n_neumes": len(tokens),
        "neumes": tokens,
        "codes": "".join(codes),
        "unknown": dict(unknown),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    neume_map = load_map()
    files = sorted(globmod.glob(args.glob))
    rows = []
    stats = {"files": 0, "neumes": 0, "empty": 0}
    for f in files:
        r = extract_neumes(Path(f), neume_map)
        rows.append(r)
        stats["files"] += 1
        stats["neumes"] += r["n_neumes"]
        if r["n_neumes"] == 0:
            stats["empty"] += 1

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
