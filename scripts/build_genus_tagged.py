#!/usr/bin/env python3
"""Tier-1 microtonal-intent experiment: add Byzantine genus to translation targets.

Takes the EXISTING prime translation datasets and produces genus-tagged copies in NEW
files (originals untouched). The mode header in each user prompt and assistant target gets
its genus appended, e.g.:  "Mode pl. 2"  ->  "Mode pl. 2 (hard chromatic)".

Genus preserves microtonal INTENT symbolically (it determines the comma pattern) without
claiming exact frequencies. Mapping: data/byzantine/mode_genus_map.json.

INTERRUPTION-SAFE:
  - Reads only; writes to new *_genus.jsonl paths (prime data never modified).
  - Deterministic + idempotent: same inputs -> byte-identical output, so re-running after
    a power loss simply reproduces the file. No partial state to reconcile.
  - Atomic writes: each output is written to <path>.tmp then os.replace()'d into place, so
    an interruption mid-write cannot leave a half-written/corrupt dataset — either the old
    file (or nothing) or the complete new file exists.

Usage:
  python scripts/build_genus_tagged.py            # build all
  python scripts/build_genus_tagged.py --check     # report what would change, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"
GENUS_MAP = json.load((DATA / "mode_genus_map.json").open())["map"]

# Prime inputs (READ ONLY) -> genus-tagged outputs (NEW files).
FILES = {
    "sft_translation_train.jsonl":   "sft_translation_train_genus.jsonl",
    "sft_translation_heldout.jsonl": "sft_translation_heldout_genus.jsonl",
    "sft_n2w_train_sub.jsonl":       "sft_n2w_train_sub_genus.jsonl",
    "sft_w2n_train_sub.jsonl":       "sft_w2n_train_sub_genus.jsonl",
}

# Match a canonical mode label at the start of a header line, capturing exactly the labels
# our data uses. Longest-first so "Mode pl. 2" wins over "Mode 2" etc.
_MODE_LABELS = sorted(GENUS_MAP.keys(), key=len, reverse=True)
_MODE_RE = re.compile("(" + "|".join(re.escape(m) for m in _MODE_LABELS) + r")(?![\w.])")


def tag_genus(text: str) -> tuple[str, bool]:
    """Append '(genus)' after the first canonical mode label in the text.
    Returns (new_text, changed). Idempotent: won't double-tag if genus already present."""
    m = _MODE_RE.search(text)
    if not m:
        return text, False
    label = m.group(1)
    genus = GENUS_MAP[label]["genus"]
    insert_at = m.end()
    # already tagged? (idempotency guard)
    if text[insert_at:insert_at + 2] == " (":
        return text, False
    return text[:insert_at] + f" ({genus})" + text[insert_at:], True


def process_row(row: dict) -> dict:
    out = dict(row)
    msgs = [dict(m) for m in row["messages"]]
    for m in msgs:
        if m["role"] in ("user", "assistant"):
            m["content"], _ = tag_genus(m["content"])
    out["messages"] = msgs
    return out


def atomic_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    for src_name, out_name in FILES.items():
        src = DATA / src_name
        if not src.exists():
            print(f"skip (missing): {src_name}")
            continue
        rows = [json.loads(l) for l in src.open() if l.strip()]
        tagged = []
        n_changed = 0
        for r in rows:
            before = json.dumps(r["messages"], ensure_ascii=False)
            nr = process_row(r)
            if json.dumps(nr["messages"], ensure_ascii=False) != before:
                n_changed += 1
            tagged.append(nr)
        if args.check:
            print(f"[check] {src_name}: {len(rows)} rows, {n_changed} would get genus tag")
            continue
        atomic_write(DATA / out_name, tagged)
        print(f"{out_name}: {len(tagged)} rows written, {n_changed} genus-tagged")


if __name__ == "__main__":
    main()
