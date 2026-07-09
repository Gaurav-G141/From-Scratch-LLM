#!/usr/bin/env python3
"""Quarantine duplicate / poor-quality rows out of sft_raw.jsonl.

Does NOT delete anything:
  - full original is copied to sft_raw.backup.jsonl
  - removed rows are written to sft_raw_rejected.jsonl with a _reject_reason field
  - clean rows are written back to sft_raw.jsonl

Two removal categories:
  1. exact-ID duplicates (resume/re-append artifact) — keep one per id
     (prefer status=accepted, else the last occurrence)
  2. poor-quality distinct rows:
       - too_short
       - no_notation_out         (assistant has no pitch and no neume)
       - degenerate_single_pitch (assistant body collapses to one pitch)
       - lyric_leak              (>=5 distinct non-notation words)
       - count_mismatch          (neume/pitch counts differ >=3x)

Usage:
  python scripts/quarantine_byzantine_garbage.py            # apply
  python scripts/quarantine_byzantine_garbage.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "byzantine"
IN = DATA / "sft_raw.jsonl"
BACKUP = DATA / "sft_raw.backup.jsonl"
REJECTED = DATA / "sft_raw_rejected.jsonl"
REPORT = DATA / "quarantine_report.json"

PITCH = re.compile(r"\b[A-G](?:[#b♭♯]|[↑↓])?\d?\b")
NEUME = re.compile(
    r"(oligon|petast|apostroph|kent[eē]ma|ison|gorgon|argon|elaphron|ypsil|"
    r"ὀλ|ἰσον|πεταστ|ἀπόστροφ|μαρτυρ)",
    re.I,
)
NOTATION = set(
    """mode ison martyria echos tone plagal grave diesis fthora sharp flat natural
    ni pa vou ga di ke zo oligon petaste apostrophos apostrophe kentema kenteme gorgon
    argon elaphron ypsile ypsili pnevma line opening direction byzantine western first
    second third fourth fifth sixth seventh eighth verse phrase ending final formula""".split()
)


def strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def msg(row: dict, role: str) -> str:
    for m in row.get("messages") or []:
        if m.get("role") == role:
            return str(m.get("content") or "").strip()
    return ""


def lyric_words(t: str) -> list[str]:
    t = PITCH.sub(" ", strip_marks(t))
    return [w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in NOTATION]


def quality_tags(row: dict) -> list[str]:
    tags: list[str] = []
    u, a = msg(row, "user"), msg(row, "assistant")
    d = row.get("direction", "byz_to_west")
    if len(u) < 20 or len(a) < 10:
        tags.append("too_short")
    if not PITCH.search(a) and not NEUME.search(a):
        tags.append("no_notation_out")
    body = re.sub(r"(?im)^.*(mode|ison).*$", "", a)
    pitches = PITCH.findall(body)
    if pitches and len(set(pitches)) <= 1:
        tags.append("degenerate_single_pitch")
    if len(set(lyric_words(u)) | set(lyric_words(a))) >= 5:
        tags.append("lyric_leak")
    if d == "byz_to_west":
        nu, pa = len(NEUME.findall(u)), len(PITCH.findall(a))
    else:
        pa, nu = len(PITCH.findall(u)), len(NEUME.findall(a))
    if nu and pa and max(nu, pa) / min(nu, pa) >= 3.0:
        tags.append("count_mismatch")
    return tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in IN.open(encoding="utf-8") if l.strip()]
    total = len(rows)

    # Pass 1: exact-ID dedup. Prefer status=accepted, else last occurrence.
    keep: "OrderedDict[str, dict]" = OrderedDict()
    dup_rows: list[dict] = []
    for r in rows:
        rid = r["id"]
        if rid in keep:
            prev = keep[rid]
            # prefer an accepted row over a raw/rejected one
            if prev.get("status") == "accepted" and r.get("status") != "accepted":
                dup_rows.append({**r, "_reject_reason": "duplicate_id"})
                continue
            dup_rows.append({**prev, "_reject_reason": "duplicate_id"})
        keep[rid] = r
    distinct = list(keep.values())

    # Pass 2: quality on distinct rows
    clean: list[dict] = []
    poor: list[dict] = []
    tag_counter: Counter = Counter()
    for r in distinct:
        tags = quality_tags(r)
        if tags:
            poor.append({**r, "_reject_reason": ",".join(tags)})
            for t in tags:
                tag_counter[t] += 1
        else:
            clean.append(r)

    rejected = dup_rows + poor

    report = {
        "input_rows": total,
        "duplicate_id_rows": len(dup_rows),
        "distinct_rows": len(distinct),
        "poor_quality_rows": len(poor),
        "clean_rows": len(clean),
        "poor_quality_breakdown": dict(tag_counter.most_common()),
        "clean_direction": dict(Counter(r.get("direction") for r in clean)),
        "clean_source": dict(Counter(r.get("source") for r in clean)),
    }

    print(json.dumps(report, indent=2))

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    # Back up original untouched, then write splits.
    shutil.copy2(IN, BACKUP)
    with REJECTED.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with IN.open("w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nBackup   → {BACKUP}  ({total} rows, untouched)")
    print(f"Rejected → {REJECTED}  ({len(rejected)} rows: {len(dup_rows)} dup + {len(poor)} poor)")
    print(f"Clean    → {IN}  ({len(clean)} rows)")
    print(f"Report   → {REPORT}")


if __name__ == "__main__":
    main()
