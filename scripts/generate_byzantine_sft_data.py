#!/usr/bin/env python3
"""Generate SFT JSONL from Byzantine scenario YAML files (Day 2 junk/smoke data).

Reads hand-crafted scenarios with reference_output and writes chat-format JSONL
for supervised fine-tuning smoke tests.

Usage:
  python scripts/generate_byzantine_sft_data.py
  python scripts/generate_byzantine_sft_data.py --min-rows 50 --out data/byzantine/sft_junk.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SCENARIO_FILES = [
    ROOT / "scenarios/byzantine_transcription_dev.yaml",
    ROOT / "scenarios/byzantine_transcription_heldout.yaml",
    ROOT / "scenarios/byzantine_transcription_break_dev.yaml",
]

SYSTEM_PROMPT = (
    "You transcribe between Byzantine (Chrysanthine / New Analytical Method) notation "
    "and Western staff notation.\n\n"
    "Convert the input notation to the requested target format. Preserve musical meaning: "
    "melodic contour, mode (echos), martyria, ison (drone), microtonal alterations "
    "(diesis, fthora), and rhythmic neume modifiers (gorgon, argon). "
    "Do NOT round to 12-TET without marking approximation. "
    "Do NOT add harmony or impose 4/4.\n\n"
    "Byzantine → Western: state mode/Ni, staff pitches (D4, E4…), preserve ison line, "
    "mark microtones.\n"
    "Western → Byzantine: martyria, interval names (oligon, petastē, apostrophos…), "
    "ison as (Ν)/(Δι)/(Κε).\n\n"
    "Output notation only — no commentary."
)


def _load_yaml_scenarios(path: Path) -> list[dict]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of scenarios")
    return data


def scenario_to_row(item: dict, *, source_file: str) -> dict | None:
    user_input = str(item.get("input") or "").strip()
    reference = str(item.get("reference_output") or "").strip()
    if not user_input or not reference:
        return None

    scenario_id = str(item.get("id") or "unknown")
    return {
        "id": scenario_id,
        "source": source_file,
        "direction": str(item.get("direction") or "byz_to_west"),
        "tags": list(item.get("tags") or []),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reference},
        ],
    }


def load_corpus_jsonl(path: Path, *, status: str = "raw") -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if status and row.get("status") != status:
                continue
            rows.append(row)
    return rows


def collect_rows(scenario_files: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for path in scenario_files:
        if not path.exists():
            print(f"Warning: skipping missing file {path}", file=sys.stderr)
            continue
        for item in _load_yaml_scenarios(path):
            row = scenario_to_row(item, source_file=path.name)
            if row is None:
                continue
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            rows.append(row)

    return rows


def pad_to_min(rows: list[dict], min_rows: int) -> list[dict]:
    """Duplicate earliest rows with suffix ids if below min_rows (junk smoke test)."""
    if len(rows) >= min_rows:
        return rows[:min_rows] if min_rows else rows

    padded = list(rows)
    idx = 0
    while len(padded) < min_rows:
        base = rows[idx % len(rows)]
        copy = dict(base)
        copy["id"] = f"{base['id']}_dup{len(padded) - len(rows) + 1}"
        copy["tags"] = list(base.get("tags") or []) + ["junk_dup"]
        padded.append(copy)
        idx += 1
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Byzantine SFT JSONL from scenario YAML")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[str(p) for p in DEFAULT_SCENARIO_FILES],
        help="Scenario YAML files to read",
    )
    parser.add_argument("--out", default=str(ROOT / "data/byzantine/sft_junk.jsonl"))
    parser.add_argument("--from-corpus", default="", help="Merge rows from sft_raw.jsonl (Day 3 corpus)")
    parser.add_argument("--corpus-status", default="raw", help="Filter corpus rows by status field")
    parser.add_argument("--corpus-only", action="store_true", help="Only use --from-corpus rows (skip YAML scenarios)")
    parser.add_argument("--no-pad", action="store_true", help="Do not pad to --min-rows")
    parser.add_argument("--min-rows", type=int, default=50, help="Minimum rows (pad with dups if needed)")
    parser.add_argument("--limit", type=int, default=0, help="Cap output rows (0 = no cap)")
    args = parser.parse_args()

    if args.corpus_only:
        if not args.from_corpus:
            raise SystemExit("--corpus-only requires --from-corpus")
        rows = load_corpus_jsonl(Path(args.from_corpus), status=args.corpus_status)
        print(f"Loaded {len(rows)} corpus rows from {args.from_corpus}", file=sys.stderr)
    else:
        scenario_files = [Path(p) for p in args.scenarios]
        rows = collect_rows(scenario_files)

        if args.from_corpus:
            corpus_path = Path(args.from_corpus)
            corpus_rows = load_corpus_jsonl(corpus_path, status=args.corpus_status)
            seen = {r["id"] for r in rows}
            for row in corpus_rows:
                if row["id"] not in seen:
                    rows.append(row)
                    seen.add(row["id"])
            print(f"Merged {len(corpus_rows)} corpus rows from {corpus_path}", file=sys.stderr)

    if not rows:
        raise SystemExit("No training rows found.")

    if not args.no_pad:
        rows = pad_to_min(rows, args.min_rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_dir = {}
    for row in rows:
        by_dir[row["direction"]] = by_dir.get(row["direction"], 0) + 1

    print(f"Wrote {len(rows)} rows → {out_path}")
    print(f"  directions: {by_dir}")
    print(f"  sample id: {rows[0]['id']}")


if __name__ == "__main__":
    main()
