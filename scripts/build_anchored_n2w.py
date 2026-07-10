#!/usr/bin/env python3
"""Add the Ison anchor to the PROMPT for byz->west directional data.

The other agent's key finding (docs/byzantine_synthetic_breakthrough_20260709.md):
putting `Ison: X4` in the *prompt* (not withheld in the target) turns an
under-determined mapping into a well-posed one. The w2n files already carry this;
the n2w files do NOT. This script produces anchored n2w copies.

Deterministic transform, no fabrication:
  - The target's line 2 is always `Ison: X4`. We lift that exact line into the user
    prompt (inserted right after the `Mode ...` line). Target is unchanged.
  - Idempotent: if the prompt already contains an `Ison:` line, the row is passed through.

Inputs  (READ ONLY): sft_n2w_train_sub.jsonl, sft_n2w_heldout.jsonl
Outputs (NEW files):  sft_n2w_train_sub_cued.jsonl, sft_n2w_heldout_cued.jsonl
"""
from __future__ import annotations
import json, os
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "byzantine"
FILES = {
    "sft_n2w_train_sub.jsonl": "sft_n2w_train_sub_cued.jsonl",
    "sft_n2w_heldout.jsonl":   "sft_n2w_heldout_cued.jsonl",
}


def anchor_row(row: dict) -> dict:
    msgs = [dict(m) for m in row["messages"]]
    user = next(m for m in msgs if m["role"] == "user")
    asst = next(m for m in msgs if m["role"] == "assistant")
    if "Ison:" in user["content"]:
        return row  # already anchored
    # Find the Ison line in the target (canonically line index 1).
    ison = next((l for l in asst["content"].splitlines() if l.startswith("Ison:")), None)
    if ison is None:
        return row  # no anchor to lift; leave as-is
    ulines = user["content"].splitlines()
    # Insert after the `Mode ...` line if present, else after the instruction line.
    idx = next((i for i, l in enumerate(ulines) if l.startswith("Mode")), 0)
    ulines.insert(idx + 1, ison)
    user["content"] = "\n".join(ulines)
    out = dict(row)
    out["messages"] = msgs
    return out


def atomic_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    for src_name, out_name in FILES.items():
        src = DATA / src_name
        if not src.exists():
            print(f"skip (missing): {src_name}"); continue
        rows = [json.loads(l) for l in src.open() if l.strip()]
        out = [anchor_row(r) for r in rows]
        n_anchored = sum(
            1 for r in out
            if "Ison:" in next(m for m in r["messages"] if m["role"] == "user")["content"]
        )
        atomic_write(DATA / out_name, out)
        print(f"{out_name}: {len(out)} rows, {n_anchored} anchored prompts")


if __name__ == "__main__":
    main()
