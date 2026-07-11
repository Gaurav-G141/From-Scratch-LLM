#!/usr/bin/env python3
"""Tabulate score_real_musical.py outputs across runs, side by side.

Scans runs/*_realscore.json, groups by run tag and direction (n2w / w2n), and
prints one comparison table per direction so decoding variants (e.g. the v3b
ngram / temp experiments) line up against v3, v2 (curr2), v1 (curr), and the
7B baseline (coder7b).

It also reports the "above-gate" musicality: score_real_musical.py force-zeroes
any row whose variety < 0.15 (the anti-drone gate). For a run like v3 that
LOOPS (variety just under the gate) the headline composite understates the real
knowledge, so this shows the mean real_musicality over only the rows that clear
the gate, plus how many rows that is — the honest knowledge signal.

Usage:
  python3 scripts/compare_realscores.py                 # all runs/*_realscore.json
  python3 scripts/compare_realscores.py --glob 'runs/v3*_realscore.json'
  python3 scripts/compare_realscores.py --order coder7b curr curr2 v3 v3b_ngram v3b_temp
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = 0.15  # keep in sync with score_real_musical.py anti-drone gate

# metrics shown per direction; only those present in a run are printed
N2W_METRICS = [
    "real_musicality_0_2", "good_rate", "variety", "set_f1", "hist_sim",
    "interval_hist_sim", "contour_sim", "ambitus_match", "ngram_f1",
    "length_ratio", "mode_correct", "ison_correct",
]
W2N_METRICS = [
    "real_musicality_0_2", "good_rate", "variety", "set_f1", "hist_sim",
    "ngram_f1", "length_ratio", "mode_correct", "ison_correct",
]


def parse_name(path: str) -> tuple[str, str]:
    """runs/v3b_ngram_n2w_realscore.json -> ('v3b_ngram', 'n2w')."""
    stem = os.path.basename(path)
    stem = re.sub(r"_?realscore\.json$", "", stem)
    stem = re.sub(r"\.json$", "", stem)
    m = re.search(r"_(n2w|w2n)$", stem)
    if not m:
        return stem, "?"
    return stem[: m.start()], m.group(1)


def above_gate(report: dict) -> tuple[float, int, int]:
    """(mean real_musicality over rows with variety>=GATE, n_above, n_total)."""
    rows = report.get("per_row") or []
    kept = [r for r in rows if float(r.get("variety", 0.0)) >= GATE]
    if not kept:
        return 0.0, 0, len(rows)
    mean = sum(float(r.get("real_musicality_0_2", 0.0)) for r in kept) / len(kept)
    return mean, len(kept), len(rows)


def block(report: dict, direction: str) -> dict:
    """Pick the metrics block matching the direction, falling back to overall."""
    key = "neume_to_west" if direction == "n2w" else "west_to_neume"
    b = report.get(key)
    if b:
        return b
    return report.get("overall", {})


def fmt(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return str(v)


def print_table(direction: str, runs: list[tuple[str, dict]], metrics: list[str]) -> None:
    # keep only metrics some run actually has
    present = [m for m in metrics if any(m in block(r, direction) for _, r in runs)]
    tags = [tag for tag, _ in runs]
    w0 = max([len("metric")] + [len(m) for m in present])
    colw = max(9, *(len(t) for t in tags))

    header = "metric".ljust(w0) + " | " + " | ".join(t.rjust(colw) for t in tags)
    print(f"\n=== {direction} ===")
    print(header)
    print("-" * len(header))
    for m in present:
        cells = []
        for _, rep in runs:
            b = block(rep, direction)
            cells.append(fmt(b[m]).rjust(colw) if m in b else "-".rjust(colw))
        print(m.ljust(w0) + " | " + " | ".join(cells))

    # above-gate honest signal
    print("-" * len(header))
    ag_row, cnt_row = [], []
    for _, rep in runs:
        mean, n_above, n_total = above_gate(rep)
        ag_row.append((f"{mean:.3f}" if n_total else "-").rjust(colw))
        cnt_row.append((f"{n_above}/{n_total}" if n_total else "-").rjust(colw))
    print("above_gate_music".ljust(w0) + " | " + " | ".join(ag_row))
    print("above_gate_rows".ljust(w0) + " | " + " | ".join(cnt_row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="runs/*_realscore.json", help="glob for realscore JSON files")
    ap.add_argument("--order", nargs="*", default=None,
                    help="explicit run-tag order (missing tags dropped, extras appended)")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(ROOT / args.glob)))
    if not paths:
        ap.error(f"no files matched {args.glob}")

    # tag -> {direction -> report}
    runs: dict[str, dict[str, dict]] = {}
    for p in paths:
        tag, direction = parse_name(p)
        with open(p) as f:
            runs.setdefault(tag, {})[direction] = json.load(f)

    tags = list(runs)
    if args.order:
        ordered = [t for t in args.order if t in runs]
        ordered += [t for t in tags if t not in ordered]
        tags = ordered

    for direction, metrics in (("n2w", N2W_METRICS), ("w2n", W2N_METRICS)):
        present = [(t, runs[t][direction]) for t in tags if direction in runs[t]]
        if present:
            print_table(direction, present, metrics)

    print("\nnote: above_gate_music = mean real_musicality_0_2 over rows with "
          f"variety >= {GATE} (the anti-drone gate); above_gate_rows = kept/total.")
    print("For a looping run this is the honest knowledge signal the gated composite hides.")


if __name__ == "__main__":
    main()
