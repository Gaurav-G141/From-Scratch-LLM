#!/usr/bin/env python3
"""Assemble ultra-hard Byzantine scenario bank from held-out + compound cases."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
HELDOUT = ROOT / "scenarios" / "byzantine_transcription_heldout.yaml"
FINAL = ROOT / "scenarios" / "byzantine_transcription_final_dev.yaml"
OUT = ROOT / "scenarios" / "byzantine_transcription_ultra_hard.yaml"

# Hardest final_dev IDs: microtonal, reverse, long, multi-ison (not in unseen diatonic set)
ULTRA_IDS = [
    "break_enharmonic_fthora_nenano",
    "break_west_to_byz_eight_notes_fthora",
    "break_west_to_byz_rhythm_reverse",
    "break_hard_chromatic_mode2_leap",
    "break_double_ison_shift",
    "break_soft_chromatic_double_diesis",
    "break_gorgon_argon_hemiolon_stack",
    "final_triple_stack_chromatic_ison_fthora",
    "final_west_long_fthora_reverse",
    "final_ten_neume_mode4",
    "final_hard_chromatic_fthora_gorgon",
    "final_enharmonic_long_phrase",
    "final_plagal2_soft_chromatic",
]


def main() -> None:
    heldout = yaml.safe_load(HELDOUT.read_text(encoding="utf-8"))
    final = yaml.safe_load(FINAL.read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in final}

    scenarios = []
    for s in heldout:
        rec = dict(s)
        rec["tags"] = list(rec.get("tags") or []) + ["ultra_hard", "heldout"]
        scenarios.append(rec)

    for sid in ULTRA_IDS:
        if sid not in by_id:
            raise SystemExit(f"Missing scenario {sid}")
        rec = dict(by_id[sid])
        tags = list(rec.get("tags") or [])
        if "ultra_hard" not in tags:
            tags.append("ultra_hard")
        rec["tags"] = tags
        scenarios.append(rec)

    OUT.write_text(yaml.dump(scenarios, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(scenarios)} ultra-hard scenarios → {OUT}")
    print(f"  heldout: {len(heldout)}, compound: {len(ULTRA_IDS)}")


if __name__ == "__main__":
    main()
