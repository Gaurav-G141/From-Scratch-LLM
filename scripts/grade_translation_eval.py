#!/usr/bin/env python3
"""Aggregate agent-graded base-vs-tuned scores for the translation SFT run.

The gpt-4o API judge was unavailable (billing), so outputs in
runs/byzantine_translation_1.7b_outputs.json were graded by an Opus agent on the
0-2 rubric from goals/byzantine_transcription.yaml, following the repo's existing
"Opus blind eval" precedent (docs/byzantine_opus_blind_eval.md).

Dimensions (order): melodic_equivalence, mode_fidelity, notation_convention,
meaning_preservation. Strict pass requires melodic>=1.5 AND meaning>=1.5.

This script just aggregates the hand-entered grades into per-suite means + deltas,
matching the schema of scripts/compare_local_sft_all_suites.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIMS = ["melodic_equivalence", "mode_fidelity", "notation_convention", "meaning_preservation"]

# grades[suite][id] = {"base": [mel, mode, notation, meaning], "tuned": [...]}
GRADES = {
    "heldout": {
        "mode4_authentic_descending": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "plagal1_ison_ni":            {"base": [0, 1, 0, 0], "tuned": [0, 1, 1, 0]},
        "west_to_byz_trisagion_snippet": {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 1, 0]},
        "hard_chromatic_mode2_fthora": {"base": [0, 1, 0, 0], "tuned": [0.5, 1, 1, 0.5]},
        "alleluia_plagal_fragment":   {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "ison_only_passage":          {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "west_to_byz_kontakion_mode4": {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "roundtrip_mode1_phrase":     {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "nenano_mode_medial":         {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "corpus_prokeimenon_stub":    {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
    },
    "unseen": {
        "unseen_apolytikia_resurrection":   {"base": [0, 0.5, 0, 0], "tuned": [0, 0.5, 0, 0]},
        "unseen_transfiguration_apolytikion": {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "unseen_pentecost_apolytikion":     {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 1, 0]},
        "unseen_annunciation_apolytikion":  {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 1, 0]},
        "unseen_circumcision_apolytikion":  {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "unseen_theophany_apolytikion":     {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "unseen_ascension_apolytikion":     {"base": [0, 0.5, 0, 0], "tuned": [0, 0.5, 0, 0]},
        "unseen_pentecost_great_prokimenon": {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "unseen_theophany_communion":       {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "unseen_dormition_lamentations":    {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
    },
    "ultra_hard": {
        # first 10 are heldout dupes (ultra_hard tag)
        "mode4_authentic_descending": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "plagal1_ison_ni":            {"base": [0, 1, 0, 0], "tuned": [0, 1, 1, 0]},
        "west_to_byz_trisagion_snippet": {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 1, 0]},
        "hard_chromatic_mode2_fthora": {"base": [0, 1, 0, 0], "tuned": [0.5, 1, 1, 0.5]},
        "alleluia_plagal_fragment":   {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "ison_only_passage":          {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "west_to_byz_kontakion_mode4": {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "roundtrip_mode1_phrase":     {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "nenano_mode_medial":         {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "corpus_prokeimenon_stub":    {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        # 13 compound break_/final_
        "break_enharmonic_fthora_nenano": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "break_west_to_byz_eight_notes_fthora": {"base": [0, 0.5, 0, 0], "tuned": [0, 1, 1, 0]},
        "break_west_to_byz_rhythm_reverse": {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "break_hard_chromatic_mode2_leap": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "break_double_ison_shift":    {"base": [0, 1, 0, 0], "tuned": [0, 0.5, 1, 0]},
        "break_soft_chromatic_double_diesis": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "break_gorgon_argon_hemiolon_stack": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "final_triple_stack_chromatic_ison_fthora": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "final_west_long_fthora_reverse": {"base": [0, 0.5, 0, 0], "tuned": [0, 0, 0, 0]},
        "final_ten_neume_mode4":      {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "final_hard_chromatic_fthora_gorgon": {"base": [0, 1, 0, 0], "tuned": [0, 0.5, 1, 0]},
        "final_enharmonic_long_phrase": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
        "final_plagal2_soft_chromatic": {"base": [0, 1, 0, 0], "tuned": [0, 0, 1, 0]},
    },
}


def arm_summary(suite: dict, arm: str) -> dict:
    n = len(suite)
    dim_means = {}
    for i, d in enumerate(DIMS):
        dim_means[d] = sum(row[arm][i] for row in suite.values()) / n
    strict = 0
    for row in suite.values():
        mel, mode, notation, meaning = row[arm]
        if mel >= 1.5 and meaning >= 1.5:
            strict += 1
    overall = sum(dim_means.values()) / len(DIMS)
    return {
        "overall_mean": overall,
        "dimensions": dim_means,
        "strict_pass": strict,
        "strict_pass_rate": f"{strict}/{n}",
        "n_scenarios": n,
    }


def main() -> None:
    summary = {
        "model": "Qwen/Qwen3-1.7B",
        "adapter_path": "models/byzantine_sft_translation_1.7b",
        "prompt_file": "prompts/byzantine_transcription_v2.txt",
        "judge": "opus-agent (manual rubric grading; gpt-4o judge unavailable - billing)",
        "train_data": "data/byzantine/sft_translation_train.jsonl (897 rows: 448 neume_to_west + 449 west_to_neume)",
        "epochs": 3,
        "suites": {"base": {}, "tuned": {}},
        "deltas": {},
        "totals": {"base": {}, "tuned": {}},
    }
    tot = {"base": {"n": 0, "strict": 0, "wsum": 0.0, "dim": {d: 0.0 for d in DIMS}},
           "tuned": {"n": 0, "strict": 0, "wsum": 0.0, "dim": {d: 0.0 for d in DIMS}}}

    for suite_key, suite in GRADES.items():
        for arm in ("base", "tuned"):
            s = arm_summary(suite, arm)
            summary["suites"][arm][suite_key] = s
            n = s["n_scenarios"]
            tot[arm]["n"] += n
            tot[arm]["strict"] += s["strict_pass"]
            tot[arm]["wsum"] += s["overall_mean"] * n
            for d in DIMS:
                tot[arm]["dim"][d] += s["dimensions"][d] * n
        b, t = summary["suites"]["base"][suite_key], summary["suites"]["tuned"][suite_key]
        summary["deltas"][suite_key] = {
            "overall_mean": t["overall_mean"] - b["overall_mean"],
            "strict_pass": t["strict_pass"] - b["strict_pass"],
            "dimensions": {d: t["dimensions"][d] - b["dimensions"][d] for d in DIMS},
        }

    for arm in ("base", "tuned"):
        n = tot[arm]["n"]
        summary["totals"][arm] = {
            "n_scenarios": n,
            "strict_pass": tot[arm]["strict"],
            "strict_pass_rate": f"{tot[arm]['strict']}/{n}",
            "overall_mean": tot[arm]["wsum"] / n,
            "dimensions": {d: tot[arm]["dim"][d] / n for d in DIMS},
        }
    summary["totals"]["delta_overall"] = (
        summary["totals"]["tuned"]["overall_mean"] - summary["totals"]["base"]["overall_mean"]
    )
    summary["totals"]["delta_dimensions"] = {
        d: summary["totals"]["tuned"]["dimensions"][d] - summary["totals"]["base"]["dimensions"][d]
        for d in DIMS
    }

    out = ROOT / "runs/byzantine_translation_1.7b_graded.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown table
    print(f"\nWrote -> {out}\n")
    hdr = f"{'suite':<11} {'arm':<6} {'melodic':>8} {'mode':>6} {'notat':>6} {'mean':>6} {'strict':>7}"
    print(hdr)
    print("-" * len(hdr))
    for suite_key in GRADES:
        for arm in ("base", "tuned"):
            s = summary["suites"][arm][suite_key]
            dd = s["dimensions"]
            print(f"{suite_key:<11} {arm:<6} {dd['melodic_equivalence']:>8.2f} "
                  f"{dd['mode_fidelity']:>6.2f} {dd['notation_convention']:>6.2f} "
                  f"{s['overall_mean']:>6.2f} {s['strict_pass_rate']:>7}")
    print("-" * len(hdr))
    for arm in ("base", "tuned"):
        t = summary["totals"][arm]
        dd = t["dimensions"]
        print(f"{'TOTAL':<11} {arm:<6} {dd['melodic_equivalence']:>8.2f} "
              f"{dd['mode_fidelity']:>6.2f} {dd['notation_convention']:>6.2f} "
              f"{t['overall_mean']:>6.2f} {t['strict_pass_rate']:>7}")
    print(f"\ndelta_overall = {summary['totals']['delta_overall']:+.3f}")
    print("delta_dimensions = " + json.dumps(
        {d: round(v, 3) for d, v in summary["totals"]["delta_dimensions"].items()}))


if __name__ == "__main__":
    main()
