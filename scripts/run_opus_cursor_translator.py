#!/usr/bin/env python3
"""Apply pre-computed Opus (Cursor agent) translations to ultra-hard scenarios."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Opus agent translations following byzantine_transcription_v2.txt
OPUS_OUTPUTS: dict[str, str] = {
    "mode4_authentic_descending": """Mode IV, Ni = F4
Ison: F4
C5 B4 A4 G4""",
    "plagal1_ison_ni": """Mode Plagal I, Ni = G4
Ison: G4
G4 A4 B4 C5""",
    "west_to_byz_trisagion_snippet": """[Mode II, Ni=Κε]
(Κε) ison
ison | oligon | oligon | kentēma""",
    "hard_chromatic_mode2_fthora": """Mode II hard chromatic, Ni = G4
Ison: G4
fthora hard chromatic
G4↑ A4↓ B4↓""",
    "alleluia_plagal_fragment": """Mode Plagal IV, Ni = D4
Ison: D4
E4 D4 A4 G4""",
    "ison_only_passage": """Mode I, Ni = D4
Ison: D4
D4 D4 D4 E4""",
    "west_to_byz_kontakion_mode4": """[Mode IV, Ni=Γα]
(Γα) ison
oligon | oligon | petastē | oligon | kentēma""",
    "roundtrip_mode1_phrase": """Mode I, Ni = D4
Ison: D4
D4 E4 D4 E4""",
    "nenano_mode_medial": """Medial (nenano) mode, Ni = D4
Ison: D4
fthora nenano
D4 E4↓ D4""",
    "corpus_prokeimenon_stub": """Mode Plagal IV, Ni = D4
Ison: D4
E4 D4 A4""",
    "break_enharmonic_fthora_nenano": """Mode III enharmonic, Ni = F4
Ison: F4
F4↓ B4 A4
fthora nenano
G4 F4""",
    "break_west_to_byz_eight_notes_fthora": """[Mode III, Ni=Γα]
(Γα) ison
oligon | oligon | kentēma | kentēma
fthora soft chromatic
kentēma (diesis) | oligon | apostrophos | apostrophos""",
    "break_west_to_byz_rhythm_reverse": """[Mode I, Ni=Πα]
(Ν) ison
oligon gorgon | petastē gorgon | oligon argon | oligon argon | kentēma""",
    "break_hard_chromatic_mode2_leap": """Mode II hard chromatic, Ni = G4
Ison: G4
fthora hard chromatic
G4↑ A4↓ B4↓ E5""",
    "break_double_ison_shift": """Mode IV, Ni = F4
Ison: F4
F4 G4
Ison: G4
A4 B4
Ison: A4
G4 A4""",
    "break_soft_chromatic_double_diesis": """Mode II soft chromatic, Ni = G4
Ison: G4
G4↑ A4↑ A4↓""",
    "break_gorgon_argon_hemiolon_stack": """Mode I, Ni = D4
Ison: D4
D4 (short) E4 (short) F#4 (long) A4""",
    "final_triple_stack_chromatic_ison_fthora": """Mode II soft chromatic, Ni = G4
Ison: G4
G4↑ A4
Ison: A4
fthora soft chromatic
C#5 B4↓""",
    "final_west_long_fthora_reverse": """[Mode III, Ni=Γα]
(Γα) ison
oligon | oligon | kentēma | kentēma
fthora soft chromatic
kentēma (diesis) | oligon | apostrophos | apostrophos | apostrophos | apostrophos""",
    "final_ten_neume_mode4": """Mode IV, Ni = F4
Ison: F4
G4 A4 Bb4 C5 D5 Eb5 F5 Eb5 D5 C5 Bb4""",
    "final_hard_chromatic_fthora_gorgon": """Mode II hard chromatic, Ni = G4
Ison: G4
fthora hard chromatic
G4↑ (short) B4↓ C5 (long)""",
    "final_enharmonic_long_phrase": """Mode III enharmonic, Ni = F4
Ison: F4
F4↓ B4 A4 G4 C5 B4""",
    "final_plagal2_soft_chromatic": """Mode Plagal II soft chromatic, Ni = E4
Ison: E4
E4↑ F#4↑ A4↓ G#4""",
}


def main() -> None:
    import yaml
    from eval_harness.config import load_goal
    from eval_harness.judge.byzantine_checks import check_byzantine_transcription

    goal = load_goal(ROOT / "goals/byzantine_transcription.yaml")
    scenarios_path = ROOT / "scenarios/byzantine_transcription_ultra_hard.yaml"
    scenarios = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))

    results = []
    for s in scenarios:
        sid = s["id"]
        out = OPUS_OUTPUTS.get(sid, "ERROR: missing opus translation")
        rule_failures = check_byzantine_transcription(
            model_output=out,
            direction=s.get("direction", "byz_to_west"),
            forbidden_extra=goal.forbidden_patterns,
        )
        results.append(
            {
                "id": sid,
                "direction": s.get("direction"),
                "echos": s.get("echos"),
                "tags": s.get("tags"),
                "input": s["input"].strip(),
                "reference_output": s["reference_output"].strip(),
                "context": s.get("context", "").strip(),
                "model_output": out,
                "rule_failures": rule_failures,
            }
        )

    out_path = ROOT / "runs/byzantine_opus_v2_ultra_hard_outputs.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} Opus translations → {out_path}")


if __name__ == "__main__":
    main()
