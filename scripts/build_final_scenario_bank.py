#!/usr/bin/env python3
"""Build final Byzantine scenario bank: break tests + new hard cases from corpus."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BREAK = ROOT / "scenarios" / "byzantine_transcription_break_dev.yaml"
OUT = ROOT / "scenarios" / "byzantine_transcription_final_dev.yaml"

NEW_SCENARIOS = [
    {
        "id": "final_anaphora_mode_kliton",
        "direction": "byz_to_west",
        "echos": "varies",
        "tags": ["liturgy", "anaphora", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Anaphora-Byzantine_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode (Reader's / Kliton oral tradition), Ni = Πα = D4]
(Πα) ison
petastē | oligon | apostrophos | oligon | petastē | kentēma
""",
        "reference_output": """Mode (Kliton), Ni = D4
Ison: D4
E4 D4 C4 D4 E4 F#4
""",
        "context": "Anaphora fragment (Cappella corpus). Prokeimenon-like contour extended.",
    },
    {
        "id": "final_post_communion_mode2_stack",
        "direction": "byz_to_west",
        "echos": "II",
        "tags": ["liturgy", "post_communion", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Post-Communion-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode II, Ni = Κε = A4]
(Κε) ison
oligon | oligon | petastē | apostrophos | oligon gorgon
""",
        "reference_output": """Mode II, Ni = A4
Ison: A4
B4 C5 D5 C5 B4 (short)
""",
        "context": "Post-communion Mode II (Cappella). Ends with gorgon.",
    },
    {
        "id": "final_entrance_chant_plagal",
        "direction": "byz_to_west",
        "echos": "Pl IV",
        "tags": ["liturgy", "entrance", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Entrance-Chant-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode Plagal IV, Ni = Πα = D4]
(Πα) ison
kentēma | apostrophos | petastē | oligon | apostrophos
""",
        "reference_output": """Mode Plagal IV, Ni = D4
Ison: D4
A4 G4 F#4 G4 F#4
""",
        "context": "Sunday entrance chant (Cappella). Formula contour with kentēma opening to A4.",
    },
    {
        "id": "final_sunday_antiphons_opening",
        "direction": "byz_to_west",
        "echos": "I",
        "tags": ["liturgy", "antiphons", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Sunday-Antiphons-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode I, Ni = Πα = D4]
(Ν) ison
oligon | kentēma | apostrophos | oligon | kentēma
""",
        "reference_output": """Mode I, Ni = D4
Ison: D4
E4 F#4 E4 F#4 A4
""",
        "context": "Sunday antiphons opening (Cappella). Alternating step and m3.",
    },
    {
        "id": "final_litany_of_peace",
        "direction": "byz_to_west",
        "echos": "IV",
        "tags": ["liturgy", "litany", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Litany-of-Peace-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode IV, Ni = Γα = F4]
(Γα) ison
oligon | oligon | petastē | apostrophos | oligon
""",
        "reference_output": """Mode IV, Ni = F4
Ison: F4
G4 A4 Bb4 A4 G4
""",
        "context": "Litany of peace (Cappella). Mode IV diatonic with Bb.",
    },
    {
        "id": "final_koukouzelis_communion",
        "direction": "byz_to_west",
        "echos": "Pl I",
        "tags": ["liturgy", "koukouzelis", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Koukouzelis-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode Plagal I, Ni = Δι = G4]
(Δι) ison
petastē | kentēma | ypsilē pnevma | apostrophos | oligon
""",
        "reference_output": """Mode Plagal I, Ni = G4
Ison: G4
A4 C5 F5 E5 D5
""",
        "context": "Koukouzelis communion verse (Cappella). Contains 4th leap.",
    },
    {
        "id": "final_responsorial_psalm",
        "direction": "byz_to_west",
        "echos": "Pl IV",
        "tags": ["liturgy", "psalm", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Responsorial-Psalm-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode Plagal IV, Ni = Πα = D4]
(Πα) ison
oligon | petastē | oligon | kentēma | apostrophos | oligon
""",
        "reference_output": """Mode Plagal IV, Ni = D4
Ison: D4
E4 F#4 G4 A4 G4 F#4
""",
        "context": "Communion responsorial psalm (Cappella).",
    },
    {
        "id": "final_general_responses",
        "direction": "byz_to_west",
        "echos": "II",
        "tags": ["liturgy", "responses", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/General-Responses-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode II, Ni = Κε = A4]
(Κε) ison
ison | oligon | apostrophos | oligon | petastē
""",
        "reference_output": """Mode II, Ni = A4
Ison: A4
A4 B4 A4 B4 C5
""",
        "context": "General responses (Cappella). Opens with ison neume repeat.",
    },
    {
        "id": "final_west_dynamis_reverse",
        "direction": "west_to_byz",
        "echos": "II",
        "tags": ["reverse", "dynamis", "formula", "final"],
        "input": """Direction: Western → Byzantine
Mode II, Ni = A4
Ison: A4
B4 A4 D5 C5 B4
""",
        "reference_output": """[Mode II, Ni=Κε]
(Κε) ison
petastē | oligon | kentēma | apostrophos | oligon
""",
        "context": "Reverse Dynamis formula exactly.",
    },
    {
        "id": "final_west_cherubic_reverse",
        "direction": "west_to_byz",
        "echos": "Pl IV",
        "tags": ["reverse", "cherubic", "formula", "final"],
        "input": """Direction: Western → Byzantine
Mode Plagal IV, Ni = D4
Ison: D4
A4 D5 E5 F#5
""",
        "reference_output": """[Mode Plagal IV, Ni=Πα]
(Πα) ison
kentēma | ypsilē pnevma | petastē | oligon
""",
        "context": "Reverse Cherubic formula.",
    },
    {
        "id": "final_west_trisagion_ison_reverse",
        "direction": "west_to_byz",
        "echos": "II",
        "tags": ["reverse", "trisagion", "ison", "final"],
        "input": """Direction: Western → Byzantine
Mode II, Ni = A4
Ison: A4
A4 (short) A4 B4 C5
Ison: B4
A4 (short) B4
""",
        "reference_output": """[Mode II, Ni=Κε]
(Κε) ison
ison gorgon | oligon | oligon | petastē
— (Ζω) ison —
apostrophos gorgon | oligon
""",
        "context": "Reverse Trisagion with ison shift and gorgon.",
    },
    {
        "id": "final_triple_stack_chromatic_ison_fthora",
        "direction": "byz_to_west",
        "echos": "II",
        "tags": ["microtonal", "ison", "fthora", "triple", "final"],
        "input": """Direction: Byzantine → Western
[Mode II soft chromatic, Ni = Δι = G4]
(Δι) ison
oligon (diesis) | petastē
— (Κε) ison —
fthora soft chromatic
kentēma | apostrophos (diesis)
""",
        "reference_output": """Mode II soft chromatic, Ni = G4
Ison: G4
G4↑ A4
Ison: A4
fthora soft chromatic
C#5 B4↓
""",
        "context": "Triple stack: diesis + ison shift Di→Ke + fthora + diesis apostrophos.",
    },
    {
        "id": "final_enharmonic_long_phrase",
        "direction": "byz_to_west",
        "echos": "III",
        "tags": ["enharmonic", "long", "final"],
        "input": """Direction: Byzantine → Western
[Mode III enharmonic, Ni = Γα = F4]
(Γα) ison
oligon (diesis) | kentēma | apostrophos | oligon | kentēma | apostrophos
""",
        "reference_output": """Mode III enharmonic, Ni = F4
Ison: F4
F4↓ B4 A4 G4 C5 B4
""",
        "context": "Long enharmonic phrase. 6 neumes — count carefully.",
    },
    {
        "id": "final_plagal2_soft_chromatic",
        "direction": "byz_to_west",
        "echos": "Pl II",
        "tags": ["plagal", "chromatic", "final"],
        "input": """Direction: Byzantine → Western
[Mode Plagal II soft chromatic, Ni = Βου = E4]
(Βου) ison
oligon (diesis) | petastē | kentēma (diesis) | apostrophos
""",
        "reference_output": """Mode Plagal II soft chromatic, Ni = E4
Ison: E4
E4↑ F#4↑ A4↓ G#4
""",
        "context": "Plagal II on Bou=E4. Multiple diesis in soft chromatic.",
    },
    {
        "id": "final_mode1_elaphron_chain",
        "direction": "byz_to_west",
        "echos": "I",
        "tags": ["elaphron", "interval", "final"],
        "input": """Direction: Byzantine → Western
[Mode I, Ni = Πα = D4]
(Ν) ison
kentēma | elaphron | apostrophos | elaphron | oligon
""",
        "reference_output": """Mode I, Ni = D4
Ison: D4
F#4 D4 C#4 A3 B3
""",
        "context": "elaphron = −3rd. Chain of leaps and descents.",
    },
    {
        "id": "final_ten_neume_mode4",
        "direction": "byz_to_west",
        "echos": "IV",
        "tags": ["long", "mode_iv", "final"],
        "input": """Direction: Byzantine → Western
[Mode IV, Ni = Γα = F4]
(Γα) ison
oligon | oligon | petastē | oligon | kentēma | oligon | apostrophos | oligon | petastē | oligon
""",
        "reference_output": """Mode IV, Ni = F4
Ison: F4
G4 A4 Bb4 C5 D5 Eb5 F5 Eb5 D5 C5 Bb4
""",
        "context": "10-neume stress test. Mode IV diatonic spelling.",
    },
    {
        "id": "final_paschal_katavasias",
        "direction": "byz_to_west",
        "echos": "Pl",
        "tags": ["corpus", "new_byzantium", "pascha", "final"],
        "source_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/paschal_katavasias_en_byz.pdf",
        "input": """Direction: Byzantine → Western
[Mode Plagal IV, Ni = Πα = D4]
(Πα) ison
petastē | oligon | kentēma | apostrophos | petastē | oligon
""",
        "reference_output": """Mode Plagal IV, Ni = D4
Ison: D4
E4 F#4 A4 G4 F#4 G4
""",
        "context": "Paschal katavasias (New Byzantium EB/GS pair).",
    },
    {
        "id": "final_hard_chromatic_fthora_gorgon",
        "direction": "byz_to_west",
        "echos": "II",
        "tags": ["hard_chromatic", "fthora", "gorgon", "final"],
        "input": """Direction: Byzantine → Western
[Mode II hard chromatic, Ni = Δι = G4]
(Δι) ison
fthora hard chromatic
oligon (diesis) gorgon | kentēma | apostrophos argon
""",
        "reference_output": """Mode II hard chromatic, Ni = G4
Ison: G4
fthora hard chromatic
G4↑ (short) B4↓ C5 (long)
""",
        "context": "fthora + diesis + gorgon + argon combined.",
    },
    {
        "id": "final_west_long_fthora_reverse",
        "direction": "west_to_byz",
        "echos": "III",
        "tags": ["reverse", "long", "fthora", "final"],
        "input": """Direction: Western → Byzantine
Mode III, Ni = F4
Ison: F4
F4 G4 A4 B4
fthora soft chromatic
B4↓ C5↑ B4 A4 G4 F4
""",
        "reference_output": """[Mode III, Ni=Γα]
(Γα) ison
oligon | oligon | kentēma | kentēma
fthora soft chromatic
kentēma (diesis) | oligon | apostrophos | apostrophos | apostrophos | apostrophos
""",
        "context": "Long reverse with fthora and 6-note chromatic tail.",
    },
    {
        "id": "final_communion_prologue",
        "direction": "byz_to_west",
        "echos": "Pl IV",
        "tags": ["liturgy", "communion", "corpus", "final"],
        "source_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Communion-Prologue-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "input": """Direction: Byzantine → Western
[Mode Plagal IV, Ni = Πα = D4]
(Πα) ison
oligon | petastē | kentēma | ypsilē pnevma | apostrophos
""",
        "reference_output": """Mode Plagal IV, Ni = D4
Ison: D4
E4 F#4 A4 D5 C5
""",
        "context": "Communion prologue (Cappella). Ends with leap + step down.",
    },
]


def main() -> None:
    with open(BREAK, encoding="utf-8") as f:
        break_scenarios = yaml.safe_load(f)
    combined = break_scenarios + NEW_SCENARIOS
    with open(OUT, "w", encoding="utf-8") as f:
        yaml.dump(
            combined,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=1000,
        )
    print(f"Wrote {len(combined)} scenarios ({len(break_scenarios)} break + {len(NEW_SCENARIOS)} new) → {OUT}")


if __name__ == "__main__":
    main()
