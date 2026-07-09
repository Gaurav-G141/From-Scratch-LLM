# Golden Set — Byzantine ↔ Western Transcription

Ten hand-selected, provenance-verified datapoints for grading. Chosen for **diversity**:
both transcription directions, all four Byzantine genera (diatonic / soft-chromatic /
hard-chromatic / enharmonic), all three corpus sources (GOA DCS, New Byzantium,
St. Anthony's), and a range of sequence lengths.

Every example is drawn from the clean `data/byzantine/sft_translation_{train,heldout}.jsonl`
data. For the `neume_to_west` examples, the target pitch line was verified to be an **exact
contiguous slice of the real Audiveris-OMR output** for that hymn (no fabrication). The
`west_to_neume` targets are the real extracted neume sequences for the same hymn.

## How to read these

- **System prompt** (identical for all 10):
  > You are a Byzantine chant notation assistant. You work with Byzantine neume sequences
  > (ison, oligon, petaste, apostrophos, gorgon, martyria, ...) and their Western
  > staff-notation transcriptions. Output the answer only — no commentary.
- **Task `neume_to_west`**: input is a neume-name sequence, target is Western staff pitches.
- **Task `west_to_neume`**: input is Western pitches, target is a neume chain.
- Targets carry a `Mode …\nIson: X4` (or `(Ison X4)`) header, matching the eval rubric format.

## Grading caveat (important)

This dataset supports **contour, mode fidelity, and notation-system** grading well. It does
**not** support exact-pitch grading: Byzantine neumes and Western notes are ~1.78:1
(melismatic), so neume count ≠ pitch count and the two sides are not per-symbol aligned. A
grader should score melodic *contour / interval pattern*, not one-to-one pitch identity, and
should **not** penalize the neume-vs-pitch count mismatch visible in these examples (it is
expected — non-pitch neumes like martyria/breath/gorgon and melismatic expansion). See
`docs/byzantine_handoff_20260709.md`.

## Provenance caveat

9 of 10 are from the `train` split (chosen to maximize diversity). For a strictly *unseen*
grade, use the hand-authored eval banks in `scenarios/byzantine_transcription_*.yaml`, which
are structurally separate from all training data.

## Diversity coverage

| # | id | task | mode | genus | source | provenance |
|---|----|------|------|-------|--------|-----------|
| 1 | `sam_m0150-lord-have-mercy-mode-1_n2w_0` | neume→west | Mode 1 | diatonic | St. Anthony's | OMR-verified |
| 2 | `dcs_alt1aposticha_n2w_0` | neume→west | Mode 2 | soft chromatic | GOA | OMR-verified |
| 3 | `newbyz_fc8de42722_n2w_0` | neume→west | Mode pl. 2 | hard chromatic | New Byzantium | OMR-verified |
| 4 | `dcs_alt1apolytikion1_n2w_0` | neume→west | Mode 3 | enharmonic | GOA | OMR-verified |
| 5 | `sam_m0640-anaphora-mode-7-hierotheos_n2w_0` | neume→west | Mode grave | enharmonic | St. Anthony's | OMR-verified |
| 6 | `dcs_idiomelon32_w2n_0` | west→neume | Mode 4 | diatonic | GOA | real neume seq |
| 7 | `dcs_crowns_w2n_0` | west→neume | Mode pl. 1 | diatonic | GOA | real neume seq |
| 8 | `dcs_magnificatf_w2n_21` | west→neume | Mode pl. 4 | diatonic | GOA | real neume seq |
| 9 | `dcs_lauds5_w2n_0` | west→neume | Mode 1 | diatonic | GOA | real neume seq |
| 10 | `dcs_palmsunday_n2w_0` | neume→west | Mode 4 | diatonic | GOA | OMR-verified |

---

### #1 — neume→west · Mode 1 (diatonic) · St. Anthony's
**Input**
```
Mode 1
breath_mark_v martyria_V petaste_4 martyria_V apostrophos oligon psifiston elaphron_apostrophos petaste_qualitative ison apostrophos gorgon_D oligon_kentemata_support apostrophos_2 digorgon_dot measure_bar breath_mark_v martyria_V oligon_kentema martyria_V psifiston apostrophos
```
**Target**
```
Mode 1
Ison: G4
G4 F4 F4 E4 D4 E4 D4 E4 F4 D4 G4
```

### #2 — neume→west · Mode 2 (soft chromatic) · GOA
**Input**
```
Mode 2
argon chamile_2 martyria_M apostrophos oligon_kentemata period_breath psifiston oligon_kentemata martyria_M apostrophos apostrophos ison martyria_measure_J oligon apostrophos_2 petaste_5 elaphron_5 ison apli breath_mark_m kentemata martyria_q oligon martyria_measure_J
```
**Target**
```
Mode 2
Ison: G4
B4 A-4 G4 A-4 A-4 F4 G4 G4 A-4 G4 G4 F4
```

### #3 — neume→west · Mode pl. 2 (hard chromatic) · New Byzantium
**Input**
```
Mode pl. 2
breath_mark_m kentemata martyria_q apostrophos_2 oligon ison ison oligon oligon oligon petaste_2 apostrophos oligon oligon petaste_qualitative oligon apostrophos apostrophos gorgon_variant kentemata_support_O apostrophos_2 apli_variant breath_mark_b kentemata martyria_q
```
**Target**
```
Mode pl. 2
Ison: G4
E4 G4 G4 G4 G4 A4 G4 G4 G4 G4 F4 E4 F4 G4 E4 E4 F4
```

### #4 — neume→west · Mode 3 (enharmonic) · GOA
**Input**
```
Mode 3
argon oligon_with_kentema_below martyria_N argon_variant petaste_3 martyria_N apostrophos martyria_measure_J petaste_qualitative ison oligon heteron oligon_hypsili_7 apostrophos breath_mark_c petaste_5 oligon apli elaphron_3 martyria_q ison martyria_measure_J
```
**Target**
```
Mode 3
Ison: G4
A4 G4 A4 A4 G4 G4 D4 E4 F4 G4 F4 A4
```

### #5 — neume→west · Mode grave / Varys (enharmonic) · St. Anthony's
**Input**
```
Mode grave
argon petaste_8 chamile_3 martyria_gt period_breath martyria_gt oligon_kentemata martyria_gt psifiston apostrophos apostrophos petaste apostrophos apostrophos_support digorgon_dot measure_bar period_breath martyria_gt petaste_qualitative oligon_hypsili martyria_gt oligon heteron apostrophos
```
**Target**
```
Mode grave
Ison: F#4
D4 C4 B3 C4 D4 E4 D4 C4 B3 F#4 E#4 F#4 G4 F#4
```

### #6 — west→neume · Mode 4 (diatonic) · GOA
**Input**
```
Mode 4
Ison: G4
E4 D4 D4 E4 F4 G4 A4 G4 F4 F4 G4 F4 E4 D4 E4 D4 D4 E4 E4 G4 E4 F4 G4 A4 G4 F4
```
**Target**
```
Mode 4
(Ison G4)
oligon | oligon | oligon | petaste | apostrophos | ison | oligon | kentemata | petaste | oligon | ison | ison | ison | apostrophos | apostrophos | ison
```

### #7 — west→neume · Mode pl. 1 (diatonic) · GOA
**Input**
```
Mode pl. 1
Ison: A4
F4 E4 F4 G4 B-4 A4 G4 F4 E4 D4 C4 D4 E4 F4 F4 B-4 A4 G4 G4 C5
```
**Target**
```
Mode pl. 1
(Ison A4)
oligon | apostrophos | ison | apostrophos | apostrophos | apostrophos | oligon | kentemata | ison | apli | ison | ison | petaste | kentemata | oligon_kentema | petaste | oligon_kentema | apli | apostrophos
```

### #8 — west→neume · Mode pl. 4 (diatonic) · GOA
**Input**
```
Mode pl. 4
Ison: G4
A4 B-4 A4 G4 G4 F4 F4 F4 G4 F4 F4 G4
```
**Target**
```
Mode pl. 4
(Ison G4)
dipli | oligon | klasma | oligon | oligon | oligon | apli | breath_mark_n | martyria_C | ison | oligon | oligon
```

### #9 — west→neume · Mode 1 (diatonic) · GOA
**Input**
```
Mode 1
Ison: F4
D4 F4 G4 G4 F4 E4 D4 E4 F4 G4 A4 G4 F4 F4 E4 D4 E4 D4 D4 D4 E4 F4 E4 D4 E4 D4
```
**Target**
```
Mode 1
(Ison F4)
oligon | oligon | oligon | oligon_kentema | oligon | apostrophos | apostrophos | apostrophos | kentemata | apli
```

### #10 — neume→west · Mode 4 (diatonic) · GOA
**Input**
```
Mode 4
breath_mark_b martyria_B petaste_3 oligon_with_kentema_below apostrophos apostrophos apostrophos apli_variant oligon ison ison psifiston apostrophos apostrophos petaste_qualitative ison breath_mark_b apostrophos gorgon_variant ison oligon oligon oligon
```
**Target**
```
Mode 4
Ison: G4
G4 F4 E4 D4 D4 E4 F4 E4 D4 E4
```
