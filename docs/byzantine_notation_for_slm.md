# Byzantine Notation Guide for an SLM

Source: `/Users/gaurav/Downloads/Detailed Byzantine Guide.pdf`, Nick Nicholas, "Unicode Technical Note: Byzantine Musical Notation", Version 1.1, February 2006.

Purpose: encode modern/post-Early Byzantine neumatic notation as explicit parsing rules. This file assumes the model may not understand advanced Western notation. It should prefer relative, modal output over hallucinated absolute pitches when context is incomplete.

## Core Model

Represent Byzantine notation as a state machine.

```text
state = {
  current_pitch_degree,
  current_pitch_letter_if_known,
  current_octave_region_if_known,
  current_mode_or_echos,
  current_scale_comma_pattern,
  current_finalis_if_known,
  current_ison_if_marked,
  pending_duration_group,
  current_meter_if_marked
}
```

Processing loop:

```text
for each sign_or_sign_group:
  identify whether it is:
    interval sign
    interval combination
    temporal sign
    consonant/ornament sign
    differentiator/local accidental
    phthora/modulator
    martyria/signature
    rhythm/meter sign
    ison marking

  if martyria/signature:
    update mode/pitch checkpoint

  if phthora/modulator:
    update current scale/mode/finalis context

  if interval sign/group:
    compute movement relative to previous pitch in current modal scale
    update pitch cursor

  if temporal sign:
    update duration of current, previous, or grouped notes

  if consonant sign:
    add accent/ornament/phrasing metadata

  if differentiator:
    alter local pitch by comma amount if known

  emit melody event, ison event, or metadata event
```

## Pitch Letters

Approximate reference mapping from the guide:

```text
Pa = D
Vou = E
Ga = F
Di = G
Ke = A
Zo = B
Ni = C
```

Rules:

- Treat these as modal pitch labels.
- Do not assume equal temperament.
- If starting pitch/martyria is missing, output relative intervals instead of absolute Western notes.
- Keep ison as a separate drone/support line, not as harmony.

## Main Interval Signs

| Symbol | Codepoint | Name | SLM pitch action | Metadata |
|---:|---|---|---|---|
| 𝁆 | U+1D046 | Ison | `pitch = previous_pitch` | repeat |
| 𝁇 | U+1D047 | Oligon | `pitch += 1 modal degree` | unaccented |
| 𝁉 | U+1D049 | Petasti | `pitch += 1 modal degree` | accented |
| 𝁎 | U+1D04E | Kentimata | `pitch += 1 modal degree` | soft/legato/tied to previous |
| 𝁏 | U+1D04F | Kentima | upward third when validly combined | leap modifier |
| 𝁐 | U+1D050 | Ypsili | upward fifth when validly combined | leap modifier |
| 𝁑 | U+1D051 | Apostrophos | `pitch -= 1 modal degree` | descending step |
| 𝁓 | U+1D053 | Hyporroi / Yporroi | emit `-1`, then `-1` | two descending steps, tied/connected |
| 𝁕 | U+1D055 | Elaphron | descending third | can stand alone in modern notation |
| 𝁖 | U+1D056 | Hamili / Chamili | descending fifth | can stand alone in modern notation |
| 𝁈 | U+1D048 | Oxeia | upward step variant | abrupt/accented; older/occasional modern |
| 𝁊 | U+1D04A | Koufisma | upward step variant | weak/hesitant or tremolo in older interpretations |
| 𝁍 | U+1D04D | Pelaston | upward step variant | more intense than petasti |
| 𝁒 | U+1D052 | Apostrofoi Syndesmos | long descending second | older/specialized |
| 𝁔 | U+1D054 | Kratimoyporroon | two descending steps | preceding note lengthened |

Do not convert `+1 modal degree` to fixed semitones until the current scale comma pattern is known.

## Interval Combination Rules

Use these rules before emitting pitch.

1. Some stacked signs are cumulative: add interval values.
2. Some stacked signs are subordinated: one sign gives interval, the host gives accent/quality.
3. In subordination, the host sign's own upward second does not count.
4. Oligon host means unaccented.
5. Petasti host means accented.
6. Kentima and ypsili should not be treated as ordinary isolated modern signs.
7. Kentimata + oligon may be a two-note sequence, not one event.

High-value combinations:

| Parsed sign/group | Emit pitch action | Metadata |
|---|---|---|
| Oligon over petasti | up third | cumulative |
| Kentima over oligon | up fourth | unaccented |
| Kentima over petasti | up fourth | accented |
| Ypsili at right/middle of oligon | up fifth | unaccented, subordinated |
| Ypsili at right/middle of petasti | up fifth | accented, subordinated |
| Ypsili at left of oligon | up sixth | unaccented, cumulative |
| Ypsili at left of petasti | up sixth | accented, cumulative |
| Ypsili next to kentima over oligon | up seventh | unaccented |
| Ypsili next to kentima over petasti | up seventh | accented |
| Ypsili over kentima over oligon | up octave | unaccented |
| Ypsili over kentima over petasti | up octave | accented |
| Two ypsili over oligon | up ninth | unaccented |
| Two ypsili over petasti | up ninth | accented |
| Elaphron over apostrophos | down fourth | cumulative descent |
| Hamili over apostrophos | down sixth | cumulative descent |
| Hamili over elaphron | down seventh | cumulative descent |
| Hamili over elaphron over apostrophos | down octave | cumulative descent |
| Hamili over hamili | down ninth | cumulative descent |
| Ison over petasti | same pitch | accented unison |
| Apostrophos over oligon | down second | slightly accented |
| Apostrophos over petasti | down second | accented |
| Elaphron over petasti | down third | accented |
| Elaphron over apostrophos over petasti | down fourth | accented |

Two-note sequence groups:

| Parsed group | Emit events |
|---|---|
| Kentimata above oligon | `+1`, `+1`; second event legato |
| Oligon above kentimata | `+1`, `+1`; first event legato |
| Oligon before kentimata | same as oligon above kentimata |
| Apostrophos under ison | `0`, then `-1` |
| Two stacked apostrophoi | `-1`, then `-1` |

## Duration Rules

Default duration:

```text
duration = 1 beat
```

| Symbol | Codepoint | Name | Duration action |
|---:|---|---|---|
| 𝁿 | U+1D07F | Klasma ano | add 1 beat |
| 𝃴 | U+1D0F4 | Klasma kato | add 1 beat |
| 𝂏 | U+1D08F | Gorgon | current note and previous note share 1 beat |
| 𝃵 | U+1D0F5 | Lower gorgon | same as gorgon |
| 𝂕 | U+1D095 | Digorgon | three-note group in 1 beat |
| 𝂖 | U+1D096 | Trigorgon | four-note group in 1 beat |
| 𝂅 | U+1D085 | Apli | add 1 beat |
| 𝂆 | U+1D086 | Dipli | add 2 beats |
| 𝂇 | U+1D087 | Tripli | add 3 beats |
| 𝂈 | U+1D088 | Tetrapli | add 4 beats |
| 𝂗 | U+1D097 | Argon | specialized slowing in oligon-kentimata contexts |
| 𝂘 | U+1D098 | Imidiargon | argon-like, longer second note |
| 𝂙 | U+1D099 | Diargon | argon-like, still longer second note |

Gorgon handling:

```text
if gorgon on note N:
  duration(N-1) = 1/2 beat
  duration(N) = 1/2 beat
```

Digorgon/trigorgon:

```text
digorgon: notes [previous, current, following] fit in 1 beat
trigorgon: notes [previous, current, following, following+1] fit in 1 beat
```

Punctuated gorgon:

- Treat as quickened group plus lengthening dot(s).
- If exact tradition is unknown, preserve "punctuated gorgon" in output instead of forcing one rhythmic value.

Rests and breath:

| Symbol | Codepoint | Name | Output |
|---:|---|---|---|
| 𝁾 | U+1D07E | Stavros | breath |
| 𝂉 | U+1D089 | Koronis | fermata |
| 𝂊 | U+1D08A | Leimma enos chronou | 1-beat rest |
| 𝂋 | U+1D08B | Leimma dyo chronon | 2-beat rest |
| 𝂌 | U+1D08C | Leimma trion chronon | 3-beat rest |
| 𝂍 | U+1D08D | Leimma tessaron chronon | 4-beat rest |
| 𝂎 | U+1D08E | Leimma imiseos chronou | 1/2-beat rest |

## Consonant / Ornament / Phrasing Signs

These signs usually modify delivery rather than the main pitch cursor. Add metadata unless the context gives a specific pitch ornament.

| Symbol | Codepoint | Name | Metadata action |
|---:|---|---|---|
| 𝁘 | U+1D058 | Varia | accent/separation; may form rests with temporal signs |
| 𝁛 | U+1D05B | Omalon | light undulation |
| 𝁜 | U+1D05C | Antikenoma | lively ascent/grace-like gesture |
| 𝁚 | U+1D05A | Psifiston | accent then diminuendo over span |
| 𝁠 | U+1D060 | Eteron | slur/connection; not identical to Western tie |
| 𝁻 | U+1D07B | Endofonon | nasal tone; obsolete/restricted |
| 𝁼 | U+1D07C | Yfen kato | tie/lengthener, no inserted grace |
| 𝁽 | U+1D07D | Yfen ano | tie/lengthener, no inserted grace |

Guardrail: do not invent exact Western ornaments for rare or uncertain hypostases.

## Mode and Scale Rules

Use comma patterns for pitch approximation.

```text
ET whole tone = 12 commas
ET semitone = 6 commas

diatonic = [10, 8, 12, 12, 10, 8, 12]
soft_chromatic = [8, 14, 8, 12, 8, 14, 8]
hard_chromatic_ascending = [6, 20, 4, 12, 6, 20, 4]
hard_chromatic_descending = [4, 20, 6, 12, 4, 20, 6]
enharmonic_1881 = [12, 12, 6, 12, 12, 12, 6]
```

Mode snapshot:

| Mode | Approximate context | Scale behavior |
|---|---|---|
| I | D-based, finalis D sometimes A | diatonic |
| Plagal I | A-based, finalis A fast or D moderate | diatonic; B often flattened |
| II | G/E/D finalis possibilities | chromatic, typically soft |
| Plagal II | D/G finalis possibilities | chromatic, typically hard |
| III | F finalis | enharmonic; often needs general sharp/flat |
| Varys | F or B variants | multiple forms |
| IV | G/D/E finalis possibilities | diatonic; Legetos variant |
| Plagal IV | C finalis; with Ni phthora F finalis | diatonic; B may flatten |

Never assume:

```text
Mode I == D major
Plagal IV == C major
sharp == +1 ET semitone
flat == -1 ET semitone
finalis == Western tonic in all behavior
```

## Differentiators: Local Accidentals

Differentiators alter local pitch. Use comma values when known; otherwise preserve the sign name.

| Symbol | Codepoint | Name | SLM action |
|---:|---|---|---|
| 𝃋 | U+1D0CB | Fthora I Yfesis Tetartimorion / Agem | 1/3-tone flat in Chrysanthus; also possible modulator |
| 𝃍 | U+1D0CD | Yfesis Tritimorion | 2/3-tone flat in Chrysanthus |
| 𝃎 | U+1D0CE | Diesis Tritimorion | 2/3-tone sharp in Chrysanthus |
| 𝃏 | U+1D0CF | Diesis Tetartimorion | 1/3-tone sharp in Chrysanthus |
| 𝃐 | U+1D0D0 | Diesis Apli Dyo Dodekata | default sharp; semitone in Chrysanthus, two commas in 1881 |
| 𝃑 | U+1D0D1 | Diesis Monogrammos | +4 commas in 1881 |
| 𝃒 | U+1D0D2 | Diesis Digrammos | +6 commas in 1881 |
| 𝃓 | U+1D0D3 | Diesis Trigrammos | +8 commas in 1881 |
| 𝃔 | U+1D0D4 | Yfesis Apli Dyo Dodekata | default flat; semitone in Chrysanthus, two commas in 1881 |
| 𝃕 | U+1D0D5 | Yfesis Monogrammos | -4 commas in 1881 |
| 𝃖 | U+1D0D6 | Yfesis Digrammos | -6 commas in 1881 |
| 𝃗 | U+1D0D7 | Yfesis Trigrammos | -8 commas in 1881 |
| 𝃘 | U+1D0D8 | Geniki Diesis | raise all E's; guide's 1881 explanation raises E by two commas |
| 𝃙 | U+1D0D9 | Geniki Yfesis | flatten all B's |

## Phthorae: Modulators

Phthorae update the current scale/mode/finalis context. They are not just local accidentals.

```text
if phthora occurs on its home pitch:
  change current scale/mode context
else:
  transpose modal frame/finalis according to phthora's associated pitch
```

| Symbol | Codepoint | Name | SLM action |
|---:|---|---|---|
| 𝂶 | U+1D0B6 | Enarxis kai phthora Vou | set/transpose to diatonic E/Vou context |
| 𝂺 | U+1D0BA | Fthora diatoniki Pa | set/transpose to diatonic D/Pa context |
| 𝂻 | U+1D0BB | Fthora diatoniki Nana | set/transpose to diatonic F/Ga/Nana context |
| 𝂽 | U+1D0BD | Fthora diatoniki Di | set/transpose to diatonic G/Di context |
| 𝂿 | U+1D0BF | Fthora diatoniki Ke | set/transpose to diatonic A/Ke context |
| 𝃀 | U+1D0C0 | Fthora diatoniki Zo | set/transpose to diatonic B/Zo context |
| 𝃁 | U+1D0C1 | Fthora diatoniki Ni kato | set/transpose to diatonic low C/Ni context |
| 𝃂 | U+1D0C2 | Fthora diatoniki Ni ano | set/transpose to diatonic high C/Ni context |
| 𝃃 | U+1D0C3 | Fthora malakon chroma difonias | soft chromatic Di/G context; also Ni/Vou/Zo without transposition |
| 𝃄 | U+1D0C4 | Fthora malakon chroma monofonias | soft chromatic Ke/A context; also pitch-indicator use |
| 𝃅 | U+1D0C5 | Fthora skliron chroma vasis | hard chromatic Pa/D base context |
| 𝃇 | U+1D0C7 | Fthora Nenano | hard chromatic Di/G or Nenano context |
| 𝃈 | U+1D0C8 | Chroa Zygos | chroa: sharpen D and F in guide's summary |
| 𝃉 | U+1D0C9 | Chroa Kliton | chroa: sharpen E and F |
| 𝃊 | U+1D0CA | Chroa Spathi | chroa: sharpen G and flatten B |
| 𝃋 | U+1D0CB | Agem | on Zo/Vou/Ga, flatten B in Plagal I and indicate enharmonic scale |
| 𝃘 | U+1D0D8 | General sharp | global E raising |
| 𝃙 | U+1D0D9 | General flat | global B flattening |

## Martyriai / Signatures

Martyriai are mode/pitch checkpoints. Use them to initialize or correct state.

| Symbol | Codepoint | Meaning |
|---:|---|---|
| 𝂢 | U+1D0A2 | First mode signature |
| 𝂣 | U+1D0A3 | Other First mode; pitch indicator for diatonic Pa/Ke |
| 𝂤 | U+1D0A4 | Second mode signature |
| 𝂥 | U+1D0A5 | Other Second mode; chromatic pitch indicator in Modes II/Plagal II |
| 𝂦 | U+1D0A6 | Third mode signature variant |
| 𝂧 | U+1D0A7 | Trifonias/Fourth; Mode III and diatonic Ga/high Ni indicator |
| 𝂨 | U+1D0A8 | Fourth mode; Modes IV/Plagal IV and diatonic Ni/Di indicator |
| 𝂩 | U+1D0A9 | Tetartos Legetos; diatonic Vu/Zo indicator |
| 𝂪 | U+1D0AA | Alternative Legetos |
| 𝂫 | U+1D0AB | Plagal marker |
| 𝂬 | U+1D0AC | Isakia remnant |
| 𝂭 | U+1D0AD | Apostrofoi/dots remnant |
| 𝂮 | U+1D0AE | Fanerosis tetrafonias |
| 𝂯 | U+1D0AF | Fanerosis monofonias, Varys-related |
| 𝂰 | U+1D0B0 | Fanerosis monofonias, Mode II-related |
| 𝂱 | U+1D0B1 | Varys mode; low Zo indicator |
| 𝂲 | U+1D0B2 | Proto-Varys |
| 𝂳 | U+1D0B3 | Plagal Fourth |
| 𝂴 | U+1D0B4 | Gorthmic single nu |
| 𝂵 | U+1D0B5 | Gorthmic double nu |

If a martyria contradicts the current pitch cursor, suspect:

- missed interval combination,
- missed phthora/modulation,
- wrong octave/register,
- incorrect assumption about starting pitch.

## Meter Signs

| Symbol | Codepoint | Name | SLM action |
|---:|---|---|---|
| 𝃚 | U+1D0DA | Diastoli apli mikri | simple/small barline |
| 𝃛 | U+1D0DB | Diastoli apli megali | simple/large barline |
| 𝃜 | U+1D0DC | Diastoli dipli | double barline |
| 𝃝 | U+1D0DD | Diastoli theseos | internal downbeat divider |
| 𝃞 | U+1D0DE | Simansis theseos | one-beat downbeat |
| 𝃟 | U+1D0DF | Simansis theseos disimou | two-beat downbeat |
| 𝃠 | U+1D0E0 | Simansis theseos trisimou | three-beat downbeat |
| 𝃡 | U+1D0E1 | Simansis theseos tetrasimou | four-beat downbeat |
| 𝃢 | U+1D0E2 | Simansis arseos | one-beat upbeat |
| 𝃣 | U+1D0E3 | Simansis arseos disimou | two-beat upbeat |
| 𝃤 | U+1D0E4 | Simansis arseos trisimou | three-beat upbeat |

Meter may be absent. Do not infer Western bar structure unless signs or text support it.

## Output Policy for an SLM

When converting Byzantine to Western-like output:

1. If starting pitch/mode is known, output approximate staff pitches plus modal/microtonal annotations.
2. If starting pitch/mode is unknown, output relative intervals.
3. Preserve ison as a separate line.
4. Preserve phthorae as scale/mode changes, not local accidentals.
5. Mark microtones with comma values, arrows, or explicit uncertainty.
6. Do not add harmony, chords, key signatures, or time signatures unless the source has them.
7. If a sign is rare/pre-modern/uncertain in the source, output the sign name and `uncertain value`.

## Hallucination Guards

- Do not assume `Mode I = D major`.
- Do not assume `Plagal IV = C major`.
- Do not assume `sharp = +1 semitone`.
- Do not assume `flat = -1 semitone`.
- Do not collapse petasti into oligon without preserving accent.
- Do not treat gorgon as affecting only the marked note.
- Do not treat phthora as a one-note accidental.
- Do not turn ison into a bass progression.
- Do not invent values for obscure Middle/Late hypostases.
- Do not output confident absolute pitches when martyria or starting pitch is absent.
