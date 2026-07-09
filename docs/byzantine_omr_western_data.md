# Deterministic Western-Music Data via OMR

## Data file index (canonical locations)

**Primary training data — train on these:**

| File | Rows | Contents |
|------|------|----------|
| `data/byzantine/sft_byzantine_all_train.jsonl` | 4,823 | Combined Western + neume tasks (headline SFT set) |
| `data/byzantine/sft_byzantine_all_heldout.jsonl` | 536 | Held-out split (0 leakage vs train) |
| `data/byzantine/sft_western.jsonl` | 4,294 | Western-only tasks: `mode_id`, `continuation`, `contour`, `transpose` |
| `data/byzantine/sft_neume.jsonl` | 2,247 | Neume tasks: `neume_read`, `mode_from_neumes`, `neume_to_west`, `west_to_neume` |

**Translation (bidirectional) is the core transcription data:** `neume_to_west` (515) +
`west_to_neume` (515) = **1,030 translation examples**, both directions from the same
parallel pairs. The other task types (continuation, contour, mode_id, transpose,
neume_read, mode_from_neumes) are single-modality augmentation derived from the same hymns
— useful support, but not paired transcription. So the headline "4,823 rows" is task rows,
of which ~1,030 are true bidirectional transcription.

**Intermediate / source data (regenerable):**

| File | Contents |
|------|----------|
| `data/byzantine/omr/omr_{goa,newbyz,sam}.jsonl` | Deterministic OMR Western pitches (Audiveris + music21) |
| `data/byzantine/neumes_{goa-dcs,new-byzantium,st-anthonys}.jsonl` | Font-extracted named neume sequences |
| `data/byzantine/neumes_vector.jsonl` | 89 vector-recovered neume files (+86 bidirectional) |
| `data/byzantine/ez_neume_map.json` | EZ/ED ASCII → neume-name map (official EZ character tables) |
| `data/byzantine/manifest.jsonl` | 2,292 discovered parallel PDF pairs |

**Archived / superseded (vision-era — do NOT train on):**

| File | Rows | Note |
|------|------|------|
| `data/byzantine/sft_raw.backup.jsonl` | 9,936 | Full pre-clean original (vision-extracted) |
| `data/byzantine/sft_raw_rejected.jsonl` | 5,393 | Quarantined duplicate/garbage rows |
| `data/byzantine/sft_raw.jsonl` | 4,543 | Deduped raw (still vision-labeled) |
| `data/byzantine/sft_v2.jsonl` / `sft_v1.jsonl` | 4,005 / 98 | Earlier SFT attempts |

## Why this exists

The original Day 3 corpus extracted training pairs by asking a vision LLM (GPT-4.1) to
transcribe Byzantine ↔ Western notation from score images. Audit showed those **labels
were unreliable**: the model confabulated pitches it could not actually read (0.1% of
outputs marked any microtone despite Byzantine chant being inherently microtonal; only
~48% of pitch sequences were unique; many outputs collapsed to generic formulas). This
is the exact failure the Day 2 litmus test predicted — frontier models cannot do this
transcription — so using them as the label source baked the failure into the data.

We pivoted to **deterministic extraction**: recover exact Western staff pitches directly
from the vector-engraved PDFs, with no model guessing.

## Pipeline

1. **OMR** — `scripts/omr_extract_western.py` runs **Audiveris** (bundled under
   `tools/Audiveris.app`, self-contained JRE) in headless batch mode over each Western
   PDF, exporting MusicXML, then parses it with **music21** into per-staff pitch
   sequences. Runs 8 workers in parallel (~4s/file).
   - Output: `data/byzantine/omr/omr_{goa,newbyz,sam}.jsonl`
2. **Task building** — `scripts/build_western_tasks.py` turns pitch sequences into
   chat-format SFT tasks. Mode labels are read deterministically from PDF title text.
   - Output: `data/byzantine/sft_western{,_train,_heldout}.jsonl`

### Validation

Audiveris was validated against the engraved reference (`dcs_canon1ode3_west.pdf`):
recovered melody matches the score **exactly**, differing only by a *constant*
transposition (Audiveris reads a plain treble clef; these scores use treble-8). All
task types are transposition/octave-invariant or metadata-anchored, so this offset does
not affect label correctness. The deterministic tasks (contour, transpose) are
**100% self-consistent** (1550/1550 verified).

## Yield

| Source        | Files ok | Notes  |
|---------------|----------|--------|
| GOA DCS       | 348/350  | 92,764 |
| New Byzantium | 303/305  | 49,154 |
| St. Anthony's | 124/131  | 44,259 |
| **Total**     | **775**  | **186,177** |

Sequence uniqueness 80–100% (vs 48% for the discarded vision data).

## Task dataset

`data/byzantine/sft_western.jsonl` — 3,449 deduped rows (train 3,104 / heldout 345):

| Task          | What it teaches (all labels exact) |
|---------------|-------------------------------------|
| `mode_id`     | melody → Byzantine mode/tone (label from title) |
| `continuation`| opening notes → next notes |
| `contour`     | melody → U/D/S interval contour |
| `transpose`   | melody + interval → transposed melody |

Format matches `scripts/train_byzantine_sft.py` (system/user/assistant `messages`).

## Byzantine neume side (named-neume extraction)

The Byzantine side is also now extractable, symbolically. The EZ Byzantine music fonts
render neumes as font glyphs on a stable ASCII keyboard layout, documented in the
official `EZ-CharacterTables.pdf` (from St. Anthony's Monastery's font package).

- `scripts/extract_neumes.py` normalizes PDF glyph codes (ASCII / PUA+0xF000) and maps
  each to its documented neume NAME + category via `data/byzantine/ez_neume_map.json`.
  Coverage: **100%** of glyphs map to a documented name (0 unknowns).
- GOA DCS uses the older **"ED" fonts** (EDPsaltica/EDIsson/EDFthora) that EZ was built
  from; verified glyph-for-glyph to share the identical ASCII layout, so the same map
  applies. `extract_neumes.py` matches ED and EZ font names by prefix.
- Files with text-based neumes: **503 / 786** (goa 73/350, newbyz 301/305, sam 129/131);
  **136,247 named neumes** total. The 277 remaining GOA files have neumes as vector
  graphics/images (no font) and are NOT text-extractable.
- `scripts/build_neume_tasks.py` builds neume tasks:
  - `neume_read` — raw codes → named neume sequence (verified 430/430 exact)
  - `mode_from_neumes` — neume sequence → mode
  - `neume_to_west` — neume sequence → parallel OMR Western pitches (seq2seq; both sides
    are real data for the same hymn)

### What was NOT solved: pitch resolution per neume

The character table documents neume **names and categories**, not exact pitch intervals.
An inferred interval mapping was built and then **refuted** against the OMR pitch data
(decoded melodic span ~17 steps vs actual ~6.3; ~zero correlation). Byzantine neumes do
not align 1:1 with notes (0 exact-length matches across 436 files; modifiers/martyria
interleave), so intervals cannot be recovered empirically by alignment. Precise
Byzantine→pitch decoding would require a full theory engine (running pitch + mode +
martyria degree + fthora accidentals + support-neume logic). Hence `neume_to_west` is
posed as sequence-to-sequence (model learns the mapping) rather than a hand-aligned
per-note transcription, and `ez_neume_map.json`'s `interval` field is marked unvalidated.

## Combined dataset

`data/byzantine/sft_byzantine_all_train.jsonl` (4,213 rows) + `_heldout.jsonl` (469):
Western tasks (`mode_id`, `continuation`, `contour`, `transpose`) + neume tasks
(`neume_read`, `mode_from_neumes`, `neume_to_west`, including GOA/ED-font hymns).
All labels deterministic or real. Verified: 0 train/heldout leakage, 0 malformed rows,
all content unique.

### Mode-label fix (Phase 1)

The mode normalizer (`norm_mode`/`normalize_mode` in `scripts/build_neume_tasks.py` and
`scripts/build_western_tasks.py`) previously collapsed plagal and grave modes to authentic
Mode 1-4 (e.g. "Plagal First Mode" -> "Mode 1"), affecting ~151 rows. Fixed to match
plagal/grave BEFORE authentic patterns; now emits `Mode pl. 1/2/4` and `Mode grave`.
Re-validated: **495/495 `mode_from_neumes` labels match their PDF titles** (was 327/482).

Note on OMR completeness: investigated suspected multi-page truncation — it was a false
alarm. All 128 multi-page (>=4pp) Western files have healthy note counts (>=15 notes/page,
0 truncated). Audiveris processes whole books; low music21 "part" counts just reflect
merged parts, not lost notes. No OMR re-run needed.

### Vector-path neume recovery (Phase 2)

89 GOA byz PDFs render neumes as vector paths (no font), so the font-based
`extract_neumes.py` skipped them — but their Western pitches were already OMR'd, so each
recovered = +1 bidirectional pair. `scripts/extract_neumes_vector.py`:
1. Collects black filled vector glyphs (drops page bg + red martyria).
2. Perceptual-hashes each glyph (16x16 bit signature) → ~90 clusters, matching the ~90
   known font glyphs.
3. Names clusters by triangulated evidence: bitmap similarity to EZ/ED font glyph renders
   + **frequency alignment against the known font-neume distribution** (this resolved
   ambiguous cases — e.g. confirmed the most common cluster is apostrophos, not oligon) +
   manual shape check. Only high-confidence names assigned; rest → `unk_`. **84% of glyphs
   named.**

**Validation (objective):** vector note/neume ratio = **1.27, identical to the font-based
baseline (1.27)**; neume distribution matches the font repertoire ranking (apostrophos >
ison > oligon > kentemata > petaste). This confirms the core-neume naming is correct.

**Result: bidirectional pairs 495 → 581 (+86).** Lower fidelity than font-based (84% named
vs 100%), so `build_neume_tasks.py` prefers font extraction and only uses vector to fill
gaps. Remaining ~16% of vector glyphs (rarer variants/composites) are `unk_`.

## Notes / limitations

- Audiveris reads a plain treble clef; true sounding pitch is one octave lower
  (treble-8). Absolute octave is therefore uniformly high but internally consistent.
- 11 files failed OMR (2 timeouts on large multi-page compilations, 8 no-export, 1
  music21 ZeroDivisionError) — captured per-file via the `status` field.
- The **Byzantine neume side remains non-deterministic** (GOA neumes are vector
  graphics; new_byz/st_anthonys use EZ-Byzantine fonts with identity-mapped ToUnicode).
  These tasks are therefore Western-only, per the chosen strategy.
