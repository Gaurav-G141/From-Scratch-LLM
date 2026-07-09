# Byzantine ↔ Western Transcription — Handoff

Written 2026-07-09 for the next agent, **especially one with ideas for
`melodic_equivalence`**. Read the "Melodic-equivalence wall" section first — it is the
one metric that has never moved, and it is where your ideas matter. The rest is context so
you don't repeat work that's already been ruled out.

---

## 1. Goal & current standing

Train a small local model (`Qwen/Qwen3-1.7B` + LoRA) to transcribe **between Byzantine
neume notation and Western staff notation**, scored on a 0–2 rubric across four dimensions
(`goals/byzantine_transcription.yaml`). Strict pass = `melodic_equivalence ≥ 1.5` AND
`meaning_preservation ≥ 1.5`.

**Where things stand:**
- Frontier prompting (GPT-4o, Opus 4) fails the task → SFT is justified (Day 2 litmus).
- We built a clean deterministic dataset and trained several LoRA adapters.
- Three of four dimensions now respond to training. **`melodic_equivalence` is stuck at
  ~0.00 across every experiment. Strict pass is 0/43 in every run.**

| Dimension | best tuned result | status |
|---|---|---|
| notation_convention | 1.39 | works (esp. after directional split) |
| mode_fidelity | 1.61 | works (after re-adding mode header) |
| meaning_preservation | 1.19 | works when output length is disciplined |
| **melodic_equivalence** | **0.00** | **never moved — the open problem** |

---

## 2. The melodic-equivalence wall (READ THIS)

**Core finding: the neume sequences we can extract structurally UNDER-SPECIFY the melody,
so no model or rule can recover exact pitches from them.** Byzantine neumes encode
*relative, context-dependent* motion with melisma (one neume held over several notes),
ison-priming, and modal/microtonal realization that the bare symbol stream does not carry.

### Why neume→note is not 1:1 (the musical reasons)

Byzantine notation is fundamentally different from Western staff notation: it is
**prescriptive of contour and gesture, not of an explicit note-for-note sequence.** It
tells a trained cantor *how to move*; the singing tradition fills in the actual notes.
Western notation is the opposite — every note is spelled out. So the conversion is
inherently **1-to-many and context-dependent.** Four concrete mechanisms cause the count
mismatch (they stack):

1. **Melisma — one neume, many notes (dominant cause).** A single neume, especially when
   combined with cheironomic/"great hypostases" signs, expands into a multi-note ornamental
   figure: 1 written symbol → 3–5 sung notes. Chant is inherently melismatic (many notes
   per syllable). This is the main driver of the observed **1.78 notes per pitch-bearing
   neume**.
2. **The ison adds notes with no neume.** The ison is a held drone; the Western
   transcription writes the drone/opening pitches out as real notes, but on the Byzantine
   side the ison is a *separate support marking*, not part of the neume chain. This is
   exactly the "+2 pitches" surplus seen in every eval reference (§2, confirmation 3).
3. **Many neume tokens carry zero pitch.** The stream includes martyria (mode signatures),
   gorgon/argon (tempo), breath marks, and fthora (modulation) — all extracted as tokens
   but none of which are notes. So neume-count and note-count diverge even before melisma.
4. **Realization depends on mode + microtonal genus.** The same neume ("up one degree")
   resolves to a whole tone, a small semitone, or a microtonal step depending on the mode's
   comma pattern (diatonic / soft-chromatic / hard-chromatic / enharmonic). This doesn't
   change the *count*, but it's why even a correct count wouldn't yield correct pitches
   without the full modal context the symbols omit.

Bottom line: the exact-pitch information **is not present in the neume symbols** — it lives
in the melismatic expansion, the ison, and the oral/modal tradition. That is why every
attempt to recover pitch from the neume stream alone has failed, and why the promising
leads below all involve sourcing data where that realization is actually captured
(audio, or a natively-authored score).

This under-specification is confirmed from **four independent angles** — please engage with
these before proposing a fix, because any idea that assumes recoverable 1:1 alignment will
fail the same way:

1. **OMR ratio.** Across 495 hymns with both sides extracted, pitch-bearing-neume : pitch
   ratio is **1.78 : 1** (median). Only 15% of hymns are within 0.9–1.1.
2. **Zero exact-length files.** Of 436 hymns, **0** have equal neume and pitch counts.
3. **Reference surplus.** In the hand-authored eval scenarios, every `reference_output`
   has ~2 **more** pitches than its input neume chain has neumes (4→6, 3→5, consistently).
4. **Neanes agrees.** The reference open-source Byzantine engine (`neanes/neanes`,
   `AnalysisService` + `getNeumeValue`) uses a strict **1:1 neume→pitch** model
   (Ison→0, Oligon→+1, OligonPlusKentimaBelow→+2 …). It only works on scores authored
   natively in Neanes where every note is explicitly placed; it cannot recover alignment
   from a bare neume-name list either.

### What has already been TRIED and FAILED for melodic_equivalence
- **Whole-hymn seq2seq** (neume seq ↔ pitch seq): model learns "emit a plausible chant-like
  run," not "transcribe *these* symbols." melodic 0.00.
- **Windowed/fragment pairs** (proportional slicing to ~24-neume windows): fixed length &
  format bugs, 12× more rows — melodic still 0.00.
- **Rules engine from `docs/byzantine_notation_for_slm.md`** (`scripts/neume_rules_engine.py`):
  deterministic relative-interval mapping. **0/8 exact, 0/8 contour** vs eval refs. Fails
  because of the +2 surplus above.
- **Directional split** (separate byz→west / west→byz adapters): fixed *direction
  confusion*, did nothing for melodic (still 0.00).

### Ideas NOT yet tried (candidate leads for you)
These are the paths that could plausibly break the wall, roughly ordered by promise:
1. **Audio-anchored alignment.** Several sources (e.g. GOA DCS, newbyz) publish/reference
   MIDI or recordings. Time-aligning audio → pitch → neume could yield true note-level
   alignment. **This is the most likely real fix** but needs an audio-alignment pipeline we
   have not built. No aligned audio is currently in the repo.
2. **Neume-native corpus.** A `.byz`/`.byzx` corpus authored in Neanes carries explicit
   per-note placement → deterministic MusicXML with exact pitches. Only 12 example files
   are public (`github.com/neanes/neanes/examples`). Sourcing a real corpus (community,
   Trisagion School) would make Neanes's engine actually usable.
3. **Restrict to the ~15% near-1:1 hymns.** Small, cleaner training set where alignment is
   approximately valid. Low yield but honest signal; good for a controlled test of whether
   *any* aligned data moves the needle.
4. **Model melisma explicitly.** Learn/annotate how many notes each neume spans (a duration
   model) so the mapping becomes 1→N instead of assumed 1→1. Hard; needs aligned data to
   train the span model — chicken-and-egg with #1.
5. **Reframe the metric.** If exact pitch is unrecoverable, score *contour* / interval-class
   equivalence instead of exact pitch. This is a deliverable change, not a model fix — but
   may be the honest endpoint. (See §6.)

---

## 3. The data pipeline (what's trustworthy)

Everything is deterministic — NO vision-model guessing (the original vision-extracted data
was proven to confabulate and was discarded; see `docs/byzantine_omr_western_data.md`).

**Western side (pitches): trustworthy, exact.**
- `scripts/omr_extract_western.py` — Audiveris OMR (`tools/Audiveris.app`, bundled JRE) +
  music21 → exact pitch sequences. 775/786 files OK.
- Output: `data/byzantine/omr/omr_{goa,newbyz,sam}.jsonl`.
- Caveat: Audiveris reads a plain treble clef; scores are treble-8, so absolute octave is
  uniformly one high but internally consistent (irrelevant for contour/interval work).

**Byzantine side (neumes): trustworthy as SYMBOLS, not as pitch.**
- `scripts/extract_neumes.py` — reads EZ/ED Byzantine music-font glyphs (stable ASCII
  layout) → named neume sequences via `data/byzantine/ez_neume_map.json` (from the official
  EZ character tables). 100% glyph→name coverage on font PDFs.
- `scripts/extract_neumes_vector.py` — recovers 89 GOA hymns whose neumes are vector paths
  (no font): perceptual-hash clustering → name via bitmap match + frequency alignment.
  ~84% named.
- Outputs: `data/byzantine/neumes_{goa-dcs,new-byzantium,st-anthonys}.jsonl`,
  `neumes_vector.jsonl`.

**Parallel pairs:** 581 hymns have both a neume sequence AND OMR pitches (validated same-hymn
pairing). This is the bidirectional corpus.

---

## 4. Training-data files (current)

Built by `scripts/build_neume_tasks.py` (neume tasks) + `scripts/build_western_tasks.py`
(Western-only tasks), split by `scripts/build_translation_split.py` (splits by HYMN stem —
important: windows of one hymn must not leak across train/heldout).

| File | Contents |
|---|---|
| `data/byzantine/sft_byzantine_all_train.jsonl` | combined all-task set (Western + neume) |
| `data/byzantine/sft_translation_train.jsonl` / `_heldout.jsonl` | windowed neume↔pitch, both directions |
| `data/byzantine/sft_n2w_train_sub.jsonl` | byz→west only (directional adapter) |
| `data/byzantine/sft_w2n_train_sub.jsonl` | west→byz only (directional adapter) |

Task types: `neume_to_west`, `west_to_neume` (the translation pairs), plus single-modality
augmentation (`continuation`, `contour`, `mode_id`, `transpose`, `neume_read`,
`mode_from_neumes`).

**Known data-construction gotchas (fixed, but know them):**
- Never truncate targets to a fixed length (v1 bug: 97% of targets were exactly 60 tokens →
  model learned "always emit 60 tokens"). Use proportional windowing.
- Targets MUST carry the `Mode …\nIson: X4` header or mode_fidelity collapses.
- `west_to_neume` targets need repeat-collapse or they degenerate into `measure_bar` loops.
- Neume:pitch is NOT 1:1 — do not hand-align windows note-for-note.

---

## 5. Trained adapters & eval

Adapters in `models/`:
- `byzantine_sft_translation_v2_1.7b` — bidirectional, windowed+fixed data (v2 run).
- `byzantine_sft_n2w_1.7b` — byz→west only. Best notation_convention (1.39); fixed
  direction confusion (26→10 wrong of 36).
- `byzantine_sft_w2n_1.7b` — west→byz only. **FAILED**: all outputs are runaway `<think>`
  chains that never answer. Needs `<think>` suppression at train/gen time.

**Eval flow** (no API judge — billing blocked; graded by in-session Opus agent on the
rubric, consistent with `docs/byzantine_opus_blind_eval.md`):
```
.venv/bin/python scripts/gen_base_vs_tuned_outputs.py \
  --adapter-path models/<adapter> --suites heldout,unseen,ultra_hard \
  --out runs/<name>_outputs.json
```
Then grade the tuned outputs against `reference_output` per the 0–2 rubric.

**Two output-discipline bugs to fix (independent of the alignment wall):**
1. `<think>` runaway — the base model and the w2n adapter loop in reasoning and never emit
   an answer. Suppress via `/no_think`, stripping the block in targets, or a stop sequence.
2. Length over-generation — n2w emits 20+ notes for a 4-note reference (windowed-training
   side effect). Needs length hint / EOS discipline / shorter windows.

---

## 6. Honest assessment & recommendation

The project has produced **trustworthy deterministic data and a working pipeline for three
of four dimensions.** The remaining wall — `melodic_equivalence` — is a property of the
*data*, not the model or training recipe: neume sequences under-specify pitch. Confirmed
four ways (§2).

Two forward paths for the metric:
- **Break the wall with genuinely aligned data** — audio-time-alignment (§2 idea 1) or a
  neume-native `.byzx` corpus (§2 idea 2). These are the only paths that can move exact
  pitch. Both require sourcing/building alignment we don't have.
- **Reframe** melodic_equivalence to contour / interval-class equivalence (§2 idea 5),
  which the current data *can* support.

If your idea assumes exact pitch is recoverable from the neume stream alone, please first
reconcile it with the four confirmations in §2 — that's where every prior attempt died.

---

## 7. Key files index
- `docs/byzantine_omr_western_data.md` — data pipeline, per-file index, why vision data was replaced
- `docs/byzantine_day3_results_20260708.md` — v1/v2/v3 training results + all deltas (most detail)
- `docs/byzantine_notation_for_slm.md` — machine-oriented neume→pitch-action spec (basis of the rules engine)
- `goals/byzantine_transcription.yaml` — rubric, dimensions, thresholds
- `scripts/neume_rules_engine.py` — the deterministic engine that scored 0/8 (don't re-derive; extend if you have a melisma model)
- `scenarios/byzantine_transcription_{heldout,unseen,ultra_hard}.yaml` — eval banks (hand-aligned ground truth)
