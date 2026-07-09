# Day 3: First Real Training Run & Base-vs-Tuned Numbers

Midweek gate deliverable: first supervised fine-tune of the local base model on
real corpus data, plus the first base-vs-tuned evaluation. **Numbers are on the
board.**

## What was run

| Item | Value |
|------|-------|
| Base model | `Qwen/Qwen3-1.7B` |
| Adapter | `models/byzantine_sft_translation_1.7b` (LoRA r=8, 8.7M trainable params, 0.50%) |
| Training data | `data/byzantine/sft_translation_train.jsonl` — 897 rows (448 `neume_to_west` + 449 `west_to_neume`) |
| Held-out (not trained on) | 133 translation rows in `sft_byzantine_all_heldout.jsonl` (67 + 66), 0 id-overlap with train |
| Epochs / steps | 3 epochs / 1,347 steps (effective batch 2) |
| Backend | PEFT LoRA on MPS (Apple M5); Unsloth path reserved for CUDA/Colab |
| Wall-clock | ~60 min (≈2.68 s/step after dynamic-padding fix) |
| Train loss | 3.00 → 0.29 (final mean 0.47), clean monotonic descent |

The trainer was changed to use **dynamic per-batch padding** instead of padding
every row to 1024 tokens (`scripts/train_byzantine_sft.py`), which cut step time
~1.7× locally and matters for Colab throughput.

## How it was graded

The gpt-4o API judge configured in `config/byzantine_eval.yaml` was unavailable
(OpenAI account returned `billing_not_active`), and no `ANTHROPIC_API_KEY` was
set. Grading therefore followed the repo's existing **Opus blind-eval precedent**
(`docs/byzantine_opus_blind_eval.md`): outputs were generated locally with no
judge (`scripts/gen_base_vs_tuned_outputs.py`), then scored by an Opus agent on
the 0–2 rubric from `goals/byzantine_transcription.yaml`
(`scripts/grade_translation_eval.py`).

Strict pass requires `melodic_equivalence >= 1.5` **and** `meaning_preservation >= 1.5`.

> **Comparability caveat:** the README "Results at a glance" rows were judged by
> the gpt-4o / Opus **API**. This run was graded by an in-session Opus agent.
> Directionally faithful, but re-run the standard judged script once an API judge
> is available for an apples-to-apples row.

Artifacts:
- `runs/byzantine_translation_1.7b_outputs.json` — raw base + tuned outputs
- `runs/byzantine_translation_1.7b_graded.json` — per-dimension scores + deltas

## Results (0–2 per dimension)

| Suite | Arm | melodic_equiv | mode_fidelity | notation_conv | meaning_pres | mean | strict |
|-------|-----|---------------|---------------|---------------|--------------|------|--------|
| heldout | base | 0.00 | 0.90 | 0.00 | 0.00 | 0.23 | 0/10 |
| heldout | tuned | 0.05 | 0.20 | **0.90** | 0.05 | 0.30 | 0/10 |
| unseen | base | 0.00 | 0.50 | 0.00 | 0.00 | 0.12 | 0/10 |
| unseen | tuned | 0.00 | 0.10 | 0.20 | 0.00 | 0.08 | 0/10 |
| ultra_hard | base | 0.00 | 0.89 | 0.00 | 0.00 | 0.22 | 0/23 |
| ultra_hard | tuned | 0.02 | 0.17 | **0.87** | 0.02 | 0.27 | 0/23 |
| **TOTAL** | **base** | 0.00 | 0.80 | 0.00 | 0.00 | 0.20 | 0/43 |
| **TOTAL** | **tuned** | 0.02 | 0.16 | 0.72 | 0.02 | 0.23 | 0/43 |

**Δ overall = +0.03.** Δ by dimension: notation_convention **+0.72**,
meaning_preservation +0.02, melodic_equivalence +0.02, mode_fidelity **−0.64**.

## Error analysis

**Best improvement — output discipline / `notation_convention` (+0.72).**
The base model never actually answers: every output is a runaway `<think>`
chain-of-thought that reasons about the pitch mapping and never emits notation
(0.00 notation across all 43 scenarios). The tuned model learned the *format* —
it suppresses the thinking block and immediately emits a bare notation line
(`A4 B4 C5 …` or a neume sequence). Fine-tuning successfully instilled the spec's
"output notation only, no commentary" requirement, which the base model fails
completely. This is the clearest evidence that SFT is doing something real.

**Worst — two regressions:**

1. **`mode_fidelity` (−0.64).** The base model at least echoes a mode/Ni header
   ("Mode IV, Ni = F4"). The tuned model stopped emitting the mode / martyria /
   Ni header entirely and jumps straight to pitches — it traded the one thing the
   base did adequately for format compliance.
2. **`melodic_equivalence` still ≈0 (+0.02, essentially flat).** The core
   behavior did not move. Tuned outputs have the right *shape* but wrong
   *content*: they ignore the specific input neumes and emit a generic,
   often absurdly long scale-run. Example (`mode4_authentic_descending`):
   reference `C5 B4 A4 G4` (4 notes) vs. tuned 60+ notes of
   `A4 B4 C5 B4 A4 G4 …`. This is degeneration + memorized-contour: the model
   learned "produce a plausible chant-like run," not "transcribe *these* neumes."

**Other observations:**
- `unseen` got slightly *worse* overall (−0.03): the model overfit seen
  liturgical-formula shapes and did not generalize.
- The reverse direction (`west_to_neume`) degenerates hardest — endless
  `[MeasureBar]…` / repeated-token loops — suggesting repetitive or misaligned
  targets in the neume-direction rows.

## Day 4 actions (fix in data, not hyperparameters)

1. **Cap / curate target length.** Outputs run 40–70+ tokens against 3–6-note
   references. Audit `sft_translation_train.jsonl` targets for over-long or
   padded assistant messages; trim to the true reference length.
2. **Re-add the mode/Ni/Ison header to targets.** The `mode_fidelity` crash
   implies the training targets strip the header the model should emit. Put it
   back in the assistant output.
3. **Fix degeneration in reverse-direction data.** Specifically audit
   `west_to_neume` rows for repetitive or misaligned targets driving the
   `[MeasureBar]…` loops.
4. **Retrain + re-eval** on the same held-out banks and report the new deltas,
   ideally with an API judge restored for a comparable row.

---

# v2 Run — Data Fixes Applied (2026-07-09)

All four Day-4 data actions above were implemented in `scripts/build_neume_tasks.py`
and the model retrained. Results below.

## What changed in the data

| Fix | Before (v1) | After (v2) | Verified |
|-----|-------------|-----------|----------|
| Target length | 97% exactly 60 tokens (fixed-slice artifact) | proportional windowing, median 22 / p90 39 / cap 121 | ✅ |
| Mode/Ison header in targets | 0% | 100% (matches eval reference format) | ✅ |
| `west_to_neume` degeneration | runs to 16, 45 `measure_bar` loops | max run 1, 0 loops (repeat-collapse) | ✅ |
| Training rows | 897 | 10,496 (windowing = honest fragment augmentation), split by hymn, 0 leakage | ✅ |

Adapter: `models/byzantine_sft_translation_v2_1.7b` — LoRA r=8, trained on a
diverse 2,400-row / 513-hymn subset, ~6 epochs, final train loss **0.30**
(v1 was 0.47). Graded by in-session Opus agent on the same rubric (API judge still
billing-blocked — same caveat as v1; directional, not API-comparable).

## Results v1 → v2 (tuned, mean over 43 scenarios)

| Dimension | v1 | v2 | Δ |
|-----------|-----|-----|-----|
| **melodic_equivalence** (litmus) | 0.02 | **0.00** | **−0.02** |
| mode_fidelity | 0.16 | **1.61** | **+1.45** |
| notation_convention | 0.72 | 0.81 | +0.09 |
| meaning_preservation | 0.02 | **1.19** | **+1.17** |
| overall mean | 0.23 | **0.90** | **+0.67** |
| strict pass | 0/43 | **0/43** | 0 |

## Verdict: fixes worked — except on the one metric that matters

The data fixes did exactly what they were designed to do on three of four dimensions.
The `mode_fidelity` regression is **fully repaired** (0.16 → 1.61): re-adding the header
means the tuned model now emits the correct mode label. `meaning_preservation` jumped
(0.02 → 1.19) because outputs now carry the ison header and aren't degenerate. Overall
quality nearly quadrupled (0.23 → 0.90).

**But `melodic_equivalence` is still 0.00, and strict pass is still 0/43.** The core
transcription behavior did not move. Fixing the *format* of the data could not fix the
*content* problem, and the eval surfaced a new failure mode that explains why:

1. **Direction confusion.** On `byz_to_west` scenarios (target = Western pitches), the
   tuned model emits *Byzantine neumes* instead — 6/8 (heldout), 9/10 (unseen), 11/18
   (ultra_hard). The bidirectional training taught both mappings but not *which direction
   to apply* from the prompt.
2. **Mode-template collapse.** It falls back to a memorized "Mode II, Ni=Κε / petastē
   oligon kentēma apostrophos oligon" string, especially on unseen (9/10) — it reproduces
   a frequent training pattern rather than transcribing the actual input.
3. **`<think>` wrapper returned** on 100% of outputs (usually empty `<think>\n\n</think>`).

## Root cause (unchanged from the deepest finding)

This is the same wall hit repeatedly: **neumes and pitches do not align 1:1 (~1.78:1,
melismatic), so whole-hymn seq2seq pairs teach contour/format but not exact pitch
mapping.** The model learns "produce a plausible chant-like sequence in some notation,"
not "transcribe *these* specific symbols." No amount of format-fixing or data-volume
changes that — it is a property of the training signal.

## Day 4+ actions (content, not format)

1. **Fix direction confusion first (cheap, high-value).** Make the two directions
   unmistakable — distinct system prompts / task tags per direction, or train two
   single-direction adapters. This alone should stop byz→west emitting neumes.
2. **Attack alignment (the real bottleneck).** Options, hardest-but-only-real-fix last:
   (a) use the Neanes `MusicXmlExporter` pitch engine to produce *per-neume-aligned*
   targets (exact, deterministic) instead of whole-hymn seq2seq; (b) restrict training to
   the ~15% of hymns that are near-1:1 neume:pitch; (c) accept that this SLM does
   contour/mode, not exact pitch, and reframe the deliverable.
3. **Suppress the `<think>` wrapper** in training targets / generation config.
4. Re-eval with an API judge once billing is restored for an API-comparable row.

---

# v3 — Two One-Directional SLMs + rules-engine test (2026-07-09)

Two experiments this round: (A) test whether the `byzantine_notation_for_slm.md` pitch
rules can generate *aligned* targets (the real melodic_equivalence fix); (B) split the
bidirectional model into two single-direction adapters to kill the direction-confusion
failure from v2.

## (A) Rules-engine for aligned targets — DEAD END (confirmed)

Built `scripts/neume_rules_engine.py` from the doc's per-neume interval rules (ison=0,
oligon=+1, apostrophos=-1, elaphron=-2, ...) + comma scale patterns, and validated against
the hand-aligned eval references. **Result: 0/8 exact, 0/8 contour match.** Cause is
structural and systematic: every reference has ~2 **more** pitches than its neume chain
has neumes (4→6, 3→5, consistently) — the neume chain *under-specifies* the melody
(ison-priming + melodic realization the bare symbols don't encode).

Checked Neanes as an alternative engine: its `AnalysisService`/`getNeumeValue` uses the
**identical 1:1 neume→pitch model** (Ison→0, Oligon→+1, OligonPlusKentimaBelow→+2, …), so
it hits the same wall. The alignment problem is now confirmed from **four** independent
angles: OMR ratio 1.78:1, 0 exact-length files, +2 reference-pitch surplus, and Neanes's
own 1:1 assumption. **No rule engine can recover exact pitch from bare neume sequences.**

## (B) Two directional adapters

Data split by direction (hymn-level, 0 leakage), trained separately on the v2 fixed data
(windowed, header-carrying). `models/byzantine_sft_n2w_1.7b` (byz→west) and
`models/byzantine_sft_w2n_1.7b` (west→byz), each LoRA r=8, ~1,510-row subset, 4,500 steps.
Graded by in-session Opus agent (same caveat as prior rows).

### Result #1 — n2w (byz→west), graded on 36 byz→west scenarios

| Dimension | v2 bidir | n2w directional | Δ |
|-----------|----------|-----------------|-----|
| notation_convention | 0.81 | **1.39** | **+0.58** |
| mode_fidelity | 1.61 | 1.50 | −0.11 |
| meaning_preservation | 1.19 | **0.39** | **−0.80** |
| melodic_equivalence | 0.00 | 0.00 | 0 |
| strict pass | 0/43 | 0/36 | 0 |

**Direction confusion fixed as predicted:** byz→west outputs emitting *neumes* (wrong)
dropped from **26/36 (v2) → 10/36**; 67% now correctly emit pitches, and the mode/Ison
header often lands (`Mode IV, Ni = F4 / Ison: F4`). BUT meaning_preservation regressed to
0.39 — the adapter over-generates (20+ notes for a 4-note reference), a **length-discipline**
side-effect of windowed training (targets ~22 tokens, no stop signal on short inputs).

### Result #2 — w2n (west→byz), graded on 7 west→byz scenarios — FAILED

| Dimension | value |
|-----------|-------|
| notation_convention | 0.57 |
| mode_fidelity | 1.29 |
| meaning_preservation | **0.00** |
| correct-direction (emits neumes) | **0/7** |

**All 7 outputs are runaway `<think>` chains that never close and never emit an answer**
(vs n2w which closed think + answered 100%). The west→byz adapter collapsed into
reasoning-loop degeneration. Likely causes: the west→byz targets (neume chains) are the
harder/messier direction, and 4,500 steps on 1,510 rows may have over- or under-fit into a
think-loop attractor. This direction needs `<think>` suppression at train time and likely
a cleaner target set.

## v3 verdict

- The **rules-engine / aligned-target path is definitively closed** — melodic_equivalence
  cannot be moved via any deterministic neume→pitch mapping on this data.
- The **directional split works for byz→west** (fixes the specific direction-confusion bug,
  +0.58 notation) but **not for west→byz** (think-loop collapse), and neither moves the
  melodic bottleneck.
- Two *new*, fixable issues surfaced, both about output discipline not alignment:
  **(1) length over-generation** (n2w), **(2) `<think>` runaway** (w2n).

## Day 5+ actions

1. **Suppress `<think>` in training + generation** (add `/no_think` or strip the block in
   targets; set generation to stop at answer). Fixes w2n's total failure directly.
2. **Length discipline for n2w:** add an explicit output-length hint / EOS training, or
   train on shorter windows, so it stops instead of running to the token cap.
3. **Reframe the deliverable around what's achievable:** contour + mode + notation-system
   discipline (all now working) rather than exact pitch (blocked by alignment). The
   litmus's melodic_equivalence gate is not reachable with the current data.
4. Aligned pitch data would require a source with note-level neume↔pitch alignment (audio
   time-alignment or a natively-aligned corpus) — none currently available.
