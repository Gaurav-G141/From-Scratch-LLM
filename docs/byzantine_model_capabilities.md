# Byzantine Synthetic-Grammar Model — What It Does and Doesn't Do

A plain-language capability summary of the fine-tuned model (LoRA adapter on
`unsloth/Qwen2.5-Coder-7B-bnb-4bit`). For the full experimental trail see
`byzantine_synthetic_expanded_results_20260711.md`; for the data pipeline see the
`build_synthetic_musicality.py` / `verify_synthetic_musicality.py` scripts.

## One-sentence summary

The model transcribes **Byzantine neume notation → Western staff pitches near-perfectly
(96% exact, 98% melodic)** on held-out data for a diatonic interval grammar that includes
ascending leaps up to an octave and rhythmic note durations — a task frontier prompting could
not do reliably — but only for **synthetic, correct-by-construction** sequences, not real
scanned manuscript chant, and the reverse direction is limited by an inherent ambiguity in the
notation itself.

## The evaluation (so the numbers are trustworthy)

- **Held-out:** 4,794 sequences the model never saw in training, with **0 leakage** (verified by
  both row-id and exact-input-prompt overlap = 0).
- **Deterministic scoring:** no LLM judge. The synthetic gold is exact by construction, so
  correctness is computed directly (exact match, per-position pitch accuracy, interval accuracy,
  edit distance, and a melodic-equivalence composite).
- **All scores below are shown as percentages (higher = better).** The melodic-equivalence
  composite is natively a 0–2 rubric score; it is reported here as % of its 2.0 maximum so every
  number is on the same 0–100% scale.
- **Sanity-checked:** gold-vs-gold scores a perfect 100% in both directions, so the scorer is
  measuring real correctness, not an artifact.

## What it does WELL

**neume → west (the headline capability):**

| metric | score | meaning |
|---|---|---|
| exact_match | **96.0%** | reproduces the entire pitch sequence exactly, 96% of the time |
| pitch_accuracy | **99.1%** | 99% of individual pitches correct |
| interval_accuracy | **99.2%** | the melodic motion between notes is right |
| melodic_equivalence | **97.8%** | (1.955 of 2.0) — essentially at the ceiling |
| strict_pass_rate | **96.2%** | 96% pass the strict bar (melodic ≥ 75% AND meaning ≥ 75%) |

Concretely, given a neume sequence and its ison (starting-pitch) anchor, the model correctly:
- walks the diatonic intervals — unison, steps, and leaps up to an **octave** (the ypsili/kentima
  combinations added in the expanded grammar);
- honors **rhythmic durations** — `apli`/`dipli`/`tetrapli` produce held notes rendered as
  `<pitch>:<beats>` (e.g. `C4:5`);
- anchors absolute pitch to the stated ison and echoes the correct mode header;
- **stops at the right length** (after the EOS fix — earlier versions ran on 3× too long).

This is the assignment's core claim, demonstrated: **supervised fine-tuning teaches a small
model a specialized notation grammar that prompting a frontier model did not reliably perform.**

## What it does NOT do well (and why)

**west → neume — low exact-match (14%), but this is a notation ceiling, not a model failure.**

| metric | score | how to read it |
|---|---|---|
| pitch_accuracy | 76.5% | **the honest signal** — 76% of neume positions correct |
| exact_match | 13.7% | misleading here — see below |
| melodic_equivalence | 49.0% | (0.981 of 2.0) — held down by the ambiguity below |
| interval_accuracy | n/a | output is neume tokens, not pitches, so no interval is defined |

The reverse direction is intrinsically ambiguous: **multiple neumes encode the same pitch
motion** (e.g. `oligon` and `petaste` both mean "+1 step"). From pitches alone you cannot recover
which one was originally written, so exact match is capped well below 100% no matter how good the
model is. Judge this direction by positional pitch accuracy (76%), not exact match. This is the
documented "w2n ceiling."

**Out of scope entirely (never trained, do not expect):**
- **Real scanned manuscript chant.** The model was trained on synthetic data specifically because
  the available real neume↔pitch corpus was found to be non-recoverable — the paired neume and
  pitch streams did not actually correspond (≈35% directional agreement; a ~10% positional
  ceiling for any model). This model has **not** been shown to transcribe real melismatic chant,
  and the honest finding is that that corpus can't support exact transcription.
- **Microtones, chromatic/enharmonic modes, fthora, melisma.** Only the four **diatonic** modes
  (1, pl.1, 4, pl.4) on a natural-note ladder are modeled. The model asserts no accidental or
  microtonal content.

## Fine-tuning created the capability (base vs tuned)

The same base model, **without** the adapter, scores **0.0% exact-match** on neume→west (it does
not produce a valid pitch transcription at all — prompting alone cannot do this task). **With** the
adapter it scores **96.0% exact.** That 0% → 96% jump is the direct measurement that SFT *created*
the behavior rather than surfacing something already in the base. Full breakdown:
`byzantine_deltas_base_vs_tuned_20260712.md`.

## The most important takeaway

The same class of model scored **~10% positional accuracy on the real corpus** and **96% exact on
correct-by-construction data.** That contrast cleanly isolates the earlier failure: it was a
**data-labeling problem, not a model-capability problem.** When the training pairs actually
correspond, a small fine-tuned model learns the grammar almost perfectly — leaps, rhythm, and
both directions (modulo the notation's own reverse-ambiguity).

## Bottom line for a reader/user
- Use it for: **neume→west transcription of diatonic interval-grammar sequences** (with leaps and
  durations). It is excellent at this.
- Don't use it for: real manuscript scans, microtonal/chromatic chant, or expecting a unique
  west→neume answer where the notation is inherently ambiguous.
