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
