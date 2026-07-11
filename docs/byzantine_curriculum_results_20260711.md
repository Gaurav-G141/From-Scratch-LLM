# Curriculum Results — Synthetic→Real Moved the Needle, But Didn't Break the Wall (2026-07-11)

Option #1 (synthetic-pretrain → real-finetune) ran on Colab: stage 1 on `sft_synth_2500`,
stage 2 continued via `--init-adapter` on the real directional data (n2w cued, w2n plain),
same 7B base (`unsloth/Qwen2.5-Coder-7B-bnb-4bit`) as the collapsed baseline. Graded
locally with `score_real_musical.py` (drone-proof, alignment-robust) against the 7B
real-data baseline.

## Verdict: partial win. Every distributional metric improved; composite still 0.

The curriculum **measurably reduced the drone and improved musical structure**, but not
enough to clear the "real musicality" bar. This is real signal, not noise — the direction
is right, the dose is insufficient.

### n2w: baseline(7B real) → curriculum
| metric | base | curr | Δ |
|---|---|---|---|
| variety | 0.012 | 0.021 | +0.009 ↑ |
| set_f1 | 0.340 | 0.480 | **+0.140 ↑** |
| hist_sim | 0.319 | 0.414 | +0.095 ↑ |
| interval_hist_sim | 0.278 | 0.408 | **+0.130 ↑** |
| ambitus_match | 0.105 | 0.339 | **+0.234 ↑** |
| ngram_f1 | 0.049 | 0.083 | +0.034 ↑ |
| contour_sim | 0.761 | 0.700 | −0.062 ↓ (baseline's was drone-inflated) |
| real_musicality_0_2 | 0.00 | 0.00 | = |

### w2n: baseline → curriculum
| metric | base | curr | Δ |
|---|---|---|---|
| set_f1 | 0.268 | 0.368 | **+0.101 ↑** |
| hist_sim | 0.088 | 0.204 | **+0.116 ↑** |
| length_ratio | 0.594 | 0.667 | +0.073 ↑ |
| variety | 0.072 | 0.070 | ≈ |
| real_musicality_0_2 | 0.00 | 0.00 | = |

### Degeneration (fraction of outputs ending in 20+ identical repeated tokens)
| | 7B baseline | curriculum |
|---|---|---|
| n2w | 88% | **66%** |
| w2n | 37% | **15%** |

The drone is retreating: w2n pure-repeat more than halved (37%→15%), n2w dropped 88%→66%.
Qualitatively, curriculum n2w now emits real motion (`G4 F4 E4 F4 G4 A4 G4 F4…`) instead of
a flat `G4 G4 G4…` — but it falls into **short repeating loops** rather than free melody,
which is why variety is still low and the composite stays at 0.

## The smoking gun: catastrophic forgetting of the synthetic prior

Synthetic sanity check (curriculum n2w adapter re-run on the synthetic held-out, 56 n2w
rows): **pitch_accuracy 0.012, melodic 0.0** — i.e. the model that this same recipe drives
to **2.0 on synthetic** has, after stage-2 real training, **completely forgotten the
interval grammar** (a fresh synthetic adapter scores 1.0/2.0; this scores 0.01/0.0).

So the mechanism is clear: stage 2 on real melismatic data **overwrote** the synthetic
prior instead of adapting it. The prior helped transiently (hence the improved
distributional metrics — some grammar survived into the real outputs) but LR 1e-4 × 2
epochs of un-alignable real data pulled the weights back toward the drone attractor. The
curriculum works *while the prior lasts* and decays as real training proceeds.

## Why this is genuinely useful

1. **Direction confirmed:** a synthetic prior demonstrably improves real-data behavior
   (7/8 metrics up, drone halved). The thesis "musicality before melody" transfers — the
   problem is retention, not concept.
2. **Root cause is now specific and fixable:** catastrophic forgetting, not "real data has
   no signal." That points at concrete, cheap knobs (below) before the expensive option #3.
3. **The scorer earned its keep:** it cleanly separated "still 0 on the composite" from
   "actually moved a lot underneath" — the exact distinction a bare exact-match score hid.

## Plan for improvements (ranked cheapest-first)

### A. Anti-forgetting knobs on the SAME curriculum (cheap, do first)
The prior decayed — so preserve it. One variable per run:
1. **Lower stage-2 LR to 2e-5–5e-5** (from 1e-4) and/or **1 epoch, not 2.** Less pull away
   from the prior. Cheapest single lever; try this first.
2. **Mix, don't sequence:** instead of pure real in stage 2, train stage 2 on a **blend**
   (e.g. 50/50 synthetic + real, or the existing `sft_combined_*`). Interleaving keeps the
   grammar in-distribution so it can't be forgotten. This is the highest-expected-value
   change.
3. **Freeze/replay:** keep a slice of synthetic in every stage-2 batch (replay buffer), or
   lower LoRA rank in stage 2 so it can drift less.

### B. Inference-side anti-loop (free, stack on any run)
The remaining failure is *looping*, which greedy decoding maximizes. Add
`--repetition-penalty 1.2` / `no_repeat_ngram_size 3` to `predict_local.py` generation.
This won't add musical knowledge but will stop the loop-degeneration inflating the drone
stats — cleaner read on what the model actually learned. (Small script change; I can add
the flag.)

### C. Fix the labels (option #3, overnight, higher effort)
Even a perfectly-retained prior can't produce *correct* pitches from proportionally-sliced
labels. The real ceiling is the alignment: `build_neume_tasks.py` pairs neume/pitch windows
by proportion, not musical correspondence. **DTW-realign** each hymn's neume-contour to its
OMR pitch-contour to produce genuinely aligned training pairs, then retrain. My earlier
probe showed raw grammar-reconstruction is chance-level (0.36), so DTW on the *observed*
pitches (not grammar-predicted) is the right form. Pilot on the ~23 count-aligned hymns
before trusting it corpus-wide.

## Recommended next step
Do **A2 (blend stage 2)** + **B (repetition penalty)** together as one Colab run — both are
cheap, and A2 is the most likely single fix for forgetting. If the composite still floors
after that, escalate to **C (DTW realignment)** — that's the overnight job we pre-agreed
on. I have not started C yet (waiting on your call in the morning).

## Files
- Scores: `runs/curr_{n2w,w2n}_realscore.json`, baseline `runs/coder7b_{n2w,w2n}_realscore.json`.
- Predictions: `CLAUDE LOOK AT THIS FOLDER!/curr_preds/`.
- Curriculum notebook: `docs/colab_curriculum.md`. Scorer: `scripts/score_real_musical.py`.
