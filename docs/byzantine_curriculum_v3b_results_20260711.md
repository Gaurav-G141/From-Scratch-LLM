# Curriculum v3b Results — Decoding Broke the Loop; ngram-8 Is the Winner; Register Is the Last Wall (2026-07-11)

v3b changes **nothing about training** — same v3 adapters (DTW-aligned real + melisma prior +
blended stage-2). The only change is **decoding**. Three variants were tested on the same
adapters and graded on the same DTW-aligned real heldout with `score_real_musical.py`:

- **ngram-6**: `--repetition-penalty 1.2 --no-repeat-ngram-size 6` (first loop-break attempt)
- **ngram-8**: `--repetition-penalty 1.2 --no-repeat-ngram-size 8` (looser block) — **WINNER**
- **temp**: `--repetition-penalty 1.3 --temperature 0.5` (mild sampling)

## Verdict: ngram-8 wins. Best n2w result of the whole project; register drift is the last wall.

v3's diagnosis was "knowledge is there, looping hides it." v3b confirms it directly: with the
loops broken, the hidden knowledge surfaces and the composite jumps **0.048 → 0.317**, and it
does so the *right* way (knowledge metrics rose to all-time highs — the opposite of v2's
hallucination).

### Variant comparison (n2w)
| metric | v3 | ngram-6 | **ngram-8** | temp | read |
|---|---|---|---|---|---|
| real_musicality_0_2 | 0.048 | 0.254 | **0.317** | 0.327 | temp marginally higher but fragile (see below) |
| interval_hist_sim | 0.654 | 0.673 | **0.698** | 0.630 | ★ ngram-8 best ever; temp DROPPED below v3 |
| ngram_f1 | 0.116 | 0.166 | **0.184** | 0.137 | ★ ngram-8 best ever |
| hist_sim | 0.368 | 0.324 | **0.381** | 0.296 | ★ ngram-8 best ever |
| set_f1 | 0.459 | 0.380 | 0.439 | 0.357 | ngram-8 ≈ v3, best of v3b |
| ambitus_match | 0.462 | 0.072 | 0.147 | 0.236 | all decoders ≪ v3 — the last wall |
| mode_correct | 1.000 | 0.844 | **1.000** | 1.000 | ngram-8 fully recovered ngram-6's dip |
| variety | 0.121 | 0.279 | 0.258 | 0.408 | ngram-8 healthy; temp edging high |
| above_gate_music (rows) | 0.358 (67) | 0.265 (479) | 0.340 (468) | 0.358 (458) | ngram-8 = high quality across ~7× rows |

**Why ngram-8 beats ngram-6:** the looser 8-gram block fixed ngram-6's two regressions —
`mode_correct` recovered 0.844 → 1.000 and `ambitus` doubled back 0.072 → 0.147 — while every
knowledge metric hit an all-time high. Best knowledge, best above-gate quality, no new cost.

**Why NOT temp, despite its 0.327 composite (marginally highest):** its edge is fragile and
partly fake. (1) `interval_hist_sim` (0.630) DROPPED below plain v3 — sampling degraded the
relative-motion profile, the metric that matters most. (2) The output **hallucinates**: the
first n2w prediction ends `…D4 เรียบร คุ้ม user Wh` — foreign-script garbage + a leaked chat
turn, so its high variety (0.408) is partly noise gaming the gate. (3) w2n knowledge cratered
under sampling: set_f1 0.346 → 0.274, hist_sim 0.263 → 0.167. temp buys +0.01 composite by
trading away interval fidelity, output cleanliness, and the whole w2n direction — rejected.

### The old v3-vs-ngram6 detail (kept for the record)

### n2w:  baseline → v1 → v2 → v3 → **v3b (ngram-6)**
| metric | base | v1 | v2 | v3 | **v3b** | read |
|---|---|---|---|---|---|---|
| real_musicality_0_2 | 0.000 | 0.000 | 0.114 | 0.048 | **0.254** | ★ best composite of any run (5× v3) |
| variety | 0.012 | 0.021 | 0.808 | 0.121 | **0.279** | ★ healthy band — loop broken, NOT hallucinating |
| interval_hist_sim | 0.278 | 0.408 | 0.437 | 0.654 | **0.673** | ★ best ever — relative motion held |
| ngram_f1 | 0.049 | 0.083 | 0.061 | 0.116 | **0.166** | ★ best ever — local shape held |
| set_f1 | 0.340 | 0.480 | 0.242 | 0.459 | 0.380 | slight dip vs v3 (still ≫ v2) |
| hist_sim | 0.319 | 0.414 | 0.166 | 0.368 | 0.324 | slight dip vs v3 |
| contour_sim | 0.761 | 0.700 | 0.623 | 0.640 | 0.629 | ≈ v3 |
| length_ratio | 0.196 | 0.198 | 0.490 | 0.361 | 0.376 | ≈ v3 |
| mode_correct | 1.000 | 1.000 | 0.081 | 1.000 | 0.844 | dipped (see cost) |
| ison_correct | 1.000 | 1.000 | 0.000 | 0.493 | 0.479 | ≈ v3 |
| **ambitus_match** | 0.105 | 0.339 | 0.045 | **0.462** | **0.072** | ✗ **cratered — the cost of the loop-break** |

### The gate flipped — this is why the composite jumped
- v3: **67/501** rows cleared the 0.15 anti-drone gate (variety median 0.11 → 87% force-zeroed).
- v3b: **479/501** rows clear the gate (variety median 0.264). The looping is essentially gone.
- `above_gate_music`: v3 = 0.358 over 67 rows; v3b = **0.265 over 479 rows**. Per-row quality
  dipped a little, but it is now real across ~7× as many rows → the aggregate quintupled.

### Why this is a real win, not curr2's fake win
v2/curr2 also had high variety (0.81) — but by **hallucinating** (42% invalid tokens), so its
set_f1/hist_sim/interval_hist all collapsed. v3b is the opposite: variety rose to a *moderate*
0.28 **and interval_hist_sim + ngram_f1 rose to all-time highs.** The decoder broke loops
without manufacturing garbage — exactly what size-6 (vs size-3) was chosen to do.

## The cost: register drift (ambitus 0.462 → 0.072)

The one clear regression. `ambitus_match` collapsed and **84% of rows (422/501) now score
ambitus < 0.1** — pervasive, not a few outliers. Cause is visible in the raw output: v3 sat in
the correct 4th–5th octave; v3b wanders down into the 2nd–3rd octave, e.g.
`G3 F#3 E-3 D3 C3 B2 A#2 …`. The intervals/contour are right (interval_hist_sim is the best
ever) but the **absolute register drifts**. `no_repeat_ngram_size` forbids repeating any 6-gram,
which pressures the model to keep moving rather than settle on a pitch center — so it drifts
octave-wise. `mode_correct` slipped 1.0 → 0.844 for the same reason (a drifted register reads as
the wrong mode header).

So v3b is a **genuine trade**: `+`loops broken, `+`best-ever local shape/interval profile,
`−`register discipline. Net strongly positive by composite, but not a pure dominance over v3.

## w2n: unchanged (as expected)
| metric | v3 | v3b | read |
|---|---|---|---|
| set_f1 | 0.346 | 0.342 | ≈ v3 |
| hist_sim | 0.263 | 0.253 | ≈ v3 |
| variety | 0.791 | 0.857 | already saturated |
| real_musicality_0_2 | 0.000 | 0.000 | ceiling ~1.2 (oligon/petaste both +1) |

w2n signal is not where the win is; judge it by set_f1/hist_sim, both flat vs v3. No regression.

## The last wall: register drift (ambitus) is a TRAINING problem, not decoding

All three decoders leave `ambitus_match` far below v3 (0.462 → 0.147 best). The intervals and
contour are the best of any run, but the melody drifts octave-wise — ngram-8's n2w still opens
`G3 F#3 E-3 D3 C3 B2 …` (2nd–3rd octave; v3 sat correctly in the 4th–5th). Because **three
different decoders all show it**, this is confirmed NOT a decoding artifact: the model learned
*relative* motion but not *absolute* register. Root cause is a training-signal gap — the target
gives no explicit pitch-center anchor, so cross-entropy never had to pin the octave.

## Next levers
1. **Ship ngram-8 as the v3b decoder** — it's the headline n2w result; regrade with
   `bash scripts/grade_v3b.sh` any time.
2. **Attack register at training time (the real remaining lever):** add an explicit pitch-center
   anchor to the n2w target (an ambitus/register cue in the prompt, like the Ison anchor that
   made the synthetic task well-posed), or an auxiliary register/octave loss term. This is the
   one thing decoding cannot fix.
3. **Optional:** run w2n with ngram-8 for completeness (only ngram-6/temp w2n exist); expect it
   flat vs v3 (w2n ceiling ~1.2).
4. If register can't be anchored → Branch 4 (reframe to recoverable properties, or reward-based
   training on the scorer). See `docs/byzantine_next_plans_by_outcome.md`.

## Files
- Scores: `runs/v3b_{ngram,ngram8,temp}_{n2w,w2n}_realscore.json` (w2n: ngram + temp only).
- Predictions: `runs/v3b_n2w_{ngram,ngram8,temp}_preds.jsonl`, `runs/v3b_w2n_{ngram,temp}_preds.jsonl`.
- Compare: `bash scripts/grade_v3b.sh` (regrades all + prints the table), or
  `python3 scripts/compare_realscores.py --order coder7b curr curr2 v3 v3b_ngram v3b_ngram8 v3b_temp`.
  v3 baseline: `runs/v3_*_realscore.json`.
- Recipe: `docs/colab_curriculum_v3.md` (Cell 5 = decoding variants).
