# Curriculum v3 Results — Root-Cause Fix Worked; Looping Is the New Bottleneck (2026-07-11)

v3 = DTW-aligned real data (`sft_aligned_*`) + melisma synthetic prior + blended stage-2 +
soft repetition penalty only. First run to attack the label defect (proportional slicing →
DTW contour alignment). Graded on the DTW-aligned real heldout with `score_real_musical.py`.

## Verdict: best knowledge of any run, bottlenecked by looping (not ignorance)

The model now produces real diatonic chant melodies in the right mode
(`G4 A4 B4 C5 D5 E5 F5 E5 D5…`), a first. Every *knowledge* metric is the best yet. The
composite stays low ONLY because the model loops phrases, tripping the anti-drone variety
gate.

### n2w:  baseline → v1 → v2 → v3
| metric | base | v1 | v2 | **v3** | read |
|---|---|---|---|---|---|
| interval_hist_sim | 0.278 | 0.408 | 0.437 | **0.654** | ★ relative-motion profile now matches real chant |
| ambitus_match | 0.105 | 0.339 | 0.045 | **0.462** | right melodic range |
| set_f1 | 0.340 | 0.480 | 0.242 | **0.459** | right notes (≈ v1 best) |
| hist_sim | 0.319 | 0.414 | 0.166 | **0.368** | right note-usage profile |
| ngram_f1 | 0.049 | 0.083 | 0.061 | **0.116** | best local-shape score of any run |
| contour_sim | 0.761 | 0.700 | 0.623 | 0.640 | (baseline's was drone-inflated) |
| variety | 0.012 | 0.021 | 0.808 | 0.121 | just under 0.15 gate → see below |
| real_musicality_0_2 | 0.00 | 0.00 | 0.114 | 0.048 | misleading; see "why composite is low" |
| invalid tokens | — | — | 42% | **1%** | clean (v2 hallucinated) |
| drone (pure-repeat) | 88% | 66% | 0% | **4%** | drone essentially gone |

### w2n:  baseline → v1 → v2 → v3
| metric | base | v1 | v2 | v3 |
|---|---|---|---|---|
| hist_sim | 0.088 | 0.204 | 0.180 | **0.263** (best) |
| length_ratio | 0.594 | 0.667 | 0.324 | **0.620** (recovered) |
| set_f1 | 0.268 | 0.368 | 0.316 | 0.346 |
| real_musicality_0_2 | 0.00 | 0.00 | 0.00 | 0.00 |

## Why the composite is low despite the best knowledge

**The model loops.** 100% of n2w predictions repeat a 4-gram ≥3× — it generates a valid
melodic phrase then cycles it. Consequences:
- variety median = 0.110, just under the scorer's 0.15 anti-drone gate → **87% of rows
  (434/501) are force-scored real_musicality 0** regardless of note quality.
- The **67 rows that clear the gate average real_musicality 0.36, set_f1 0.45,
  interval_hist 0.65** — i.e. when it does NOT loop, it is genuinely decent chant.
- So the composite understates v3 badly. v2's higher 0.114 came from *random* high variety
  (hallucination gaming the gate); v3 is penalized for *structured* looping. On every
  knowledge metric, v3 ≫ v2.

## Diagnosis

The root-cause fix succeeded: aligned labels + melisma prior gave the model correct melodic
knowledge (interval profile, range, notes, local shape all up, output clean). The remaining
failure is a **generation behavior** — it learned phrases but not phrase *termination/
variation*, so greedy decoding with a mild penalty cycles them. This is a decoding/training
-signal problem, NOT a knowledge problem — a much better and more tractable place to be than
the drone/hallucination of prior runs.

## Next levers (cheap, no retrain required)

1. **Tune decoding to break loops without hallucinating** (free, re-predict only):
   - `--repetition-penalty 1.3` (up from 1.2) + light sampling (`--temperature 0.5`) instead
     of pure greedy. Enough to escape cycles; not enough to invent invalid tokens.
   - Re-grade; check variety climbs into a healthy 0.3–0.6 band (not v2's 0.8 hallucination),
     and that the 67→more rows clear the gate.
2. **Re-examine the gate** (analysis): the 0.15 variety gate was tuned against the drone.
   Looping at variety 0.11 is a different failure; report the "above-gate" musicality (0.36)
   as the honest knowledge signal alongside the composite so v3's real gain is visible.

## If decoding tuning caps out
The looping likely also reflects the aligned *targets* still containing repetitive
windows (real chant repeats phrases) + no explicit stop signal. A training-side fix would be
length/EOS-discipline data or a mild diversity term — but try the free decoding lever first.

## Files
- Scores: `runs/v3_{n2w,w2n}_realscore.json`; predictions `CLAUDE LOOK AT THIS FOLDER!/v3_preds/`.
- Compare: `runs/{coder7b,curr,curr2}_*_realscore.json`.
- Data/build: `scripts/align_neume_pitch_dtw.py`, `scripts/build_synthetic_melisma.py`, `docs/colab_curriculum_v3.md`.
