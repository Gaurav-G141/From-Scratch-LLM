# Curriculum v2 Results — Drone Killed, Hallucination Introduced; Labels Are the Ceiling (2026-07-11)

Curriculum v2 (`docs/colab_curriculum_v2.md`) applied two fixes over v1: **step A** blended
synthetic+real in stage 2 (anti-forgetting), **step B** decoded with
`--repetition-penalty 1.2 --no-repeat-ngram-size 3` (anti-loop). Graded with
`score_real_musical.py` against the 7B baseline and curriculum v1.

## Verdict: not better than v1. We traded the drone for hallucination.

The two fixes had opposite effects that cancelled out. The drone is gone; the output is now
non-repetitive garbage instead of a repetitive tone. Net real-musicality is still ~0.

### base(7B real) → curr v1 → curr v2

| metric (n2w) | 7B base | v1 | v2 | v2 vs v1 |
|---|---|---|---|---|
| variety | 0.012 | 0.021 | **0.808** | +0.787 |
| set_f1 | 0.340 | 0.480 | **0.242** | −0.238 |
| hist_sim | 0.319 | 0.414 | **0.166** | −0.248 |
| ambitus_match | 0.105 | 0.339 | **0.045** | −0.294 |
| ngram_f1 | 0.049 | 0.083 | 0.061 | −0.022 |
| interval_hist_sim | 0.278 | 0.408 | 0.437 | +0.029 |
| length_ratio | 0.196 | 0.198 | 0.490 | +0.292 |
| real_musicality_0_2 | 0.00 | 0.00 | 0.114 | +0.114 |
| good_rate | 0% | 0% | **0%** | = |

| metric (w2n) | 7B base | v1 | v2 |
|---|---|---|---|
| variety | 0.072 | 0.070 | 0.990 |
| set_f1 | 0.268 | 0.368 | 0.316 |
| hist_sim | 0.088 | 0.204 | 0.180 |
| length_ratio | 0.594 | 0.667 | 0.324 |
| real_musicality_0_2 | 0.00 | 0.00 | 0.00 |

### Degeneration → hallucination
- Pure-repeat (drone): 66% (v1) → **0%** (v2). Anti-loop fully eliminated the drone.
- BUT **42% of n2w pitch tokens are now INVALID** (`G8`, `A9`, `D------6`, runaway
  accidentals, corrupted `Isole` header, OOV neumes `apostrophus`/`oligon_hypsili_6`).

## Diagnosis: two separable problems

1. **`no_repeat_ngram_size=3` forces hallucination.** Forbidding any repeated 3-gram means
   when the model should emit a naturally repeated melodic phrase, it is *compelled* to
   mutate tokens — manufacturing invalid octaves/accidentals to satisfy the constraint.
   Variety went up because the tokens are novel, not because they're musical. The scorer
   correctly caught this: variety ↑↑ but set_f1 / hist_sim / ambitus all ↓ (right-note
   metrics collapsed).

2. **The knowledge was weak the whole time.** set_f1/ngram_f1 never got high in ANY run
   (base, v1, v2). Decoding changes only redistribute the same weak knowledge — they cannot
   add the right notes. The model does not reliably know which pitches a neume sequence maps
   to, because the **training labels mis-pair neumes and pitches** (proportional slicing in
   `build_neume_tasks.py`; ~1.78:1 melisma).

## Conclusion: Branch 3 (labels are the ceiling)

Per `docs/byzantine_next_plans_by_outcome.md`, this is the Branch-3 outcome: recipe and
decoding levers are exhausted, and every run points at the same root cause — the labels are
positionally wrong, so cross-entropy has no correct target to learn. No sampling knob fixes
"doesn't know the notes."

## Next steps (decided)

1. **Cheap re-predict (free, no retrain):** re-run the SAME v2 adapters with
   `--repetition-penalty 1.2` ONLY (drop `--no-repeat-ngram-size`). The soft penalty
   discourages repeats without forcing token mutation, so this reveals the model's honest
   output. Expected: fewer invalid tokens than 42%, set_f1 recovering toward v1. Diagnostic
   only — separates "decoding artifact" from "weak knowledge."
2. **Stage A — anchor-segmented alignment** (the real fix): segment neume+pitch at breath
   marks / martyria so within-phrase pairs are near-1:1, rebuild training data, retrain.
   First direct attack on the label defect. See `docs/byzantine_realdata_escalation_plan.md`.

## Files
- Scores: `runs/curr2_{n2w,w2n}_realscore.json`; predictions `CLAUDE LOOK AT THIS FOLDER!/curr2_preds/`.
- Compare: `runs/curr_*_realscore.json` (v1), `runs/coder7b_*_realscore.json` (baseline).
