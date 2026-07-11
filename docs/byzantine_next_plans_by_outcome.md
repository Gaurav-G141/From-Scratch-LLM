# Next-Step Plans, Branched by curr2 Outcome

Decision tree for what to do after grading the curriculum-v2 predictions
(`curr2_{n2w,w2n}_preds.jsonl` → `score_real_musical.py`). **Planning only — no code/data
changes until the numbers are in.** Match the observed result to a branch below.

Baselines to compare against:
- 7B real (collapsed): n2w variety 0.012, set_f1 0.34, ngram_f1 0.05, real_musicality 0.0.
- curriculum v1: n2w set_f1 0.48, ambitus 0.34, still real_musicality 0.0; drone 88%→66%.
- Scorer ceiling on real gold ≈ 1.74. Directional ceilings: n2w can reach high; **w2n
  intrinsically ~1.2** (oligon/petaste both = +1), judge w2n by set_f1/hist/ngram.

Cross-references: `docs/byzantine_realdata_escalation_plan.md` (Stages A/B/C detail),
`docs/byzantine_curriculum_results_20260711.md` (v1 analysis).

---

## Branch 1 — BREAKTHROUGH: real_musicality_0_2 > 0, good_rate > 0 on n2w

*Meaning:* the blend fixed forgetting and the model produces genuinely musical real output.
The wall is cracked.

**Do (cheap consolidation, one variable each):**
1. **Tune the blend ratio / LR / epochs** to push the number higher — e.g. 60/40 synth:real,
   or 3 epochs at LR 3e-5. One knob per run; smoke-gate each.
2. **Confirm on w2n** with the directional-ceiling lens; if w2n lags, that's expected, not a
   regression.
3. **Run the LLM-judge (System A harness) on a small real slice** for the holistic
   melodic/mode/meaning dimensions — now worth the API spend because the deterministic score
   says there's real signal to judge.
4. **Write the result up** (`docs/byzantine_breakthrough_real_YYYYMMDD.md`) with base→v1→v2
   deltas and the judge numbers. This is the assignment's headline result.

*Skip all alignment/RL work.* If it's working, don't add risk.

---

## Branch 2 — PARTIAL: metrics up materially vs v1, but real_musicality still 0

*Meaning:* the lever (synthetic prior + blend) works, the dose is short — same shape as v1
but further along. variety off the floor, set_f1/ngram_f1 up, drone % down, composite still
under the 0.70 core threshold.

**Do (push the same lever harder before changing strategy):**
1. **More synthetic weight in the blend** (the prior is what's helping) — try 2:1 synth:real,
   and/or a 3rd epoch. Cheapest high-value knob.
2. **Stack decoding:** if pure-repeat % is still high, raise `--repetition-penalty` to 1.3
   and `--no-repeat-ngram-size` to 2; confirm it's not just masking a still-weak model by
   checking set_f1/ngram (knowledge), not only variety.
3. **If knobs plateau → Stage A (anchor-segmented alignment)** from the escalation plan.
   Cheap, CPU-only, attacks the label defect that a better recipe can't overcome. Gate on
   the per-phrase neume:pitch ratio tightening toward 1.0.

---

## Branch 3 — FLOORED: ≈ v1 or baseline, composite 0, variety ~0.02

*Meaning:* recipe changes have run out of road. The ceiling is the **labels** (proportional
slicing mis-pairs neumes and pitches), not the training method.

**Do (attack the labels; cheapest-first with hard gates):**
1. **Stage A — anchor-segmented alignment** (half day, CPU). Segment at breath marks/martyria
   so within-phrase pairs are near-1:1. Retrain v2 recipe on segmented data.
   - *Gate:* per-phrase ratio must tighten vs 1.78, and real metrics must move. If yes → B.
     If no → landmarks too sparse → Branch 4.
2. **Stage B — DTW realignment** (the all-night job) *only if A showed alignment is the
   lever.* Constrained/contour cost (NOT naive interval — that probed at chance 0.36).
   **Pilot on the ~23 count-aligned hymns before any overnight retrain.**

---

## Branch 4 — FLOORED and alignment doesn't help (Stage A flat / DTW pilot bad)

*Meaning:* exact real-pitch transcription is genuinely not recoverable from this melismatic
data. Stop chasing it; get an honest result a different way.

**Two independent options (can do either/both):**

### 4a — Stage C: reframe the target to recoverable properties
Train/eval real data on **contour / interval histogram / mode / ambitus / phrase count** —
the well-posed properties `score_real_musical.py` already measures. Guaranteed honest result
instead of zeros; weaker as a "transcription" claim. Builder gets new target variants.

### 4b — Stage D: reward-based training (scorer-as-reward) — the principled escape
Optimize the metric **directly** over the model's own generations, so there are NO
per-position teacher-forced labels — which sidesteps the alignment defect entirely (the exact
trap that trics cross-entropy).
- **DPO (cheaper, stable):** generate candidate completions, rank pairs with
  `score_real_musical.py`, train on the preference. This is stretch-ladder rung 1.
- **GRPO/PPO (stronger, finicky):** generate → score → push toward high-reward completions
  online.
- *Risks to guard:* reward-hacking (e.g. gaming the variety gate with noise) — add sanity
  penalties; slower (generation inside the loop); needs a good reward, which we have.
- *Why it's arguably the best endgame here:* since the alignment probe came back at chance,
  a method that doesn't need per-position labels may beat one that tries to reconstruct them.

---

## Orthogonal upgrade (any branch except pure Branch 1): melodic-aware token loss

Independent of the above, cross-entropy can be made to mimic the metric on
correctly-aligned/synthetic data:
- **Melodic label smoothing:** target distribution puts side-mass on neighbor pitches
  (F4/A4 around G4), so near-misses cost less — mirrors melodic_equivalence. Cheap; pitch
  vocab is ~25 tokens.
- **Distance-weighted CE / auxiliary interval loss:** penalize octave errors > step errors;
  reward contour directly.
Helps synthetic and any aligned real data; does NOT fix mislabeled melismatic pairs (only
alignment or reward-based training does). Fold in wherever cross-entropy training continues.

---

## Recommended reading order tomorrow
1. Grade curr2 → pick the branch by the n2w composite + variety + drone %.
2. Branch 1 → consolidate & write up. Branch 2 → push knobs, then Stage A.
   Branch 3 → Stage A → (gate) Stage B pilot. Branch 4 → Stage C and/or Stage D.
3. Never skip a gate; never start the overnight DTW without the 23-hymn pilot passing.
4. Push any new script/data to `main` before a Colab run.
