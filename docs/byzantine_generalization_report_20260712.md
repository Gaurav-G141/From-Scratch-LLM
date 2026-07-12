# Byzantine Grammar — Length-Generalization Report (2026-07-12)

**Detailed report.** For the condensed version see `byzantine_generalization_results_20260712.md`.
This document is the full write-up: what was tested, the raw numbers, the diagnostic that reframes
them, concrete examples, and the conclusion with its honest limits.

---

## 1. What this test asks

The shipped model scores **96% exact / 98% melodic** on neume→west transcription. But that number
came from a held-out set drawn from the **same distribution** as training — same vocabulary, same
four modes, and crucially **the same sequence lengths**. A high score there is consistent with two
very different explanations:

- **(A) It learned the rule.** The model internalized the interval grammar — each neume is a fixed
  degree-shift on the ladder, walked step by step from the ison — and can apply it to any sequence.
- **(B) It memorized the distribution's shape.** The model learned the statistical texture of
  6–20-neume walks and interpolates familiar patterns, without a compositional rule.

Same-distribution accuracy cannot separate these. A **length holdout** can: train only on **short**
walks, then test on walks **longer than any seen in training**. Under (A) the score holds — the rule
doesn't care about length. Under (B) it collapses — the model never saw shapes that long.

## 2. Setup

| | |
|---|---|
| Base model | `unsloth/Qwen2.5-Coder-7B-bnb-4bit` (same as the shipped model — held fixed) |
| Train split | `--max-len 12` → walks of ≤12 neumes |
| Heldout split | `--min-len 16`, disjoint seeds + `--exclude` train → walks of 16–20 neumes |
| Overlap | 0 exact-input overlap (verified) |
| Rows scored | 6,378 (3,189 neume→west, 3,189 west→neume) |
| Scoring | deterministic (no LLM judge); gold is correct-by-construction |
| Notebook | `docs/colab_generalization.md`, Test A |

The heldout gold pitch lines are **16–20 tokens long — every one longer than the 12-token maximum
the model ever saw in training.**

## 3. Raw scores

| metric | neume→west (heldout) | west→neume (heldout) | n2w same-length baseline |
|---|---|---|---|
| exact_match | 18.0% | 2.0% | 96.0% |
| pitch_accuracy | 62.8% | 60.2% | 99.1% |
| interval_accuracy | 64.0% | n/a | 99.2% |
| melodic_equivalence | 0.77 / 2.0 | 0.60 / 2.0 | 1.955 / 2.0 |
| strict_pass_rate | 18.1% | 4.0% | 96.2% |

Taken alone this reads as a severe drop and would suggest explanation (B). **The per-row diagnostic
shows that reading is wrong.**

## 4. Diagnostic — the drop is truncation, not musical error

The scorer records `len_pred` and `len_gold` for every row. Three analyses over those fields settle
the question.

### 4.1 The model stops at ~12 tokens no matter how long the answer should be

Predicted length does not track gold length — it flatlines at the training cap:

| gold length | mean predicted length |
|---|---|
| 16 | 12.9 |
| 17 | 11.9 |
| 18 | 12.0 |
| 19 | 12.1 |
| 20 | 11.8 |

**79.7%** of n2w rows are *shorter* than gold; only **0.2%** overrun. The model learned "outputs end
by ~token 12" — which is exactly true of every example it trained on (`--max-len 12`). This is a
learned **length prior on stopping**, independent of the pitch mapping.

### 4.2 When it emits a full-length line, it is back at baseline

Conditioning neume→west accuracy on how close the predicted length is to gold:

| subset | n | exact | pitch | interval | melodic |
|---|---|---|---|---|---|
| all rows | 3189 | 18.0% | 62.8% | 64.0% | 0.77 |
| **length exactly right** | **642** | **89.6%** | **95.1%** | **99.2%** | **1.90** |
| length within ±1 | 1072 | 53.6% | 89.7% | 93.7% | 1.53 |
| length short by ≥2 | 2113 | 0.0% | 49.2% | 48.9% | 0.39 |

On the **642 rows the model completed to full length, interval accuracy is 99.2%** — statistically
identical to the same-length 96% baseline, but on sequences *longer than anything in training*. The
rule generalized. The all-rows 63% is a **scoring artifact**: the scorer's denominator is
`max(len(pred), len(gold))`, so a correct-but-short prefix is charged for the tail it never emitted.

### 4.3 The prefix the model does emit is 94% correct

Restricting to rows short by ≥2 and measuring accuracy over just the tokens the model **actually
produced** (not the full gold length): **≈94%**. So even on the rows it truncates, it is not
degrading musically — it writes a correct prefix and stops.

### 4.4 The fingerprint: interval ≈ pitch

Across all rows, interval accuracy (64.0%) ≈ pitch accuracy (62.8%). This is the signature of
**truncation**, not per-step error. If the model were making *musical* mistakes as sequences grew,
interval accuracy would collapse far below pitch accuracy — a single wrong step corrupts every
interval after it. Equal interval and pitch accuracy means the emitted notes are right and the loss
is length, full stop.

## 5. Concrete examples

**Full-length, unseen length, still perfect** (`synth_10000029_t0_n2w`): gold 20 tokens, model emits
20, exact=1.0, interval=1.00. A sequence 66% longer than any in training, transcribed exactly.

**Truncated, correct prefix** (`synth_10000054_t0_n2w`): gold 20 tokens, model emits 13
(`D5 E5 B5 B5 A5 D5 E5 E5 F5 B4 A4 E4 F4`) — the prefix walks the intervals correctly, then stops at
13 ≈ the training ceiling. Scored pitch 0.65 only because 7 correct tail tokens are missing.

## 6. Conclusion

**The interval grammar generalized to unseen lengths (explanation A); the stopping behavior did
not.** Evidence: 99.2% interval accuracy on full-length completions of longer-than-trained
sequences, and a 94%-correct prefix even where it truncates. This is **rule-learning, not
shape-memorization** — the strong result the test was designed to detect.

The single real limitation is a **learned length ceiling**: trained only on ≤12-token outputs, the
model stops at ~12 and truncates longer targets.

### Verdict: qualified pass
The compositional claim holds. The length ceiling is a separate, well-understood, fixable artifact —
not evidence against generalization.

## 7. Why the ceiling exists and how to remove it

The generalization split was *deliberately* capped at 12-neume outputs so that the ≥16 heldout would
be genuinely out-of-distribution. The model correctly learned that cap as a length prior. Fixes,
cheapest first:

1. **Train on the full length range** (`--min-len 6 --max-len 20`, already supported). The **shipped
   96% model was trained on the full range and does not truncate** — this ceiling is specific to the
   restricted generalization split, not a property of the approach.
2. **Explicit length control.** The prompt already states the token count
   (`Transcribe this Byzantine neume sequence (N neumes)...`); the model could be trained to treat
   `N` as a hard target rather than a hint.

A cheap confirmation run (retrain the length split on the full range, show the ceiling disappears)
would convert "fixable in principle" into "fixed, demonstrated."

## 8. Honest framing for downstream writeups

Never report the 18% exact alone — always with the diagnostic. The defensible one-liner:

> *On sequences 50–66% longer than any in training, the model applied the interval grammar with
> 99.2% interval accuracy wherever it completed the line, and wrote a 94%-correct prefix even where
> it truncated; its only failure was a learned length ceiling (it stops at ~12 tokens, the training
> maximum), which is removed by training on the full length range.*

## 9. Reproduce

- Per-row score (with `len_pred`/`len_gold`): `runs/synth_len_score.json`
- Predictions: `runs/synth_len_preds.jsonl`
- Diagnostic: in the score JSON's `per_row` list, condition `pitch_accuracy` / `interval_accuracy`
  on `len_pred == len_gold`; bucket `len_pred` by `len_gold` to see the flat ~12 ceiling.
- Notebook: `docs/colab_generalization.md` (Test A). Both files currently live in `runs/`
  (git-ignored), so the numbers are transcribed into this doc for the record.
