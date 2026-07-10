# Synthetic Melodic-Equivalence Result — and Its Limits (2026-07-09)

A local LoRA run moved byz→west `melodic_equivalence` from **0.00** (every prior
experiment, per `docs/byzantine_handoff_20260709.md`) to **2.00 / 100% exact** on a
held-out slice. This doc records the result, the audit that confirmed it's real, and —
importantly — **why it does NOT solve the original real-chant task.** Read the "Limits"
section before citing this anywhere.

## Results (byz→west, held-out synthetic slice, `score_synthetic_eval.py`)

| run | training data | pitch_acc | exact | melodic_equiv (0–2) | strict pass |
|---|---|---|---|---|---|
| base (untrained) | — | 0.00 | 0.00 | 0.00 | 0% |
| tiny500 | 500 synth, **no anchor in prompt** | 0.12 | 0.00 | ~0.00 | 0% |
| 2500 ep1 | 2500 synth, **anchor in prompt** | 0.99 | 0.925 | 1.925 | 92.5% |
| **2500 ep2** | 2500 synth, anchor in prompt | **1.00** | **1.00** | **2.00** | **100%** |

west→neume (ep2): pitch_acc 0.88, exact 0.20, melodic 1.2 — lags n2w. (Its
`interval_accuracy 0.0` is a known metric artifact: the metric is defined on pitch
tokens, not neume-name tokens.)

## What the data actually was

**Both train and test are fully synthetic** (`scripts/build_synthetic_musicality.py`) —
**not real Byzantine chant.**
- Train: `data/byzantine/sft_synth_2500.jsonl` (2,500 rows, direction-balanced, drawn
  from the synthetic *train* file).
- Test: `data/byzantine/sft_synthetic_musicality_heldout.jsonl` (300 walks × 2 dirs,
  generated with disjoint seeds `10_000_000+` and `--exclude` the train file).
- Adapter: `models/byzantine_synth_2500` (Qwen3-1.7B + LoRA r=8, PEFT, response-only
  loss, 2 epochs, batch 4).

**The task is deterministic interval arithmetic over 9 fixed-step neumes:**
`ison`=0, `oligon`/`petaste`=+1, `apostrophos`=−1, `elaphron`=−2, `oligon_kentema`=+3,
`oligon_hypsili`=+4, `elaphron_apostrophos`=−3, `chamile`=−4. Given the Ison anchor and a
neume list, the pitch sequence is a running sum. A 1.7B model reaching 100% on this is
**expected**, not surprising — it memorizes 9 offsets and adds.

## What changed to cause 0.00 → 2.00 (ranked)

1. **Anchor-in-prompt fix (dominant).** The n2w prompt previously gave Mode + neumes but
   **omitted the Ison anchor.** Because transposition re-anchors identical neumes to
   different start pitches, one prompt mapped to **4 different valid pitch targets** —
   absolute pitch was mathematically undetermined. Adding `Ison: X4` to the prompt made
   it a well-posed function. Evidence: tiny500 *without* anchor = 0.12 pitch_acc;
   2500-ep1 *with* anchor = 0.99. The task definition changed, not just the data volume.
2. **Training AND testing on the clean 1:1 synthetic distribution.** Prior 0.00 numbers
   came from adapters trained on real melismatic data and/or scored against real refs.
3. **Epochs (minor).** ep1→ep2 (1.925→2.0) only cleaned up a few octave slips.

## Audit — why this is real, not an artifact

Investigated because the jump looked suspiciously large. All read-only:
- **Leakage: none.** 0 ID overlap, 0 (prompt,target) content overlap, and 0 prompt
  overlap between `sft_synth_2500.jsonl` and the held-out slice. The model never saw a
  held-out prompt.
- **Musical duplication: none.** 0 of 300 held-out neume-sequences appear in training
  (different seeds/walks), so it isn't regurgitating memorized melodies.
- **Scorer is honest.** Prediction files contain only `{id, prediction}` (the model's
  own generated text); `score_synthetic_eval.py` re-derives gold independently. Verified
  a prediction is model output, not copied gold.
- **Task triviality acknowledged.** The strong score reflects an easy, well-posed task —
  not a hard one solved.

## Limits — READ THIS before citing the result

**This does NOT solve real Byzantine transcription.** The wall in
`docs/byzantine_handoff_20260709.md` §2 is untouched:
- Real neumes are **melismatic (~1.78:1)** and structurally **under-specify pitch**;
  `docs/byzantine_near1to1_findings_20260709.md` shows even length-matched *real* windows
  fail (contour 0.305), so the gap is the data's information content, not the model.
- The synthetic task hands the model exactly the information real notation omits (a clean
  1:1 mapping + explicit anchor).

**Correct framing:** model capacity was never the bottleneck. The SLM performs exact
neume→pitch transcription **when the mapping is 1:1 and the anchor is given.** Real chant
provides neither. This is a **scoping/diagnostic result** that sharpens the project's
conclusion — it does not overturn the melismatic wall.

## Relation to other docs
- Supersedes the blanket "melodic_equivalence never moved" reading of the handoff — but
  ONLY for synthetic/aligned data. The real-corpus wall stands.
- Consistent with `docs/byzantine_near1to1_findings_20260709.md`.

## Reproduce
```
# train (Qwen3-1.7B, MPS ~2-3h): scripts/train_byzantine_sft.py \
#   --data data/byzantine/sft_synth_2500.jsonl --epochs 2 --batch-size 4 --force-peft \
#   --out models/byzantine_synth_2500
python3 scripts/predict_local.py --adapter-path models/byzantine_synth_2500 \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --out runs/synth_2500_ep2_preds.jsonl --batch-size 12
python3 scripts/score_synthetic_eval.py --pred runs/synth_2500_ep2_preds.jsonl
```
