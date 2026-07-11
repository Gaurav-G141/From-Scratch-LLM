# 7B Colab Results — Real-Data Adapters Hit the Melisma Wall (2026-07-10)

Grading of the two directional LoRA adapters trained on Colab
(`unsloth/Qwen2.5-Coder-7B-bnb-4bit`, r=8, 1 epoch, effective batch 16 → 95 steps):

- `coder7b_n2w` — neume→west, trained on `sft_n2w_train_sub_cued.jsonl` (1510 rows)
- `coder7b_w2n` — west→neume, trained on `sft_w2n_train_sub.jsonl` (1510 rows)

Predictions generated on Colab (greedy, 4-bit) over the matched held-out sets, graded
locally with `scripts/score_synthetic_eval.py` (deterministic; self-test passes).

## Headline: no learning of melodic content — same wall, now at 7B

| Model | n | exact | pitch_acc | interval_acc | melodic_equiv (0–2) | strict pass |
|---|---|---|---|---|---|---|
| n2w **tuned** | 555 | 0.00 | 0.062 | 0.047 | **0.00** | 0% |
| n2w **base**\* | 48 | 0.00 | 0.048 | 0.060 | 0.00 | 0% |
| w2n **tuned** | 555 | 0.00 | 0.027 | 0.000 | **0.00** | 0% |

\*Cell 5 (base) was interrupted after 48/555 rows; enough to establish the base is also
at chance. Tuned ≈ base → **the fine-tune added no melodic skill.**

Held-out sets matched to each adapter's training recipe: n2w vs
`sft_n2w_heldout_cued.jsonl`, w2n vs `sft_w2n_heldout.jsonl`. Score JSONs in
`runs/coder7b_{n2w,w2n,n2w_base}_score.json`.

## What actually happened: format learned, content collapsed

The model learned the **notation scaffolding perfectly** and the **melody not at all**:

- **n2w Mode line exactly correct: 555/555 (100%).**
- **n2w Ison line exactly correct: 555/555 (100%).**
- **n2w melodic body: 486/555 (88%) degenerate** — the pitch body is one tone repeated,
  almost always the modal tone `G4` (`Mode 4\nIson: G4\nG4 G4 G4 G4 …`).
- **w2n: 203/555 (37%) degenerate** — repeated `oligon`; also emits a non-training format
  (`(Ison G4)`, `|`-separated) it hallucinated rather than the trained target shape.

So the adapter reliably reproduces mode, anchor, output length, and vocabulary — every
*conventional* aspect — while filling the melodic content with the safest high-frequency
token. This is textbook mode collapse under a training signal with no learnable structure.

## Why (consistent with the documented melisma wall)

Real neume↔pitch data aligns ~1.78:1 (melismatic): one neume spans several pitches, so
the input **under-specifies** the exact pitch sequence position-by-position (see
`docs/byzantine_handoff_20260709.md`, `docs/byzantine_synthetic_breakthrough_20260709.md`).
With no recoverable 1:1 mapping, response-only loss is minimized by emitting the modal
drone — exactly the degeneration observed. A larger model cannot supply information the
input lacks; as the runbook predicted, **capacity was never the bottleneck, alignment is.**

## Why this run was still worth it

1. **Replicates the wall at 7B**, ruling out "the model was too small" — the wall held at
   1.7B locally and now at 7B on Colab.
2. **Clean format-vs-content dissociation**: 100% header accuracy with 0.0 melodic
   equivalence isolates *where* the information gap is — the notation conventions are
   learnable from this data; the melody is not.
3. **Empirical justification for the synthetic-data direction**: the same recipe on
   correct-by-construction 1:1 synthetic data reaches melodic 2.0 (breakthrough doc). The
   contrast — **synthetic 1:1 → 2.0, real melismatic → 0.0, at two model scales** — is the
   core scientific result of the project.

## Synthetic proof: the same pipeline CAN do the task when data is 1:1

To prove the zeros above are a **data property**, not a broken model/pipeline, the
synthetic adapter (`models/byzantine_synth_2500`, Qwen3-1.7B + LoRA) was scored on the
synthetic held-out set **with the exact same `score_synthetic_eval.py`** that produced
the 7B zeros. Same scorer, same metrics, opposite result:

| Adapter | Data (train→eval) | Direction | pitch_acc | melodic (0–2) | strict pass |
|---|---|---|---|---|---|
| coder7b_n2w (7B) | **real melismatic** | neume→west | 0.062 | **0.00** | 0% |
| coder7b_w2n (7B) | **real melismatic** | west→neume | 0.027 | **0.00** | 0% |
| synth_2500 (1.7B) | **synthetic 1:1** | neume→west | **1.00** | **2.00** | 100% |
| synth_2500 (1.7B) | **synthetic 1:1** | west→neume | 0.878 | 1.20 | 20% |

The *smaller* model on aligned data beats the *larger* model on real data by the entire
range of the metric (0.00 → 2.00 on byz→west). This is the decisive control: capacity is
not the bottleneck, **data alignment is**. (w2n is intrinsically harder even on synthetic
— multiple neume spellings map to one pitch move — hence 1.2 not 2.0; still infinitely
above the real-data 0.0.)

Score JSON: `runs/synth_2500_ep2_score.json`. Fuller synthetic write-up:
`docs/byzantine_synthetic_breakthrough_20260709.md`.

### Optional further step (costs GPU)
A fresh **7B** Colab run on `sft_synth_2500.jsonl` would close the last gap (same model
size, only data differs). Not required — the cross-scale contrast above already isolates
the cause — but it would make the "one variable changed" story literally exact.

## Reproduce the grading (local, no GPU)
```bash
python3 scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_n2w_heldout_cued.jsonl \
  --pred "<preds>/coder7b_n2w_preds.jsonl" --out runs/coder7b_n2w_score.json
python3 scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_w2n_heldout.jsonl \
  --pred "<preds>/coder7b_w2n_preds.jsonl" --out runs/coder7b_w2n_score.json
```
