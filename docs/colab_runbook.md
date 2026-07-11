# Colab Runbook — Byzantine SFT on a bigger model ($10 budget)

> **For the runnable, copy-paste, cell-by-cell notebook see `docs/colab_notebook.md`.**
> This file is the background/rationale (budget math, model choice, why each step).

Budget-aware, vetted step sequence for training the Byzantine transcription adapter on
Colab. Guiding principle: **spend zero GPU money on anything verifiable on CPU first,
then make ONE clean paid run — not five debugging runs.**

## Budget math (~$10 ≈ ~100 compute units)

| GPU | ~units/hr | ~hrs for $10 | QLoRA fit |
|-----|-----------|--------------|-----------|
| T4 (16GB) | ~2 | ~50h | ≤9B ok, 14B tight |
| **L4 (24GB)** | ~5 | **~20h** | **9–14B (sweet spot)** |
| A100 (40GB) | ~12 | ~8h | 14–20B, burns budget |

**Recommended: L4 + Gemma-2-9B.** No `<think>` wrapper (Day-3's most persistent bug),
fits 24GB at seq 1024 in 4-bit with headroom, ~10–15 units for the run → leaves budget
for the smoke gate + one re-run. A100 only for a single 14–20B shot with no re-run room.
A bigger model will NOT lift the melisma ceiling (a data-alignment property, not
capacity) — so don't pay A100 prices to test learnability.

Data is stored as chat `messages`, so switching model = just changing `--model`; no data
regeneration and no re-vetting needed.

---

## Step 0 — Pre-flight on CPU (FREE, do before any GPU)

Already done / re-runnable locally at zero cost:
- Synthetic data vetted: `python3 scripts/verify_synthetic_musicality.py data/byzantine/sft_synthetic_musicality.jsonl` → 0 errors.
- Held-out slice vetted + zero train overlap (see `docs/byzantine_scoring_methods.md`).
- Scorer self-test: `python3 scripts/score_synthetic_eval.py --self-test` → ALL PASS.
- predict→score plumbing verified (fake perfect predictor → exact 1.0).

Pin dependency versions before opening Colab (Colab drifts; a mismatch mid-run wastes
units). Known-good target: recent `transformers` + `peft` + `trl` + `bitsandbytes`, and
`unsloth` if using its fast path.

---

## Step 1 — Colab session setup (cheap)

```python
# 1a. Mount Drive FIRST so a disconnect never vaporizes a paid run.
from google.colab import drive; drive.mount('/content/drive')

# 1b. Install deps (pin versions once confirmed working).
!pip -q install "transformers>=4.44" "peft>=0.13" "trl>=0.12" "datasets>=3.0" bitsandbytes accelerate
# Optional CUDA fast path: !pip -q install unsloth

# 1c. Get the repo + data onto the box (git clone, or upload just these files):
#   scripts/train_byzantine_sft.py
#   scripts/predict_local.py
#   scripts/score_synthetic_eval.py
#   data/byzantine/sft_combined_train.jsonl          (train)
#   data/byzantine/sft_synthetic_musicality_heldout.jsonl  (eval)
```

Files needed: the combined training set (`sft_combined_train.jsonl`, interleaved) OR the
curriculum variant, plus the held-out slice for scoring.

---

## Step 2 — GPU smoke gate (~2 min, ~1 unit) — DO NOT SKIP

Run the REAL model for a handful of steps before committing to the full run. This is the
single most budget-protective step.

```bash
python3 scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_combined_train.jsonl \
  --model google/gemma-2-9b-it \
  --out /content/drive/MyDrive/byz/_smoke \
  --max-steps 20 --seq-length 1024
```

Confirm ALL of these in the log before proceeding:
- Model loads in 4-bit within VRAM (no OOM). If OOM → `--seq-length 768`, then 512.
- `train_on_responses_only` markers printed and look right for the model
  (`response-only markers: instruction=… response=…`). If the WARN about markers fires,
  the loss is over full text — stop and fix before spending on a full run.
- Train loss is finite and trending down.
- A checkpoint writes under the Drive path.

---

## Step 3 — Full training run (the paid run)

```bash
python3 scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_combined_train.jsonl \
  --model google/gemma-2-9b-it \
  --out /content/drive/MyDrive/byz/gemma2_9b_v1 \
  --epochs 2 --seq-length 1024 --lr 2e-4
```

Notes:
- `--epochs 2` overrides `--max-steps`; bf16 auto-selects on L4/A100 (fp16 on T4).
- 34,496 rows × 2 epochs on L4 ≈ 2–3h ≈ 10–15 units.
- Watch the first ~50 steps for OOM; if it survives that, it will finish.
- **Change ONE variable per run.** A result from a run that changed model+data+epochs
  together tells you nothing you can afford to re-test.
- After it saves: `Runtime → Disconnect` (idle Colab still bills).

Curriculum alternative (try only if interleave stalls): swap `--data` to
`sft_combined_curriculum_train.jsonl`.

---

## Step 4 — Eval: FREE deterministic first, PAID judge only if warranted

### 4a. Deterministic melodic score (no API, no cost) — the key experiment
```bash
python3 scripts/predict_local.py \
  --model google/gemma-2-9b-it \
  --adapter-path /content/drive/MyDrive/byz/gemma2_9b_v1 \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --out runs/synth_heldout_preds.jsonl \
  --load-4bit

python3 scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --pred runs/synth_heldout_preds.jsonl \
  --out runs/synth_heldout_score.json
```
Read the **per-direction** breakdown. Success = `neume_to_west` `melodic_equivalence_0_2`
clearly above the Day-3 wall of 0.00 (target ≥1.5, ideally 2.0) with high
`pitch_accuracy`. This answers "did the synthetic data teach the interval ladder?" with
zero API spend and zero Opus variance.

Also run the BASE model (omit `--adapter-path`) once for a base-vs-tuned delta.

### 4b. LLM judge on the REAL corpus (costs API) — only if 4a looks good
Don't spend judge money if 4a says the model didn't learn the ladder. If it did, run the
existing harness (`docs/byzantine_scoring_methods.md`, System A) against the real banks
for the melodic/mode/meaning dimensions on melismatic data.

---

## Guardrails recap
- Smoke gate before every full run.
- One variable per run.
- Disconnect after saving; save to Drive.
- Deterministic eval (free) gates the paid judge eval.
- Keep A100 for a final run only, if at all.

## Related docs
- `docs/byzantine_scoring_methods.md` — deterministic vs Opus scoring, exact functions.
- `docs/byzantine_day3_results_20260708.md` — the 0.00 melodic wall this run targets.
