# Colab Notebook — Byzantine SFT (copy-paste, cell by cell)

The single source of truth for running training + prediction on Colab. Each block below
is ONE Colab cell. Shell cells are marked `%%bash`; comments explain intent. The MODEL
does all GPU work here (train + generate predictions → JSONL). **Grading is done back in
the local repo** (or handed to the assistant) — see the last section.

Supersedes the prose in `docs/colab_runbook.md` (that file is background/rationale; THIS
file is the runnable steps).

---

## 0. Runtime & budget

- Runtime → Change runtime type → **GPU** (L4 24 GB recommended; A100 burns the $10 fast).
- Budget: L4 + a 7–9B model, 2 epochs on the ~2.5k–34k row sets ≈ 10–15 units. Leaves
  room for a smoke run + one real run.
- **After each run: Runtime → Disconnect** (idle GPU still bills).

---

## 1. Repo + deps (Cell 1)

```python
%%bash
cd /content
# clone once; if it exists, pull latest (contains the save-strategy fix)
if [ -d From-Scratch-LLM ]; then
  cd From-Scratch-LLM && git pull
else
  git clone <YOUR_REPO_URL> From-Scratch-LLM
fi
```

```python
# Cell 1b — deps. Unsloth pulls compatible torch/transformers/peft/trl/bitsandbytes.
!pip -q install unsloth
!pip -q install "trl<0.12" "peft>=0.13" datasets   # pin trl to avoid SFTConfig churn
```

> **Note on the `SFTConfig` pickle crash you already hit:** it's an Unsloth/TRL bug at
> mid-training checkpoint save. The training script now uses `save_strategy="no"` on the
> Unsloth path, so it never triggers — the final adapter is saved via
> `model.save_pretrained`. Do NOT re-add `save_strategy="epoch"`.

---

## 2. Set the working dir for all later cells (Cell 2)

```python
import os
# the repo may clone nested; point to the dir that actually contains scripts/
ROOT = "/content/From-Scratch-LLM"
if not os.path.isdir(f"{ROOT}/scripts"):
    ROOT = "/content/From-Scratch-LLM/From-Scratch-LLM"  # nested-clone fallback
os.chdir(ROOT)
print("cwd:", os.getcwd())
assert os.path.isdir("scripts"), "scripts/ not found — fix ROOT"
```

---

## 3. (Optional) mount Drive to persist adapters (Cell 3)

```python
from google.colab import drive
drive.mount('/content/drive')
# then use --out /content/drive/MyDrive/byz/<name> in training cells so a disconnect
# doesn't lose the adapter. If you skip Drive, adapters live in ./models and vanish on
# disconnect — download them before disconnecting.
```

---

## 4. GPU smoke gate — DO NOT SKIP (Cell 4)

Confirms the model loads in 4-bit, response-only markers auto-detect, loss drops, and the
final adapter saves — before you spend on a full run. ~2 min.

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synth_2500.jsonl \
  --model unsloth/Qwen2.5-Coder-7B-bnb-4bit \
  --out models/_smoke \
  --max-steps 20 --batch-size 8 --grad-accum 1 --seq-length 1024
```

Check the log for: `response-only markers: instruction=... response=...`, a decreasing
loss, and `Saved adapter → models/_smoke`. If it OOMs, drop `--seq-length 768` then `512`.

---

## 5. Training runs (Cell 5)

Pick the data by experiment. `--epochs 2` overrides `--max-steps`. bf16 auto-selects on
L4/A100. Base models with no chat template get ChatML injected automatically.

```python
# 5a. SYNTHETIC (proves interval grammar; expect near-perfect byz->west on held-out)
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synth_2500.jsonl \
  --model unsloth/Qwen2.5-Coder-7B-bnb-4bit \
  --out models/coder7b_synth --epochs 2 --batch-size 8 --grad-accum 1
```

```python
# 5b. COMBINED (synthetic + real translation, the main recipe)
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_combined_train.jsonl \
  --model unsloth/Qwen2.5-Coder-7B-bnb-4bit \
  --out models/coder7b_combined --epochs 2 --batch-size 8 --grad-accum 1
```

```python
# 5c. NEAR-1:1 REAL (handoff idea #3 — honest real-data test)
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_near1to1_train_cued.jsonl \
  --model unsloth/Qwen2.5-Coder-7B-bnb-4bit \
  --out models/coder7b_near1to1 --epochs 2 --batch-size 8 --grad-accum 1
```

One variable per run. Change model OR data OR epochs, not all at once.

---

## 6. Generate predictions (Cell 6) — this is the "eval" the model runs

Produces `{id, prediction}` JSONL that gets graded later. Use `--load-4bit` to match the
training base. Match `--eval` held-out set to the training data:
- synthetic / combined trained → `sft_synthetic_musicality_heldout.jsonl`
- near-1:1 trained → `sft_near1to1_heldout_cued.jsonl`

```python
# 6a. base-vs-tuned: run BOTH (base = no --adapter-path) for a delta
!python scripts/predict_local.py \
  --model unsloth/Qwen2.5-Coder-7B-bnb-4bit \
  --adapter-path models/coder7b_synth \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --out runs/coder7b_synth_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 96

!python scripts/predict_local.py \
  --model unsloth/Qwen2.5-Coder-7B-bnb-4bit \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --out runs/coder7b_base_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 96
```

---

## 7. Bring results back for grading

The prediction JSONLs (`runs/*_preds.jsonl`) are small — download them or save to Drive.
Grading happens in the local repo (or hand the file to the assistant):

```bash
# LOCAL (or a fresh Colab cell — scorer is pure Python, no model/GPU):
python3 scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --pred runs/coder7b_synth_preds.jsonl \
  --out runs/coder7b_synth_score.json
```

Read the **per-direction** breakdown. Report to the assistant: the two `runs/*_preds.jsonl`
(base + tuned) and/or the score JSON. The assistant grades and interprets (base-vs-tuned
delta, format vs melodic, real-data caveats).

> Grading needs the eval file to match the format the adapter was trained on. All current
> synthetic/near-1:1 sets use the **anchor-in-prompt** format (`Ison: X` line in n2w
> prompts). If an adapter was trained on pre-anchor data, byz→west will score falsely low
> — flag it and the assistant will re-grade against the right eval.

---

## Gotchas seen so far
- `SyntaxError: invalid decimal literal` → you pasted a shell command into a Python cell.
  Prefix with `!` or use a `%%bash` cell.
- `PicklingError: SFTConfig` → fixed via `save_strategy="no"`; don't revert it.
- Nested clone path `/content/From-Scratch-LLM/From-Scratch-LLM` → handled by Cell 2.
- `qwen2.5-coder-7b` is a BASE model (no chat template) → `predict_local.py` injects
  ChatML automatically; training uses the same via the tokenizer.
```
