# Colab — Curriculum: Synthetic-Pretrain → Real-Finetune

The experiment your synthetic result set up but the 7B run skipped. The 7B adapters went
**straight to real melismatic data and collapsed to a drone** (melodic 0.0, variety ~0.01
— see `docs/byzantine_coder7b_results_20260710.md`). The project thesis is *musicality
before melody*: teach the interval grammar on correct-by-construction synthetic data
first, THEN adapt to real chant. This notebook does exactly that in two stages per
direction, using `--init-adapter` to continue stage-1 weights into stage-2.

**Hypothesis:** the synthetic prior gives the model a reason not to drone, so stage-2 real
training moves the real-data musical-property scores off the floor (variety ↑, set_f1 ↑,
ngram_f1 ↑, real_musicality > 0) even though exact per-position pitch stays walled.

Everything here reuses patterns already verified in `docs/colab_eval_download.md`
(auto-discover paths, install-if-absent, quoting for the `models 2` space). Grading is
local — bring back `preds.zip`.

---

## What runs where

| stage | data | script | output |
|---|---|---|---|
| 1. synthetic pretrain | `sft_synth_2500.jsonl` (2500, both dirs, 1:1) | train | `models/curr_synth` |
| 2a. real n2w finetune | `sft_n2w_train_sub_cued.jsonl` (1510) | train `--init-adapter models/curr_synth` | `models/curr_n2w` |
| 2b. real w2n finetune | `sft_w2n_train_sub.jsonl` (1510) | train `--init-adapter models/curr_synth` | `models/curr_w2n` |
| 3. predict | real held-out per dir | predict | `runs/curr_*_preds.jsonl` |

One shared synthetic prior (grammar is direction-general), then a directional real
finetune each — matching the two directional adapters we already have, so the ONLY changed
variable vs. the collapsed 7B run is "was there a synthetic prior first."

Base model = `unsloth/Qwen2.5-Coder-7B-bnb-4bit` (same as the 7B run, for a clean compare).

---

## Cell 1 — Locate repo, install deps if absent

```python
import os, subprocess, importlib, glob

hits = glob.glob("/content/**/scripts/train_byzantine_sft.py", recursive=True)
assert hits, ("repo not found under /content — clone it:\n"
              "  !cd /content && git clone <YOUR_REPO_URL> From-Scratch-LLM")
ROOT = os.path.dirname(os.path.dirname(hits[0]))
os.chdir(ROOT)
print("cwd:", os.getcwd())
print(subprocess.run(["git", "pull", "--ff-only"], capture_output=True, text=True).stdout)

need = [
    "scripts/train_byzantine_sft.py",
    "scripts/predict_local.py",
    "data/byzantine/sft_synth_2500.jsonl",
    "data/byzantine/sft_n2w_train_sub_cued.jsonl",
    "data/byzantine/sft_w2n_train_sub.jsonl",
    "data/byzantine/sft_n2w_heldout_cued.jsonl",
    "data/byzantine/sft_w2n_heldout.jsonl",
]
for p in need:
    assert os.path.isfile(p), f"MISSING: {p}"
print("all required files present")

missing = [m for m in ["bitsandbytes", "accelerate"] if importlib.util.find_spec(m) is None]
if missing:
    subprocess.run(["pip", "-q", "install", *missing], check=True)
    importlib.invalidate_caches()
for m in ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]:
    importlib.import_module(m)
print("imports OK")

BASE = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
```

---

## Cell 2 — Smoke gate (DO NOT SKIP, ~2 min)

Confirms training loads in 4-bit, response-only markers auto-detect, loss drops, and the
adapter saves — before committing GPU to the full curriculum.

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synth_2500.jsonl \
  --model {BASE} \
  --out models/_smoke \
  --max-steps 20 --batch-size 8 --grad-accum 1 --seq-length 1024
```

Look for: `response-only markers: instruction=... response=...`, decreasing loss, and
`Saved adapter → models/_smoke`. If OOM → add `--seq-length 768` (then 512).

---

## Cell 3 — Stage 1: synthetic pretrain (the shared prior)

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synth_2500.jsonl \
  --model {BASE} \
  --out models/curr_synth \
  --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024
```

`--epochs 2` overrides max-steps; bf16 auto-selects on L4/A100. ~2500 rows × 2 epochs is
short. This adapter alone should already ace the SYNTHETIC held-out (that's the known
result); the point is to carry its grammar into stage 2.

---

## Cell 4 — Stage 2: continue on REAL data, per direction

`--init-adapter models/curr_synth` reattaches the stage-1 LoRA as **trainable** and keeps
adapting it — this is the curriculum hinge. Lower LR (`1e-4`) so real data adapts the
prior rather than overwriting it.

```python
# 4a. real n2w, continuing from the synthetic prior
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_n2w_train_sub_cued.jsonl \
  --model {BASE} \
  --init-adapter models/curr_synth \
  --out models/curr_n2w \
  --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024 --lr 1e-4
```

```python
# 4b. real w2n, continuing from the same synthetic prior
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_w2n_train_sub.jsonl \
  --model {BASE} \
  --init-adapter models/curr_synth \
  --out models/curr_w2n \
  --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024 --lr 1e-4
```

Expect the log to print `Continuing from adapter: models/curr_synth` — if it doesn't, the
flag didn't take and you're training fresh (stop and check).

---

## Cell 5 — Discover adapters + generate predictions on REAL held-out

```python
import glob, os

def find_adapter(root):
    if os.path.isfile(os.path.join(root, "adapter_config.json")):
        return root
    hits = glob.glob(os.path.join(root, "**", "adapter_config.json"), recursive=True)
    assert hits, f"no adapter under {root}"
    def step(p):
        d = os.path.basename(os.path.dirname(p))
        return int("".join(c for c in d if c.isdigit()) or 0)
    return os.path.dirname(sorted(hits, key=step)[-1])

N2W = find_adapter("models/curr_n2w")
W2N = find_adapter("models/curr_w2n")
print("n2w:", N2W, "\nw2n:", W2N)
```

```python
# predict on the REAL held-out sets (matched recipe: n2w cued, w2n plain)
!python scripts/predict_local.py --model {BASE} --adapter-path "{N2W}" \
  --eval data/byzantine/sft_n2w_heldout_cued.jsonl \
  --out runs/curr_n2w_preds.jsonl --load-4bit --batch-size 16 --max-new-tokens 256

!python scripts/predict_local.py --model {BASE} --adapter-path "{W2N}" \
  --eval data/byzantine/sft_w2n_heldout.jsonl \
  --out runs/curr_w2n_preds.jsonl --load-4bit --batch-size 16 --max-new-tokens 256
```

Optional — also predict on the SYNTHETIC held-out to confirm the prior survived stage 2
(sanity that the model didn't forget the grammar):

```python
!python scripts/predict_local.py --model {BASE} --adapter-path "{N2W}" \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --out runs/curr_n2w_on_synth_preds.jsonl --load-4bit --batch-size 16 --max-new-tokens 256
```

---

## Cell 5b — Download prediction files NOW (before anything else)

Grab the result files the instant they exist, so a disconnect can't lose them. These are
the ONLY files needed for grading — the grader is a local pure-Python script (no API, no
GPU, no cost), so downloading these is the whole handoff.

```python
import shutil, os
from google.colab import files

os.makedirs("/content/dl", exist_ok=True)
# zip only the prediction JSONLs (tiny — a few hundred KB total)
preds = [f for f in [
    "runs/curr_n2w_preds.jsonl",
    "runs/curr_w2n_preds.jsonl",
    "runs/curr_n2w_on_synth_preds.jsonl",   # present only if you ran the optional cell
] if os.path.isfile(f)]
assert preds, "no prediction files found — run Cell 5 first"

import zipfile
zp = "/content/dl/curr_preds.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in preds:
        z.write(p, arcname=os.path.basename(p))
print("zipped:", preds)
files.download(zp)          # <-- browser download of the result files for grading
```

Bring `curr_preds.zip` back and hand it to the assistant (or drop it in the repo). That's
all the grader needs. The adapter download below is optional (for versioning the models).

---

## Cell 6 — (Optional) download the adapters too

Predictions were already downloaded in Cell 5b. Run this ONLY if you also want the trained
adapters saved off-box (for versioning / later reuse). Skip it otherwise.

```python
import shutil, os
os.makedirs("/content/dl", exist_ok=True)
from google.colab import files
for name, path in [("curr_n2w", N2W), ("curr_w2n", W2N), ("curr_synth", find_adapter("models/curr_synth"))]:
    shutil.make_archive(f"/content/dl/{name}", "zip", path)
    files.download(f"/content/dl/{name}.zip")
print("done — Runtime → Disconnect after downloads")
```

---

## After download — grade LOCALLY (no API, no cost; this is where we see if it worked)

Grading is a **pure-Python local script** — `score_real_musical.py` loads no model and
makes no API calls, so it's free and deterministic. Hand `curr_preds.zip` to the assistant
(or run this yourself). It grades the alignment-robust properties a drone can't fake, and
compares against the 7B baseline (`runs/coder7b_*_realscore.json`: variety ~0.01,
real_musicality 0.0).

```bash
# unzip curr_preds.zip into runs/ first, then:
python3 scripts/score_real_musical.py \
  --eval data/byzantine/sft_n2w_heldout_cued.jsonl \
  --pred runs/curr_n2w_preds.jsonl --out runs/curr_n2w_realscore.json

python3 scripts/score_real_musical.py \
  --eval data/byzantine/sft_w2n_heldout.jsonl \
  --pred runs/curr_w2n_preds.jsonl --out runs/curr_w2n_realscore.json

# sanity: synthetic still solved after stage 2?
python3 scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --pred runs/curr_n2w_on_synth_preds.jsonl --out runs/curr_n2w_on_synth_score.json
```

**Success signals (curriculum beat the collapsed baseline):**
- `variety` climbs off ~0.01 toward the gold's natural variety.
- `set_f1`, `hist_sim`, `ngram_f1` rise materially above the 7B baseline (~0.34 / ~0.32 / ~0.05).
- `real_musicality_0_2` > 0 and `good_rate` > 0 on real held-out.
- Synthetic sanity still high (prior not forgotten).

If all of these stay at the baseline floor, the curriculum did not transfer → that's the
signal to try option #3 (DTW-realigned real pairs) overnight.

---

## Gotchas (pre-empted)
- `--init-adapter` must print `Continuing from adapter: …`; absence = training fresh.
- `models 2` space → all adapter vars are quoted (`"{N2W}"`).
- `Can't find adapter_config.json` → adapters nest under `checkpoints/`; `find_adapter` handles it.
- Shell in a Python cell needs `!`; a bare shell line throws `SyntaxError: invalid decimal literal`.
