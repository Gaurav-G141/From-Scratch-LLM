# Colab — Curriculum v3: DTW-Aligned Real Data + Melisma Synthetic

The first run that attacks the ROOT CAUSE. Every prior run (baseline, v1, v2) failed
because the real training labels were built by **proportional slicing** — neumes and pitches
paired by position, which is wrong under melisma. v3 fixes the labels and the prior:

1. **DTW-aligned real data** (`sft_aligned_*`, quality ≥0.6, 556 hymns): each neume window is
   paired with the pitches it ACTUALLY governs (DTW contour alignment, median quality 0.75),
   not a proportional guess. This is the fix v1/v2 could not provide.
2. **Melisma synthetic prior** (`sft_synth_melisma`, pitch:neume 1.50): teaches
   one-neume→many-pitches on correct-by-construction data, matching real melisma density
   (~1.78) far better than the old 1:1 synthetic.
3. **Blended stage 2** (v2's anti-forgetting lesson) + **soft repetition penalty only**
   (v2's ngram block caused 42% hallucinated tokens — dropped here).

All data built + verified locally: DTW re-derivation and melisma re-derivation both 0
mismatches; blends single-direction, well-mixed, well-formed; originals untouched. All files
pushed to `main`.

Base model = `unsloth/Qwen2.5-Coder-7B-bnb-4bit` (unchanged, for clean comparison).

---

## Training files (prebuilt, in the repo)

| file | rows | contents |
|---|---|---|
| `sft_v3_stage1.jsonl` | 6058 | shared prior: 1:1 synth (2500) + melisma synth (3558), both directions |
| `sft_v3_n2w.jsonl` | 8188 | n2w blend: synth n2w + melisma n2w + DTW-aligned real n2w |
| `sft_v3_w2n.jsonl` | 8188 | w2n blend: synth w2n + melisma w2n + DTW-aligned real w2n |
| `sft_aligned_n2w_heldout.jsonl` | 501 | DTW-aligned real n2w eval (stem-disjoint from train) |
| `sft_aligned_w2n_heldout.jsonl` | 501 | DTW-aligned real w2n eval |

---

## Cell 1 — Repo, deps, sanity

```python
import os, subprocess, importlib, glob
hits = glob.glob("/content/**/scripts/train_byzantine_sft.py", recursive=True)
assert hits, ("repo not found — clone:\n"
              "  !cd /content && git clone https://github.com/Gaurav-G141/From-Scratch-LLM From-Scratch-LLM")
ROOT = os.path.dirname(os.path.dirname(hits[0])); os.chdir(ROOT)
print("cwd:", os.getcwd())
print(subprocess.run(["git","pull","--ff-only"], capture_output=True, text=True).stdout)

need = [
    "scripts/train_byzantine_sft.py", "scripts/predict_local.py",
    "data/byzantine/sft_v3_stage1.jsonl",
    "data/byzantine/sft_v3_n2w.jsonl", "data/byzantine/sft_v3_w2n.jsonl",
    "data/byzantine/sft_aligned_n2w_heldout.jsonl",
    "data/byzantine/sft_aligned_w2n_heldout.jsonl",
]
for p in need:
    assert os.path.isfile(p), f"MISSING (push + git pull?): {p}"
assert "--init-adapter" in open("scripts/train_byzantine_sft.py").read()
assert "repetition-penalty" in open("scripts/predict_local.py").read()
print("all files + flags present")

missing = [m for m in ["bitsandbytes","accelerate"] if importlib.util.find_spec(m) is None]
if missing: subprocess.run(["pip","-q","install",*missing], check=True); importlib.invalidate_caches()
for m in ["torch","transformers","peft","bitsandbytes","accelerate"]: importlib.import_module(m)
BASE = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
print("ready")
```

---

## Cell 2 — Smoke gate (DO NOT SKIP)

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_v3_stage1.jsonl --model {BASE} \
  --out models/_smoke --max-steps 20 --batch-size 8 --grad-accum 1 --seq-length 1024
```

Require: `response-only markers: …`, decreasing loss, `Saved adapter → models/_smoke`.
OOM → add `--seq-length 768` (then 512) to every train cell.

---

## Cell 3 — Stage 1: shared prior (1:1 + melisma synthetic)

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_v3_stage1.jsonl --model {BASE} \
  --out models/v3_prior --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024
```

---

## Cell 4 — Stage 2: blended finetune per direction, continuing from the prior

Blended data (anti-forgetting) + LR 3e-5 (adapt, don't overwrite). Must log
`Continuing from adapter: models/v3_prior` — if absent, STOP.

```python
# 4a. n2w
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_v3_n2w.jsonl --model {BASE} \
  --init-adapter models/v3_prior --out models/v3_n2w \
  --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024 --lr 3e-5
```

```python
# 4b. w2n
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_v3_w2n.jsonl --model {BASE} \
  --init-adapter models/v3_prior --out models/v3_w2n \
  --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024 --lr 3e-5
```

---

## Cell 4b-persist — Save adapters to Drive (do this the moment training finishes)

Colab has wiped these adapters once already, forcing a full retrain. Copy them to Drive
**right after Cell 4b** so a disconnect can never cost you the ~L4 retrain again. Predictions
(Cell 5) then read from the local copies as usual; Drive is just the durable backup.

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil, os
DST = "/content/drive/MyDrive/byz_v3_adapters"
os.makedirs(DST, exist_ok=True)
for name in ["v3_prior", "v3_n2w", "v3_w2n"]:
    src = f"models/{name}"
    if os.path.isdir(src):
        shutil.copytree(src, f"{DST}/{name}", dirs_exist_ok=True)
        print("saved", name, "->", f"{DST}/{name}")
```

To RESTORE on a fresh runtime (skip Cells 2–4, jump straight to prediction): mount Drive, then
`shutil.copytree("/content/drive/MyDrive/byz_v3_adapters/<name>", "models/<name>", dirs_exist_ok=True)`
for each of the three, and run the `find_adapter` cell.

---

## Cell 5 — v3b: predict two decoding variants to break the looping

v3's knowledge was the best of any run, but it **loops** (variety 0.11, just under the 0.15
anti-drone gate → 87% of rows force-scored 0). The fix is decoding, not retraining, so predict
BOTH variants on the same v3 adapters in one session and grade offline to pick the winner:

- **ngram** (recommended primary, deterministic): `--repetition-penalty 1.2 --no-repeat-ngram-size 6`.
  Size 6 blocks only LONG verbatim loops; natural short chant repeats survive. (v2's size-3 was
  too aggressive → 42% hallucinated tokens — do NOT go below 6.)
- **temp** (mild sampling, breaks loops a different way): `--repetition-penalty 1.3 --temperature 0.5`.

`--max-new-tokens 160` = safe headroom, ~2× faster than 256.

```python
import glob, os
def find_adapter(root):
    if os.path.isfile(os.path.join(root,"adapter_config.json")): return root
    h=glob.glob(os.path.join(root,"**","adapter_config.json"), recursive=True)
    assert h, f"no adapter under {root}"
    return os.path.dirname(sorted(h,key=lambda p:int("".join(c for c in os.path.basename(os.path.dirname(p)) if c.isdigit()) or 0))[-1])
N2W=find_adapter("models/v3_n2w"); W2N=find_adapter("models/v3_w2n")
print("n2w:",N2W,"\nw2n:",W2N)
```

```python
# Variant A — ngram: gentle long-loop block (deterministic; recommended primary)
!python scripts/predict_local.py --model {BASE} --adapter-path "{N2W}" \
  --eval data/byzantine/sft_aligned_n2w_heldout.jsonl \
  --out runs/v3b_n2w_ngram_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 \
  --repetition-penalty 1.2 --no-repeat-ngram-size 6

!python scripts/predict_local.py --model {BASE} --adapter-path "{W2N}" \
  --eval data/byzantine/sft_aligned_w2n_heldout.jsonl \
  --out runs/v3b_w2n_ngram_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 \
  --repetition-penalty 1.2 --no-repeat-ngram-size 6
```

```python
# Variant B — temp: mild sampling
!python scripts/predict_local.py --model {BASE} --adapter-path "{N2W}" \
  --eval data/byzantine/sft_aligned_n2w_heldout.jsonl \
  --out runs/v3b_n2w_temp_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 \
  --repetition-penalty 1.3 --temperature 0.5

!python scripts/predict_local.py --model {BASE} --adapter-path "{W2N}" \
  --eval data/byzantine/sft_aligned_w2n_heldout.jsonl \
  --out runs/v3b_w2n_temp_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 \
  --repetition-penalty 1.3 --temperature 0.5
```

> Want the original v3 baseline too (soft penalty only)? Re-add a block with
> `--out runs/v3_{n2w,w2n}_preds.jsonl --repetition-penalty 1.2` and no other decode flag —
> but the v3 scores are already saved in `runs/v3_*_realscore.json`, so this is optional.

---

## Cell 6 — Download predictions NOW (let Cell 5 finish first — one cell at a time)

```python
import os, zipfile
from google.colab import files
os.makedirs("/content/dl", exist_ok=True)
preds=[f for f in [
    "runs/v3b_n2w_ngram_preds.jsonl","runs/v3b_w2n_ngram_preds.jsonl",
    "runs/v3b_n2w_temp_preds.jsonl","runs/v3b_w2n_temp_preds.jsonl",
] if os.path.isfile(f)]
assert preds, "run Cell 5 first"
with zipfile.ZipFile("/content/dl/v3b_preds.zip","w",zipfile.ZIP_DEFLATED) as z:
    for p in preds: z.write(p, arcname=os.path.basename(p))
files.download("/content/dl/v3b_preds.zip")
```

Then locally: unzip into `runs/` and run `bash scripts/grade_v3b.sh` — it grades all four files
and prints the comparison table (v3b variants vs v3 / curr2 / curr / coder7b).

**Persist the adapters this time (see Cell 4b-persist below) BEFORE you disconnect, so a wipe
can't cost you another retrain.** After downloads: Runtime → Disconnect.

---

## After download — grade LOCALLY (free, deterministic)

Unzip `v3b_preds.zip` into `runs/`, then one command grades all four files and prints the
comparison table (v3b variants vs v3 / curr2 / curr / coder7b):

```bash
# unzip v3b_preds.zip into runs/ first, then:
bash scripts/grade_v3b.sh
```

`grade_v3b.sh` wraps `score_real_musical.py` (per direction, DTW-aligned heldout) and calls
`compare_realscores.py`. Read `above_gate_music` / `above_gate_rows` in the table — that is the
honest knowledge signal the anti-drone gate hides for a looping run (v3 was 0.358 over 67/501).

**Pick the winning decoder:** highest `above_gate_music` with `above_gate_rows` climbing well
past v3's 67/501, variety in a healthy ~0.3–0.6 band (NOT ~0 drone, NOT curr2's ~0.8
hallucination). If variety rises but set_f1 / hist_sim / interval_hist_sim FALL vs v3, the
decoder is manufacturing novel-but-wrong tokens (v2's failure) — reject it. n2w leads; w2n
ceiling ~1.2 (oligon/petaste both +1) — judge w2n by set_f1 / hist_sim.

Also worth a glance: grade a v3b n2w file on the SYNTHETIC melisma heldout to confirm the
melisma prior held (`sft_synth_melisma_heldout.jsonl`, use `score_synthetic_eval.py`).

If BOTH decoders cap out (looping persists or knowledge metrics drop) → the fix is training-side
(length/EOS discipline or a diversity term), then Branch 4 (reframe / reward-based training).
See `docs/byzantine_next_plans_by_outcome.md`.

---

## Gotchas (pre-empted)
- Cell 1 asserts files + flags before any GPU spend; fails loud if the pull didn't land.
- `--init-adapter` must log `Continuing from adapter: …`.
- Cell 5 v3b: ngram size **6** (not 3 — size-3 caused v2's 42% hallucination). temp variant
  stays mild (0.5) to break loops without inventing invalid tokens.
- Save adapters to Drive (Cell 4b-persist) BEFORE disconnecting — Colab has wiped them once.
- Shell in a Python cell needs `!`.
```
