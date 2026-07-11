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

## Cell 5 — Predict on DTW-aligned real heldout (SOFT penalty only)

`--repetition-penalty 1.2` only — NO `--no-repeat-ngram-size` (that caused v2's
hallucination). `--max-new-tokens 160` = safe headroom, ~2× faster than 256.

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
!python scripts/predict_local.py --model {BASE} --adapter-path "{N2W}" \
  --eval data/byzantine/sft_aligned_n2w_heldout.jsonl \
  --out runs/v3_n2w_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 --repetition-penalty 1.2

!python scripts/predict_local.py --model {BASE} --adapter-path "{W2N}" \
  --eval data/byzantine/sft_aligned_w2n_heldout.jsonl \
  --out runs/v3_w2n_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 --repetition-penalty 1.2
```

---

## Cell 6 — Download predictions NOW (let Cell 5 finish first — one cell at a time)

```python
import os, zipfile
from google.colab import files
os.makedirs("/content/dl", exist_ok=True)
preds=[f for f in ["runs/v3_n2w_preds.jsonl","runs/v3_w2n_preds.jsonl"] if os.path.isfile(f)]
assert preds, "run Cell 5 first"
with zipfile.ZipFile("/content/dl/v3_preds.zip","w",zipfile.ZIP_DEFLATED) as z:
    for p in preds: z.write(p, arcname=os.path.basename(p))
files.download("/content/dl/v3_preds.zip")
```

Optional adapters:
```python
import shutil
from google.colab import files
for name,path in [("v3_n2w",N2W),("v3_w2n",W2N),("v3_prior",find_adapter("models/v3_prior"))]:
    shutil.make_archive(f"/content/dl/{name}","zip",path); files.download(f"/content/dl/{name}.zip")
```

**After downloads: Runtime → Disconnect.**

---

## After download — grade LOCALLY (free, deterministic)

Grade against the DTW-aligned heldout (the labels the model was actually trained toward),
and compare to v1/v2/baseline.

```bash
# unzip v3_preds.zip into runs/ first, then:
python3 scripts/score_real_musical.py --eval data/byzantine/sft_aligned_n2w_heldout.jsonl \
  --pred runs/v3_n2w_preds.jsonl --out runs/v3_n2w_realscore.json
python3 scripts/score_real_musical.py --eval data/byzantine/sft_aligned_w2n_heldout.jsonl \
  --pred runs/v3_w2n_preds.jsonl --out runs/v3_w2n_realscore.json
```

**Success = the labels-fix worked:** set_f1 / hist_sim / ngram_f1 materially above v1's best
(n2w set_f1 0.48), `real_musicality_0_2 > 0` with `good_rate > 0`, variety in a healthy mid
range (NOT ~0 drone, NOT ~0.9 hallucination). n2w should lead; w2n ceiling ~1.2 (one-to-many
oligon/petaste) — judge it by set_f1/ngram, not exact match.

Also worth a glance: grade v3 n2w on the SYNTHETIC melisma heldout to confirm the melisma
prior held (`sft_synth_melisma_heldout.jsonl`, use `score_synthetic_eval.py`).

If v3 still floors → the ceiling is deeper than alignment (Branch 4: reframe to recoverable
properties, or reward-based training). See `docs/byzantine_next_plans_by_outcome.md`.

---

## Gotchas (pre-empted)
- Cell 1 asserts files + flags before any GPU spend; fails loud if the pull didn't land.
- `--init-adapter` must log `Continuing from adapter: …`.
- SOFT penalty only in Cell 5 — do NOT re-add `--no-repeat-ngram-size` (v2 hallucination).
- Shell in a Python cell needs `!`.
```
