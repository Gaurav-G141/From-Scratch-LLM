# Colab — Curriculum v2: Blended Stage-2 + Anti-Loop Decoding

Fixes the two failure modes found in curriculum v1 (`docs/byzantine_curriculum_results_20260711.md`):
- **Catastrophic forgetting** — v1 sequenced synthetic→real, and pure-real stage 2 erased
  the interval grammar (synthetic sanity fell 2.0→0.0). **Fix (step A): stage 2 trains on a
  BLENDED file** (synthetic + real interleaved), so the grammar stays in-distribution and
  can't be forgotten.
- **Loop degeneration** — v1 broke the flat drone but fell into short repeating loops.
  **Fix (step B): decode with `--repetition-penalty 1.2 --no-repeat-ngram-size 3`.**

Everything below was verified locally before this doc was written:
- Blend files built + validated (2760 rows each, perfect synth/real alternation, all
  well-formed): `data/byzantine/sft_blend_n2w.jsonl`, `sft_blend_w2n.jsonl`.
- Prompt formats confirmed identical across synthetic/real/held-out (all carry `Mode` +
  `Ison:` lines) — the blend is coherent, the held-out eval matches training.
- `predict_local.py` repetition flags added, parse-checked; scorer grading path dry-run
  with a perfect predictor (real_musicality 1.74, the realistic ceiling on real gold).
- All new files committed and pushed to `main` (Colab pulls `main`).

Base model = `unsloth/Qwen2.5-Coder-7B-bnb-4bit` (same as every prior run — one variable
changes: the stage-2 data is now blended, not pure real).

---

## What runs

| stage | data | note |
|---|---|---|
| 1. synthetic pretrain | `sft_synth_2500.jsonl` | shared grammar prior (both directions) |
| 2a. n2w blended finetune | `sft_blend_n2w.jsonl` | `--init-adapter models/curr2_synth`, low LR |
| 2b. w2n blended finetune | `sft_blend_w2n.jsonl` | `--init-adapter models/curr2_synth`, low LR |
| 3. predict (anti-loop) | real held-out per dir | `--repetition-penalty 1.2 --no-repeat-ngram-size 3` |

---

## Cell 1 — Locate repo, pull latest, install deps if absent

```python
import os, subprocess, importlib, glob

hits = glob.glob("/content/**/scripts/train_byzantine_sft.py", recursive=True)
assert hits, ("repo not found under /content — clone it:\n"
              "  !cd /content && git clone https://github.com/Gaurav-G141/From-Scratch-LLM From-Scratch-LLM")
ROOT = os.path.dirname(os.path.dirname(hits[0]))
os.chdir(ROOT)
print("cwd:", os.getcwd())
# MUST pull — this run depends on files just pushed (blends + predict_local repetition flags)
print(subprocess.run(["git", "pull", "--ff-only"], capture_output=True, text=True).stdout)

need = [
    "scripts/train_byzantine_sft.py",
    "scripts/predict_local.py",
    "data/byzantine/sft_synth_2500.jsonl",
    "data/byzantine/sft_blend_n2w.jsonl",          # step-A blended n2w
    "data/byzantine/sft_blend_w2n.jsonl",          # step-A blended w2n
    "data/byzantine/sft_n2w_heldout_cued.jsonl",   # n2w eval (matches training recipe)
    "data/byzantine/sft_w2n_heldout.jsonl",        # w2n eval
]
for p in need:
    assert os.path.isfile(p), f"MISSING (did you push + git pull?): {p}"
print("all required files present")

# verify --init-adapter and the repetition flag are actually in the pulled code
assert "--init-adapter" in open("scripts/train_byzantine_sft.py").read(), "train script missing --init-adapter — pull didn't land"
assert "repetition-penalty" in open("scripts/predict_local.py").read(), "predict script missing --repetition-penalty — pull didn't land"
print("code has --init-adapter and --repetition-penalty")

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

## Cell 2 — Smoke gate (DO NOT SKIP, ~3 min incl. one-time model download)

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synth_2500.jsonl \
  --model {BASE} \
  --out models/_smoke \
  --max-steps 20 --batch-size 8 --grad-accum 1 --seq-length 1024
```

Require in the log: `response-only markers: instruction=... response=...`, decreasing loss,
`Saved adapter → models/_smoke`. OOM → add `--seq-length 768` (then 512) to every train cell.

---

## Cell 3 — Stage 1: synthetic pretrain (shared prior)

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synth_2500.jsonl \
  --model {BASE} \
  --out models/curr2_synth \
  --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024
```

---

## Cell 4 — Stage 2: BLENDED finetune per direction (step A fix)

Blended data keeps synthetic grammar present in every span of training. **LR 3e-5** (much
lower than v1's 1e-4) so real data adapts, not overwrites. Must log
`Continuing from adapter: models/curr2_synth` — if absent, STOP (training fresh, not
continuing).

```python
# 4a. n2w blended (synthetic n2w + real n2w cued, interleaved)
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_blend_n2w.jsonl \
  --model {BASE} \
  --init-adapter models/curr2_synth \
  --out models/curr2_n2w \
  --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024 --lr 3e-5
```

```python
# 4b. w2n blended (synthetic w2n + real w2n, interleaved)
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_blend_w2n.jsonl \
  --model {BASE} \
  --init-adapter models/curr2_synth \
  --out models/curr2_w2n \
  --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 1024 --lr 3e-5
```

---

## Cell 5 — Predict with anti-loop decoding (step B fix)

`--max-new-tokens 160` (real targets max ~121 words, so this is safe headroom and ~2× faster
than 256). `--repetition-penalty 1.2 --no-repeat-ngram-size 3` breaks the looping.

```python
import glob, os
def find_adapter(root):
    if os.path.isfile(os.path.join(root, "adapter_config.json")):
        return root
    h = glob.glob(os.path.join(root, "**", "adapter_config.json"), recursive=True)
    assert h, f"no adapter under {root}"
    return os.path.dirname(sorted(h, key=lambda p: int("".join(c for c in os.path.basename(os.path.dirname(p)) if c.isdigit()) or 0))[-1])

N2W = find_adapter("models/curr2_n2w")
W2N = find_adapter("models/curr2_w2n")
print("n2w:", N2W, "\nw2n:", W2N)
```

```python
!python scripts/predict_local.py --model {BASE} --adapter-path "{N2W}" \
  --eval data/byzantine/sft_n2w_heldout_cued.jsonl \
  --out runs/curr2_n2w_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 \
  --repetition-penalty 1.2 --no-repeat-ngram-size 3

!python scripts/predict_local.py --model {BASE} --adapter-path "{W2N}" \
  --eval data/byzantine/sft_w2n_heldout.jsonl \
  --out runs/curr2_w2n_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 \
  --repetition-penalty 1.2 --no-repeat-ngram-size 3
```

Each ends with `Wrote 555 predictions`. Progress prints `N/555` to stderr.

---

## Cell 6 — Download prediction files NOW (before anything else)

One cell runs at a time in Colab, so let Cell 5 finish, THEN run this. It downloads the
only files needed for grading (grading is free local Python — no API, no GPU).

```python
import os, zipfile
from google.colab import files
os.makedirs("/content/dl", exist_ok=True)
preds = [f for f in ["runs/curr2_n2w_preds.jsonl", "runs/curr2_w2n_preds.jsonl"] if os.path.isfile(f)]
assert preds, "no prediction files — run Cell 5 first"
with zipfile.ZipFile("/content/dl/curr2_preds.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in preds:
        z.write(p, arcname=os.path.basename(p))
print("zipped:", preds)
files.download("/content/dl/curr2_preds.zip")
```

Optional — also save the adapters off-box:

```python
import shutil
from google.colab import files
for name, path in [("curr2_n2w", N2W), ("curr2_w2n", W2N)]:
    shutil.make_archive(f"/content/dl/{name}", "zip", path)
    files.download(f"/content/dl/{name}.zip")
```

**After downloads: Runtime → Disconnect** (idle GPU still bills).

---

## After download — grade LOCALLY (free, deterministic)

Hand `curr2_preds.zip` to the assistant, or run it yourself. Compares against v1 curriculum
(`runs/curr_*_realscore.json`) and the collapsed 7B baseline (`runs/coder7b_*_realscore.json`).

```bash
# unzip curr2_preds.zip into runs/ first, then:
python3 scripts/score_real_musical.py \
  --eval data/byzantine/sft_n2w_heldout_cued.jsonl \
  --pred runs/curr2_n2w_preds.jsonl --out runs/curr2_n2w_realscore.json

python3 scripts/score_real_musical.py \
  --eval data/byzantine/sft_w2n_heldout.jsonl \
  --pred runs/curr2_w2n_preds.jsonl --out runs/curr2_w2n_realscore.json
```

**Success = beats v1 on:** variety (off the ~0.02 floor), set_f1 / hist_sim / ngram_f1 up,
lower pure-repeat %, and ideally `real_musicality_0_2 > 0` with `good_rate > 0`. The scorer's
ceiling on real gold is ~1.74, so that's the target to approach, not 2.0.

### Read n2w and w2n on DIFFERENT scales — the directions are not equally well-posed

The two directions have different intrinsic ceilings *by the music*, not by a data bug, so
do not expect symmetric numbers:

- **neume→west (n2w) is a true function.** Each neume maps to exactly one pitch step
  (`oligon`=+1, `apostrophos`=−1, …), so given the Ison anchor the pitch sequence is
  uniquely determined. On synthetic this hits melodic **2.0**. Expect n2w to climb highest.
- **west→neume (w2n) is one-to-many.** `oligon` AND `petaste` both encode "+1" (an accent
  distinction with no Western-pitch counterpart), so a rising second maps to two valid
  neumes and the input can't disambiguate. Even on *perfect* synthetic, w2n tops out around
  **1.2**, not 2.0 (`docs/byzantine_synthetic_breakthrough_20260709.md`).

Therefore, when grading `curr2`:
- A lower w2n number is **not** failure — it reflects a real notational ambiguity. Judge w2n
  by `set_f1` / `hist_sim` / `ngram_f1` (right neume vocabulary and local shape); exact-match
  is unfairly capped there.
- The clean 1:1 correctness the synthetic prior teaches is inherently n2w-directional. w2n
  benefits from the prior (right vocabulary, non-degeneracy) but cannot reach n2w's ceiling.

If it still floors → escalate to option #3 (DTW label realignment), the overnight job.

---

## Gotchas (all pre-empted)
- **Depends on a fresh `git pull`** — Cell 1 asserts the new files + flags are present and
  fails loudly if the pull didn't land. No silent old-code runs.
- `--init-adapter` must print `Continuing from adapter: …`; absence = fresh training, stop.
- `Can't find adapter_config.json` → nested under `checkpoints/`; `find_adapter` handles it.
- Shell in a Python cell needs `!`; a bare line throws `SyntaxError: invalid decimal literal`.
- One cell at a time — don't try to run the download while Cell 5 is generating.
```
