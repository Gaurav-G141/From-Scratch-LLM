# Colab — Evaluate & Download the Two Trained Models

Self-contained, copy-paste notebook for the **two real directional adapters you trained**
(`coder7b_n2w`, `coder7b_w2n`). It (1) generates predictions on the matching held-out
sets and (2) downloads the adapters + predictions. **Grading is done locally** — bring
back `preds.zip` and hand it over.

Every value below is verified against the actual checkpoint on disk, not assumed:

| fact | verified value | why it matters |
|---|---|---|
| base model | `unsloth/Qwen2.5-Coder-7B-bnb-4bit` (LoRA r=8) | both adapters share it |
| adapter location | `<out>/checkpoints/checkpoint-95/` (NESTED) | top-level `<out>` is empty → the not-found crash you hit |
| n2w recipe | trained on `sft_n2w_train_sub_cued.jsonl` | eval MUST be the **cued** held-out set |
| w2n recipe | trained on `sft_w2n_train_sub.jsonl` | eval is the **plain** w2n held-out set |
| chat template | checkpoint bundles ChatML, identical to what the scripts inject | inference formatting matches training |

> **Root cause of the earlier crash:** `--adapter-path` pointed at the `--out` dir, but the
> final top-level `save_pretrained` never landed — only `checkpoints/checkpoint-95/` has
> `adapter_config.json`. Cell 2 auto-discovers the real dir, so this can't recur.

---

## Cell 1 — Repo up to date + environment sanity (no reinstall)

Your session already has a working env (the last run loaded the 7B fine). We do **not**
reinstall — that risks version churn on a working box. We only pull the repo (the eval
files + scripts are git-tracked) and assert the imports exist.

> If a fresh Colab session wiped `/content`, clone first:
> `!cd /content && git clone <YOUR_REPO_URL> From-Scratch-LLM`

```python
import os, subprocess, importlib, glob

# find the repo by locating scripts/predict_local.py anywhere under /content
# (works whether the clone is flat, nested, or elsewhere — no hardcoded path)
hits = glob.glob("/content/**/scripts/predict_local.py", recursive=True)
assert hits, ("repo not found under /content — clone it first:\n"
              "  !cd /content && git clone <YOUR_REPO_URL> From-Scratch-LLM")
ROOT = os.path.dirname(os.path.dirname(hits[0]))   # .../scripts/predict_local.py -> repo root
os.chdir(ROOT)
print("cwd:", os.getcwd())

# pull latest (brings tracked eval files + scripts); ignore failure if offline
print(subprocess.run(["git", "pull", "--ff-only"], capture_output=True, text=True).stdout)

# verify the exact files this notebook needs are present
need = [
    "scripts/predict_local.py",
    "data/byzantine/sft_n2w_heldout_cued.jsonl",   # n2w eval (cued)
    "data/byzantine/sft_w2n_heldout.jsonl",        # w2n eval (plain)
]
for p in need:
    assert os.path.isfile(p), f"MISSING: {p}"
print("all required files present")

# imports the run needs. torch/transformers/peft are preinstalled on Colab.
# bitsandbytes (for --load-4bit) is usually NOT in a fresh session; accelerate (for
# device_map="auto") usually IS but isn't guaranteed. Install either if absent.
missing = [m for m in ["bitsandbytes", "accelerate"]
           if importlib.util.find_spec(m) is None]
if missing:
    subprocess.run(["pip", "-q", "install", *missing], check=True)
    importlib.invalidate_caches()
for m in ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]:
    importlib.import_module(m)
print("imports OK")
```

---

## Cell 2 — Config + auto-discover the real adapter dirs

Set the two `--out` roots you used during training. The helper walks into
`checkpoints/checkpoint-95/` (or the newest checkpoint) and returns the dir that actually
contains `adapter_config.json`.

```python
import glob, os

BASE = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"

# The trained adapters are committed IN THE REPO under "models 2/" (note the space),
# so no Drive mount is needed — a git pull brought them onto the Colab box.
# If yours live elsewhere (e.g. Drive), point these at those --out roots instead.
N2W_ROOT = "/content/From-Scratch-LLM/models 2/coder7b_n2w"
W2N_ROOT = "/content/From-Scratch-LLM/models 2/coder7b_w2n"

def find_adapter(root):
    """Return the dir holding adapter_config.json: root itself, else newest checkpoint."""
    if os.path.isfile(os.path.join(root, "adapter_config.json")):
        return root
    hits = glob.glob(os.path.join(root, "**", "adapter_config.json"), recursive=True)
    if not hits:
        raise SystemExit(
            f"No adapter_config.json anywhere under {root}. "
            f"Check the path — is this the --out you trained to?")
    def step(p):  # sort by checkpoint step number, take the highest
        d = os.path.basename(os.path.dirname(p))
        return int("".join(c for c in d if c.isdigit()) or 0)
    return os.path.dirname(sorted(hits, key=step)[-1])

N2W = find_adapter(N2W_ROOT)
W2N = find_adapter(W2N_ROOT)

# eval files, matched to the recipe each adapter was trained on
N2W_EVAL = "data/byzantine/sft_n2w_heldout_cued.jsonl"   # n2w trained cued
W2N_EVAL = "data/byzantine/sft_w2n_heldout.jsonl"        # w2n trained plain

print("N2W adapter:", N2W)
print("W2N adapter:", W2N)
print("N2W eval   :", N2W_EVAL)
print("W2N eval   :", W2N_EVAL)
```

---

## Cell 3 — Pre-flight assertions (catches problems BEFORE spending GPU)

```python
import os
for tag, adir, ev in [("n2w", N2W, N2W_EVAL), ("w2n", W2N, W2N_EVAL)]:
    assert os.path.isfile(os.path.join(adir, "adapter_config.json")), f"{tag}: no adapter_config.json in {adir}"
    assert os.path.isfile(os.path.join(adir, "adapter_model.safetensors")), f"{tag}: no weights in {adir}"
    assert os.path.isfile(ev), f"{tag}: eval file missing {ev}"
    print(f"{tag}: adapter + eval OK")
print("pre-flight passed — safe to generate")
```

---

## Cell 4 — Generate predictions (the "eval" the model runs)

Greedy (temperature 0) for reproducible scoring. `--load-4bit` matches the training base
(the base is already a bnb-4bit checkpoint; the config is reused — the "already has a
quantization_config" warning is expected and harmless). `--max-new-tokens 256` is safe
headroom: verified longest targets are 121 words (n2w) / 60 words (w2n); a trained model
emits EOS well before the cap.

```python
# tuned n2w
!python scripts/predict_local.py --model {BASE} --adapter-path "{N2W}" \
  --eval {N2W_EVAL} --out runs/coder7b_n2w_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 256

# tuned w2n
!python scripts/predict_local.py --model {BASE} --adapter-path "{W2N}" \
  --eval {W2N_EVAL} --out runs/coder7b_w2n_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 256
```

Expect a `Wrote 555 predictions -> runs/coder7b_<dir>_preds.jsonl` line for each.

---

## Cell 5 — (Optional) base-model predictions for a base-vs-tuned delta

Same eval files, **no `--adapter-path`** = untuned base. Skip if you only want the tuned
numbers.

```python
!python scripts/predict_local.py --model {BASE} \
  --eval {N2W_EVAL} --out runs/coder7b_n2w_base_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 256

!python scripts/predict_local.py --model {BASE} \
  --eval {W2N_EVAL} --out runs/coder7b_w2n_base_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 256
```

---

## Cell 6 — Download the two models + predictions

Zips the **resolved adapter dirs** (the ones with `adapter_config.json`), so the download
is a valid, reloadable adapter — a few MB of LoRA weights, not the 7B base. `preds.zip`
holds every `runs/*_preds.jsonl` to grade locally.

```python
import shutil, os
os.makedirs("/content/dl", exist_ok=True)

shutil.make_archive("/content/dl/coder7b_n2w", "zip", N2W)   # zips the checkpoint dir
shutil.make_archive("/content/dl/coder7b_w2n", "zip", W2N)
shutil.make_archive("/content/dl/preds", "zip", "runs")

from google.colab import files
for f in ["coder7b_n2w.zip", "coder7b_w2n.zip", "preds.zip"]:
    files.download(f"/content/dl/{f}")

print("done — after downloads finish: Runtime → Disconnect (idle GPU still bills)")
```

---

## After the download — grading (local, no GPU)

Bring back `preds.zip` (and the adapter zips if you want them versioned). Grading runs on
CPU with the deterministic scorer — hand `preds.zip` over, or run it yourself:

```bash
# unzip preds.zip into runs/ first, then per direction:
python3 scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_n2w_heldout_cued.jsonl \
  --pred runs/coder7b_n2w_preds.jsonl \
  --out runs/coder7b_n2w_score.json

python3 scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_w2n_heldout.jsonl \
  --pred runs/coder7b_w2n_preds.jsonl \
  --out runs/coder7b_w2n_score.json
```

Read the per-direction `melodic_equivalence_0_2` / `pitch_accuracy` / `exact_match`. If
you ran Cell 5, grade the `*_base_preds.jsonl` too for the base-vs-tuned delta.

---

## Gotchas (all pre-empted above)
- **`Can't find adapter_config.json`** → the adapter is nested in `checkpoints/checkpoint-95/`;
  Cell 2's `find_adapter` resolves it. Never pass the bare `--out` root.
- **`SyntaxError: invalid decimal literal`** → a shell command in a Python cell. All shell
  here is under `!` inside Python cells, so it's fine.
- **`quantization_config` warning** → expected; the base is already 4-bit, its config wins.
- **Recipe mismatch** → n2w uses the **cued** held-out set, w2n the **plain** one, matching
  what each was trained on. Do not swap them or n2w scores falsely low.
