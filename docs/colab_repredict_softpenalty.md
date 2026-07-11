# Colab — Re-predict curr2 with SOFT penalty only (diagnostic, no retrain)

curr2 used `--repetition-penalty 1.2 --no-repeat-ngram-size 3`. The hard ngram block forced
the model to mutate tokens to avoid repeats, manufacturing 42% invalid pitch tokens
(`G8`, `D------6`, `Isole`…). This re-runs the **same already-trained v2 adapters** with the
**soft penalty ONLY** (no ngram block) to see the model's honest output.

**No training.** Just prediction on the existing `models/curr2_*` adapters — a few minutes.
If those adapters aren't still on the box (fresh session wiped `/content`), you'll need the
`curr2_*.zip` you downloaded, or skip this and go straight to Stage A.

---

## Cell 1 — Locate repo, deps, base

```python
import os, subprocess, importlib, glob
hits = glob.glob("/content/**/scripts/predict_local.py", recursive=True)
assert hits, "repo not found — clone: !cd /content && git clone https://github.com/Gaurav-G141/From-Scratch-LLM From-Scratch-LLM"
ROOT = os.path.dirname(os.path.dirname(hits[0])); os.chdir(ROOT)
print(subprocess.run(["git","pull","--ff-only"], capture_output=True, text=True).stdout)
missing = [m for m in ["bitsandbytes","accelerate"] if importlib.util.find_spec(m) is None]
if missing: subprocess.run(["pip","-q","install",*missing], check=True); importlib.invalidate_caches()
for m in ["torch","transformers","peft","bitsandbytes"]: importlib.import_module(m)
BASE = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
print("ready")
```

## Cell 2 — Find the v2 adapters (must still exist from the v2 run)

```python
def find_adapter(root):
    if os.path.isfile(os.path.join(root,"adapter_config.json")): return root
    h = glob.glob(os.path.join(root,"**","adapter_config.json"), recursive=True)
    assert h, f"no adapter under {root} — was the box wiped? re-upload curr2_*.zip or skip to Stage A"
    return os.path.dirname(sorted(h, key=lambda p:int("".join(c for c in os.path.basename(os.path.dirname(p)) if c.isdigit()) or 0))[-1])
N2W = find_adapter("models/curr2_n2w"); W2N = find_adapter("models/curr2_w2n")
print("n2w:", N2W, "\nw2n:", W2N)
```

## Cell 3 — Predict with SOFT penalty only (note: NO --no-repeat-ngram-size)

```python
!python scripts/predict_local.py --model {BASE} --adapter-path "{N2W}" \
  --eval data/byzantine/sft_n2w_heldout_cued.jsonl \
  --out runs/curr2soft_n2w_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 --repetition-penalty 1.2

!python scripts/predict_local.py --model {BASE} --adapter-path "{W2N}" \
  --eval data/byzantine/sft_w2n_heldout.jsonl \
  --out runs/curr2soft_w2n_preds.jsonl \
  --load-4bit --batch-size 16 --max-new-tokens 160 --repetition-penalty 1.2
```

## Cell 4 — Download for grading

```python
import zipfile
from google.colab import files
os.makedirs("/content/dl", exist_ok=True)
with zipfile.ZipFile("/content/dl/curr2soft_preds.zip","w",zipfile.ZIP_DEFLATED) as z:
    for p in ["runs/curr2soft_n2w_preds.jsonl","runs/curr2soft_w2n_preds.jsonl"]:
        if os.path.isfile(p): z.write(p, arcname=os.path.basename(p))
files.download("/content/dl/curr2soft_preds.zip")
```

---

## Grade locally
```bash
python3 scripts/score_real_musical.py --eval data/byzantine/sft_n2w_heldout_cued.jsonl \
  --pred runs/curr2soft_n2w_preds.jsonl --out runs/curr2soft_n2w_realscore.json
python3 scripts/score_real_musical.py --eval data/byzantine/sft_w2n_heldout.jsonl \
  --pred runs/curr2soft_w2n_preds.jsonl --out runs/curr2soft_w2n_realscore.json
```

**What we're checking:** invalid-token % should fall well below curr2's 42%, and set_f1
should recover toward v1's 0.48. This tells us how much of curr2's collapse was the decoding
artifact vs. genuinely weak knowledge — either way, Stage A (label realignment) is the next
real lever.
```
