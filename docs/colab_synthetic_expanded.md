# Colab — Expanded Synthetic Musicality on a Real Model

Train a larger model on the **expanded** correct-by-construction synthetic grammar
(octave-range ascending leaps + reversible duration/rhythm channel) and score it on the
disjoint synthetic heldout with the deterministic scorer. No API, no LLM judge — the
synthetic gold is exact, so grading is free and objective.

**Why this is the honest target:** forensics proved the real neume↔pitch corpus is not
recoverable (0.56 pitch-bearing:pitch ratio, ~35% directional agreement corpus-wide, ~10%
label ceiling the model already hit). The synthetic grammar is the one place there is a real,
provable capability — and it now includes leaps up to an octave and note durations, in both
directions, all verified 0-error / 0-leakage.

Base model = `unsloth/Qwen2.5-Coder-7B-bnb-4bit` (same base as the curriculum runs, for
comparison). Swap `BASE` for a larger model if you have the VRAM (see Cell 1 note).

---

## Data (already in the repo, pushed to main)

| file | rows | contents |
|---|---|---|
| `data/byzantine/sft_synthetic_musicality.jsonl` | 23,942 | full expanded train (both dirs, 4 modes, leaps+durations) |
| `data/byzantine/sft_synthetic_musicality_heldout.jsonl` | 4,794 | disjoint heldout (seed-start 10M, 0 leakage) |

Both regenerate + self-verify via `scripts/build_synthetic_musicality.py` and
`scripts/verify_synthetic_musicality.py` (independent re-derivation).

---

## Cell 1 — Repo, deps, sanity

```python
import os, subprocess, importlib, glob
hits = glob.glob("/content/**/scripts/train_byzantine_sft.py", recursive=True)
if not hits:
    subprocess.run(["git","clone","https://github.com/Gaurav-G141/From-Scratch-LLM",
                    "/content/From-Scratch-LLM"], check=True)
    hits = glob.glob("/content/**/scripts/train_byzantine_sft.py", recursive=True)
ROOT = os.path.dirname(os.path.dirname(hits[0])); os.chdir(ROOT)
print("cwd:", os.getcwd())
print(subprocess.run(["git","pull","--ff-only"], capture_output=True, text=True).stdout)

need = [
    "scripts/train_byzantine_sft.py", "scripts/predict_local.py",
    "scripts/score_synthetic_eval.py", "scripts/verify_synthetic_musicality.py",
    "data/byzantine/sft_synthetic_musicality.jsonl",
    "data/byzantine/sft_synthetic_musicality_heldout.jsonl",
]
for p in need:
    assert os.path.isfile(p), f"MISSING (push + pull?): {p}"
# EOS fix must be present (else predictions run on to max_new_tokens)
assert "eos_token_id=_stop_ids" in open("scripts/predict_local.py").read(), "predict_local missing EOS fix"
print("all files + EOS fix present")

missing = [m for m in ["bitsandbytes","accelerate"] if importlib.util.find_spec(m) is None]
if missing: subprocess.run(["pip","-q","install",*missing], check=True); importlib.invalidate_caches()
for m in ["torch","transformers","peft","bitsandbytes","accelerate"]: importlib.import_module(m)

BASE = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
# Larger option if VRAM allows (A100): "unsloth/Qwen2.5-Coder-14B-bnb-4bit" or a 32B 4-bit.
# The task is small-vocab interval arithmetic; 7B is already plenty, bigger mainly speeds
# convergence. Keep 7B for a clean comparison to the curriculum runs.
print("BASE =", BASE)
```

---

## Cell 2 — Verify the data is correct-by-construction (free, ~10s)

Never train without this — it re-derives every row independently.

```python
!python scripts/verify_synthetic_musicality.py data/byzantine/sft_synthetic_musicality.jsonl
!python scripts/verify_synthetic_musicality.py data/byzantine/sft_synthetic_musicality_heldout.jsonl
```

Require `PASS: 100% ... correct by construction` and `0 content / 0 reversibility errors` on
both. If either fails, STOP — do not train.

---

## Cell 3 — Smoke gate (DO NOT SKIP)

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synthetic_musicality.jsonl --model {BASE} \
  --out models/_smoke --max-steps 20 --batch-size 8 --grad-accum 1 --seq-length 512
```

Require: `response-only markers: …`, decreasing loss, `Saved adapter → models/_smoke`.
OOM → drop `--seq-length` to 384 (rows are short: ~6–20 tokens of neumes/pitches).

---

## Cell 4 — Train on the full expanded synthetic

24k rows is small; 2 epochs on an L4/A100 is minutes. The interval grammar + durations are
deterministic, so it should converge hard.

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synthetic_musicality.jsonl --model {BASE} \
  --out models/synth_expanded --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 512
```

---

## Cell 5 — Predict on the disjoint heldout (EOS fix + short cap)

`--max-new-tokens 96`: synthetic targets are short (≤~20 pitches + a few duration marks), so
this is ample and fast. The EOS fix (pushed) makes generation stop at `<|im_end|>` instead of
running on. Pure greedy — synthetic has one correct answer, so no sampling/penalty.

```python
import glob, os
def find_adapter(root):
    if os.path.isfile(os.path.join(root,"adapter_config.json")): return root
    h=glob.glob(os.path.join(root,"**","adapter_config.json"), recursive=True)
    assert h, f"no adapter under {root}"
    return os.path.dirname(sorted(h,key=lambda p:int("".join(c for c in os.path.basename(os.path.dirname(p)) if c.isdigit()) or 0))[-1])
ADP = find_adapter("models/synth_expanded"); print("adapter:", ADP)

!python scripts/predict_local.py --model {BASE} --adapter-path "{ADP}" \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --out runs/synth_expanded_preds.jsonl \
  --load-4bit --batch-size 32 --max-new-tokens 96
```

---

## Cell 6 — Score (deterministic, free) — the actual result

```python
!python scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
  --pred runs/synth_expanded_preds.jsonl \
  --out runs/synth_expanded_score.json
```

Read the **per-direction** block:
- **neume_to_west** is the real test: `exact_match`, `pitch_accuracy`, and
  `melodic_equivalence_0_2`. The prior 1:1 synthetic hit ~2.0 melodic; this set is harder
  (octave leaps + `pitch:beats` durations), so exact_match tests whether the model absorbed
  the added complexity, not just the easy stepwise grammar.
- **west_to_neume** recovers neumes incl. duration signs (bijective, so fully scorable here —
  unlike the real corpus where oligon/petaste collide).

A strong result: n2w `exact_match` high (≥0.8) and `melodic_equivalence_0_2` near 2.0, w2n
similar. That is a clean, honest capability claim: *the model learned the Byzantine interval
grammar including leaps and rhythmic durations, bidirectionally, on held-out sequences.*

---

## Cell 7 — Save the adapter to Drive (so a disconnect can't wipe it)

```python
from google.colab import drive; drive.mount('/content/drive')
import shutil, os
DST="/content/drive/MyDrive/byz_synth_expanded"; os.makedirs(DST, exist_ok=True)
shutil.copytree(ADP, f"{DST}/adapter", dirs_exist_ok=True)
shutil.copy("runs/synth_expanded_preds.jsonl", DST)
shutil.copy("runs/synth_expanded_score.json", DST)
print("saved adapter + preds + score to", DST)
```

Optional local download:
```python
import shutil
from google.colab import files
shutil.make_archive("/content/synth_expanded_adapter","zip",ADP)
files.download("/content/synth_expanded_adapter.zip")
files.download("runs/synth_expanded_score.json")
```

---

## Cell 8 — (optional) Push to HuggingFace

```python
from huggingface_hub import login; login()   # write-scope token
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, ADP)
model.push_to_hub("YOUR_USERNAME/byzantine-synthetic-grammar-lora")   # adapters only (small)
```

For a standalone merged model, load a NON-4bit base (`Qwen/Qwen2.5-Coder-7B`) first, then
`model.merge_and_unload().push_to_hub(...)` — merging into a 4-bit base is lossy.

---

## Gotchas (pre-empted)
- Cell 1 asserts files + the EOS fix before any GPU spend.
- Cell 2 is the correctness gate — never skip; synthetic's whole value is being exact.
- Shell in a Python cell needs `!`.
- The EOS fix is what stops the run-on; if you see predictions far longer than the gold,
  confirm `eos_token_id=_stop_ids` is in `predict_local.py` (Cell 1 checks this).
- Compare to the local MPS result in `docs/byzantine_synthetic_expanded_results_*.md`.
```
