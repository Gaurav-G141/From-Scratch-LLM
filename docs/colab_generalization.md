# Colab — Generalization Test (does the model learn the grammar, or memorize patterns?)

Same base as the successful run (`unsloth/Qwen2.5-Coder-7B-bnb-4bit`). The only change is the
**data split**: train on sequences from one region of the space, test on a region the model
never saw. A high score on the unseen region means the model applied the *rule* compositionally;
a sharp drop means it memorized the training distribution's shape.

Two tests. **Test A (length) is primary and needs no code change. Test B (construct) is the
higher-value stretch and needs a tiny, verified generator change.** Run A alone if compute is
tight; run both if units allow.

- **Test A — length:** train on ≤12-neume walks, test on ≥16-neume walks. The model literally
  never saw a sequence that long, so success = it walks the intervals step-by-step (the rule),
  not memorized shapes.
- **Test B — construct:** train with the octave-leap token `ypsili_over_kentima_oligon` removed
  entirely, test only on sequences that contain it. Success = it inferred the token's meaning
  from the grammar's structure despite zero training examples.

Keep the base model FIXED so the result isolates *generalization* (changing base + test at once
would confound it).

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

for p in ["scripts/build_synthetic_musicality.py","scripts/verify_synthetic_musicality.py",
          "scripts/train_byzantine_sft.py","scripts/predict_local.py","scripts/score_synthetic_eval.py"]:
    assert os.path.isfile(p), f"MISSING: {p}"
assert "eos_token_id=_stop_ids" in open("scripts/predict_local.py").read(), "predict_local missing EOS fix"

missing = [m for m in ["bitsandbytes","accelerate"] if importlib.util.find_spec(m) is None]
if missing: subprocess.run(["pip","-q","install",*missing], check=True); importlib.invalidate_caches()
for m in ["torch","transformers","peft","bitsandbytes","accelerate"]: importlib.import_module(m)
BASE = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"     # SAME base as the successful run — do not change
print("ready; BASE =", BASE)
```

---

## Cell 2A — Generate the LENGTH split (Test A; no code change)

Train ≤12 neumes; heldout ≥16 neumes, disjoint (large seed offset + exclude the train file).

```python
# TRAIN: short walks
!python scripts/build_synthetic_musicality.py --n 3000 --max-len 12 \
  --out data/byzantine/sft_synth_len_train.jsonl
# HELDOUT: long walks, disjoint from train
!python scripts/build_synthetic_musicality.py --n 800 --min-len 16 \
  --seed-start 10000000 --exclude data/byzantine/sft_synth_len_train.jsonl \
  --out data/byzantine/sft_synth_len_heldout.jsonl
```

## Cell 2A-verify — correctness + leakage gate (never skip)

```python
!python scripts/verify_synthetic_musicality.py data/byzantine/sft_synth_len_train.jsonl
!python scripts/verify_synthetic_musicality.py data/byzantine/sft_synth_len_heldout.jsonl
import json
def sigs(p): return {(json.loads(l)["task"], json.loads(l)["messages"][1]["content"]) for l in open(p)}
tr, ho = sigs("data/byzantine/sft_synth_len_train.jsonl"), sigs("data/byzantine/sft_synth_len_heldout.jsonl")
print("exact-input overlap (must be 0):", len(tr & ho))
# confirm the split really separates lengths
import statistics as st
def lens(p):
    out=[]
    for l in open(p):
        d=json.loads(l)
        if d["task"]=="neume_to_west":
            out.append(len(d["messages"][1]["content"].split("\n")[-1].split()))
    return out
print("train n2w neume-count median:", st.median(lens("data/byzantine/sft_synth_len_train.jsonl")))
print("heldout n2w neume-count median:", st.median(lens("data/byzantine/sft_synth_len_heldout.jsonl")))
```

Require: both PASS 0-error, overlap 0, train median well below heldout median.

---

## Cell 3 — Smoke gate (DO NOT SKIP)

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synth_len_train.jsonl --model {BASE} \
  --out models/_smoke --max-steps 20 --batch-size 8 --grad-accum 1 --seq-length 512
```

Require: `response-only markers`, decreasing loss, `Saved adapter`. OOM → `--seq-length 384`.

## Cell 4 — Train on the SHORT (seen) split

```python
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_synth_len_train.jsonl --model {BASE} \
  --out models/synth_len --epochs 2 --batch-size 8 --grad-accum 1 --seq-length 512
```

## Cell 5 — Predict on the LONG (unseen) split

Longer targets → allow more tokens (`--max-new-tokens 160`). EOS fix stops generation.

```python
import glob, os
def find_adapter(root):
    if os.path.isfile(os.path.join(root,"adapter_config.json")): return root
    h=glob.glob(os.path.join(root,"**","adapter_config.json"), recursive=True); assert h
    return os.path.dirname(sorted(h,key=lambda p:int("".join(c for c in os.path.basename(os.path.dirname(p)) if c.isdigit()) or 0))[-1])
ADP = find_adapter("models/synth_len"); print("adapter:", ADP)

!python scripts/predict_local.py --model {BASE} --adapter-path "{ADP}" \
  --eval data/byzantine/sft_synth_len_heldout.jsonl \
  --out runs/synth_len_preds.jsonl \
  --load-4bit --batch-size 32 --max-new-tokens 160
```

## Cell 6 — Score (the Test A result)

```python
!python scripts/score_synthetic_eval.py \
  --eval data/byzantine/sft_synth_len_heldout.jsonl \
  --pred runs/synth_len_preds.jsonl \
  --out runs/synth_len_score.json
```

**Read `neume_to_west`:** if `exact_match` / `melodic_equivalence_0_2` stay near the same-length
baseline (0.96 / 1.95), the model generalizes to longer sequences ⇒ learned the rule. A large
drop ⇒ length-pattern memorization. Report either honestly.

---

## Cell 2B — (STRETCH) Generate the CONSTRUCT split (Test B; needs the generator change)

**Prerequisite — a small generator edit (do locally, verify, push before this cell):** add an
opt-in `--exclude-construct <token>` to `scripts/build_synthetic_musicality.py`:
- TRAIN exclusion (two edits): drop `("ypsili_over_kentima_oligon",1)` from `_MOVE_WEIGHTS`
  (~L143) AND filter it out of `_ASC` (~L146) — else the ambit-clamp fallback (~L169-170)
  reintroduces it. Leave `INTERVAL_NEUMES` (L82) intact.
- HELDOUT require (one line): in the `build()` loop after `gen_neumes` (~L354), `continue` unless
  the token is present.
The independent verifier passes unchanged for both (it re-declares its own STEP table with the
token, and only asserts data-tokens ∈ STEP). After editing, run the verifier locally and push.

```python
# after the generator supports --exclude-construct:
!python scripts/build_synthetic_musicality.py --n 3000 \
  --exclude-construct ypsili_over_kentima_oligon \
  --out data/byzantine/sft_synth_noleap_train.jsonl
!python scripts/build_synthetic_musicality.py --n 400 \
  --require-construct ypsili_over_kentima_oligon \
  --seed-start 20000000 --exclude data/byzantine/sft_synth_noleap_train.jsonl \
  --out data/byzantine/sft_synth_onlyleap_heldout.jsonl
!python scripts/verify_synthetic_musicality.py data/byzantine/sft_synth_noleap_train.jsonl
!python scripts/verify_synthetic_musicality.py data/byzantine/sft_synth_onlyleap_heldout.jsonl
```

Then repeat Cells 3–6 with these two files (`--out models/synth_noleap`, predict on
`onlyleap_heldout`). **Result:** octave leaps correct despite zero training on them ⇒ strong
compositional generalization.

> Note: the octave token is rare (weight 1/22), so the heldout keep-rate is low and generation
> may hit the `attempts > n_walks*200` cap — fine for a modest heldout size.

---

## Cell 7 — Download the RESULTS directly (predictions + score)

The results are small JSON files (a few MB) — download them straight to your Mac's
`~/Downloads`, no Drive or model download needed. This is all you need to analyze/record the
run.

```python
from google.colab import files
import os
for f in ["runs/synth_len_score.json", "runs/synth_len_preds.jsonl"]:
    # (for Test B use synth_noleap_* / synth_onlyleap_* names)
    if os.path.isfile(f):
        print("downloading", f, f"({os.path.getsize(f)/1e6:.2f} MB)")
        files.download(f)
    else:
        print("MISSING (run predict/score first):", f)
```

Tip: `synth_len_score.json` alone is the headline number; `synth_len_preds.jsonl` is the raw
per-row predictions if you want to inspect specific cases.

## Cell 7b — (optional) Save adapter to Drive

Only if you want to keep the trained weights. The adapter is ~80 MB (or ~600 MB if you copy the
`checkpoints/` too — the top-level adapter alone is enough to reuse). Drive is better than a
browser download for something this size.

```python
from google.colab import drive; drive.mount('/content/drive')
import shutil, os
DST="/content/drive/MyDrive/byz_generalization"; os.makedirs(DST, exist_ok=True)
# copy ONLY the top-level adapter files, not the big checkpoints/ dir:
os.makedirs(f"{DST}/synth_len_adapter", exist_ok=True)
for fn in os.listdir(ADP):
    src=os.path.join(ADP, fn)
    if os.path.isfile(src):                       # skips checkpoints/ subdir
        shutil.copy(src, f"{DST}/synth_len_adapter/")
for f in ["runs/synth_len_preds.jsonl","runs/synth_len_score.json"]:
    if os.path.isfile(f): shutil.copy(f, DST)
print("saved adapter (top-level only) + results to", DST)
```

To also download the adapter to your Mac as a zip (instead of / in addition to Drive):
```python
import shutil
from google.colab import files
shutil.make_archive("/content/synth_len_adapter","zip",ADP)   # includes checkpoints -> larger
files.download("/content/synth_len_adapter.zip")
```

---

## After the run — record results
Add scores to `docs/byzantine_generalization_results_<date>.md`: Test A (length) and, if run,
Test B (construct), each vs the 96%/1.95 same-length baseline, with the honest read (rule vs
memorization). Optionally also run the **free** real-transfer check locally:
`python scripts/score_real_musical.py` on the existing Qwen adapter's real-heldout predictions.

## Gotchas
- Same base as before → the EOS path is identical to the successful run; Cell 1 asserts the fix.
- Never skip the verify/leakage gate (Cell 2A-verify) — the whole value is exactness + no leak.
- Test B requires the generator change pushed FIRST; Test A needs nothing beyond existing flags.
- Shell in a Python cell needs `!`.
```
