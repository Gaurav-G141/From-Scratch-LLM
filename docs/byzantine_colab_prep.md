# Pre-Colab Checklist — Byzantine SLM Training

Everything to do **before** opening Google Colab, plus the exact commands to run once
there. Goal: train the Byzantine transcription LoRA on a real GPU (Unsloth QLoRA) instead
of the slow local MPS path.

---

## 0. TL;DR

1. Commit + push the training data and scripts (data JSONL is git-tracked; `models/` is not).
2. Pick base model (§3b bake-off — target 7–8B) and dataset (§3).
3. On Colab: GPU runtime → `git clone` → `pip install` → run `train_byzantine_sft.py` with
   `--epochs` and `--model` (Unsloth path auto-activates on CUDA).
4. Download the adapter from `models/…` back out of Colab (it is NOT auto-committed).

### Sequencing note (current plan)
The bigger/better training run is gated on **more data from the other agent** (it's
producing the near-1:1 subset / additional aligned data). Order of operations:
1. Wait for the other agent's data → re-run the §1d sanity check + a quick audit on it.
2. Rebuild the training files to include it (same `build_*` scripts).
3. Then hand off Colab credentials so training can run on GPU with the full dataset.
Until the new data lands, the §3b bake-off can run on the existing directional data to
pick the base model — that decision doesn't need the new data.

---

## 1. Local prep (do these before Colab)

### 1a. Decide what to train, and confirm the files exist
| Dataset | File | Rows | Use |
|---|---|---|---|
| Bidirectional (windowed) | `data/byzantine/sft_translation_train.jsonl` | 10,496 | one model, both directions |
| byz→west only | `data/byzantine/sft_n2w_train_sub.jsonl` | 1,510 | directional (best notation result so far) |
| west→byz only | `data/byzantine/sft_w2n_train_sub.jsonl` | 1,510 | directional (failed last time — see §5) |
| Genus-tagged (experiment) | `data/byzantine/sft_translation_train_genus.jsonl` | 10,496 | microtonal-intent experiment |
| Near-1:1 subset (in progress) | `data/byzantine/sft_near1to1_train_cued.jsonl` | 1,916 | alignment experiment (from the other process) |

Held-out counterparts exist for each (`*_heldout*.jsonl`). Verify before pushing:
```bash
wc -l data/byzantine/sft_translation_train.jsonl \
      data/byzantine/sft_n2w_train_sub.jsonl \
      data/byzantine/sft_w2n_train_sub.jsonl
```

### 1b. Commit and push (CRITICAL — Colab pulls via git)
The training JSONL files **are** git-tracked (`.gitignore` only excludes `models/` and
`corpus/*.pdf`), so they travel with a `git clone`. But confirm the newest ones are
committed — some were created this session and may be untracked:
```bash
git status --short data/byzantine/*.jsonl
git add data/byzantine/*.jsonl scripts/ docs/ requirements*.txt
git commit -m "Byzantine training data + scripts for Colab run"
git push
```
If you do NOT want to push the ~9 MB data through git, alternative: upload the specific
JSONL to Google Drive and mount it in Colab (see §2, option B).

### 1c. Note what will NOT come through git
- `models/` is gitignored → trained adapters must be downloaded out of Colab manually.
- `data/byzantine/corpus/*.pdf` is gitignored → **not needed for training** (training uses
  the extracted JSONL, not the PDFs). Only re-extraction needs PDFs, which you won't do on
  Colab.
- `tools/Audiveris.app` → not needed on Colab (OMR already done; only relevant if you
  re-extract Western pitches, which you won't).

### 1d. Sanity-check the data one last time (optional but cheap)
```bash
.venv/bin/python - <<'PY'
import json
for f in ["sft_translation_train","sft_n2w_train_sub","sft_w2n_train_sub"]:
    rows=[json.loads(l) for l in open(f"data/byzantine/{f}.jsonl")]
    bad=sum(1 for r in rows if len(r.get("messages",[]))!=3)
    print(f"{f}: {len(rows)} rows, {bad} malformed")
PY
```

---

## 2. Colab environment setup

### Runtime
- **Runtime → Change runtime type → GPU.** A **T4** (free tier, 16 GB) is plenty for
  Qwen3-1.7B 4-bit QLoRA (~2–3 GB weights + activations). L4/A100 (Pro) are faster and let
  you raise `--batch-size`.
- Confirm the GPU:
  ```python
  !nvidia-smi
  ```

### Option A — clone from git (recommended if you pushed the data)
```bash
!git clone https://github.com/<you>/From-Scratch-LLM.git
%cd From-Scratch-LLM
```

### Option B — data via Google Drive (if data not in git)
```python
from google.colab import drive; drive.mount('/content/drive')
# copy your JSONL from Drive into data/byzantine/ after cloning code
```

### Install deps
```bash
!pip install -q -r requirements.txt
!pip install -q -r requirements-train.txt
!pip install -q unsloth        # CUDA-only fast QLoRA path; auto-used when CUDA present
```
Note: the script auto-selects **Unsloth on CUDA** (2× faster QLoRA, 4-bit). It falls back
to PEFT if `unsloth` isn't installed. Do NOT pass `--force-peft` on Colab — you want Unsloth.

---

## 3. Training commands (on Colab)

The script (`scripts/train_byzantine_sft.py`) takes `--epochs` for real runs (use this, not
`--max-steps` which is for smoke tests). Key args: `--data --out --epochs --batch-size
--grad-accum --lr --seq-length`.

**Recommended first run — the byz→west directional adapter (cleanest result to date):**
```bash
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_n2w_train_sub.jsonl \
  --model Qwen/Qwen3-1.7B \
  --out models/byzantine_sft_n2w_colab \
  --epochs 3 --batch-size 8 --grad-accum 2 --lr 2e-4
```
On a T4 this is ~1,510 rows × 3 epochs ≈ a few minutes with Unsloth. Bump `--batch-size`
to 16 on L4/A100; drop to 4 if you hit OOM.

**Full bidirectional run:**
```bash
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_translation_train.jsonl \
  --out models/byzantine_sft_translation_colab \
  --epochs 3 --batch-size 8 --grad-accum 2
```

**Genus experiment (only if you want to test microtonal-intent tagging):**
```bash
!python scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_translation_train_genus.jsonl \
  --out models/byzantine_sft_genus_colab --epochs 3 --batch-size 8
```

---

## 3b. Base-model bake-off (which model to use)

The task is ~90% **format discipline + structured-symbol seq2seq** (pipe-delimited neume
tokens ↔ pitch tokens with a fixed `Mode/Ison` header) and needs almost no world knowledge
or reasoning. So the base model should be strong at *emitting an exact structured sequence
and stopping* — NOT a reasoning model. Recommended target size: **7–8B** (1.5B you can run
locally; the point of Colab is to go bigger). QLoRA 4-bit fits all of these on a T4 (16 GB);
raise `--batch-size` on L4/A100.

### Candidates (train the same dataset on each, then compare)

| Rank | Model | Why it fits this task | Watch-out |
|------|-------|-----------------------|-----------|
| 1 | **Qwen2.5-Coder-7B** | Code training = strongest "emit exactly this structured sequence and stop" prior — directly targets our format failures. Qwen tokenizer handles `F#4`/`A-4`/`oligon_kentema` efficiently. | slightly terse (irrelevant after fine-tune) |
| 2 | **Qwen2.5-7B-Instruct** | Same tokenizer strengths, best-in-class instruction following, **no `<think>` mode**, hyperparameters already known from prior runs. Lowest-risk. | marginally less symbolic bias than Coder |
| 3 | **Llama-3.1-8B-Instruct** | Excellent format adherence; different tokenizer = the hedge if Qwen tokenizes the notation poorly. Huge Unsloth support. | re-tune LR; 8B vs 7B (still T4-4bit OK) |

**Avoid:** Qwen3-8B or any "thinking" model — the `<think>` runaway is exactly what tanked
the w2n adapter. Gemma-2-9B — tokenizer most different from our character set, heavier, no
task-specific edge.

### Bake-off commands (same data + hyperparams, only `--model` changes)
Use the clean byz→west directional set for a fast, comparable run:
```bash
for M in "Qwen/Qwen2.5-Coder-7B" "Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct"; do
  OUT="models/bakeoff_$(echo $M | tr '/' '_')"
  python scripts/train_byzantine_sft.py \
    --data data/byzantine/sft_n2w_train_sub.jsonl \
    --model "$M" --out "$OUT" \
    --epochs 3 --batch-size 8 --grad-accum 2 --lr 2e-4
done
```
Notes:
- Llama-3.1 is gated on HF — accept the license and set `HF_TOKEN` in Colab first
  (`from huggingface_hub import login; login()`), or it 401s on download.
- VRAM/batch: T4 → `--batch-size 8` (drop to 4 if OOM on 8B); L4 → 16; A100 → 16–32.
- Each 7–8B QLoRA on ~1,510 rows × 3 epochs is a few minutes with Unsloth.

### How to pick the winner
Eval all three (see §6) on the golden set (`docs/byzantine_golden_set.md`) + held-out, and
compare on the metrics that actually respond to training:
**notation_convention, mode_fidelity, meaning_preservation, and `<think>`-failure rate.**
Do NOT compare on `melodic_equivalence` — it stays ~0 for every model (alignment wall, §5).

---

## 4. After training — get the adapter OUT of Colab

`models/` is gitignored and Colab is ephemeral, so **download or push the adapter yourself**:
```python
# Option A: zip + download
!cd models && zip -r n2w_colab.zip byzantine_sft_n2w_colab
from google.colab import files; files.download('models/n2w_colab.zip')

# Option B: copy to mounted Drive
!cp -r models/byzantine_sft_n2w_colab /content/drive/MyDrive/
```
The adapter is small (~35 MB LoRA safetensors), so download is quick.

---

## 5. Known issues to carry into the run (don't relearn the hard way)

- **`<think>` runaway.** Qwen3 emits a `<think>` block; the base model and the failed
  `w2n` adapter looped in it forever and never answered. **Before/while training w2n,
  suppress thinking** (add `/no_think` to the system prompt, or strip `<think>…</think>`
  from targets, or set a stop sequence at generation). This is the #1 fix for the west→byz
  direction, which totally failed last run.
- **Length over-generation.** The n2w adapter over-produced (20+ notes for a 4-note ref).
  If it recurs, lower `--seq-length` toward the true target length or add an EOS-discipline
  pass. Windowed targets are short (median ~22 tokens).
- **Don't trust `sft_byzantine_all_*` for eval.** That combined file has a train/heldout
  hymn-leak bug and stale pre-header format (see audit in the plan file). Use the
  `sft_translation_*` / directional files, which are clean.
- **melodic_equivalence will stay ~0.** This is the fundamental alignment wall (neumes↔notes
  ≈1.78:1, not 1:1). Colab/GPU changes speed, not this. Judge success on notation_convention,
  mode_fidelity, and meaning_preservation — not exact pitch. See
  `docs/byzantine_handoff_20260709.md`.

---

## 6. Eval after Colab (back on local, or on Colab)

Generation is judge-free; grading uses the rubric (API judge is billing-blocked, so grade
by agent as before):
```bash
python scripts/gen_base_vs_tuned_outputs.py \
  --adapter-path models/byzantine_sft_n2w_colab \
  --suites heldout,unseen,ultra_hard \
  --out runs/n2w_colab_outputs.json
```
Then grade `runs/n2w_colab_outputs.json` targets vs `reference_output` on the 0–2 rubric in
`goals/byzantine_transcription.yaml`. Compare against the v2/v3 numbers in
`docs/byzantine_day3_results_20260708.md`.

---

## 7. Quick pre-flight checklist

- [ ] Training JSONL committed + pushed (`git status` clean for `data/byzantine/*.jsonl`)
- [ ] Scripts + `requirements*.txt` pushed
- [ ] Decided which dataset(s) to train
- [ ] Colab runtime set to GPU (`!nvidia-smi` shows a card)
- [ ] `unsloth` installed on Colab (for the fast path)
- [ ] Plan to download the adapter out (models/ won't persist)
- [ ] `<think>` suppression decided for the w2n run
- [ ] Eval expectations set: judge contour/mode, not exact pitch
