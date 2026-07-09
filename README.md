# From-Scratch-LLM

Working through the "Train Your Own Small Learning Model" assignment, now focused on a narrow music-notation behavior:

> Transcribe between Byzantine/Chrysanthine neumatic notation and Western staff notation while preserving melodic contour, mode, martyria, ison, microtonal intent, and rhythmic modifiers.

Base local model: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B).

## Current status

The repo contains a working eval harness, Byzantine behavior spec, prompt banks, corpus discovery scripts, a deterministic data pipeline (Audiveris OMR for Western pitches + EZ/ED font and vector decoding for Byzantine neumes), the resulting SFT task datasets, and local LoRA adapters.

The core result so far: frontier prompting can improve notation formatting and memorized liturgical formulas, but it still fails on melodic equivalence for unseen and adversarial Byzantine transcription cases. That keeps supervised fine-tuning justified for this behavior.

## Repository map

| Path | Purpose |
|------|---------|
| [`eval_harness/`](eval_harness/) | CLI for litmus runs, single evals, scenario generation, and local model comparisons |
| [`goals/`](goals/) | Behavior specs, rubrics, pass thresholds, and scenario paths |
| [`scenarios/`](scenarios/) | Dev, held-out, final-dev, break, ultra-hard, and unseen eval banks |
| [`prompts/`](prompts/) | Byzantine transcription system prompt versions |
| [`config/`](config/) | Model, judge, threshold, and generation settings |
| [`scripts/`](scripts/) | Corpus discovery, extraction, pruning, SFT data generation, training, and sweep utilities |
| [`data/byzantine/`](data/byzantine/) | Corpus manifests, OMR pitch data, extracted neume sequences, SFT task JSONL files, and local rendered score assets |
| `runs/` | Local eval outputs and sweep summaries; generated artifacts may be untracked |
| `models/` | Local LoRA adapters; generated artifacts may be untracked |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For training workflows:

```bash
pip install -r requirements-train.txt
```

Create a `.env` file for API-backed evals:

```bash
OPENAI_API_KEY=sk-...
# Optional: only needed for Anthropic/Opus API judging.
ANTHROPIC_API_KEY=sk-ant-...
```

If `ANTHROPIC_API_KEY` is not set, see [`docs/byzantine_opus_blind_eval.md`](docs/byzantine_opus_blind_eval.md) and [`docs/byzantine_opus_sweep.md`](docs/byzantine_opus_sweep.md) for the Cursor-agent Opus workflows used in the existing reports.

## Quick workflows

### Local inference

Model weights download from Hugging Face on first run.

```bash
python3 run_inference.py
python3 run_inference.py "Explain gradient descent in one sentence."
python3 run_inference.py --chat
```

### Prompt litmus harness

The harness asks: can a well-prompted frontier model already do the target behavior reliably? If held-out performance stays below the training threshold, the behavior is worth fine-tuning.

```bash
# Optimize prompt on dev set, then run the held-out litmus verdict.
python3 -m eval_harness --config config/byzantine_eval.yaml litmus \
  --goal goals/byzantine_transcription.yaml

# Run one eval round without prompt editing.
python3 -m eval_harness --config config/byzantine_eval.yaml eval \
  --goal goals/byzantine_transcription.yaml \
  --prompt-file prompts/byzantine_transcription_v2.txt \
  --backend openai --verbose

# Compare the local base model on held-out scenarios.
python3 -m eval_harness --config config/byzantine_eval.yaml compare \
  --goal goals/byzantine_transcription.yaml \
  --prompt-file prompts/byzantine_transcription_v2.txt \
  --split heldout

# Compare a local LoRA adapter against the base model.
python3 -m eval_harness --config config/byzantine_eval.yaml compare \
  --goal goals/byzantine_transcription.yaml \
  --prompt-file prompts/byzantine_transcription_v2.txt \
  --split heldout \
  --adapter-path models/byzantine_sft_v1 \
  --compare-base

# Generate more scenarios from the behavior spec.
python3 -m eval_harness --config config/byzantine_eval.yaml generate-scenarios \
  --goal goals/byzantine_transcription.yaml \
  --count 10 \
  --split dev
```

Use [`config/byzantine.yaml`](config/byzantine.yaml) for Anthropic/Opus API judging when `ANTHROPIC_API_KEY` is available. Use [`config/byzantine_eval.yaml`](config/byzantine_eval.yaml) for OpenAI-backed smoke evals.

### Corpus and SFT data (deterministic pipeline)

The current pipeline builds training data deterministically — no vision-model guessing.
See [`docs/byzantine_omr_western_data.md`](docs/byzantine_omr_western_data.md) for details.

```bash
# 1. Discover parallel Byzantine/Western PDF pairs from all configured sources.
python3 scripts/scrape_all_byzantine_sources.py discover

# 2. Western side: OMR the staff-notation PDFs to exact pitches (Audiveris + music21).
python3 scripts/omr_extract_western.py \
  --glob 'data/byzantine/corpus/goa-dcs/*_west.pdf' \
  --out data/byzantine/omr/omr_goa.jsonl --workers 8
#    (repeat for new-byzantium -> omr_newbyz.jsonl, st-anthonys -> omr_sam.jsonl)

# 3. Byzantine side: extract named neumes from EZ/ED music fonts, then recover
#    vector-drawn neume PDFs that have no font text.
python3 scripts/extract_neumes.py \
  --glob 'data/byzantine/corpus/new-byzantium/*_byz.pdf' \
  --out data/byzantine/neumes_new-byzantium.jsonl
python3 scripts/extract_neumes_vector.py \
  --out data/byzantine/neumes_vector.jsonl --names /tmp/named_ref_hashes.json

# 4. Build SFT task JSONL (Western tasks + neume tasks) from the extracted data.
python3 scripts/build_western_tasks.py --out data/byzantine/sft_western.jsonl
python3 scripts/build_neume_tasks.py   --out data/byzantine/sft_neume.jsonl
#    Combine + dedup + split into sft_byzantine_all_{train,heldout}.jsonl.
```

### Training

```bash
# Train a local LoRA adapter. Uses PEFT on MPS/CPU and Unsloth on CUDA unless --force-peft is set.
python3 scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_byzantine_all_train.jsonl \
  --out models/byzantine_sft_v1

# Evaluate the adapter against the base model.
python3 -m eval_harness --config config/byzantine_eval.yaml compare \
  --goal goals/byzantine_transcription.yaml \
  --prompt-file prompts/byzantine_transcription_v2.txt \
  --split heldout \
  --adapter-path models/byzantine_sft_v1 \
  --compare-base
```

### Full sweeps

Broader multi-bank frontier-model sweeps live in scripts, not the harness CLI:

```bash
python3 scripts/run_byzantine_full_sweep.py \
  --provider openai \
  --model gpt-4o \
  --prompt v2 \
  --suite all \
  --judge
```

## Results at a glance

Strict pass requires `melodic_equivalence >= 1.5`, `meaning_preservation >= 1.5`, and passing rule checks from [`goals/byzantine_transcription.yaml`](goals/byzantine_transcription.yaml).

| Translator / judge | Prompt | `final_dev` strict | `unseen` strict | Takeaway |
|--------------------|--------|--------------------|-----------------|----------|
| GPT-4o / GPT-4o | v2 | 11/36 | 0/10 | Better rule compliance and formatting, but no unseen generalization |
| Claude Opus 4 / Claude Opus 4 | v2 | 11/36 | 0/10 | Same strict-pass pattern; melody remains the bottleneck |

Across the documented sweeps, models usually produce plausible notation shape, mode labels, and ison lines. They still miss pitch sequences, mode-specific anchors, leading ison neumes, and microtonal spellings often enough that the SFT verdict is unchanged.

## Current data snapshot

The training data was rebuilt on deterministic extraction (Audiveris OMR for Western
pitches + EZ/ED font and vector decoding for Byzantine neumes), replacing the earlier
vision-extracted rows. See [`docs/byzantine_omr_western_data.md`](docs/byzantine_omr_western_data.md)
for the full pipeline and a per-file data index.

**Primary training data (use these):**

| Artifact | Current local state |
|----------|---------------------|
| [`data/byzantine/sft_byzantine_all_train.jsonl`](data/byzantine/sft_byzantine_all_train.jsonl) | 4,823 rows — combined Western + neume tasks (headline SFT set) |
| [`data/byzantine/sft_byzantine_all_heldout.jsonl`](data/byzantine/sft_byzantine_all_heldout.jsonl) | 536 rows — held-out split (no train leakage) |
| [`data/byzantine/sft_western.jsonl`](data/byzantine/sft_western.jsonl) | 4,294 Western-only task rows (`mode_id`, `continuation`, `contour`, `transpose`) |
| [`data/byzantine/sft_neume.jsonl`](data/byzantine/sft_neume.jsonl) | 2,247 neume task rows: `neume_read`, `mode_from_neumes`, and **bidirectional translation** `neume_to_west` (581) + `west_to_neume` (581) = **1,162 translation examples** |

**Intermediate / source data:**

| Artifact | Current local state |
|----------|---------------------|
| `data/byzantine/omr/omr_{goa,newbyz,sam}.jsonl` | Deterministic OMR Western pitches (Audiveris + music21) |
| `data/byzantine/neumes_{goa-dcs,new-byzantium,st-anthonys}.jsonl` | Font-extracted named neume sequences |
| [`data/byzantine/neumes_vector.jsonl`](data/byzantine/neumes_vector.jsonl) | 89 vector-recovered neume files (+86 bidirectional) |
| [`data/byzantine/ez_neume_map.json`](data/byzantine/ez_neume_map.json) | EZ/ED ASCII → neume-name map (from official EZ character tables) |
| [`data/byzantine/manifest.jsonl`](data/byzantine/manifest.jsonl) | 2,292 discovered parallel PDF pairs |
| `models/byzantine_sft_v1/`, `models/byzantine_sft_smoke/` | Local LoRA adapters (trained on earlier data) |

**Archived / superseded** (kept for provenance — do **not** train on these): the
vision-era `data/byzantine/sft_raw.jsonl` (4,543), `sft_raw.backup.jsonl` (9,936 original),
`sft_raw_rejected.jsonl` (5,393 quarantined dup/garbage), and earlier `sft_v1.jsonl` (98) /
`sft_v2.jsonl` (4,005). Rationale in the report below.

Keep these eval banks out of training data:

- [`scenarios/byzantine_transcription_heldout.yaml`](scenarios/byzantine_transcription_heldout.yaml)
- [`scenarios/byzantine_transcription_unseen.yaml`](scenarios/byzantine_transcription_unseen.yaml)
- [`scenarios/byzantine_transcription_ultra_hard.yaml`](scenarios/byzantine_transcription_ultra_hard.yaml)

## Reports and handoff docs

- [`docs/byzantine_omr_western_data.md`](docs/byzantine_omr_western_data.md) — **current** deterministic data pipeline (OMR + neume extraction), per-file data index, and why the vision-era data was replaced
- [`docs/byzantine_day2_litmus_report.md`](docs/byzantine_day2_litmus_report.md) — behavior spec, eval design, Day 2 litmus verdict, and educational framing
- [`docs/byzantine_day3_corpus.md`](docs/byzantine_day3_corpus.md) — historical vision-extraction corpus flow (superseded; see the OMR doc above)
- [`docs/byzantine_day3_results_20260708.md`](docs/byzantine_day3_results_20260708.md) — **Day 3 midweek gate**: first real SFT run (Qwen3-1.7B, 897 translation rows) and base-vs-tuned numbers with error analysis
- [`docs/byzantine_gpt4o_sweep.md`](docs/byzantine_gpt4o_sweep.md) — GPT-4o full translator sweep
- [`docs/byzantine_opus_sweep.md`](docs/byzantine_opus_sweep.md) — Claude Opus full translator sweep and Opus judge workflow
- [`docs/byzantine_opus_blind_eval.md`](docs/byzantine_opus_blind_eval.md) — blind translator instructions for Opus-agent evals

## Verdict meanings

| Verdict | Meaning |
|---------|---------|
| **PASS** | Frontier prompting cannot reliably hit threshold after prompt edits; proceed to SFT |
| **FAIL** | Frontier prompting reaches high adherence; pick a harder behavior |
| **BORDERLINE** | Scores land in the middle band; tighten the spec or add adversarial scenarios |
