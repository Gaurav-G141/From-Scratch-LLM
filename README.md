# From-Scratch-LLM

Working through the "Train Your Own Small Learning Model" assignment, now focused on a narrow music-notation behavior:

> Transcribe between Byzantine/Chrysanthine neumatic notation and Western staff notation while preserving melodic contour, mode, martyria, ison, microtonal intent, and rhythmic modifiers.

Base local model: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B).

## Current status

The repo contains a working eval harness, Byzantine behavior spec, prompt banks, corpus discovery/extraction scripts, SFT data rows, and local LoRA adapters.

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
| [`data/byzantine/`](data/byzantine/) | Corpus manifests, extracted rows, SFT JSONL files, and local rendered score assets |
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

### Corpus and SFT data

```bash
# Discover parallel Byzantine/Western PDF pairs from all configured sources.
python3 scripts/scrape_all_byzantine_sources.py discover
python3 scripts/scrape_all_byzantine_sources.py stats

# Vision-extract chat-format training rows from rendered PDF pages.
python3 scripts/extract_byzantine_training_data.py --download --resume \
  --max-pages 1 \
  --fragments-per-page 2

# Prune extracted rows with local rules only, or add --judge for OpenAI scoring.
python3 scripts/prune_byzantine_corpus.py --rules-only

# Build SFT JSONL from accepted corpus rows.
python3 scripts/generate_byzantine_sft_data.py \
  --from-corpus data/byzantine/sft_raw.jsonl \
  --corpus-status accepted \
  --out data/byzantine/sft_v1.jsonl
```

### Training

```bash
# Train a local LoRA adapter. Uses PEFT on MPS/CPU and Unsloth on CUDA unless --force-peft is set.
python3 scripts/train_byzantine_sft.py \
  --data data/byzantine/sft_v1.jsonl \
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

| Artifact | Current local state |
|----------|---------------------|
| [`data/byzantine/manifest.jsonl`](data/byzantine/manifest.jsonl) | 2,120 discovered parallel PDF pairs |
| Source counts | GOA DCS 1,825; New Byzantium 277; Cappella Romana 16; St. Anthony's 2 |
| [`data/byzantine/sft_raw.jsonl`](data/byzantine/sft_raw.jsonl) | 108 extracted chat-format rows |
| [`data/byzantine/sft_v1.jsonl`](data/byzantine/sft_v1.jsonl) | 98 accepted training rows |
| `models/byzantine_sft_smoke/` | Local smoke LoRA adapter |
| `models/byzantine_sft_v1/` | Local v1 corpus LoRA adapter |

Keep these eval banks out of training data:

- [`scenarios/byzantine_transcription_heldout.yaml`](scenarios/byzantine_transcription_heldout.yaml)
- [`scenarios/byzantine_transcription_unseen.yaml`](scenarios/byzantine_transcription_unseen.yaml)
- [`scenarios/byzantine_transcription_ultra_hard.yaml`](scenarios/byzantine_transcription_ultra_hard.yaml)

## Reports and handoff docs

- [`docs/byzantine_day2_litmus_report.md`](docs/byzantine_day2_litmus_report.md) — behavior spec, eval design, Day 2 litmus verdict, and educational framing
- [`docs/byzantine_day3_corpus.md`](docs/byzantine_day3_corpus.md) — real corpus sources, extraction flow, pruning, and holdout guidance
- [`docs/byzantine_gpt4o_sweep.md`](docs/byzantine_gpt4o_sweep.md) — GPT-4o full translator sweep
- [`docs/byzantine_opus_sweep.md`](docs/byzantine_opus_sweep.md) — Claude Opus full translator sweep and Opus judge workflow
- [`docs/byzantine_opus_blind_eval.md`](docs/byzantine_opus_blind_eval.md) — blind translator instructions for Opus-agent evals

## Verdict meanings

| Verdict | Meaning |
|---------|---------|
| **PASS** | Frontier prompting cannot reliably hit threshold after prompt edits; proceed to SFT |
| **FAIL** | Frontier prompting reaches high adherence; pick a harder behavior |
| **BORDERLINE** | Scores land in the middle band; tighten the spec or add adversarial scenarios |
