# From-Scratch-LLM

Working through the "Train Your Own Small Learning Model" assignment.

## Status

- Day 1 (first part): base model runs and responds via [`run_inference.py`](run_inference.py)
- Day 2 (litmus harness): prompt optimization + eval loop via [`eval_harness/`](eval_harness/)

Base model: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file for the eval harness (OpenAI API):

```bash
OPENAI_API_KEY=sk-...
```

Model weights for local inference download from Hugging Face on first run.

## Run local inference

```bash
python run_inference.py
python run_inference.py "Explain gradient descent in one sentence."
python run_inference.py --chat
```

## Prompt litmus harness

The harness answers: **can a well-prompted frontier model already do your target behavior reliably?**

If prompt optimization plateaus below the reliability bar, the behavior is worth fine-tuning. If it hits the ceiling, pick a harder behavior.

### Commands

```bash
# Main litmus run: optimize prompt on dev set, verdict on held-out
python -m eval_harness litmus --goal goals/tutor.yaml

# Single eval round (no prompt editing)
python -m eval_harness eval --goal goals/tutor.yaml --prompt-file prompts/tutor_v0.txt

# Compare local base model against a saved prompt
python -m eval_harness compare --goal goals/tutor.yaml --prompt-file runs/<run>/best_prompt.txt

# Generate more scenarios from a behavior spec
python -m eval_harness generate-scenarios --goal goals/tutor.yaml --count 10 --split dev
```

For `eval`, pass a plain-text system prompt file (e.g. copy `initial_system_prompt` into `prompts/tutor_v0.txt`).

### Behavior goals

Goals live in [`goals/`](goals/). Each goal defines:

- A falsifiable behavior spec
- Initial system prompt
- Optional forbidden/required regex patterns (and JSON schema for structured-output goals)
- Dev and held-out scenario file paths

Example goals:

- [`goals/tutor.yaml`](goals/tutor.yaml) — Socratic tutor that never gives the answer
- [`goals/structured_output.yaml`](goals/structured_output.yaml) — strict JSON-only output
- [`goals/connections.yaml`](goals/connections.yaml) — NYT Connections-style word grouping (16 → 4×4)
- [`goals/sanitization.yaml`](goals/sanitization.yaml) — rewrite rude text to be school-appropriate while preserving meaning
- [`goals/latex_transcription.yaml`](goals/latex_transcription.yaml) — word problems → LaTeX, preserving student errors (test prompts only)

Scenarios live in [`scenarios/`](scenarios/). Dev scenarios drive prompt editing; held-out scenarios determine the litmus verdict.

For Connections puzzles, scenarios include `words` and `expected_groups` so the harness reports objective **groups correct** and **puzzles solved** scores alongside the LLM judge.

### Config

Thresholds and model names are in [`config/default.yaml`](config/default.yaml):

- `success_threshold` (default 1.85): held-out spec adherence at or above → **FAIL** litmus (behavior is promptable)
- `train_threshold` (default 1.2): held-out spec adherence at or below → **PASS** litmus (worth fine-tuning)
- `max_iterations`, `patience`, `min_delta`: prompt optimization stop conditions

### Reports

Each litmus run writes to `runs/<goal_name>_<timestamp>/`:

- `summary.md` — round-by-round scores and verdict
- `best_prompt.txt` — highest-scoring dev prompt
- `heldout_eval.json` — final held-out evaluation
- `rounds/` — per-round prompts and scores

### Verdict meanings

| Verdict | Meaning |
|---------|---------|
| **PASS** | Frontier model cannot reliably hit threshold even after prompt edits → proceed to SFT |
| **FAIL** | Frontier model reaches high adherence with optimized prompt → pick a harder behavior |
| **BORDERLINE** | Scores in the middle band → tighten spec or add adversarial scenarios |

### Byzantine notation transcription

Transcribe between **Byzantine (Chrysanthine) neumes** and **Western staff notation** while preserving mode, ison, and microtonal intent. Uses **Claude Opus** as judge (`config/byzantine.yaml`).

```bash
# 1. Discover parallel PDF corpus (Cappella Romana + New Byzantium)
pip install requests beautifulsoup4 anthropic
python scripts/scrape_byzantine_corpus.py discover
python scripts/scrape_byzantine_corpus.py discover --download   # optional PDFs

# 2. Eval with Opus judge (set ANTHROPIC_API_KEY in .env)
python -m eval_harness eval \
  --config config/byzantine.yaml \
  --goal goals/byzantine_transcription.yaml \
  --prompt-file prompts/byzantine_transcription_v0.txt \
  --backend openai --verbose

# 3. Compare local SLM on held-out set
python -m eval_harness compare \
  --config config/byzantine.yaml \
  --goal goals/byzantine_transcription.yaml \
  --prompt-file prompts/byzantine_transcription_v0.txt \
  --split heldout
```

**Corpus:** `data/byzantine/manifest.jsonl` — paired PDF URLs from liturgical sources.  
**Text eval:** `scenarios/byzantine_transcription_{dev,heldout}.yaml` — hand-crafted neume sequences with gold references for the judge.  
**Litmus hypothesis:** If GPT-4o + v0 prompt scores high on microtonal/chromatic cases, Byzantine may be too Western-isomorphic for SLM fine-tuning (per your knowledge tree). The harness tests that empirically.
