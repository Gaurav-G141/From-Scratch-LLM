#!/usr/bin/env python3
"""Publish the Byzantine synthetic-grammar LoRA adapter to the HuggingFace Hub.

Adapter-only upload: pure file transfer, NO GPU and NO base-model download needed.
The adapter is a small (~tens of MB) LoRA diff; anyone loads it on top of the base
`unsloth/Qwen2.5-Coder-7B-bnb-4bit` to reproduce the eval behavior.

Auth (either one):
  export HF_TOKEN=hf_...            # a WRITE-scope token from hf.co/settings/tokens
  # or pass --token hf_...
  # or run `huggingface-cli login` first

Usage:
  python scripts/push_to_hf.py \
    --adapter ~/Downloads/synth_expanded_adapter \
    --repo YOUR_USERNAME/byzantine-synthetic-grammar-lora
  # add --private for a private repo; --dry-run to validate without uploading.

What it does:
  1. Validates the adapter dir (adapter_config.json + adapter_model.safetensors).
  2. Writes a README.md model card (unless one exists / --no-card).
  3. Creates the repo if missing and uploads the whole folder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_CARD = """---
base_model: unsloth/Qwen2.5-Coder-7B-bnb-4bit
library_name: peft
tags:
  - lora
  - byzantine-chant
  - music-notation
  - qwen2.5
license: apache-2.0
---

# Byzantine Synthetic-Grammar LoRA (neume ↔ Western pitch)

A LoRA adapter for `unsloth/Qwen2.5-Coder-7B-bnb-4bit` that transcribes **Byzantine
(Chrysanthine) neume notation ↔ Western staff pitches** for the diatonic interval grammar,
including ascending leaps up to an octave and rhythmic note durations.

Trained purely on **correct-by-construction synthetic data** (a deterministic, verifier-checked
interval grammar), not on scanned scores. See "Scope" below for exactly what that means.

## Results (held-out, 4,794 rows, zero-leakage, deterministic scoring)

| metric | neume→west | west→neume |
|---|---|---|
| exact_match | **0.960** | 0.137 |
| pitch_accuracy | **0.991** | 0.765 |
| interval_accuracy | **0.992** | n/a |
| melodic_equivalence (0–2) | **1.955** | 0.981 |
| strict_pass_rate | **0.962** | 0.169 |

## What it does well
- **neume→west transcription: near-perfect (0.96 exact, 1.95/2.0 melodic).** Reproduces the
  exact pitch sequence from a neume sequence + ison anchor, including:
  - the full diatonic step/leap vocabulary (unison through octave-range ascending leaps),
  - **rhythmic durations** (`apli`/`dipli`/`tetrapli` → held notes rendered `<pitch>:<beats>`),
  - correct mode header and ison echo.
- Absolute pitch is correctly anchored to the stated ison (transposition-invariant grammar).

## What it does NOT do well (by design / by the notation)
- **west→neume exact-match is low (0.14) — this is a notation ceiling, not a model failure.**
  Multiple neumes encode the same interval (e.g. `oligon` and `petaste` are both +1), so a
  rising step is genuinely two-valued and cannot be uniquely recovered from pitches alone. Judge
  this direction by **pitch_accuracy (0.76)**, i.e. positional correctness, not exact match.
- **Real scanned chant is out of scope.** This adapter was trained on synthetic data because the
  available real neume↔pitch corpus was found to be non-recoverable to exact pitch (the paired
  neume and pitch streams did not correspond; ~10% positional ceiling). It has NOT been shown to
  transcribe real melismatic manuscript chant.
- **Microtones / chromatic & enharmonic modes / fthora / melisma are excluded.** Only the four
  diatonic modes (1, pl.1, 4, pl.4) on a natural-note ladder are modeled. No accidental or
  microtonal intent is asserted.

## Intended use
Educational / research demonstration that supervised fine-tuning teaches a small model a
notation grammar that frontier prompting did not reliably perform. Not a production transcription
tool for real manuscripts.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "%%REPO%%")

prompt = ("Transcribe this Byzantine neume sequence (6 neumes) to Western staff pitches:\\n"
          "Mode 1\\nIson: D4\\noligon_kentema petaste elaphron petaste ison oligon_hypsili")
msgs = [{"role": "system", "content": "You are a Byzantine chant notation assistant."},
        {"role": "user", "content": prompt}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=96, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Training
- Base: `unsloth/Qwen2.5-Coder-7B-bnb-4bit` (4-bit), LoRA (PEFT), response-only loss.
- Data: 23,942 rows of verifier-checked synthetic grammar (both directions, 4 diatonic modes,
  leaps + durations), disjoint 4,794-row held-out slice (0 id/content leakage).
- Eval: deterministic, no LLM judge (synthetic gold is exact).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="path to the LoRA adapter folder")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. username/byzantine-synth-lora")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (or set HF_TOKEN / hf login)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--no-card", action="store_true", help="do not write/overwrite README.md")
    ap.add_argument("--dry-run", action="store_true", help="validate + write card, but do not upload")
    args = ap.parse_args()

    adapter = Path(args.adapter).expanduser().resolve()
    if not adapter.is_dir():
        ap.error(f"adapter dir not found: {adapter}")

    cfg = adapter / "adapter_config.json"
    weights = list(adapter.glob("adapter_model.safetensors")) or list(adapter.glob("adapter_model.bin"))
    if not cfg.is_file():
        ap.error(f"missing adapter_config.json in {adapter} — is this really a LoRA adapter dir?")
    if not weights:
        ap.error(f"missing adapter_model.safetensors/.bin in {adapter} — download may be incomplete")

    # sanity: report base model + size so the user sees what they're publishing
    base = json.loads(cfg.read_text()).get("base_model_name_or_path", "?")
    wsize = weights[0].stat().st_size / 1e6
    print(f"adapter: {adapter}")
    print(f"  base_model_name_or_path: {base}")
    print(f"  weights: {weights[0].name} ({wsize:.1f} MB)")
    print(f"  target repo: {args.repo} ({'private' if args.private else 'public'})")

    # model card
    card = adapter / "README.md"
    if not args.no_card and not card.exists():
        card.write_text(MODEL_CARD.replace("%%REPO%%", args.repo))
        print(f"  wrote model card -> {card}")
    elif card.exists():
        print(f"  README.md already present, leaving as-is")

    if args.dry_run:
        print("dry-run: validated, card written; NOT uploading.")
        return 0

    if not args.token:
        ap.error("no HF token: set HF_TOKEN, pass --token, or run `huggingface-cli login` first")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        ap.error("huggingface_hub not installed: pip install huggingface_hub")

    create_repo(args.repo, token=args.token, private=args.private, exist_ok=True, repo_type="model")
    print(f"  repo ready: {args.repo}")
    api = HfApi()
    api.upload_folder(
        folder_path=str(adapter),
        repo_id=args.repo,
        token=args.token,
        repo_type="model",
        commit_message="Upload Byzantine synthetic-grammar LoRA adapter",
    )
    print(f"DONE → https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
