---
base_model: unsloth/Qwen2.5-Coder-7B-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - byzantine-chant
  - music-notation
  - neume
  - transcription
  - qwen2.5
language:
  - en
license: apache-2.0
---

# Byzantine Synthetic-Grammar LoRA — Neume ↔ Western Pitch Transcription

A LoRA adapter for `unsloth/Qwen2.5-Coder-7B-bnb-4bit` that transcribes between **Byzantine
(Chrysanthine) neume notation** and **Western staff pitches** for a diatonic interval grammar —
including ascending leaps up to an octave and rhythmic note durations, in both directions.

It is a small fine-tuned model that reliably performs a specialized notation task that
frontier-model prompting did **not**: on a held-out, zero-leakage test set the neume→west
direction reaches **96% exact-match / 98% melodic equivalence**.

> **This adapter was trained on synthetic, correct-by-construction data**, not on scanned
> manuscripts. Read the Scope section before using it — it does not transcribe real melismatic
> chant, and the reverse direction is limited by an inherent ambiguity in the notation.

---

## TL;DR

| | neume → west | west → neume |
|---|---|---|
| **exact_match** | **96.0%** | 13.7% |
| **pitch_accuracy** | **99.1%** | 76.5% |
| **interval_accuracy** | **99.2%** | n/a (output is neumes) |
| **melodic_equivalence** | **97.8%** (1.955 / 2.0) | 49.0% (0.981 / 2.0) |
| **strict_pass_rate** | **96.2%** | 16.9% |

Held-out: 4,794 sequences, zero leakage, deterministic scoring (no LLM judge). All metrics shown
as percentages; melodic-equivalence is a 0–2 rubric reported as % of its 2.0 maximum.

---

## What it does well

**neume → west transcription (the headline capability).** Given a neume sequence and its ison
(starting-pitch) anchor, the model produces the exact Western pitch sequence 96% of the time,
correctly handling:

- the diatonic interval vocabulary — unison, steps, and **ascending leaps up to an octave**
  (`oligon_kentema` +3, `oligon_hypsili` +4, and the ypsili combinations up to +7);
- **rhythmic durations** — `apli` / `dipli` / `tetrapli` produce held notes, rendered inline as
  `<pitch>:<beats>` (e.g. `C4:5`);
- the correct mode header and ison echo;
- **sequence length** — it stops at the right place rather than running on (a ChatML `<|im_end|>`
  stop-token was used at inference).

## What it does NOT do well (and why)

- **west → neume exact-match is low (14%) — a notation ceiling, not a model failure.** Several
  neumes encode the same pitch motion (e.g. `oligon` and `petaste` are both "+1 step"), so a
  rising step is genuinely two-valued and cannot be uniquely recovered from pitches alone. Judge
  this direction by **positional pitch accuracy (76%)**, not exact match.
- **Real scanned manuscript chant is out of scope.** The available real neume↔pitch corpus was
  found to be non-recoverable to exact pitch (the paired neume and pitch streams did not
  correspond; ~10% positional ceiling for any model), which is *why* this adapter was trained on
  synthetic data. It has **not** been shown to transcribe real melismatic chant.
- **Microtones, chromatic/enharmonic modes, fthora, and melisma are excluded by design.** Only
  the four **diatonic** modes (1, pl.1, 4, pl.4) on a natural-note ladder are modeled; the model
  asserts no accidental or microtonal content.

## Why the results are trustworthy

- **Zero leakage:** the 4,794-row test set shares no row-id and no exact input prompt with
  training (both verified = 0).
- **Deterministic scoring, no LLM judge:** the synthetic gold is exact by construction, so
  correctness is computed directly (exact match, per-position pitch accuracy, interval accuracy,
  edit distance, and a 0–2 melodic composite). Gold-vs-gold scores a perfect 100%, confirming the
  scorer measures real correctness.
- **Correct-by-construction data:** every training/eval pair is generated from a vetted interval
  grammar and re-derived by an independent verifier (0 content + 0 reversibility errors).

## Headline finding

The **same class of model** scored **~10% positional accuracy on the real corpus** and **96%
exact on correct-by-construction data**. That contrast isolates the earlier failure as a
**data-labeling problem, not a model-capability problem**: when the neume↔pitch pairs actually
correspond, a small fine-tuned model learns the grammar almost perfectly.

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "YOUR_USERNAME/byzantine-synthetic-grammar-lora")

prompt = ("Transcribe this Byzantine neume sequence (6 neumes) to Western staff pitches:\n"
          "Mode 1\nIson: D4\noligon_kentema petaste elaphron petaste ison oligon_hypsili")
msgs = [{"role": "system", "content": "You are a Byzantine chant notation assistant."},
        {"role": "user", "content": prompt}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=96,
                     eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
# -> Mode 1 / Ison: D4 / G4 A4 F4 G4 G4 D5   (example shape)
```

**Input format.** Line 1: an instruction naming the token count. Line 2: `Mode X`. Line 3:
`Ison: <pitch>` (the starting-pitch anchor — pitches are relative, so this is required). Line 4:
the space-separated neume (or, for west→neume, pitch) tokens.

## Vocabulary (diatonic, degree-shift)

`ison` 0 · `oligon`/`petaste` +1 · `apostrophos` −1 · `elaphron` −2 · `elaphron_apostrophos` −3 ·
`chamile` −4 · `oligon_kentema` +3 · `oligon_hypsili` +4 · `ypsili_left_oligon` +5 ·
`ypsili_kentima_oligon` +6 · `ypsili_over_kentima_oligon` +7 (octave). Duration signs:
`apli` (2 beats) · `dipli` (3) · `tetrapli` (5). Breath/barline signs are no-ops on pitch
(neume→west only).

## Training

- **Base:** `unsloth/Qwen2.5-Coder-7B-bnb-4bit` (4-bit), LoRA via PEFT, response-only loss.
- **Data:** 23,942 rows of verifier-checked synthetic grammar (both directions, 4 diatonic
  modes, leaps + durations); 4,794-row disjoint held-out slice (0 id/content leakage).
- **Objective:** exact bidirectional transcription of the interval grammar; the task is small,
  finite-vocabulary interval arithmetic anchored to a stated ison.

## Intended use & limitations

Educational / research demonstration that supervised fine-tuning teaches a small model a
notation grammar that prompting a frontier model did not reliably perform. **Not** a production
transcription tool for real manuscripts, and not a source of musicological ground truth for
microtonal or chromatic repertoire.

## License

Apache-2.0 (adapter). The base model retains its own license.
