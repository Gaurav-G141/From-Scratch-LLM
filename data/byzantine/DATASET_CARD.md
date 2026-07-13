---
license: apache-2.0
task_categories:
  - translation
  - text-generation
language:
  - en
tags:
  - byzantine-chant
  - music-notation
  - neume
  - transcription
  - synthetic
  - correct-by-construction
pretty_name: Byzantine Synthetic Interval-Grammar (Neume ↔ Western Pitch)
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/sft_synthetic_musicality_train.jsonl
      - split: heldout
        path: data/sft_synthetic_musicality_heldout.jsonl
---

# Byzantine Synthetic Interval-Grammar — Neume ↔ Western Pitch

A **correct-by-construction** supervised fine-tuning dataset for transcribing between **Byzantine
(Chrysanthine) neume notation** and **Western staff pitches**, in both directions, over a diatonic
interval grammar with ascending leaps up to an octave and rhythmic note durations.

This dataset trains the model
[`FableMogger9000/byzantine-synthetic-grammar-lora`](https://huggingface.co/FableMogger9000/byzantine-synthetic-grammar-lora),
which reaches **96% exact-match / 98% melodic equivalence** on the held-out neume→west direction.

## Why "correct-by-construction"

Byzantine neumes are **intervallic**: each sign is a fixed degree-shift on the modal ladder. Instead
of scanning manuscripts (whose neume and pitch streams are melismatic and do **not** align 1:1), each
melody here is **generated** as a walk over those intervals — so the neume sequence and the pitch
sequence are **1:1 aligned by a mathematical identity**. There is nothing to recover and no labeling
noise: the pairing is exact by definition.

Every row is re-derived by an **independent verifier** that does not import the generator (it
re-declares its own interval table and re-computes each pitch), and only rows that pass are written.

## Contents

| split | rows | neume→west | west→neume |
|---|---|---|---|
| `train` | 23,942 | 11,971 | 11,971 |
| `heldout` | 4,794 | 2,397 | 2,397 |

- **Balanced across the four diatonic modes** (Mode 1, pl. 1, 4, pl. 4) — ~25% each in both splits.
- **Zero leakage:** the held-out split shares **0** exact input prompts with train (verified).
- **Deterministic gold:** correctness is computable directly (no LLM judge); gold-vs-gold scores 100%.

## Format

Chat-style JSONL. Each row:

```json
{
  "id": "synth_000000_t0_n2w",
  "task": "neume_to_west",
  "synthetic": true,
  "messages": [
    {"role": "system", "content": "You are a Byzantine chant notation assistant. ..."},
    {"role": "user", "content": "Transcribe this Byzantine neume sequence (6 neumes) to Western staff pitches:\nMode 1\nIson: D4\noligon_hypsili ison oligon petaste apostrophos petaste"},
    {"role": "assistant", "content": "Mode 1\nIson: D4\nA4 A4 B4 C5 B4 C5"}
  ]
}
```

- **`task`** is `neume_to_west` or `west_to_neume`.
- The **Ison** line is the starting-pitch anchor — neume pitches are *relative*, so absolute pitch is
  only fixed once the ison is known. It is a reference, not the answer.
- Held notes render inline as `<pitch>:<beats>` (e.g. `C4:5`).

## Vocabulary (diatonic, degree-shift)

`ison` 0 · `oligon`/`petaste` +1 · `apostrophos` −1 · `elaphron` −2 · `elaphron_apostrophos` −3 ·
`chamile` −4 · `oligon_kentema` +3 · `oligon_hypsili` +4 · `ypsili_left_oligon` +5 ·
`ypsili_kentima_oligon` +6 · `ypsili_over_kentima_oligon` +7 (octave). Durations: `apli` (2 beats) ·
`dipli` (3) · `tetrapli` (5). Breath/barline signs are no-ops on pitch (neume→west only).

## Scope and limitations

- **Diatonic only.** Microtones, chromatic/enharmonic modes, fthora, and melisma are **excluded by
  design** — they cannot be faked deterministically and would break the correct-by-construction
  guarantee.
- **Not real manuscripts.** This is synthetic grammar data, deliberately chosen because the available
  real neume↔pitch corpus does not correspond 1:1. It teaches the interval grammar, not real
  melismatic chant transcription.
- **west→neume is intrinsically ambiguous** — multiple neumes encode the same pitch motion (e.g.
  `oligon` and `petaste` are both +1), so exact-match is capped in that direction; judge it by
  positional pitch accuracy.

## Reproduce

Generated and checked by two scripts in the
[project repo](https://github.com/Gaurav-G141/From-Scratch-LLM):

```bash
python scripts/build_synthetic_musicality.py  --n 3000 --out data/byzantine/sft_synthetic_musicality.jsonl
python scripts/verify_synthetic_musicality.py data/byzantine/sft_synthetic_musicality.jsonl   # 0 errors
```

## License

Apache-2.0.
