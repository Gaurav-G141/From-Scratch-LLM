# Byzantine blind translator eval — Opus agent instructions

You are a **translator**, not a judge. Your job is to transcribe each scenario input using only the system prompt and the scenario `input` field. A different agent will grade your work later.

---

## What you are testing

**Behavior goal:** `goals/byzantine_transcription.yaml`

Transcribe between Byzantine (Chrysanthine) neumatic notation and Western staff notation while preserving:

- Melodic contour and note count
- Mode (echos), martyria, ison (drone)
- Microtonal intervals (diesis, fthora) — mark with ↑ ↓, do not silently round to 12-TET
- Rhythm modifiers (gorgon → short, argon → long)

**Do NOT:** add harmony, prose commentary, key signatures like "G major", or markdown fences around output.

---

## System prompt (read in full)

Load and follow **every line** of:

```
prompts/byzantine_transcription_v2.txt
```

That file is your system prompt. Do not use v3 or other prompt versions unless explicitly instructed.

---

## Scenario inputs (blind — no answers)

**Only read:**

```
runs/blind_eval_inputs_ultra_hard_unseen.json
```

Each entry has `id`, `direction`, and `input`. Translate every scenario in order.

### Hard rules — do not cheat the eval

| Do NOT open | Why |
|-------------|-----|
| `reference_output` in any `scenarios/*.yaml` | That is the gold standard |
| `context` fields in scenario YAML | Often hints at the expected contour |
| `runs/byzantine_*_outputs.json` | Prior model runs |
| `runs/byzantine_opus_v2_ultra_hard_outputs.json` | Prior Opus run |
| `runs/byzantine_ultra_hard_eval_summary.json` | Contains pass/fail analysis |

If you already saw reference material in this chat, say so in `honor_code` and restart from blind inputs only.

---

## How to translate each scenario

For each scenario in `blind_eval_inputs_*.json`:

1. Read `direction` (`byz_to_west` or `west_to_byz`).
2. Pass the full `input` string as the **user message**.
3. Apply the system prompt from `prompts/byzantine_transcription_v2.txt`.
4. Reply with **notation only** — no explanation, no markdown code fences.
5. Record your output under that scenario's `id`.

### Output format examples

**Byzantine → Western:**

```
Mode II, Ni = A4
Ison: A4
B4 A4 D5 C5 B4
```

**Western → Byzantine:**

```
[Mode II, Ni=Κε]
(Κε) ison
petastē | oligon | kentēma | apostrophos | oligon
```

---

## How to submit answers

### Step 1 — Copy the template

```
runs/byzantine_opus_blind_submission.template.json
```

### Step 2 — Fill in metadata

| Field | Value |
|-------|--------|
| `translator` | `claude-opus-4 (Cursor agent)` |
| `translator_model_note` | Your exact model ID if known |
| `prompt_file` | `prompts/byzantine_transcription_v2.txt` |
| `bank` | `ultra_hard+unseen` (or whichever bank you ran) |
| `submitted_at` | ISO-8601 timestamp |
| `honor_code` | Confirm you did not read references |

### Step 3 — Add one result per scenario

```json
{
  "id": "unseen_apolytikia_resurrection",
  "model_output": "Mode Plagal IV, Ni = F4\nIson: F4\n..."
}
```

- **`id`** must match the blind input file exactly.
- **`model_output`** is your raw notation string (use `\n` for newlines in JSON).
- Include **all** scenarios — no skips. If you cannot translate one, put `"model_output": "ERROR: <reason>"`.

### Step 4 — Save submission file

Save as:

```
runs/byzantine_opus_blind_submission.json
```

Do not overwrite `runs/byzantine_opus_blind_submission.template.json`.

### Step 5 — (Optional) Merge into harness format

If Python is available:

```bash
cd /path/to/From-Scratch-LLM
PYTHONPATH=. python3 scripts/merge_blind_submission.py \
  --submission runs/byzantine_opus_blind_submission.json \
  --inputs runs/blind_eval_inputs_ultra_hard_unseen.json \
  --out runs/byzantine_opus_blind_outputs.json
```

This merges your answers with scenario metadata for the grading agent. If the merge script is missing, the JSON submission alone is enough.

---

## Scenario banks in this eval

| Bank | File | Count | Difficulty |
|------|------|-------|------------|
| Ultra-hard | `scenarios/byzantine_transcription_ultra_hard.yaml` | 23 | Held-out + microtonal/reverse/long |
| Unseen | `scenarios/byzantine_transcription_unseen.yaml` | 10 | Fresh corpus (Cappella + New Byzantium) |
| **Combined blind input** | `runs/blind_eval_inputs_ultra_hard_unseen.json` | **33** | Run this unless told otherwise |

To run only ultra-hard (23 cases), use `runs/blind_eval_inputs_ultra_hard.json` instead.

Regenerate blind inputs anytime:

```bash
PYTHONPATH=. python3 scripts/export_blind_eval_inputs.py --bank both
```

---

## Grading (for the follow-up agent — not you)

The user will return to the **grading agent** with:

```
Please grade runs/byzantine_opus_blind_submission.json
```

The grader will:

1. Load references from `scenarios/byzantine_transcription_ultra_hard.yaml` and `scenarios/byzantine_transcription_unseen.yaml`
2. Score each scenario on `melodic_equivalence`, `mode_fidelity`, `notation_convention`, `meaning_preservation` (see goal YAML)
3. Apply pass thresholds: melodic ≥ 1.5 AND meaning ≥ 1.5
4. Write `runs/byzantine_opus_blind_graded_summary.json`
5. Compare to GPT-4.1 baseline in `runs/byzantine_gpt-4.1_v2_ultra_hard_outputs.json` and `runs/byzantine_gpt-4.1_v2_unseen_outputs.json`

---

## Quick checklist before submitting

- [ ] Read `prompts/byzantine_transcription_v2.txt` in full
- [ ] Translated all scenarios from blind inputs JSON only
- [ ] Did not read `reference_output` or prior run files
- [ ] Every `id` in submission matches blind inputs
- [ ] Outputs are notation-only (no prose, no markdown fences)
- [ ] Saved to `runs/byzantine_opus_blind_submission.json`

---

## One-line prompt to paste into the other Opus agent

```
You are the Byzantine transcription translator for a blind eval. Read and follow
docs/byzantine_opus_blind_eval.md exactly. Load prompts/byzantine_transcription_v2.txt
as your system prompt. Translate every scenario in runs/blind_eval_inputs_ultra_hard_unseen.json
without opening reference_output or any runs/*_outputs.json files. Submit your answers to
runs/byzantine_opus_blind_submission.json using the template at
runs/byzantine_opus_blind_submission.template.json.
```
