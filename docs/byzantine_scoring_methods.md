# Byzantine Scoring & Checks — Reference

How model outputs are scored for the Byzantine transcription task. There are **two
separate systems**: (A) the existing eval harness — a deterministic rule/format gate
plus an LLM judge for quality dimensions; and (B) a new standalone deterministic scorer
for the correct-by-construction synthetic slice.

Function names and line numbers below were verified against the live code. Lines may
drift with edits; the function/constant names are the stable anchors.

---

## System A — Existing eval harness

Grades ANY goal (used for the real corpus and hand-built scenarios). Two layers.

### A1. Deterministic rule / format checks (code, NO LLM)

- **`eval_harness/judge/rule_checks.py` → `run_rule_checks(goal, text)`** (rule_checks.py:12)
  - Applies `goal.forbidden_patterns` and `goal.required_patterns` (regex,
    IGNORECASE|MULTILINE) declared in `goals/byzantine_transcription.yaml`.
  - Also validates `goal.json_schema` when set (not used by the Byzantine goal).
  - Returns `RuleCheckResult(passed, failures)`.

- **`eval_harness/judge/byzantine_checks.py` → `check_byzantine_transcription(...)`** (byzantine_checks.py:26)
  - `WESTERN_BIAS_PATTERNS` (byzantine_checks.py:6) — flags chord/triad/harmony,
    "C major/G major", "4/4|3/4|6/8", "quarter note…".
  - `BYZ_MARKERS` (byzantine_checks.py:14) / `WEST_MARKERS` (byzantine_checks.py:19) —
    enforce the correct notation SYSTEM per direction: `byz_to_west` output must contain
    staff pitches (`[A-G][#b]?\d`, clefs, time sigs) (byzantine_checks.py:48);
    `west_to_byz` must contain neume/martyria/ison tokens (byzantine_checks.py:51).
  - Prose/commentary detection (byzantine_checks.py:57, e.g. "the answer is",
    "note that", "I transcribed"); empty-output check.

These produce pass/fail FLAGS only. They do not compute dimension scores.

### A2. LLM judge — the 4 dimension scores (0–2)

- **`eval_harness/backends/anthropic_api.py` → `score(...)`** (anthropic_api.py:73) —
  Claude Opus (`claude-opus-4-20250514` by config), `temperature=0`.
- **`eval_harness/backends/openai_api.py` → `score(...)`** (openai_api.py:92) —
  `gpt-4o`, `temperature=0`, JSON response.
- Dimensions (from `goals/byzantine_transcription.yaml`):
  `melodic_equivalence`, `mode_fidelity`, `notation_convention`, `meaning_preservation`.
- **No code computes these numbers** — the judge model returns them (clamped to int 0–2).
  The `dimension_guidance` rubric text is interpolated into the judge prompt. The A1
  failures are also injected into the prompt as a "Deterministic rule checks failed:"
  section, but the judge weighs them freely.

### A3. Orchestration + strict-pass gate

- **`eval_harness/judge/runner.py` → `evaluate_scenario(...)`** (runner.py:17) — runs A1,
  then A2; attaches rule failures to the result and the judge prompt.
- **`eval_harness/cli.py` → `cmd_eval(...)`** (cli.py:62) — **strict pass** (cli.py:105) =
  all `pass_thresholds` met **AND** `result.rule_check.passed`. So an A1 failure forces
  FAIL regardless of judge scores.
- **`eval_harness/cli.py` → `_make_judge(config)`** (cli.py:19) — selects backend.
- Thresholds (`goals/byzantine_transcription.yaml`):
  `melodic_equivalence >= 1.5` AND `meaning_preservation >= 1.5`.
- Configs: **`config/byzantine.yaml`** → `anthropic` / Opus (production);
  **`config/byzantine_eval.yaml`** → `openai` / gpt-4o (smoke/fallback).

### Not wired into scoring
- `scripts/neume_rules_engine.py` (`neumes_to_pitches`) — a deterministic neume→pitch
  engine, but nothing in `eval_harness/` imports it.
- `scripts/grade_translation_eval.py` — does NO scoring; aggregates a hardcoded dict of
  hand-grades (Opus-agent grades from the billing-outage period).
- `scenario.reference_output` (gold) is never string-compared in code — it is only
  pasted into the judge prompt for the LLM to eyeball.

**Bottom line for System A:** rule-following / notation-system / format = deterministic
(code, can force FAIL). All four quality dimensions, INCLUDING melodic accuracy =
LLM-judged (Opus in prod). There is no deterministic pitch-correctness metric here.

---

## System B — Standalone deterministic scorer (synthetic slice only)

Fills the gap that melodic accuracy in System A is Opus-only (noisy, costs API calls).
The synthetic data (`scripts/build_synthetic_musicality.py`) is correct by construction,
so its melodic accuracy can be scored EXACTLY in code.

**`scripts/score_synthetic_eval.py`** — does **NOT** import `eval_harness/` and does
**NOT** load a model. Safe to run alongside a training/eval job.

Functions:
- `extract_pitches(text)` (score_synthetic_eval.py:68) — pulls the predicted pitch line
  from raw output; strips `<think>…</think>`, ignores the `Ison:` header.
- `extract_neumes(text, vocab)` (score_synthetic_eval.py:89) — pulls the predicted neume
  line, restricted to known vocab.
- `levenshtein(a, b)` (score_synthetic_eval.py:116); `intervals(pitches)` (score_synthetic_eval.py:130).
- `score_seq(pred, gold)` (score_synthetic_eval.py:149) — core metrics.
- `score_file(eval_path, pred_path)` (score_synthetic_eval.py:196) — aggregates overall
  + per-direction.
- `self_test()` (score_synthetic_eval.py:248) — validation, no files/model.

Metrics (per row + aggregated):
- `exact_match` — predicted sequence identical to gold.
- `pitch_accuracy` — position-aligned fraction correct (len-normalised).
- `interval_accuracy` — consecutive-interval (contour) match; catches transposition.
- `norm_edit_distance` — Levenshtein / max(len); 0.0 is perfect.
- `melodic_equivalence_0_2` — 0/1/2 mirroring the System-A rubric, for comparability.

Inputs:
- Predictions JSONL: one object per line, `{"id": "<row id>", "prediction": "<raw model text>"}`;
  ids must match the eval file (e.g. `synth_010000000_t0_n2w`).
- Eval file (exact gold): `data/byzantine/sft_synthetic_musicality_heldout.jsonl`.

Companion (data correctness, not scoring):
**`scripts/verify_synthetic_musicality.py`** re-derives every gold row from scratch.

### Usage
```bash
# self-test (no files, no model) — should print "SELF-TEST: ALL PASS"
python3 scripts/score_synthetic_eval.py --self-test

# score predictions against the held-out slice
python3 scripts/score_synthetic_eval.py \
    --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
    --pred runs/my_predictions.jsonl \
    --out  runs/synth_score.json
```

### Caveats
- Valid **only** on the synthetic (correct-by-construction) data. Do NOT use on the real
  melismatic corpus (neumes:pitches ~1.78:1 — exact match not expected there; that is
  what System A / the Opus judge is for).
- Read the **per-direction** breakdown. Pooled `interval_accuracy` blends
  `neume_to_west` (contour is meaningful, =1.0 when perfect) with `west_to_neume` (tokens
  are neume names, so the numeric interval metric is ~0 by design). On a perfect
  prediction set: `neume_to_west` interval_accuracy = 1.0, `west_to_neume` = 0.0, both
  with exact_match = 1.0.
