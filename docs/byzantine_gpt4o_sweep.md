# GPT-4o Full Translator Sweep

**Date:** 2026-07-07  
**Translator:** OpenAI `gpt-4o`  
**Judge:** GPT-4o via [`config/byzantine_eval.yaml`](../config/byzantine_eval.yaml)  
**Script:** [`scripts/run_byzantine_full_sweep.py`](../scripts/run_byzantine_full_sweep.py)  
**Summary:** [`runs/byzantine_gpt-4o_sweep_summary.json`](../runs/byzantine_gpt-4o_sweep_summary.json)

214 generation calls (107 scenarios × 2 prompts) across six scenario banks, plus 214 judge rescoring calls. No API errors.

## Per-suite results

Strict pass requires `melodic_equivalence ≥ 1.5`, `meaning_preservation ≥ 1.5`, and rule checks passing (see [`goals/byzantine_transcription.yaml`](../goals/byzantine_transcription.yaml)).

| Suite | N | v0 rule | v0 strict | v0 overall | v2 rule | v2 strict | v2 overall |
|-------|---|---------|-----------|------------|---------|-----------|------------|
| dev | 12 | 11/12 | 1/12 | 1.00 | 12/12 | 3/12 | 1.44 |
| heldout | 10 | 8/10 | 1/10 | 1.00 | 10/10 | 0/10 | 1.23 |
| final_dev | 36 | 32/36 | 1/36 | 0.85 | 36/36 | **11/36** | **1.47** |
| break_dev | 16 | 14/16 | 0/16 | 0.72 | 16/16 | 6/16 | 1.47 |
| ultra_hard | 23 | 17/23 | 2/23 | 0.78 | 23/23 | 1/23 | 1.25 |
| unseen | 10 | 7/10 | 0/10 | 0.60 | 10/10 | 0/10 | 0.80 |

### Dimension means (v2 prompt)

| Suite | melodic | mode | notation | meaning |
|-------|---------|------|----------|---------|
| dev | 0.92 | 2.00 | 1.83 | 1.00 |
| heldout | 0.40 | 1.80 | 1.80 | 0.90 |
| final_dev | 0.89 | 1.86 | 1.92 | 1.19 |
| break_dev | 0.88 | 1.94 | 1.88 | 1.19 |
| ultra_hard | 0.48 | 1.83 | 1.83 | 0.87 |
| unseen | 0.30 | 1.10 | 1.40 | 0.40 |

## Comparison to GPT-4.1 baselines (Day 2)

| Benchmark | GPT-4.1 v2 (prior) | GPT-4o v2 (this sweep) |
|-----------|-------------------|------------------------|
| final_dev strict | **18/36 (50%)** | 11/36 (31%) |
| final_dev overall | 1.70 | 1.47 |
| unseen strict | **0/10** | 0/10 |
| ultra_hard exact pitch | 6/23 (26%) | 1/23 strict; melodic mean 0.48 |

GPT-4o with v2 **underperforms GPT-4.1 v2** on the main litmus bank (11/36 vs 18/36 strict). Both models fail completely on unseen corpus scenarios. v2 prompt dramatically improves GPT-4o over v0 on rule compliance (36/36 vs 32/36 on final_dev) and overall judge scores (+0.62 on final_dev).

## v0 vs v2 takeaway

- **v0:** High rule-pass on easy cases but almost no strict passes (1/36 on final_dev). Melodic equivalence remains the bottleneck (0.31 mean on final_dev).
- **v2:** Perfect rule compliance on final_dev, break_dev, heldout, unseen, ultra_hard. Strict passes rise to 11/36 on final_dev and 6/16 on break_dev — still far below a production bar, and **0/10 on unseen**.
- **Notation vs melody:** v2 pushes notation_convention to ~1.8–1.9 on most suites while melodic_equivalence stays below 1.0 on held-out and unseen material — same pattern as blind Opus eval.

## Output files

Per-suite JSON under `runs/`:

```
runs/byzantine_gpt-4o_v0_{dev,heldout,final_dev,break_dev,ultra_hard,unseen}_outputs.json
runs/byzantine_gpt-4o_v2_{dev,heldout,final_dev,break_dev,ultra_hard,unseen}_outputs.json
```

Re-run judge only on saved outputs:

```bash
python scripts/run_byzantine_full_sweep.py --model gpt-4o --judge-only --judge
```

## Caveats

- `ultra_hard` is a subset of `final_dev` plus held-out cases — do not sum pass rates across files.
- Judge and translator are the same model (GPT-4o); scores are comparable to the Qwen base-vs-tuned compare that also used GPT-4o judge.
- Strict pass uses harness thresholds, not Opus blind grading from Day 2.

## Litmus implication (unchanged)

Even GPT-4o + v2 on the full 107-scenario sweep does not generalize to unseen corpus material (0/10 strict). Combined with prior GPT-4.1 and blind Opus results, **SFT remains justified** for reliable Byzantine transcription.
