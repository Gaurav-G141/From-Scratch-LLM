# Claude Opus Full Translator Sweep

**Date:** 2026-07-07 (updated 2026-07-08)  
**Translator:** Claude Opus 4 (`claude-opus-4-20250514`) via **Cursor agents** (blind)  
**Judge:** Claude Opus 4 via **Cursor agents** (rubric scoring with gold `reference_output`)  
**Summary:** [`runs/byzantine_claude-opus-4-20250514_sweep_summary.json`](../runs/byzantine_claude-opus-4-20250514_sweep_summary.json)  
**Prior GPT-4o judge run:** [`runs/byzantine_claude-opus-4-20250514_sweep_summary_gpt4o_judge.json`](../runs/byzantine_claude-opus-4-20250514_sweep_summary_gpt4o_judge.json)

## Methodology

`ANTHROPIC_API_KEY` is not set in `.env`, so both translation and judging use Cursor Opus agents:

1. `python scripts/export_opus_sweep_inputs.py` → blind inputs per suite  
2. Opus agents translate → `runs/opus_sweep_submissions/`  
3. `python scripts/merge_opus_sweep_submissions.py` → harness output JSON  
4. `python scripts/export_opus_judge_batches.py` → judge task batches  
5. Opus agents score → `runs/opus_judge_submissions/`  
6. `python scripts/merge_opus_judge_submissions.py` → updated summary  

Translators did **not** read `reference_output` during translation. Judges **did** use gold references (same rubric as [`goals/byzantine_transcription.yaml`](../goals/byzantine_transcription.yaml)).

Re-judge only (no re-translation):

```bash
python scripts/export_opus_judge_batches.py
# Opus agents grade runs/opus_judge_inputs/*.json → runs/opus_judge_submissions/
python scripts/merge_opus_judge_submissions.py --preserve-generations
```

With `ANTHROPIC_API_KEY` set, API judging:  
`python scripts/run_byzantine_full_sweep.py --provider anthropic --judge-only --judge --config config/byzantine.yaml`

## Per-suite results (v2 prompt — primary, Opus judge)

| Suite | N | Rule pass | Strict pass | Overall | Melodic | Meaning |
|-------|---|-----------|-------------|---------|---------|---------|
| dev | 12 | 12/12 | 4/12 | 1.67 | 0.83 | 1.83 |
| heldout | 10 | 10/10 | 1/10 | 1.55 | 0.50 | 1.90 |
| final_dev | 36 | 36/36 | **11/36** | 1.56 | 0.92 | 1.67 |
| break_dev | 16 | 16/16 | 6/16 | 1.61 | 0.88 | 1.81 |
| ultra_hard | 23 | 23/23 | 1/23 | 1.51 | 0.57 | 1.61 |
| unseen | 10 | 10/10 | **0/10** | 1.55 | 0.40 | 1.80 |

## Opus v0 vs v2 (strict pass, Opus judge)

| Suite | v0 strict | v2 strict |
|-------|-----------|-----------|
| dev | 0/12 | 4/12 |
| heldout | 1/10 | 1/10 |
| final_dev | 4/36 | 11/36 |
| break_dev | 0/16 | 6/16 |
| ultra_hard | 0/23 | 1/23 |
| unseen | 0/10 | 0/10 |

v2 improves strict pass on formula-heavy banks. Unseen corpus remains **0/10** for both prompts.

## Opus judge vs GPT-4o judge (same translations, v2)

| Suite | Opus judge strict | GPT-4o judge strict | Δ melodic |
|-------|-------------------|---------------------|-----------|
| final_dev | 11/36 | 11/36 | +0.03 |
| break_dev | 6/16 | 6/16 | 0 |
| heldout | 1/10 | 1/10 | +0.10 |
| ultra_hard | **1/23** | 2/23 | +0.09 |
| unseen | 0/10 | 0/10 | +0.10 |

Strict pass counts are nearly identical on main banks; Opus judge is slightly stricter on ultra_hard (1 vs 2) but gives higher melodic/meaning dimension means on unseen (still 0 strict — melodic bottleneck).

## Artifacts

| Path | Description |
|------|-------------|
| `runs/opus_sweep_inputs/` | Blind translation inputs |
| `runs/opus_sweep_submissions/` | Opus translator submissions |
| `runs/opus_judge_inputs/` | Judge task batches |
| `runs/opus_judge_submissions/` | Opus judge score submissions |
| `runs/byzantine_claude-opus-4-20250514_v{0,2}_{suite}_outputs.json` | Merged harness outputs (12 files) |
| [`scripts/export_opus_judge_batches.py`](../scripts/export_opus_judge_batches.py) | Export judge batches |
| [`scripts/merge_opus_judge_submissions.py`](../scripts/merge_opus_judge_submissions.py) | Merge judge scores into summary |

## Takeaway

With Opus as both translator and judge, v2 reaches **11/36** on final_dev and **0/10** on unseen — same strict-pass pattern as GPT-4o-judged runs. Melodic equivalence remains the bottleneck. **SFT verdict unchanged.**
