# Stretch-Goal Plan — Do the Adversarial/Robustness Eval First

Handoff doc. Self-contained so a fresh agent can execute it without re-deriving context.
Written 2026-07-10. **Status: planned, not started** (may sit a while pending SLM results).

## TL;DR

The assignment's stretch ladder (`Train Your Own Small Learning Model.pdf`) lists, in
"roughly this order": **(1) DPO, (2) Adversarial/robustness eval, (3) Composed behavior.**
Recommendation: **do rung 2 (adversarial eval) FIRST, likely instead of DPO.** The order
is explicitly "roughly," so this is allowed. Optimizes for **cheapest to add well**.

## Why rung 2 over rung 1 (DPO) — justification

1. **Only zero-GPU-cost rung.** Adversarial eval reuses the trained adapters +
   `scripts/predict_local.py` + `scripts/score_synthetic_eval.py` unchanged — it is
   data-generation + scoring, no new training. DPO needs a full new training run per
   direction (preference pairs + DPO pass) — the opposite of cheap on a ~$10 Colab budget.
2. **No dependency.** DPO is not a prerequisite; rung 2 stands alone.
3. **DPO risks a null result HERE.** DPO sharpens spec adherence beyond SFT, but (a) on
   synthetic the SFT model already scores 2.0/100% (no headroom), and (b) on real
   melismatic data melodic_equivalence is walled by information content — DPO cannot add
   information absent from the input. Likely outcome: no movement on the metric that
   matters. Expensive, weak deliverable.
4. **Fills a NAMED win condition.** Appendix A's rubric has a **Robustness** row, and the
   doc states: "a tuned model that beats the base on Spec adherence and Robustness is a
   win." Spec adherence is already covered (melodic/mode/format); Robustness is the
   missing half. Adversarial eval produces exactly that number.
5. **Gradeable EXACTLY, for free.** The synthetic data is correct-by-construction, so
   adversarial synthetic inputs still have known-correct gold → `score_synthetic_eval.py`
   grades them deterministically (no LLM-judge cost, no variance).
6. **Buildable NOW, CPU-only** — the adversarial dataset is deterministic generation; it
   does not need the models or a GPU to build. DPO can't start until SFT finishes.

DPO remains a viable later rung if a GPU frees up. Composed behavior (rung 3, "hardest")
is out of scope for "cheapest."

## Background a fresh agent needs

- Project: SFT Qwen (LoRA/QLoRA) to transcribe Byzantine neumes ↔ Western pitches, two
  single-direction adapters (n2w = byz→west, w2n = west→byz).
- The core finding (`docs/byzantine_synthetic_breakthrough_20260709.md`,
  `docs/byzantine_handoff_20260709.md`): real neumes are melismatic (~1.78:1) and
  under-specify pitch → melodic_equivalence is walled on REAL data. On SYNTHETIC data
  (1:1 by construction, anchor given in prompt) the model hits melodic 2.0/100% exact.
- Synthetic data + deterministic scorer already exist and are vetted:
  - `scripts/build_synthetic_musicality.py` — generator (interval walk over 9 fixed-step
    neumes: ison=0, oligon/petaste=+1, apostrophos=-1, elaphron=-2, oligon_kentema=+3,
    oligon_hypsili=+4, elaphron_apostrophos=-3, chamile=-4). n2w prompt includes the Ison
    anchor line; that anchor-in-prompt fix is what makes absolute pitch learnable.
  - `scripts/verify_synthetic_musicality.py` — independent re-derivation verifier (0/0).
  - `scripts/predict_local.py` — batched, `--load-4bit`, injects ChatML for base models.
  - `scripts/score_synthetic_eval.py` — deterministic metrics (exact/pitch/interval/
    melodic_equivalence_0_2), `--self-test`, per-direction breakdown.
- Colab workflow: `docs/colab_notebook.md` (runnable cells), `docs/colab_runbook.md`
  (rationale). The model trains + generates prediction JSONLs on Colab; grading is done
  with `score_synthetic_eval.py` (CPU, no GPU).

## Implementation

### 1. `scripts/build_synthetic_adversarial.py` (NEW)
Reuse the interval-walk core, `INTERVAL_NEUMES`, `LADDER`, and MODES from
`build_synthetic_musicality.py` (import or copy). Emit rows in the SAME schema
(`{id, task, synthetic, messages}`) with the SAME anchor-in-prompt format, each tagged
with an `adv_category`, and each still correct-by-construction (deterministic gold):

| category | what it stresses |
|---|---|
| `length_extrapolation` | sequences 40–60 neumes, far longer than trained windows → tests length discipline under extrapolation |
| `anchor_follow` | same neumes at unusual/extreme Ison anchors → honors given anchor vs memorized register |
| `anchor_conflict` / `no_anchor` | omit/contradict the anchor → graceful degradation |
| `unseen_combos` | neume n-grams held out from the training generator's seeds |
| `oov_noise` | inject junk / non-vocab tokens mid-sequence → skip vs derail |
| `range_stress` | walks pushed to ladder edges → octave-slip behavior |

Keep it disjoint from training seeds (use a distinct `--seed-start`, e.g. 20_000_000, and
`--exclude` the training files) so it is a true held-out robustness probe.

### 2. Verifier
Extend `scripts/verify_synthetic_musicality.py` (or a sibling) to re-derive the
adversarial gold from scratch and require 0 content + 0 reversibility errors. For
`oov_noise`, the verifier must treat injected junk as skipped (no pitch) — mirror the
BREATH_NOOPS handling.

### 3. Predict + grade
- Colab: `predict_local.py --load-4bit --batch-size 16` over the adversarial file, per
  adapter (base + tuned for the delta).
- Grade with `score_synthetic_eval.py`. Add per-`adv_category` aggregation (small
  extension: group rows by category tag and report each) so the output is a
  **base-vs-tuned robustness table, per category**.

### 4. `docs/byzantine_adversarial_eval.md` (NEW)
Categories, rationale, and the results table → drops into the Appendix-A Robustness row
and the final results table.

## Optional add-on (costs LLM-judge calls)
A small REAL adversarial slice (most-melismatic hymns + malformed neume strings) graded
by the existing harness Robustness dimension (`eval_harness/judge/byzantine_checks.py`).
Lead with synthetic (free, exact); treat real robustness as a smaller supplement — it
partly inherits the melisma wall so expect lower, noisier numbers.

## Verification checklist
- Adversarial generator output passes independent re-derivation (0/0 errors).
- `score_synthetic_eval.py --self-test` still passes.
- Dry run: perfect-predictor JSONL (predictions = gold assistant text) over the
  adversarial set → exact 1.0, proving the scorer parses the new categories, BEFORE
  spending Colab GPU on real predictions.

## Scope / safety
- Adversarial dataset + verifier are CPU-only and correct-by-construction — buildable
  without a GPU and without touching training.
- No changes to existing datasets or `scripts/train_byzantine_sft.py`.
- Do NOT re-run DPO or composed-behavior rungs under "cheapest to add well" — they're
  documented here only as deferred alternatives.
