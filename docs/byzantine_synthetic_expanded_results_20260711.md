# Expanded Synthetic Musicality — Data Built + Verified; Local Train Blocked by Hardware (2026-07-11)

## Context

After forensics proved the real neume↔pitch corpus is **not recoverable** to exact pitch
(0.56 pitch-bearing-neume:pitch ratio, ~35% neume-vs-pitch directional agreement corpus-wide,
~10% label ceiling the model had already reached), the project pivoted to the one honest,
demonstrable capability: the **correct-by-construction synthetic interval grammar**. This run
expanded that grammar's complexity and attempted a local capability check before a real-model
Colab run.

## What was built and verified (done, committed, pushed)

**Expanded synthetic grammar** (`74dfc1f`), two guide-vouched additions, both kept exactly
reversible:
1. **Ascending-leap ladder** (guide L122–133): `ypsili_left_oligon` +6, `ypsili_kentima_oligon`
   +7, `ypsili_over_kentima_oligon` +octave. Fills the old +5..+7 gap; AMBIT 9→11.
2. **Duration/rhythm channel** (guide L157–174): `apli`/`dipli`/`tetrapli` lengthen the
   preceding note (2/3/5 beats), rendered `<pitch>:<beats>`. **Bijective** beats↔token, so it
   is exactly recoverable in **both** directions (unlike breath marks, which stay n2w-only).

**Verification** (independent re-derivation, tables re-declared not imported):
- Train `sft_synthetic_musicality.jsonl`: **23,942 rows, 0 content + 0 reversibility errors.**
- Heldout `sft_synthetic_musicality_heldout.jsonl`: **4,794 rows, 0 errors.**
- **0 leakage** (id + exact-input) between train and heldout.
- Scorer (`score_synthetic_eval.py`) updated to parse `pitch:beats`; **gold-vs-gold =
  exact_match 1.0 / melodic 2.0 both directions** (confirms the eval measures the new
  complexity, not an artifact).

**EOS run-on fix** (`7d2331f`): `predict_local.py` now stops on `<|im_end|>` (id 151645, the
ChatML terminator training targets end with) instead of the base default `<|endoftext|>`
(151643). This was the root cause of the earlier 3.35× run-on. Confirmed active locally:
`stop token ids: [151645]`, no run-on in local predictions.

## Local training: blocked by hardware (honest finding)

The target machine is a **MacBook Air (24 GB unified memory, MPS)**. Every attempt to train
Qwen3-1.7B LoRA locally degraded into **swap-thrash**:
- Runs start fast (~3–4 s/step) but the MPS allocator cache + activations accrete in unified
  memory; after ~15–18 steps the system tips into constant swap (free pages → ~13 MB,
  compressor/swapouts spike) and step time collapses to **70–120 s/step** (→ 8+ hour ETA).
- Reproduced across 400, 600, 1000, and 3000-row sets at seq-length 320–512, batch 4–8. Row
  count only changes *when* thrash starts, not *whether*. Memory fully recovers the instant the
  process is killed, and the trainer's own RSS stays small (~100–290 MB) — this is MPS
  unified-memory behavior, not a leak in our code.
- The only run that **completed** was an early **50-step / 400-row** attempt (finished in ~4
  min before thrash). Its adapter was **undertrained**: structurally correct output (right
  Mode/Ison headers, valid pitch tokens, no run-on) but it did not follow the interval walk and
  stopped short → exact_match 0.0, melodic 0.0 on the 200-row disjoint slice. That is an
  undertraining result (50 steps cannot learn a grammar with octave leaps), **not** a data or
  correctness defect — the data verifies perfectly and gold-vs-gold scores 2.0.

**Conclusion:** this laptop cannot sustain even a small MPS LoRA run to convergence. Local
training is not the path; the run belongs on Colab.

## Deliverable: Colab notebook (ready, pushed)

`docs/colab_synthetic_expanded.md` (`db67ace`) runs the real-model version end to end:
clone/pull → **data correctness gate** → smoke → train `Qwen2.5-Coder-7B` (4-bit) on the full
23,942-row expanded set (2 epochs, minutes on L4/A100) → predict with the EOS fix + short token
cap → deterministic per-direction score → save adapter to Drive → optional HF push. The prior
1:1 synthetic hit melodic 2.0; this tests whether a real model also absorbs the added octave
leaps + rhythmic durations, bidirectionally, on held-out sequences.

## Status of each piece
| piece | state |
|---|---|
| Expanded synthetic train + heldout | built, verified 0-error, 0-leakage, pushed |
| Scorer handles `pitch:beats` | fixed, gold-vs-gold 2.0 both dirs |
| EOS run-on fix | committed, confirmed active |
| Local MPS training | blocked (swap-thrash); not viable on this Air |
| Colab notebook | written, pushed, ready to run tonight/tomorrow |

## Next step
Run `docs/colab_synthetic_expanded.md` on Colab. Real number lands there; record it back here
under a "Colab result" section when it completes.
