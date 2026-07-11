# Real-Data Escalation Plan — What to Do If Curriculum v2 Still Floors

Staged plan for pushing real-data performance past the melisma wall, to execute **only
after** grading `curr2` predictions. **Planned, not started. No code/data changes until the
v2 results are in.** Cheapest-first, with a decision gate between each stage so we never
burn a night on the wrong lever.

## Where we are (context for a fresh agent)

- Task: SFT a 7B model (Qwen2.5-Coder, LoRA) to transcribe Byzantine neumes ↔ Western
  pitches, two directional adapters (n2w, w2n).
- **The wall:** real neume↔pitch data is melismatic (~1.78:1). Worse, the training pairs in
  `scripts/build_neume_tasks.py` are built by **proportional slicing** (`_windows()`): cut
  neumes and pitches into the same *number* of chunks, pair by position. Because the counts
  differ, chunk k of neumes does NOT correspond to chunk k of pitches — the labels are
  positionally wrong. The script itself says "we don't hand-align."
- **Result so far:** 7B on real data → drone collapse (melodic 0.0, variety ~0.01).
  Curriculum v1 (synth→real) reduced the drone but forgot the prior. Curriculum v2 (blended
  stage-2 + anti-loop decode, `docs/colab_curriculum_v2.md`) is the run being graded now.
- **Instruments that already exist:** `scripts/score_real_musical.py` (drone-proof,
  alignment-robust real metrics; ceiling on real gold ~1.74), `scripts/score_synthetic_eval.py`
  (exact, synthetic only), `scripts/build_neume_tasks.py` (the builder to fix),
  `data/byzantine/omr/omr_*.jsonl` (OMR pitches with `staves`), `neumes_*.jsonl` (neume seqs).
- **Directional ceilings (do not misread):** n2w is a true function (→2.0 achievable);
  **w2n is one-to-many** (`oligon` and `petaste` both = +1 step) so it tops ~1.2 even on
  perfect data. Judge w2n by set_f1/hist_sim/ngram_f1, not exact match.

## Decision gate (read tomorrow's numbers first)

Compare `curr2` to v1 (`runs/curr_*_realscore.json`) and baseline (`runs/coder7b_*_realscore.json`):

- **v2 clearly improved (variety off floor, set_f1/ngram_f1 up, real_musicality_0_2 > 0):**
  the curriculum lever works — consider one more cheap tuning pass (LR/epoch/blend ratio)
  before any realignment. May not need this plan at all.
- **v2 still floored (variety ~0.02, composite 0):** the ceiling is the labels, not the
  recipe → start Stage A below.

---

## Stage A — Anchor-segmented alignment (CHEAP, do FIRST, ~half day, CPU-only)

**Idea.** Byzantine notation has reliable structural landmarks that appear in BOTH the
neume stream and (as rests/barlines) the pitch stream: **breath marks** (`breath_mark_*`,
`comma_breath`) and **martyria** (`martyria_*`). Segment each hymn at these landmarks;
within each short phrase the neume:pitch count ratio is far closer to 1:1, so pairing is
much less wrong than slicing the whole hymn proportionally.

**Why it could work.** It attacks the same root cause as DTW (mis-paired windows) but with
ZERO learned cost function — it only trusts tokens we can actually identify. Lower ceiling
than perfect alignment, but low-risk and fast.

**Build (when triggered).**
1. New `scripts/build_neume_tasks_segmented.py` (or a `--segment` flag on the existing
   builder) that splits neume + pitch sequences at aligned landmark boundaries, then windows
   WITHIN each phrase. Reuse the OMR/neume loaders already in `build_neume_tasks.py`.
2. Verify: report per-phrase neume:pitch ratio distribution — it should be tighter around
   1.0 than the whole-hymn 1.78. If it's not, landmarks don't co-occur reliably → skip to #3.
3. Rebuild n2w/w2n train + held-out from the segmented pairs (NEW files, never overwrite
   existing data — another agent may use it).
4. Retrain the curriculum-v2 recipe on the segmented data (one Colab run), grade with
   `score_real_musical.py`.

**Gate:** if segmented data moves the real metrics up materially → alignment IS the lever,
proceed to Stage B (DTW) to push further. If it does nothing → alignment landmarks are too
sparse/noisy; jump to Stage C (reframe).

---

## Stage B — DTW realignment (the "all-night" job, higher effort/risk)

**Idea.** Dynamic Time Warping finds the lowest-cost monotonic alignment between two
different-length sequences (same tool as speech-to-transcript). Warp each hymn's neume
sequence against its pitch sequence so each neume maps to the actual pitch(es) it governs,
respecting melisma (one neume → several pitches), instead of forcing proportional 1:1.

**Why it could work.** If pairs become genuinely aligned, the position-level target finally
matches the input — gradient descent gets real structure instead of noise. It's the only
lever that fixes the labeling defect directly.

**The hard part — the DTW cost function (this is where it can fail).**
- Naive cost = "neume's implied interval vs pitch's interval" is CHANCE-LEVEL: an earlier
  probe measured 0.36 agreement (OMR noise + martyria-as-absolute-reset + melisma break the
  running-sum). A DTW on that signal would produce garbage alignments.
- Better cost: align on OBSERVED CONTOUR with slack (does the pitch move up/down where the
  neume says up/down, allowing repeats for melisma), and/or anchor on the Stage-A landmarks
  and DTW only BETWEEN them (constrained warping — much smaller, more reliable subproblems).

**Build (when triggered).**
1. `scripts/align_neume_pitch_dtw.py`: per hymn, DTW-align using the constrained/contour cost;
   emit per-neume→pitch-span assignments + an alignment-quality score per hymn.
2. **Pilot gate:** run ONLY on the ~23 count-aligned hymns first (abs ratio diff <15%).
   Eyeball/score alignment quality. If it's visibly good, scale to the corpus; if not, stop —
   do NOT spend the night retraining on bad alignments.
3. Rebuild SFT windows from the alignment, retrain, grade.

**Cost/time:** per-hymn DTW over the corpus + verifier + a retrain = the overnight job.
Only worth it if Stage A showed alignment is the lever AND the pilot looks clean.

---

## Stage C — Reframe the real target to what's recoverable (FALLBACK, guaranteed result)

**Idea.** If alignment proves too noisy to fix (Stage A flat AND Stage B pilot bad), stop
chasing exact pitch and make the real-data TASK match what melismatic input can actually
support: **contour, interval histogram, mode, ambitus, phrase count** — exactly what
`score_real_musical.py` already measures.

**Why it works.** These properties are well-posed from melismatic input (they don't require
per-position alignment), so the model can genuinely learn and be graded on them. Weaker as a
"transcription" claim, but a real, honest, defensible result instead of a wall of zeros.

**Build (when triggered).**
1. New task variants in the builder: target = contour string / interval histogram / mode+
   ambitus summary, rather than the raw pitch sequence.
2. Train + grade. `score_real_musical.py` already covers the metrics; may add a dedicated
   "structured summary" exact-match for the reframed targets.

---

## Recommended path

1. **Grade curr2.** If it improved, tune the cheap knobs; you may be done.
2. If floored → **Stage A** (anchor-segmented, half day, low risk). Gate on per-phrase ratio.
3. If A helps → **Stage B** (DTW), but pilot on 23 hymns BEFORE the overnight retrain.
4. If A does nothing / B pilot bad → **Stage C** (reframe) for a guaranteed honest result.

Do not skip straight to B: it's the most expensive and the riskiest, and A tells us for
free whether alignment is even the right lever.

## Hard constraints (unchanged)
- Never overwrite existing datasets or `build_neume_tasks.py` output — emit NEW files
  (another agent may be using the originals).
- Every new data file gets an independent verifier before any training.
- One variable per Colab run. Smoke-gate before any paid run.
- Push new scripts/data to `main` before a Colab run (Colab pulls `main`) — the v2 miss
  where uncommitted files would have failed the run must not recur.
