# Near-1:1 Subset & Rules-Engine Cross-Check — Findings (2026-07-09)

Follow-up to `docs/byzantine_handoff_20260709.md` (idea #3: "restrict to the ~15%
near-1:1 hymns" as a controlled test of whether *any* aligned real data moves melodic).
All work here is deterministic, CPU-only, read-only on existing data.

## What was built

`scripts/build_near1to1_subset.py` — isolates hymns whose pitch : pitch-bearing-neume
ratio is in [0.9, 1.1], then filters the existing (already-vetted) windowed translation
rows to those hymns. Reuses built windows; asserts no new alignment.

Result:
- **61 near-1:1 hymns** (12% of 495 parallel hymns) — matches the handoff's ~15% estimate.
- Outputs: `data/byzantine/sft_near1to1_train.jsonl` (1,916 rows / 52 hymns),
  `sft_near1to1_heldout.jsonl` (370 rows / 9 hymns), `near1to1_stems.json` (audit).
- 0 stem leakage; balanced neume_to_west / west_to_neume.

## Rules-engine cross-check (the key finding)

Tested `scripts/neume_rules_engine.py` (deterministic neume→pitch, per-neume intervals)
against the real OMR gold on the 958 near-1:1 `neume_to_west` windows — i.e. where the
handoff expected alignment to approximately hold.

| Metric | Result |
|---|---|
| exact pitch match | 0 / 958 (0.0%) |
| mean position pitch accuracy | 0.059 |
| mean contour accuracy | 0.305 (≈ below 3-way chance of 0.33) |
| length ratio pred:gold | 1.01 (median 16 vs 16 — lengths DO match here) |

Controls run:
- Starting the engine at gold's OWN first pitch → contour still 0.308 (start-offset is
  NOT the cause).
- Concrete example inspection: the neume window contains many non-pitch tokens
  (`argon`, `martyria_N`, `martyria_measure_J`, `heteron` → engine step = None) while the
  gold has a pitch at nearly every position. The proportional windowing paired slices of
  equal-ish *count* but NOT equal *position*.

## Interpretation — sharpens the handoff's conclusion

The handoff attributed the engine's failure to the ~1.78:1 melisma length mismatch. This
cross-check **controls for length** (near-1:1 windows, pred:gold ≈ 1.01) and the engine
**still fails, even on contour**. Therefore:

- The wall is **not only melisma / length**. Even when neume and pitch COUNTS nearly
  match, they are **not aligned position-by-position** — non-pitch neumes (martyria,
  breath, time signs) occupy sequence slots with no pitch, and pitch realization
  (ison-priming, melodic filling) does not track the bare symbol order.
- "Near-1:1 in aggregate count" ≠ "per-note aligned." So idea #3, on its own, does **not**
  supply clean aligned pairs — it supplies length-matched but still-unaligned pairs.
- This is further confirmation that exact pitch is unrecoverable from the extracted neume
  stream alone (now from a 5th independent angle), and it reinforces the handoff's real
  fixes: genuinely aligned data (audio time-alignment or a neume-native corpus) or
  reframing the metric to contour/interval-class.

## Caveat / honest limits

- The near-1:1 subset is still useful as a **training** slice (a model may learn a softer
  statistical mapping the rigid rules engine cannot), but it is not a source of exact
  per-note ground truth.
- Contrast with the SYNTHETIC data (`scripts/build_synthetic_musicality.py`), which IS
  exactly 1:1 and position-aligned by construction — that remains the clean instrument for
  proving the model *can* learn interval grammar when alignment genuinely exists.

## Reproduce
```
python3 scripts/build_near1to1_subset.py
# cross-check numbers were produced by an inline script; see this doc's table.
```
