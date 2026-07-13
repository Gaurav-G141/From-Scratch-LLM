# Live demo — Byzantine neume → Western pitch

`byzantine_live_demo.ipynb` — a Colab notebook for demoing the synthetic-grammar LoRA live:
type Byzantine neumes → the model returns Western staff pitches, shown next to the
ground-truth answer so the audience sees they match.

**▶ Open in Colab:** https://colab.research.google.com/drive/1c4o_tVMIN0CUhcbTG44i5SB8yYgnodes?usp=sharing

## Running it
1. Open the Colab link above (or upload `byzantine_live_demo.ipynb`, or open it from GitHub).
2. Runtime → Change runtime type → **GPU** (T4 is enough).
3. In **Cell 1**, set `ADAPTER_SOURCE`:
   - `"hf"` + `HF_REPO` — load from your published HuggingFace adapter (simplest once uploaded).
   - `"drive"` + `DRIVE_DIR` — load from the adapter you saved to Google Drive.
   - `"upload"` — upload an adapter `.zip` when prompted.
4. Runtime → Run all (~2–3 min setup). Use the preset cells, then the interactive form cell.

## What it demonstrates
- **neume → west** (the model's strong direction: 96% exact / 98% melodic on held-out data).
- Presets: ascending octave scale, descending triad arpeggio, octave leap, leaps + steps,
  a chant phrase with held-note durations, and a stepwise wave. All verified on-ladder.
- Each result prints the model output **and** the correct pitches (from the repo's own
  interval rule) with a ✅/❌ — so it's self-verifying, not "trust me."

## Notes for the presenter
- The prompt format is reproduced **exactly** from training (`build_synthetic_musicality.py`
  + `predict_local.py`) — verified. Straying from it (different wording, wrong neume-count in
  the header) degrades output.
- **No bare ascending 3rd** exists in the grammar, so a rising do-mi-sol arpeggio isn't
  expressible; use descending thirds (`elaphron` −2) or the guide-vouched leaps. This is a
  property of the notation subset, not a model flaw — worth saying out loud if asked.
- Scope caveats to state: synthetic *diatonic* grammar only (no microtones / chromatic modes /
  fthora / melisma), not real manuscripts; reverse direction is intrinsically ambiguous.

## Editing the notebook
Don't hand-edit the `.ipynb` JSON. Edit `_build_demo_notebook.py` and re-run it:
```bash
.venv/bin/python demo/_build_demo_notebook.py
```
