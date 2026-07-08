# Day 3: Real Training Corpus

## Sources scraped

Run discovery (updates `data/byzantine/manifest.jsonl`):

```bash
python scripts/scrape_all_byzantine_sources.py discover
python scripts/scrape_all_byzantine_sources.py stats
```

| Source | Pairs | Notes |
|--------|-------|-------|
| **GOA Digital Chant Stand** (`goa_dcs`) | ~1,825 | Dedes/AGES; URL `/b/` ↔ `/w/` |
| **New Byzantium** (`new_byzantium`) | ~320 | Menaion, Triodion, liturgy pages; EB + GS/ES |
| **Cappella Romana** (`cappella_romana`) | ~16 | Divine Liturgy separate PDFs + bi-notational combined |
| **St. Anthony's** (`st_anthonys`) | ~131 | Divine Music Project; BFS + `download.php` Western PDFs |

## Extract training rows (vision)

PDFs are mostly image scores — extraction uses GPT-4.1 vision on rendered page PNGs.

```bash
# Small test batch
python scripts/extract_byzantine_training_data.py --fresh --download \
  --limit-pairs 15 --source cappella_romana

# Full corpus (resumable; run overnight)
python scripts/extract_byzantine_training_data.py --download --resume \
  --max-pages 1 --fragments-per-page 2
```

**Outputs:**
- `data/byzantine/sft_raw.jsonl` — chat-format rows, `"status": "raw"` (for pruning)
- `data/byzantine/extract_log.jsonl` — per-page success/empty/error log
- `data/byzantine/corpus/{source}/` — downloaded PDFs (gitignored)
- `data/byzantine/corpus/png/` — rendered pages (gitignored)

Each melodic fragment yields **two rows**: `byz_to_west` and `west_to_byz`.

## Prune (you + frontier LLM)

Raw rows include metadata for filtering:

```json
{
  "id": "cappella_dynamis_p0_opening_b2w",
  "direction": "byz_to_west",
  "status": "raw",
  "pair_id": "cappella_dynamis",
  "source": "cappella_romana",
  "title": "Dynamis",
  "page_idx": 0,
  "extraction_model": "gpt-4.1",
  "messages": [ ... ]
}
```

Suggested prune checks:
1. Melodic equivalence — do Western pitches match Byzantine neume chain?
2. Mode/Ni anchor correct?
3. Empty/title-page extractions (`extract_log.jsonl` → `"status": "empty"`)
4. Dedupe near-identical fragments

After pruning, set `"status": "accepted"` on kept rows.

## Build training JSONL

```bash
# Merge accepted corpus + hand-crafted YAML (optional)
python scripts/generate_byzantine_sft_data.py \
  --from-corpus data/byzantine/sft_raw.jsonl \
  --corpus-status accepted \
  --out data/byzantine/sft_v1.jsonl
```

## Eval holdout (do not train on)

Keep these out of `sft_v1.jsonl`:
- `scenarios/byzantine_transcription_heldout.yaml`
- `scenarios/byzantine_transcription_unseen.yaml`
- `scenarios/byzantine_transcription_ultra_hard.yaml`
