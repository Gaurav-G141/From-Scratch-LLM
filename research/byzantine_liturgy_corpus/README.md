# Byzantine Liturgy Corpus — research sources

Free, publicly-available settings of the **Divine Liturgy of St. John Chrysostom** in both
Byzantine (neume) and Western (staff) notation. Collected as a candidate **parallel corpus** for a
future, second attempt at real (non-synthetic) neume↔pitch transcription.

Origin: the user found a Scribd document
([Liturgy of St. John (Eliz. English) — Staff Notation](https://www.scribd.com/document/21060751/Liturgy-of-St-John-Eliz-English-staff-notation),
paywalled). It traces back to **New Byzantium Publications** (newbyz.org), which offers the same
material free. These are those free equivalents.

## Why this might work where the last real corpus failed

The prior real-corpus attempt failed because the paired neume and pitch streams **did not actually
correspond** (~35% directional agreement, ~10% positional ceiling) — see
`docs/byzantine_synthetic_expanded_results_20260711.md`. This material is more promising because
the publisher explicitly produced the **same hymns** in both notations as companion editions
(the Byzantine PDF's own cover references its staff-notated harmony counterpart). If the two
editions are truly note-for-note equivalent, they form a genuine parallel corpus rather than
mismatched pairs.

## Downloaded PDFs (`pdfs/`, git-ignored via `*.pdf`)

| file | pages | notation | notes |
|---|---|---|---|
| `liturgy_goaa_byzantine.pdf` | 90 | **Byzantine (neume)** | Full liturgy, Greek + official GOAA English. Cover states it "may [be] used with the staff-notated Three-Part Harmony version" → this is the **neume half of the parallel pair**. |
| `complete_liturgy_book.pdf` | 122 | staff ("New Byzantine Chant") | Modern English + Greek, Sunday worship. Candidate staff companion / harmony version. |
| `goarch_hieratikon_chrys.pdf` | 13 | text only (no notation) | GOARCH DCS Hieratikon skeleton — clean English/Greek text, useful as a lyric/alignment anchor. |

The PDFs are **not committed** (`.gitignore` has `*.pdf`) — they're large binaries. Re-download
with the URLs below if the folder is empty.

## Source URLs

- Byzantine-notation full liturgy (primary):
  https://newbyz.weebly.com/uploads/1/4/7/1/147110798/liturgy_book_goaa_byzantine.pdf
- Staff/harmony companion (mirror):
  https://s33939bc9149089cf.jimcontent.com/download/version/1296127037/module/3029448150/name/complete_liturgy_book.pdf
- GOARCH Hieratikon skeleton (text):
  https://dcs.goarch.org/goa/dcs/p/b/skeleton/liturgy/chrys/en/bk.skeleton.liturgy.chrys.pdf
- New Byzantium — per-hymn EB (Byzantine) and ES (staff) downloads:
  https://newbyz.weebly.com/standard-liturgy.html
- Internet Archive — Divine Liturgy, Greek + English (borrowable):
  https://archive.org/details/divineliturgyofs0000orth_k2j5

## Before trusting this as training data — verify

1. **Equivalence.** Confirm the EB (Byzantine) and ES (staff) editions are note-for-note the same
   setting, not two independent musical arrangements. Spot-check a few hymns by hand.
2. **Diatonic scope.** The current model excludes microtones, chromatic/enharmonic modes, fthora,
   and melisma. Liturgical hymns contain plenty of these. Filter to diatonic passages, or the
   scope has to expand (and the correct-by-construction guarantee is lost).
3. **Alignment mechanism.** These are engraved PDFs, not machine-readable neume/pitch token
   streams. Extracting aligned pairs needs OMR (optical music recognition) for the staff side and
   neume OCR/segmentation for the Byzantine side — a substantial pipeline, not a parse.

## Status

Reference material only. No pipeline built. The shipped model remains the
synthetic-grammar adapter (`docs/model_card.md`); that is unaffected by anything here.
