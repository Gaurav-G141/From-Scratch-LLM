# Byzantine Notation Translation — Day 2 Litmus Report

**Assignment checkpoint:** Day 2 — Spec, eval, & smoke test  
**Project:** Train a small learning model (SLM) to transcribe between Byzantine (Chrysanthine) neumes and Western staff notation  
**Date:** July 2026

This document records the Day 2 deliverables for the assignment: a falsifiable behavior spec, an eval harness with LLM-as-judge scoring, prompt optimization smoke tests, and a litmus verdict on whether frontier prompting already solves the task. It also summarizes what we learned about frontier-model limits, where an SLM earns its place, and how the system could support music education.

---

## 1. Day 2 deliverables (assignment alignment)

The assignment’s Day 2 row requires:

| Requirement | Status | Artifact |
|-------------|--------|----------|
| Write Behavior Spec | Done | [`goals/byzantine_transcription.yaml`](../goals/byzantine_transcription.yaml) |
| Build eval harness | Done | [`eval_harness/`](../eval_harness/) + [`config/byzantine.yaml`](../config/byzantine.yaml) |
| Build data-gen pipeline (corpus) | Started | [`scripts/scrape_byzantine_corpus.py`](../scripts/scrape_byzantine_corpus.py), [`data/byzantine/manifest.jsonl`](../data/byzantine/manifest.jsonl) |
| Run smoke tests (50+ scenarios) | Done | 84 hand-crafted + 10 unseen + 23 ultra-hard = **117 eval cases** across banks |
| Full loop scaffold | Ready | `eval` → judge → report; SFT step wired via `compare` against [`Qwen/Qwen3-0.6B`](../run_inference.py) |

The **litmus question** from the assignment:

> Can a well-prompted frontier model already do this behavior reliably?

If yes → fine-tuning is pointless. If no → the behavior is worth training into a small model.

---

## 2. Behavior spec

**Target behavior (pass/fail for a stranger):**

> Transcribe between Byzantine (Chrysanthine) neumatic notation and Western staff notation while preserving musical meaning: melodic contour, mode (echos), martyria, ison (drone), microtonal intervals (diesis, fthora), and rhythmic neume modifiers (gorgon, argon). Do **not** add Western harmony, do **not** simplify to 12-TET without marking approximation, do **not** impose fixed time signatures. Output notation only.

**Strict pass rule:** melodic_equivalence ≥ 1.5 **and** meaning_preservation ≥ 1.5 (on a 0–2 rubric per dimension).

**Judge dimensions:**

| Dimension | What it measures |
|-----------|------------------|
| `melodic_equivalence` | Pitch sequence and contour match (primary litmus dimension) |
| `mode_fidelity` | Echos, martyria, fthora preserved |
| `notation_convention` | Correct target notation system and formatting |
| `meaning_preservation` | No Western bias (harmony, 4/4 grid, silent ison drops) |

**Forbidden patterns (regex):** chord progressions, explanatory prose, “simplified to C major,” etc.

---

## 3. Eval harness and test banks

### Harness

- **Translator:** OpenAI GPT-4.1 / GPT-4o (prompted); Claude Opus 4 (blind Cursor-agent run)
- **Judge:** Claude Opus 4 (Cursor agent when `ANTHROPIC_API_KEY` unavailable)
- **Prompts:** [`byzantine_transcription_v0.txt`](../prompts/byzantine_transcription_v0.txt) → v1 → **v2** (liturgical formulas) → v3 (failure-targeted cheat sheet)
- **Reports:** JSON summaries under [`runs/`](../runs/)

### Scenario banks

| Bank | Cases | Purpose |
|------|-------|---------|
| `dev` + `heldout` + `break_dev` | 38 | Prompt optimization during harness build |
| `final_dev` | 36 | Full liturgical formula coverage (Cappella patterns) |
| `ultra_hard` | 23 | Compound microtonal, reverse, long-phrase stress tests |
| `unseen` | 10 | Vision-extracted fragments from held-out PDF corpus (Apolytikia, New Byzantium) |
| **Blind combined** | **33** | ultra_hard + unseen, no reference answers shown to translator |

### Corpus for future SFT

- **17 paired PDFs** (Cappella Romana + New Byzantium) in [`data/byzantine/manifest.jsonl`](../data/byzantine/manifest.jsonl)
- Bi-notational liturgical sources: Trisagion, Cherubic, Dynamis, Prokeimenon, Apolytikia, etc.
- ~239 additional New Byzantium pairs discovered for Day 3+ distillation

---

## 4. Results summary

### 4.1 Prompt optimization arc (36-case final dev)

| Stage | Model | Prompt | Overall mean | Strict pass |
|-------|-------|--------|--------------|-------------|
| Baseline | GPT-4o | v0 | 0.81 | low |
| Break set | GPT-4o | v1 | 1.44 | — |
| **Final** | **GPT-4.1** | **v2** | **1.70** | **18/36 (50%)** |

**Litmus on dev set alone:** **FAIL** (behavior appears promptable — liturgical formulas like Dynamis, Cherubic, Trisagion, Prokeimenon are memorizable in a long system prompt).

### 4.2 Failure-only re-test (9 stubborn cases)

| Prompt | Strict pass | Interpretation |
|--------|-------------|----------------|
| v2 (generic rules) | **0/9** | Generic interval logic insufficient |
| v3 (embedded reference contours) | **8/9** | Cheat-sheet prompting fixes known cases, not generalization |

v3 passes because it **embeds exact gold pitch chains** for microtonal and west→byz failures — oracle behavior, not portable skill.

### 4.3 Unseen corpus (10 cases from real PDFs)

| Model | Prompt | Strict pass |
|-------|--------|-------------|
| GPT-4.1 | v2 | **0/10** |
| GPT-4.1 | v3 | **0/10** |

**No generalization** to vision-extracted Apolytikion / Prokeimenon fragments the prompt never saw.

### 4.4 Ultra-hard bank (23 compound cases)

| Translator | Prompt | Exact pitch match |
|------------|--------|-------------------|
| GPT-4.1 | v2 | **6/23 (26%)** |
| Opus (Cursor, *with references*) | v2 | 23/23 — **invalid benchmark** (not blind) |

### 4.5 Blind Opus eval (honest frontier ceiling)

| Metric | Value |
|--------|-------|
| Cases | 33 (ultra_hard + unseen) |
| Translator | Claude Opus 4, blind (no gold answers) |
| Prompt | v2 only |
| **Strict pass** | **1/33** (`ison_only_passage`) |
| Melodic mean | **0.42** |
| Mode fidelity mean | 1.38 |
| Notation convention mean | 1.85 |
| Meaning preservation mean | 1.91 |

**Final litmus verdict:** **PASS for SFT** — even Opus blind does not reliably translate Byzantine notation. Frontier prompting plateaus far below the reliability bar on held-out and adversarial material.

---

## 5. Strengths and shortcomings of LLMs (including Opus)

### What frontier models do well

1. **Notation formatting and convention**  
   Opus blind scored **1.85** on notation_convention and **1.91** on meaning_preservation. Models reliably emit the *shape* of correct output: mode headers, ison lines, neume names, fthora labels, and “notation only” discipline. They rarely add SATB harmony or explanatory prose when forbidden.

2. **Mode vocabulary**  
   Most outputs correctly name Mode I–IV, plagal variants, and martyria *labels*. Mode fidelity mean was **1.38** — better than melody, worse than formatting.

3. **Memorized liturgical formulas (with a long prompt)**  
   GPT-4.1 + v2 passes **13/36** “perfect formula” cases on the dev set: Dynamis, Cherubic, Trisagion ison-shift, Alleluia plagal cadence, Kontakion ascending patterns, etc. These are **stereotyped Cappella contours** that fit in a cheat-sheet prompt.

4. **Simple and ison-heavy passages**  
   The single blind Opus strict pass was `ison_only_passage` (D4 D4 D4 E4). Borderline cases (`unseen_theophany_apolytikion`, `unseen_dormition_lamentations`) show partial credit when the opening is `(Pa) ison` + repeated ison neume — a pattern v2 explicitly teaches.

5. **West→Byz on short, formula-adjacent snippets**  
   GPT-4.1 sometimes passes reverse translation when the Western line maps cleanly to a taught formula (Trisagion snippet). This is pattern matching, not interval inference from first principles.

### Where frontier models fail (including Opus)

1. **Melodic equivalence is the bottleneck**  
   Blind Opus melodic mean **0.42**. The model *looks* like it transcribed (correct symbols, plausible pitches) but the **pitch sequence is wrong**. This is the core task and the primary litmus dimension.

2. **Leading ison neume skipped**  
   Repeated failure: after `(Pa) ison`, the first melodic neume is `ison` (repeat Ni) — models jump straight to `oligon` and drop D4. Seen in Annunciation, Ascension, Nenano medial, and many ultra-hard cases.

3. **Grave / Plagal IV Ni anchoring**  
   Models default Ni to **C4** instead of **D4** for Grave and Plagal IV passages (`unseen_transfiguration`, `unseen_pentecost_great_prokimenon`), collapsing entire phrases by a step.

4. **Microtones flattened to 12-TET**  
   Soft chromatic (`A4↑ B4↑ A4↓`), hard chromatic fthora, and enharmonic (`F4↓` vs `G4↑`) intervals are systematically rounded. v2 warns against this; models still emit plain naturals.

5. **West→Byzantine neume inference**  
   Long Western→Byzantine reverse passes collapse to generic `oligon` chains, miss `kentēma` on diesis leaps, and lose martyria on Mode IV kontakion-style ascent.

6. **Long phrases (8–10 neumes)**  
   Error compounds: `final_ten_neume_mode4`, `final_enharmonic_long_phrase`, and paschal katavasias diverge by bar 2. Context window is not the issue — **interval state tracking** is.

7. **Rhythmic modifiers without pitch fidelity**  
   Models attach `gorgon` / `argon` labels while getting underlying pitches wrong (`break_gorgon_argon_hemiolon_stack`). Rhythm is decorative, not grounded.

8. **Prompt length vs. generalization tradeoff**  
   v3 “fixes” 8/9 failures by memorizing exact contours — but **0/10 on unseen**. Larger prompts buy dev-set scores, not portable competence.

9. **Reference-assisted Opus ≠ deployable Opus**  
   In-agent Opus with full reasoning scored 23/23 ultra-hard; blind Opus scored 1/33. The gap shows **lookup/reasoning with implicit oracle access**, not reliable zero-shot translation.

### High-level takeaway

Frontier LLMs are strong **notation stylists** and **formula reciters** but weak **modal interval computers**. Byzantine transcription is not “translate symbols to English prose” — it is **maintaining a melodic cursor under mode-specific, microtonal, and ison-aware rules**. That stateful computation is exactly what prompting fails to stabilize, even at Opus scale.

---

## 6. Where a small language model (SLM) helps

The assignment thesis: *behavior from data*, not *smarter than GPT*. An SLM is justified here because the litmus test **passes** — prompting does not reliably hit the bar.

### 6.1 What SFT should compress into weights

| Capability | Why prompting fails | Why SFT can help |
|------------|--------------------|--------------------|
| Ison-neume vs ison-line distinction | Rule is stated but ignored under load | Hundreds of paired examples reinforce “first neume after `(Pa) ison`” |
| Mode-specific Ni anchors (Grave = D4) | Models default to C major logic | Corpus pairs ground martyria → pitch |
| Microtonal spellings (↑ ↓ diesis) | Flattened without persistent penalty | Gold Western lines from Cappella PDFs show exact spellings |
| Liturgical formula families | v2 prompt is 130+ lines and still incomplete | Distill formulas from 17+ bi-notational PDFs into weights |
| West→Byz reverse mapping | Requires inverse interval table + mode context | Parallel PDF pairs provide reverse direction gold |

**Base model candidate:** [`Qwen/Qwen3-0.6B`](../run_inference.py) (assignment default); compare via `eval_harness compare`.

### 6.2 SLM deployment niches (not replacing Opus)

An SLM is not meant to beat Opus on raw capability. It is meant to do **one narrow thing locally, cheaply, and consistently**:

1. **Offline classroom tool** — transcribe a neume exercise on a laptop with no API key or internet (Greek parishes, summer camps, mission fields).

2. **First-pass draft for choir directors** — suggest a Western staff draft from Byzantine input for review, with known failure modes documented.

3. **Exercise checker** — student submits neume sequence → SLM proposes Western line → teacher/judge confirms (human-in-the-loop).

4. **Notation router** — tiny classifier decides byz→west vs west→byz vs “needs human” before calling a frontier model on hard cases.

5. **Prompt compression** — replace a 130-line v2 prompt with a fine-tuned 0.6B–1.7B model that encodes the same rules in weights, freeing context for longer input phrases.

### 6.3 Data pipeline (Day 3+)

1. **Scrape** paired PDFs → [`manifest.jsonl`](../data/byzantine/manifest.jsonl)
2. **Extract** aligned neume ↔ staff fragments (vision teacher model on image PDFs)
3. **Filter** with the same Opus judge rubric used here
4. **SFT** (QLoRA via Unsloth) on accepted pairs
5. **Compare** base Qwen3-0.6B vs tuned on held-out + unseen banks

The **dataset is the deliverable**; this Day 2 report proves the eval exists before training.

---

## 7. Educational applications

Byzantine chant is the primary liturgical music tradition in Greek, Antiochian, and many Eastern European churches. Western staff notation dominates academic music education in the Americas and Western Europe. A reliable byz↔west transcription tool — even an imperfect SLM with known limits — supports several pedagogical scenarios.

### 7.1 Teaching Byzantine notation where Western staff is the literacy default

**Problem:** Students in the U.S., UK, or Western Europe often learn music through treble-clef, 12-TET thinking. Byzantine neumes encode **mode, ison, and microtones** that have no Western symbol.

**Use case:**

- Display a neume line from a hymnal → SLM proposes a **Western approximation** with explicit microtonal markers (`F4↑`, `A4↓`)
- Student hears the chant recording, compares, and learns *where Western notation lies*
- Teacher overrides wrong pitches (documented failure modes: leading ison, Grave Ni)

This mirrors how Cappella Romana publishes **bi-notational** scores — the SLM automates what those editions do by hand.

### 7.2 Teaching Western staff in Byzantine-dominant regions

**Problem:** In Greece, Romania, or parish schools using Byzantine notation exclusively, Western staff is unfamiliar. Conservatories and ecumenical choirs still need it for mixed ensembles.

**Use case:**

- Reverse direction (west→byz): a Western arrangement of a Trisagion snippet → neume sequence for Byzantine choir
- SLM gives a **first draft**; chanter corrects mode headers and fthora
- Emphasizes **interval names** (oligon, petastē, kentēma) tied to mode, not absolute pitch

Our eval shows west→byz is **harder** than byz→west for frontier models — a focused SLM trained on parallel PDFs targets exactly this gap.

### 7.3 Mode and ison as “physics,” not key signatures

**Pedagogical framing the eval supports:**

- **Ni is not tonic in the Western sense** — it is the ison anchor; melody moves by mode-specific intervals
- **fthora is a modulation event**, not an accidental — models that skip fthora lines teach the wrong lesson; the judge penalizes this
- **Microtones are structural** in soft/hard chromatic genres — flattening to F# vs F4↑ hides the diesis concept

An SLM trained on filtered pairs reinforces these semantics better than a generic chat model that drifts toward C major.

### 7.4 Curriculum integration ideas

| Level | Activity | Tool role |
|-------|----------|-----------|
| Intro | Match 4-neume Mode I phrases | SLM generates Western line; student sings against ison drone |
| Intermediate | Identify fthora type from notation | Compare SLM output with/without fthora line — discuss errors |
| Advanced | Transcribe Apolytikion formula | Use unseen-bank cases as **exam items**; SLM as optional scaffold |
| Choir | Bi-notational rehearsal packets | Batch-transcribe Cappella-style excerpts for mixed-notation choirs |

### 7.5 Honest limitations for educators

Ship with an **error analysis appendix** (Section 5 of this document). Teachers should know:

- Leading `ison` neume after martyria is often dropped
- Grave Ni = D4, not C4
- Long phrases need human review
- SLM is a **learning aid**, not an authoritative urtext

Transparency matches the assignment’s requirement for eval-before-train and falsifiable claims.

---

## 8. Error analysis (where models still fail)

| Failure cluster | Example IDs | Root cause | Data fix |
|-----------------|-------------|------------|----------|
| Leading ison neume | `unseen_annunciation`, `unseen_ascension`, `nenano_mode_medial` | Cursor starts at oligon, not ison repeat | More `(X) ison \| ison \| …` openers in SFT data |
| Grave Ni = D4 | `unseen_transfiguration`, `unseen_pentecost_great_prokimenon` | Western major-key prior (C4) | Tag Grave martyria explicitly in training pairs |
| Microtonal stacks | `break_soft_chromatic_double_diesis`, `final_plagal2_soft_chromatic` | 12-TET rounding | Gold ↑↓ examples from soft chromatic PDFs |
| Enharmonic fthora | `break_enharmonic_fthora_nenano`, `final_enharmonic_long_phrase` | Confusion of F4↓ vs G4↑ | Paired enharmonic passages from corpus |
| West→byz long reverse | `west_to_byz_kontakion_mode4`, `final_west_long_fthora_reverse` | Neume type collapse | Reverse-direction pairs from bi-notational PDFs |
| Formula vs generic stepping | `unseen_apolytikia_resurrection` | Generic ascending steps vs formula contour | More Apolytikion-family fragments in v2 dataset |

---

## 9. Litmus verdict and next steps

### Verdict: **PASS** → proceed to SFT

| Test | Result | Implication |
|------|--------|-------------|
| GPT-4.1 + v2 on 36 dev | 18/36 strict; overall 1.70 | Formulas are promptable — **misleading alone** |
| GPT-4.1 on 9 failures, v2 | 0/9 | Generic rules fail compound cases |
| GPT-4.1 on 9 failures, v3 | 8/9 | Oracle cheat sheet, not generalization |
| GPT-4.1 on 10 unseen | 0/10 | No corpus generalization |
| Opus blind on 33 hard+unseen | **1/33** strict | Frontier ceiling confirmed |

**Conclusion:** Byzantine neume ↔ Western staff transcription is a **behavior worth training**. Reliability — not raw intelligence — is the gap. A well-prompted frontier model (including Opus) cannot do it reliably on held-out liturgical material.

### Day 3 actions

1. Generate 500–2000 filtered parallel fragments from [`manifest.jsonl`](../data/byzantine/manifest.jsonl)
2. First QLoRA SFT run on Qwen3-0.6B
3. `eval_harness compare` — base vs tuned on `heldout`, `unseen`, `ultra_hard`
4. Report delta on **melodic_equivalence** (primary) and strict pass rate

---

## 10. Artifact index

| File | Description |
|------|-------------|
| [`goals/byzantine_transcription.yaml`](../goals/byzantine_transcription.yaml) | Behavior spec + rubric |
| [`prompts/byzantine_transcription_v2.txt`](../prompts/byzantine_transcription_v2.txt) | Primary eval prompt |
| [`runs/byzantine_final_opus_summary.json`](../runs/byzantine_final_opus_summary.json) | 36-case GPT-4.1 results |
| [`runs/byzantine_failures_v2_v3_comparison.json`](../runs/byzantine_failures_v2_v3_comparison.json) | Failure re-test v2 vs v3 |
| [`runs/byzantine_unseen_eval_summary.json`](../runs/byzantine_unseen_eval_summary.json) | 10-case unseen corpus |
| [`runs/byzantine_ultra_hard_eval_summary.json`](../runs/byzantine_ultra_hard_eval_summary.json) | 23-case ultra-hard bank |
| [`runs/byzantine_opus_blind_graded_summary.json`](../runs/byzantine_opus_blind_graded_summary.json) | 33-case blind Opus grade |
| [`docs/byzantine_opus_blind_eval.md`](byzantine_opus_blind_eval.md) | Blind eval protocol for separate agent |

---

## 11. One-paragraph thesis (Brainlift seed)

**Behavior from data:** Byzantine chant transcription requires mode-aware, microtonal interval tracking that frontier LLMs approximate in format but fail in melody — even Claude Opus scores 1/33 strict on blind hard+unseen eval. Liturgical formulas are promptable on dev sets, which falsely suggests the task is easy; held-out PDF fragments expose the real ceiling. A small fine-tuned model, distilled from bi-notational Cappella and New Byzantium parallel scores and filtered through this harness, can encode ison logic and formula families into weights for offline, low-cost use in Byzantine chant education — especially where Western staff literacy is the bridge students already have, or where Western scores must be converted back into neumes for traditional choir.
