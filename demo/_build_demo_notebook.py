#!/usr/bin/env python3
"""Generate demo/byzantine_live_demo.ipynb — a Colab notebook for a LIVE demo of the
Byzantine synthetic-grammar LoRA (neume -> Western pitch).

Why a generator: hand-writing .ipynb JSON is error-prone; this emits guaranteed-valid
JSON. Edit the cell strings here, re-run: python demo/_build_demo_notebook.py

The notebook reproduces the TRAINING prompt format character-for-character (verified
against scripts/build_synthetic_musicality.py + scripts/predict_local.py) so the model
behaves in the demo exactly as it did in eval. It also imports the repo's own
`pitches_from` to show ground truth beside the model output — so the audience sees
"model == correct answer", not a claim to trust.
"""
import json, os

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}

def _src(lines):
    # join with newlines; each element may itself contain newlines
    text = "\n".join(lines)
    out = text.split("\n")
    return [l + "\n" for l in out[:-1]] + [out[-1]]

cells = []

cells.append(md(
    "# 🎼 Byzantine Neume → Western Pitch — Live Demo",
    "",
    "Type Byzantine neumes, get Western staff pitches from a small fine-tuned model",
    "(LoRA on `unsloth/Qwen2.5-Coder-7B-bnb-4bit`). On held-out data this direction scored",
    "**96% exact / 98% melodic**.",
    "",
    "**How to run:** Runtime → Change runtime type → **GPU (T4 is fine)**, then Runtime →",
    "Run all. Setup takes ~2–3 min. Then use the **interactive cell** at the bottom, or the",
    "preset scale / arpeggio / chant examples.",
    "",
    "The demo shows the model's output **next to the ground-truth answer** (computed by the",
    "repo's own interval rule), so you can see live that they match.",
))

cells.append(md(
    "## 1 · Setup — clone repo, install, pick where the adapter lives",
    "",
    "Set `ADAPTER_SOURCE` below. Three options, in order of convenience:",
    "- `\"hf\"` — load from your HuggingFace repo (once you've published it). Set `HF_REPO`.",
    "- `\"drive\"` — mount Google Drive and load from a folder you saved the adapter to.",
    "- `\"upload\"` — upload an adapter `.zip` from your computer when prompted.",
))

cells.append(code(
    "# ---- CONFIGURE ME ----",
    "ADAPTER_SOURCE = \"hf\"        # \"hf\" | \"drive\" | \"upload\"",
    "HF_REPO   = \"YOUR_USERNAME/byzantine-synthetic-grammar-lora\"   # if ADAPTER_SOURCE=='hf'",
    "DRIVE_DIR = \"/content/drive/MyDrive/byz_synth_expanded/adapter\" # if 'drive'",
    "BASE = \"unsloth/Qwen2.5-Coder-7B-bnb-4bit\"   # same base as training — do not change",
    "# -----------------------",
    "",
    "import os, subprocess, glob, sys",
    "if not glob.glob('/content/From-Scratch-LLM/scripts/build_synthetic_musicality.py'):",
    "    subprocess.run(['git','clone','--depth','1',",
    "                    'https://github.com/Gaurav-G141/From-Scratch-LLM',",
    "                    '/content/From-Scratch-LLM'], check=True)",
    "os.chdir('/content/From-Scratch-LLM'); sys.path.insert(0, 'scripts')",
    "print('repo ready at', os.getcwd())",
    "",
    "import importlib",
    "need = [m for m in ['bitsandbytes','accelerate','peft'] if importlib.util.find_spec(m) is None]",
    "if need:",
    "    subprocess.run(['pip','-q','install',*need], check=True)",
    "print('deps ready')",
))

cells.append(md(
    "## 2 · Load the model (base + LoRA adapter, 4-bit)",
    "",
    "Runs once. Reuses the exact inference path from `scripts/predict_local.py`: ChatML",
    "template injected, greedy decoding, stop on `<|im_end|>`.",
))

cells.append(code(
    "import torch",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig",
    "from peft import PeftModel",
    "",
    "assert torch.cuda.is_available(), 'Enable a GPU runtime (Runtime > Change runtime type).'",
    "",
    "# --- resolve adapter dir from the chosen source ---",
    "if ADAPTER_SOURCE == 'drive':",
    "    from google.colab import drive; drive.mount('/content/drive')",
    "    ADAPTER = DRIVE_DIR",
    "elif ADAPTER_SOURCE == 'upload':",
    "    from google.colab import files; import zipfile, io",
    "    up = files.upload()                       # pick your adapter .zip",
    "    name = next(iter(up)); os.makedirs('/content/adapter', exist_ok=True)",
    "    zipfile.ZipFile(io.BytesIO(up[name])).extractall('/content/adapter')",
    "    hit = glob.glob('/content/adapter/**/adapter_config.json', recursive=True)",
    "    assert hit, 'no adapter_config.json in the zip'",
    "    ADAPTER = os.path.dirname(hit[0])",
    "else:  # 'hf'",
    "    ADAPTER = HF_REPO",
    "print('adapter:', ADAPTER)",
    "",
    "tok = AutoTokenizer.from_pretrained(BASE)",
    "if tok.pad_token is None: tok.pad_token = tok.eos_token",
    "if not getattr(tok, 'chat_template', None):",
    "    tok.chat_template = (\"{% for message in messages %}\"",
    "        \"{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}\"",
    "        \"{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}\")",
    "tok.padding_side = 'left'",
    "",
    "# stop tokens: <|im_end|> (ChatML turn end used in training) + tokenizer eos backup",
    "_stop = []",
    "_ie = tok.convert_tokens_to_ids('<|im_end|>')",
    "if isinstance(_ie, int) and _ie >= 0 and _ie != tok.unk_token_id: _stop.append(_ie)",
    "if tok.eos_token_id is not None and tok.eos_token_id not in _stop: _stop.append(tok.eos_token_id)",
    "",
    "bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',",
    "    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)",
    "model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map='auto')",
    "model = PeftModel.from_pretrained(model, ADAPTER)",
    "model.eval()",
    "print('MODEL READY. stop ids =', _stop)",
))

cells.append(md(
    "## 3 · The transcribe function",
    "",
    "`transcribe(neumes, mode, ison)` builds the **exact** training prompt, generates, and",
    "(when the tokens are all in the known grammar) prints the **ground-truth** pitches",
    "beside the model output so you can see they match.",
))

cells.append(code(
    "from build_synthetic_musicality import (SYSTEM_PROMPT, pitches_from,",
    "    LADDER_INDEX, INTERVAL_NEUMES, DURATION_BEATS, BREATH_NOOPS)",
    "",
    "def _ground_truth(neumes, ison):",
    "    \"\"\"Correct pitches via the repo's own interval rule, or None if out-of-grammar.\"\"\"",
    "    toks = neumes.split()",
    "    known = set(INTERVAL_NEUMES) | set(DURATION_BEATS) | set(BREATH_NOOPS)",
    "    if ison not in LADDER_INDEX or any(t not in known for t in toks):",
    "        return None",
    "    try:    return ' '.join(pitches_from(LADDER_INDEX[ison], toks))",
    "    except Exception: return None",
    "",
    "def transcribe(neumes, mode='Mode 1', ison='D4', show_prompt=False, max_new_tokens=128):",
    "    neumes = neumes.strip()",
    "    n = len(neumes.split())",
    "    user = (f'Transcribe this Byzantine neume sequence ({n} neumes) to Western staff pitches:\\n'",
    "            f'{mode}\\nIson: {ison}\\n{neumes}')",
    "    msgs = [{'role':'system','content':SYSTEM_PROMPT}, {'role':'user','content':user}]",
    "    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)",
    "    if show_prompt: print('--- prompt ---\\n' + text + '\\n--------------')",
    "    enc = tok(text, return_tensors='pt').to(model.device)",
    "    with torch.no_grad():",
    "        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,",
    "                             pad_token_id=tok.pad_token_id, eos_token_id=_stop)",
    "    out = tok.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()",
    "    gt = _ground_truth(neumes, ison)",
    "    print('🎼 INPUT   :', neumes)",
    "    print('   mode/ison:', mode, '/', ison)",
    "    print('🤖 MODEL   :')",
    "    print('   ' + out.replace('\\n', '\\n   '))",
    "    if gt is not None:",
    "        model_pitches = out.split('\\n')[-1].strip()",
    "        ok = model_pitches == gt",
    "        print(('✅ MATCHES' if ok else '❌ differs from') + ' ground truth:')",
    "        print('   ' + gt)",
    "    else:",
    "        print('   (ground truth: n/a — a token or ison is outside the known grammar)')",
    "    print()",
    "    return out",
))

cells.append(md(
    "## 4 · Preset examples — scale, arpeggio, octave leap, chant",
    "",
    "All verified on-ladder against the repo derivation. Run and watch each match ✅.",
    "",
    "> **Grammar note:** the ascending vocabulary has steps (`oligon`/`petaste` = +1) and",
    "> guide-vouched *leaps* (`oligon_kentema` +3 … octave +7) but **no bare ascending 3rd**,",
    "> so a rising major-triad arpeggio (do-mi-sol) isn't expressible — descending thirds",
    "> (`elaphron` −2) are. That's a property of the notation subset, not a model limitation.",
))

cells.append(code(
    "# Ascending diatonic scale across an octave (D4 -> D5)",
    "transcribe('ison oligon oligon oligon oligon oligon oligon oligon', 'Mode 1', 'D4')",
))
cells.append(code(
    "# Descending D-minor triad arpeggio (A4 -> F4 -> D4)",
    "transcribe('ison elaphron elaphron', 'Mode 1', 'A4')",
))
cells.append(code(
    "# The showpiece: a single octave leap (C4 -> C5)",
    "transcribe('ison ypsili_over_kentima_oligon', 'Mode pl. 4', 'C4')",
))
cells.append(code(
    "# Leaps: up a 4th, then stepwise — Mode 4 on G",
    "transcribe('ison oligon_kentema apostrophos apostrophos oligon', 'Mode 4', 'G4')",
))
cells.append(code(
    "# A simple chant-like phrase WITH rhythmic durations (held notes render as pitch:beats)",
    "transcribe('ison oligon apli petaste apostrophos dipli ison', 'Mode pl. 4', 'C4')",
))
cells.append(code(
    "# Gentle stepwise wave — the most chant-like shape",
    "transcribe('ison oligon petaste apostrophos apostrophos oligon petaste', 'Mode 1', 'D4')",
))

cells.append(md(
    "## 5 · 🎤 Interactive — type your own",
    "",
    "Edit the strings and run. Use the vocabulary cheat-sheet below. Everything is relative",
    "to the **Ison** (starting pitch), so the same neumes at a different ison transpose.",
))

cells.append(code(
    "#@title Enter neumes and run { display-mode: 'form' }",
    "neumes = 'ison oligon petaste oligon apostrophos elaphron'  #@param {type:'string'}",
    "mode   = 'Mode 1'  #@param ['Mode 1','Mode pl. 1','Mode 4','Mode pl. 4']",
    "ison   = 'D4'      #@param ['C3','D3','E3','F3','G3','A3','B3','C4','D4','E4','F4','G4','A4','B4','C5','D5','E5']",
    "_ = transcribe(neumes, mode, ison)",
))

cells.append(md(
    "## 6 · Base vs fine-tuned — the money slide",
    "",
    "The headline claim is *SFT taught a behavior prompting couldn't*. This cell measures it",
    "directly: the **same** base model, LoRA **off** vs **on**, on the real held-out synthetic",
    "set (`sft_synth_musicality_heldout_cap.jsonl`, git-tracked — same file the 96% came from).",
    "",
    "It reuses the already-loaded model and just toggles the adapter with PEFT's",
    "`disable_adapter()`, so there's **no second model load** (T4-safe). Sample size is small",
    "(`N_EVAL`) to stay cheap — bump it if you have compute.",
))

cells.append(code(
    "N_EVAL = 40   #@param {type:'integer'}   # rows to score per side; low = cheap",
    "import json, re",
    "",
    "# gold pitches for a row = the last line of the gold assistant message",
    "rows = [json.loads(l) for l in open('data/byzantine/sft_synth_musicality_heldout_cap.jsonl')]",
    "rows = [r for r in rows if r.get('task') == 'neume_to_west'][:N_EVAL]",
    "print(f'scoring {len(rows)} neume->west rows, base vs tuned...')",
    "",
    "def _last_line(s): return s.strip().split('\\n')[-1].strip()",
    "def _gen(msgs):",
    "    text = tok.apply_chat_template([m for m in msgs if m['role']!='assistant'],",
    "                                   tokenize=False, add_generation_prompt=True)",
    "    enc = tok(text, return_tensors='pt').to(model.device)",
    "    with torch.no_grad():",
    "        g = model.generate(**enc, max_new_tokens=128, do_sample=False,",
    "                           pad_token_id=tok.pad_token_id, eos_token_id=_stop)",
    "    return tok.decode(g[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)",
    "",
    "def _pitch_acc(pred, gold):",
    "    p, g = pred.split(), gold.split()",
    "    if not g: return 0.0",
    "    hit = sum(1 for i in range(min(len(p),len(g))) if p[i]==g[i])",
    "    return hit / max(len(p), len(g))   # length mismatch is penalized",
    "",
    "def evaluate(tag):",
    "    exact = tot_pitch = 0.0",
    "    for r in rows:",
    "        gold = _last_line(r['messages'][-1]['content'])",
    "        pred = _last_line(_gen(r['messages']))",
    "        exact += (pred == gold)",
    "        tot_pitch += _pitch_acc(pred, gold)",
    "    n = len(rows)",
    "    print(f'  {tag:10s}  exact={exact/n:6.1%}   pitch_acc={tot_pitch/n:6.1%}')",
    "    return exact/n, tot_pitch/n",
    "",
    "print('\\n=== BASE (adapter OFF) — what prompting alone gets you ===')",
    "with model.disable_adapter():",
    "    base_x, base_p = evaluate('BASE')",
    "print('\\n=== FINE-TUNED (adapter ON) ===')",
    "tuned_x, tuned_p = evaluate('TUNED')",
    "print(f'\\n>>> exact-match lift: {base_x:.1%}  ->  {tuned_x:.1%}   (+{tuned_x-base_x:.1%})')",
    "print(f'>>> pitch-acc   lift: {base_p:.1%}  ->  {tuned_p:.1%}   (+{tuned_p-base_p:.1%})')",
    "print('\\n(Small N — indicative, not the full 200-row 96% figure. Raise N_EVAL for more.)')",
))

cells.append(md(
    "## 7 · Vocabulary cheat-sheet",
    "",
    "Every token the model knows. Pitch action = a fixed diatonic **degree-shift** from the",
    "running cursor (starts at the ison).",
    "",
    "**Steps & unison**",
    "- `ison` = repeat (0) · `oligon` = up a step (+1) · `petaste` = up a step (+1, accented)",
    "- `apostrophos` = down a step (−1)",
    "",
    "**Ascending leaps** (no bare +2)",
    "- `oligon_kentema` +3 (4th) · `oligon_hypsili` +4 (5th) · `ypsili_left_oligon` +5 (6th)",
    "- `ypsili_kentima_oligon` +6 (7th) · `ypsili_over_kentima_oligon` +7 (octave)",
    "",
    "**Descending leaps**",
    "- `elaphron` −2 (3rd) · `elaphron_apostrophos` −3 (4th) · `chamile` −4 (5th)",
    "",
    "**Durations** (lengthen the preceding note; render inline as `pitch:beats`)",
    "- `apli` = 2 beats · `dipli` = 3 · `tetrapli` = 5",
    "",
    "**Breath / barline no-ops** (ignored on pitch): `breath_mark_m` · `comma_breath` · `measure_bar`",
    "",
    "**Modes** (canonical ison): Mode 1 → D · Mode pl. 1 → A · Mode 4 → G · Mode pl. 4 → C.",
    "You can set any Ison you like — pitches are relative to it.",
    "",
    "---",
    "**Caveats to mention when demoing:** trained on synthetic *diatonic* grammar only — no",
    "microtones / chromatic modes / fthora / melisma, and not real scanned manuscripts. The",
    "reverse direction (pitch→neume) is intrinsically ambiguous (e.g. oligon vs petaste both",
    "= +1), so this demo shows the strong neume→pitch direction.",
))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "byzantine_live_demo.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, "with", len(cells), "cells")
