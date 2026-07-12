# Publishing the Byzantine Synthetic-Grammar Adapter to HuggingFace

Step-by-step to publish the trained LoRA adapter yourself. **You run every command** — no
credentials are ever shared. Uploading needs no GPU (it is a plain file transfer).

## What you're publishing

The **LoRA adapter** (`~/Downloads/synth_expanded_adapter/`, ~tens of MB) — the fine-tuned
weights, not the full 7B model. Anyone loads it on top of the base
`unsloth/Qwen2.5-Coder-7B-bnb-4bit` to reproduce the results (n2w 96% exact / 98% melodic).
This is the standard, correct artifact for a LoRA project.

The folder should contain: `adapter_model.safetensors` (the weights), `adapter_config.json`
(names the base model + LoRA config), and tokenizer files. If your download is incomplete, use
the copy saved to Google Drive by the Colab run (`MyDrive/byz_synth_expanded/adapter/`).

---

## Step 1 — Create a write token (once)

1. Go to **https://huggingface.co/settings/tokens**
2. **New token** → Type: **Write** → Create → copy it (starts with `hf_`).

## Step 2 — Install the client (once)

```bash
cd ~/From-Scratch-LLM
.venv/bin/pip install -q huggingface_hub
```

## Step 3 — Dry-run first (validates, uploads nothing)

Confirms the adapter folder is intact and previews what will be published:

```bash
.venv/bin/python scripts/push_to_hf.py \
  --adapter ~/Downloads/synth_expanded_adapter \
  --repo YOUR_USERNAME/byzantine-synthetic-grammar-lora \
  --dry-run
```

Replace `YOUR_USERNAME` with your HF username. Expect it to print the base model, the weights
file + size (tens of MB), and `dry-run: validated`. If it errors on a missing file, the download
is incomplete — use the Drive copy.

## Step 4 — Real upload

```bash
export HF_TOKEN=hf_your_token_here
.venv/bin/python scripts/push_to_hf.py \
  --adapter ~/Downloads/synth_expanded_adapter \
  --repo YOUR_USERNAME/byzantine-synthetic-grammar-lora
```

Add `--private` to start private. The script creates the repo, writes a model card (metrics +
what the model does / doesn't do well), and uploads the folder. On success it prints the URL.

- The token lives only in your shell session (`export`) — it is not written to the repo, the
  git history, or sent anywhere except HuggingFace's API.
- To make it private/public later, use the repo Settings on huggingface.co.

---

## Manual fallback (if you'd rather not use the script)

```bash
huggingface-cli login          # paste your write token when prompted
huggingface-cli upload YOUR_USERNAME/byzantine-synthetic-grammar-lora \
  ~/Downloads/synth_expanded_adapter .
```

This uploads the files but does **not** write the model card — you'd add a README yourself. The
script (Step 3–4) is preferred because it generates the documented card automatically.

---

## After publishing — how someone loads it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/Qwen2.5-Coder-7B-bnb-4bit"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "YOUR_USERNAME/byzantine-synthetic-grammar-lora")

prompt = ("Transcribe this Byzantine neume sequence (6 neumes) to Western staff pitches:\n"
          "Mode 1\nIson: D4\noligon_kentema petaste elaphron petaste ison oligon_hypsili")
msgs = [{"role": "system", "content": "You are a Byzantine chant notation assistant."},
        {"role": "user", "content": prompt}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=96,
                     eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

---

## Optional — publish a standalone (merged) model instead of the adapter

Only if you want a self-contained model users can load without PEFT. Merge into a **non-4-bit**
base (merging into a 4-bit base is lossy), which produces a ~15 GB model:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B")   # full precision
merged = PeftModel.from_pretrained(base, "~/Downloads/synth_expanded_adapter").merge_and_unload()
merged.push_to_hub("YOUR_USERNAME/byzantine-synthetic-grammar-merged")
```

For most purposes the adapter (Steps 1–4) is the better artifact: tiny, and it makes the
base-model dependency explicit.
