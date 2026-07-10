#!/usr/bin/env python3
"""Generate predictions from a local base+LoRA model over a JSONL eval slice.

Bridges the trained adapter to the deterministic scorer: reads an eval JSONL whose rows
carry the exact chat `messages` (system+user+assistant), regenerates the assistant turn
from the model, and writes a predictions JSONL of {id, prediction} that
scripts/score_synthetic_eval.py consumes directly.

Designed for the synthetic held-out slice (data/byzantine/sft_synthetic_musicality_heldout.jsonl)
but works on any messages-style JSONL. The gold assistant message is dropped from the
prompt (we only feed system+user + a generation prompt), so there is no leakage.

Colab notes:
  --load-4bit uses bitsandbytes NF4 to fit a 9-20B model on an L4/A100. Requires CUDA.
  --enable-thinking is OFF by default (suppresses the Qwen3 <think> wrapper that polluted
  Day-3 outputs); harmless on non-Qwen models.

Usage:
  python scripts/predict_local.py \
    --model google/gemma-2-9b-it \
    --adapter-path /content/drive/MyDrive/byz/adapter \
    --eval data/byzantine/sft_synthetic_musicality_heldout.jsonl \
    --out runs/synth_heldout_preds.jsonl \
    --load-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build_prompt_messages(messages: list[dict]) -> list[dict]:
    """Keep everything up to (not including) the assistant turn — no gold leakage."""
    return [m for m in messages if m["role"] != "assistant"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA adapter dir; omit to run the base model")
    ap.add_argument("--eval", default=str(ROOT / "data/byzantine/sft_synthetic_musicality_heldout.jsonl"))
    ap.add_argument("--out", required=True, help="predictions JSONL {id, prediction}")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy/deterministic (recommended for scoring)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="rows generated per batch (left-padded); 1 = unbatched")
    ap.add_argument("--load-4bit", action="store_true", help="NF4 4-bit (CUDA only)")
    ap.add_argument("--limit", type=int, default=0, help="predict only first N rows (0=all)")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="allow Qwen3 <think> block (default off)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu")
    print(f"device={device} model={args.model} adapter={args.adapter_path} "
          f"4bit={args.load_4bit}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Base models (e.g. Qwen2.5-Coder-7B) ship NO chat_template. Inject the same ChatML
    # template used at train time so inference formatting matches training exactly.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        print("No chat_template (base model) -> injected ChatML.", file=sys.stderr)
    # Decoder-only batched generation requires LEFT padding, else right-pad tokens shift
    # the position of the generated continuation and corrupt short-prompt rows.
    tokenizer.padding_side = "left"

    model_kwargs: dict = {"torch_dtype": "auto"}
    if args.load_4bit:
        if device != "cuda":
            raise SystemExit("--load-4bit requires CUDA")
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.load_4bit:
        model = model.to(device)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    def template(messages: list[dict]) -> str:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking)
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    rows = load_rows(Path(args.eval))
    if args.limit:
        rows = rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bs = max(1, args.batch_size)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), bs):
            batch = rows[start:start + bs]
            texts = [template(build_prompt_messages(r["messages"])) for r in batch]
            # left-padded batch encode
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=(args.temperature if args.temperature > 0 else None),
                    pad_token_id=tokenizer.pad_token_id,
                )
            # with left padding, every row's continuation starts at the same padded width
            gen_only = gen[:, enc["input_ids"].shape[1]:]
            completions = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            for r, completion in zip(batch, completions):
                f.write(json.dumps({"id": r["id"], "prediction": completion},
                                   ensure_ascii=False) + "\n")
            n += len(batch)
            print(f"  {n}/{len(rows)}", file=sys.stderr, flush=True)

    print(f"Wrote {n} predictions -> {out_path}")


if __name__ == "__main__":
    main()
