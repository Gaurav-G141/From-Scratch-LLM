#!/usr/bin/env python3
"""Smoke-test SFT on Byzantine junk JSONL (Day 2 full-loop checkpoint).

Trains a LoRA adapter on Qwen3-0.6B for 1 epoch on junk data.
Uses Unsloth + QLoRA when CUDA is available; falls back to PEFT LoRA on MPS/CPU.

Usage:
  python scripts/generate_byzantine_sft_data.py
  python scripts/train_byzantine_sft.py --data data/byzantine/sft_junk.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _apply_template(tokenizer, messages, **kwargs) -> str:
    """apply_chat_template with Qwen3 thinking-mode suppressed. Day-3 v2 found a
    <think>…</think> wrapper on 100% of outputs; enable_thinking=False stops the model
    re-learning it. The kwarg is Qwen3-specific, so we retry without it on templates
    that don't accept it (Gemma-2, Llama-3.1, etc.)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)


def rows_to_text(rows: list[dict], tokenizer) -> list[str]:
    texts = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        texts.append(_apply_template(tokenizer, messages, add_generation_prompt=False))
    return texts


def detect_turn_markers(tokenizer) -> tuple[str, str]:
    """Auto-detect the (instruction_part, response_part) delimiters for THIS tokenizer's
    chat template, so response-only masking works on any base (Qwen3/ChatML, Gemma-2,
    Llama-3.1, …) without hardcoding markers.

    Both returned strings are template-fixed substrings that appear verbatim in every
    real (system, user, assistant) row, which is what train_on_responses_only matches on:

      response_part    — what the template appends after the user turn to open the
                         assistant turn. Found by rendering a user-only probe with
                         add_generation_prompt=True and taking the text AFTER the user
                         sentinel. This is the marker that actually drives masking.
      instruction_part — the turn boundary between system and user (system-close +
                         user-open). Found by rendering [system, user] and taking the
                         text BETWEEN the two sentinels. Fixed by the template regardless
                         of the system/user content, so it matches our real rows.
    """
    S, U = "␞_SYS_␞", "␞_USR_␞"  # sentinels unlikely in any template

    # response_part: text the template adds to begin the assistant turn.
    probe = _apply_template(tokenizer, [{"role": "user", "content": U}],
                            add_generation_prompt=True)
    if U not in probe:
        raise ValueError("user sentinel not found in rendered template")
    response_part = probe.split(U, 1)[1]
    if not response_part.strip():
        raise ValueError("empty response delimiter detected")

    # instruction_part: the fixed boundary between a system turn and the following user
    # turn. Falls back to the user-open portion of response-less render if no system slot.
    try:
        both = _apply_template(
            tokenizer,
            [{"role": "system", "content": S}, {"role": "user", "content": U}],
            add_generation_prompt=False,
        )
        instruction_part = both.split(S, 1)[1].split(U, 1)[0]
        if not instruction_part.strip():
            raise ValueError("empty instruction delimiter")
    except (ValueError, Exception):  # noqa: BLE001 — templates without a system role
        # fall back to everything before the user content in the user-only probe
        instruction_part = probe.split(U, 1)[0]

    return instruction_part, response_part


def _bf16_supported() -> bool:
    """bf16 on Ampere+ (A100/L4) is more stable than fp16; T4 lacks it. Falls back to
    fp16 when bf16 is unavailable."""
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        return False


def train_with_unsloth(
    *,
    model_name: str,
    rows: list[dict],
    output_dir: Path,
    max_steps: int,
    epochs: float,
    learning_rate: float,
    seq_length: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    texts = rows_to_text(rows, tokenizer)
    ds = Dataset.from_dict({"text": texts})
    bf16 = _bf16_supported()

    # Epoch-based when epochs given (real runs); max_steps only for smoke tests.
    sched = {"num_train_epochs": epochs} if epochs and epochs > 0 else {"max_steps": max_steps}
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=seq_length,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            logging_steps=5,
            save_strategy="epoch" if epochs and epochs > 0 else "no",
            report_to="none",
            bf16=bf16,
            fp16=not bf16,
            **sched,
        ),
    )

    # C1: response-only loss. Mask everything up to the assistant turn so gradient lands
    # on the answer, not the prompt. Markers are auto-detected from the tokenizer's own
    # chat template, so this works on Qwen3/ChatML, Gemma-2, Llama-3.1, … unchanged.
    try:
        instruction_part, response_part = detect_turn_markers(tokenizer)
        print(f"response-only markers: instruction={instruction_part!r} "
              f"response={response_part!r}", file=sys.stderr)
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
    except Exception as e:  # noqa: BLE001
        print(f"WARN: train_on_responses_only not applied ({e}); "
              "loss will cover full text.", file=sys.stderr)

    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def train_with_peft(
    *,
    model_name: str,
    rows: list[dict],
    output_dir: Path,
    max_steps: int,
    epochs: float,
    learning_rate: float,
    seq_length: int,
    batch_size: int,
    grad_accum: int,
) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    print(f"PEFT train on {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # C1: response-only loss. We render the prompt (system+user) alone and the full
    # conversation, tokenize both, and set labels=-100 over the prompt-length prefix so
    # loss is computed ONLY on the assistant answer. No fixed-length padding: the
    # seq2seq collator pads each batch to its own longest sequence.
    def encode(row: dict) -> dict | None:
        messages = row.get("messages")
        if not messages:
            return None
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        full_text = _apply_template(tokenizer, messages, add_generation_prompt=False)
        prompt_text = _apply_template(tokenizer, prompt_msgs, add_generation_prompt=True)
        full_ids = tokenizer(full_text, truncation=True, max_length=seq_length)["input_ids"]
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=seq_length)["input_ids"]
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(n_prompt):
            labels[i] = -100  # mask the prompt; train only on the completion
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    encoded = [e for e in (encode(r) for r in rows) if e is not None]
    tokenized = Dataset.from_list(encoded)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

    bf16 = bool(device == "cuda" and torch.cuda.is_bf16_supported())
    sched = {"num_train_epochs": epochs} if epochs and epochs > 0 else {"max_steps": max_steps}
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            logging_steps=5,
            save_strategy="epoch" if epochs and epochs > 0 else "no",
            report_to="none",
            use_cpu=(device == "cpu"),
            bf16=bf16,
            fp16=(device == "cuda" and not bf16),
            **sched,
        ),
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT on Byzantine JSONL")
    parser.add_argument("--data", default=str(ROOT / "data/byzantine/sft_junk.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out", default=str(ROOT / "models/byzantine_sft_smoke"))
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Training steps (smoke default: 30; ignored when --epochs > 0)")
    parser.add_argument("--epochs", type=float, default=0,
                        help="Epochs for a real run (e.g. 2-3); overrides --max-steps when > 0")
    parser.add_argument("--seq-length", type=int, default=1024,
                        help="Max sequence length (lower to 768 if VRAM-tight on Colab)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="per-device train batch size (4-8 on L4/A100; lower if OOM). "
                             "Dynamic padding keeps short rows cheap.")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="gradient accumulation steps; effective batch = batch-size * this")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--force-peft", action="store_true", help="Skip Unsloth even on CUDA")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run: python scripts/generate_byzantine_sft_data.py")

    rows = load_jsonl(data_path)
    if len(rows) < 50:
        print(f"Warning: only {len(rows)} rows (Day 2 target is ≥50)", file=sys.stderr)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    use_unsloth = torch.cuda.is_available() and not args.force_peft
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel  # noqa: F401
        except ImportError:
            use_unsloth = False
            print("Unsloth not installed; using PEFT fallback.", file=sys.stderr)

    if use_unsloth:
        train_with_unsloth(
            model_name=args.model,
            rows=rows,
            output_dir=output_dir,
            max_steps=args.max_steps,
            epochs=args.epochs,
            learning_rate=args.lr,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
    else:
        train_with_peft(
            model_name=args.model,
            rows=rows,
            output_dir=output_dir,
            max_steps=args.max_steps,
            epochs=args.epochs,
            learning_rate=args.lr,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )

    meta = {
        "model": args.model,
        "data": str(data_path),
        "n_rows": len(rows),
        "epochs": args.epochs,
        "max_steps": args.max_steps if not (args.epochs and args.epochs > 0) else None,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "response_only_loss": True,
        "backend": "unsloth" if use_unsloth else "peft",
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved adapter → {output_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
