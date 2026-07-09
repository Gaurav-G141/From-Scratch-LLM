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


def rows_to_text(rows: list[dict], tokenizer) -> list[str]:
    texts = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return texts


def train_with_unsloth(
    *,
    model_name: str,
    texts: list[str],
    output_dir: Path,
    max_steps: int,
    learning_rate: float,
) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
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

    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=1024,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=5,
            save_steps=max_steps,
            report_to="none",
            fp16=True,
        ),
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def train_with_peft(
    *,
    model_name: str,
    texts: list[str],
    output_dir: Path,
    max_steps: int,
    learning_rate: float,
) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )
    print(f"PEFT smoke train on {device}", file=sys.stderr)

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

    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        # No fixed-length padding here: the collator pads each batch to its own
        # longest sequence, so short fragments don't cost a full 1024-token step.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=5,
            save_steps=max_steps,
            report_to="none",
            use_cpu=(device == "cpu"),
        ),
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke SFT on Byzantine junk JSONL")
    parser.add_argument("--data", default=str(ROOT / "data/byzantine/sft_junk.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out", default=str(ROOT / "models/byzantine_sft_smoke"))
    parser.add_argument("--max-steps", type=int, default=30, help="Training steps (smoke default: 30)")
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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        texts = rows_to_text(rows, tokenizer)
        train_with_unsloth(
            model_name=args.model,
            texts=texts,
            output_dir=output_dir,
            max_steps=args.max_steps,
            learning_rate=args.lr,
        )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        texts = rows_to_text(rows, tokenizer)
        train_with_peft(
            model_name=args.model,
            texts=texts,
            output_dir=output_dir,
            max_steps=args.max_steps,
            learning_rate=args.lr,
        )

    meta = {
        "model": args.model,
        "data": str(data_path),
        "n_rows": len(rows),
        "max_steps": args.max_steps,
        "backend": "unsloth" if use_unsloth else "peft",
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved adapter → {output_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
