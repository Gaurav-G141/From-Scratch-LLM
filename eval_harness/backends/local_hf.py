from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalHFBackend:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        adapter_path: str | None = None,
        load_4bit: bool = False,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        label = f"{model_name}+adapter" if adapter_path else model_name
        self.name = f"local:{label}"
        self.device = device or pick_device()
        print(f"Loading {label} on {self.device} (4bit={load_4bit}) ...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs: dict = {"torch_dtype": "auto"}
        if load_4bit:
            # NF4 4-bit fits a 7B on a 16 GB T4; CUDA-only. device_map handles placement,
            # so we must NOT call .to(device) afterwards.
            if self.device != "cuda":
                raise SystemExit("load_4bit=True requires CUDA")
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not load_4bit:
            self.model = self.model.to(self.device)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
