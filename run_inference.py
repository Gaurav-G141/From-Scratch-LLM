"""
Day 1 (first part): get a small open base model running inference.

Loads a small open-weights instruct model and lets it respond to a prompt.
This is intentionally minimal: no fine-tuning, no eval, no data generation.
It only proves that the base model runs and responds.

Usage:
    python run_inference.py                       # single demo prompt
    python run_inference.py "Your prompt here"    # one-off prompt
    python run_inference.py --chat                # interactive chat loop
"""

import argparse
import sys

from eval_harness.backends.local_hf import DEFAULT_MODEL_NAME, LocalHFBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a small open base model.")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt to send to the model.")
    parser.add_argument("--chat", action="store_true", help="Interactive chat loop.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model id.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    backend = LocalHFBackend(model_name=args.model)

    if args.chat:
        print("Chat mode. Type 'exit' or Ctrl-C to quit.\n")
        try:
            while True:
                prompt = input("you> ").strip()
                if prompt.lower() in {"exit", "quit"}:
                    break
                if not prompt:
                    continue
                reply = backend.generate(
                    "You are a helpful assistant.",
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                print(f"model> {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print()
        return

    prompt = args.prompt or "In one sentence, explain what a large language model is."
    print(f"\nPrompt: {prompt}\n")
    reply = backend.generate(
        "You are a helpful assistant.",
        prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Response:\n{reply}\n")


if __name__ == "__main__":
    main()
