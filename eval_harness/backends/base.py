from __future__ import annotations

from typing import Protocol


class ModelBackend(Protocol):
    name: str

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        ...
