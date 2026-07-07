from __future__ import annotations

import re

# Phrases that often signal passive-aggression or backhanded tone in "sanitized" text.
PASSIVE_AGGRESSIVE_PATTERNS: list[tuple[str, str]] = [
    (r"\bwith all due respect\b", "passive-aggressive opener"),
    (r"\bno offense\b", "passive-aggressive opener"),
    (r"\bjust saying\b", "passive-aggressive distancing"),
    (r"\bif we're being (honest|real)\b", "passive-aggressive framing"),
    (r"\bif you say so\b", "dismissive/passive-aggressive"),
    (r"\bmust be nice\b", "backhanded/sarcastic"),
    (r"\bgood luck with that\b", "dismissive/sarcastic"),
    (r"\bas I'm sure you know\b", "condescending"),
    (r"\bclearly you\b", "condescending"),
    (r"\bobviously you\b", "condescending"),
    (r"\bi'm sure it makes sense to you\b", "condescending"),
    (r"\binteresting (choice|approach|decision)\b", "backhanded"),
    (r"\bwhatever you think is best\b", "passive-aggressive capitulation"),
    (r"\bsure, if you\b", "sarcastic agreement"),
    (r"\bthanks for (finally|at least)\b", "sarcastic gratitude"),
    (r"\bnice (going|job|work)\b", "sarcastic praise"),
    (r"\bwhat a (great|brilliant|fantastic) (idea|move|plan)\b", "sarcastic praise"),
    (r"\bfor once\b", "dismissive dig"),
    (r"\bas always\b", "accusatory pattern"),
    (r"\bsome people\b", "veiled group insult when criticizing"),
]


def check_passive_aggressive(text: str) -> list[str]:
    failures: list[str] = []
    for pattern, label in PASSIVE_AGGRESSIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            failures.append(f"Passive-aggressive or backhanded tone detected ({label}): /{pattern}/")
    return failures
