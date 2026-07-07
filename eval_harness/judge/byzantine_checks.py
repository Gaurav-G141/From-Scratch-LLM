from __future__ import annotations

import re

# Western bias patterns: model imposes harmony, wrong key simplification, etc.
WESTERN_BIAS_PATTERNS = [
    (r"\b(chord|triad|harmony|voicing|bass line)\b", "invented Western harmony"),
    (r"\b(C major|G major|F major|key of C)\b", "oversimplified to common-practice key"),
    (r"\b(4/4|3/4|6/8)\b", "imposed fixed time signature without source basis"),
    (r"\b(quarter note|half note|whole note)\b", "prose rhythm instead of neume mapping"),
]

# Required structural markers for text-format transcriptions
BYZ_MARKERS = re.compile(
    r"(martyria|echos|mode|ison|\(Ν\)|\(Π\)|\(Δι\)|\(Κε\)|oligon|petaste|apostrophos|"
    r"kentema|fthora|gorgon|argon|diesis|ison)",
    re.I,
)
WEST_MARKERS = re.compile(
    r"(\b[A-G][#b♭♯]?\d\b|\b\d/\d\b|treble|bass clef|staff|key sig|time sig|"
    r"whole|half|quarter|eighth|sharp|flat|natural)",
    re.I,
)


def check_byzantine_transcription(
    *,
    model_output: str,
    direction: str,
    forbidden_extra: list[str] | None = None,
) -> list[str]:
    """Deterministic checks for notation transcription outputs."""
    failures: list[str] = []
    text = model_output.strip()
    if not text:
        return ["empty output"]

    for pattern, msg in WESTERN_BIAS_PATTERNS:
        if re.search(pattern, text, re.I):
            failures.append(f"Western bias: {msg}")

    if forbidden_extra:
        for pattern in forbidden_extra:
            if re.search(pattern, text, re.I):
                failures.append(f"matched forbidden pattern: {pattern}")

    if direction == "byz_to_west":
        if not WEST_MARKERS.search(text):
            failures.append("byz→west output lacks Western staff notation markers")
    elif direction == "west_to_byz":
        if not BYZ_MARKERS.search(text):
            failures.append("west→byz output lacks Byzantine notation markers")
    elif direction == "roundtrip_check":
        pass

    # Discourage commentary
    if re.search(r"\b(the answer is|note that|keep in mind|I (transcribed|converted))\b", text, re.I):
        failures.append("commentary detected — output should be notation only")

    return failures
