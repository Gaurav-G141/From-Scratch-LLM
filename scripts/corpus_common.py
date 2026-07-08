"""Shared corpus types and helpers for Byzantine training data pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "byzantine"
CORPUS_DIR = DATA_DIR / "corpus"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"

SYSTEM_PROMPT = (
    "You transcribe between Byzantine (Chrysanthine / New Analytical Method) notation "
    "and Western staff notation.\n\n"
    "Convert the input notation to the requested target format. Preserve musical meaning: "
    "melodic contour, mode (echos), martyria, ison (drone), microtonal alterations "
    "(diesis, fthora), and rhythmic neume modifiers (gorgon, argon). "
    "Do NOT round to 12-TET without marking approximation. "
    "Do NOT add harmony or impose 4/4.\n\n"
    "Byzantine → Western: state mode/Ni, staff pitches (D4, E4…), preserve ison line, "
    "mark microtones.\n"
    "Western → Byzantine: martyria, interval names (oligon, petastē, apostrophos…), "
    "ison as (Ν)/(Δι)/(Κε).\n\n"
    "Output notation only — no commentary."
)

USER_AGENT = "From-Scratch-LLM-corpus/1.0"


@dataclass
class ScorePair:
    id: str
    title: str
    source: str
    byzantine_url: str
    western_url: str
    echos: str = ""
    tags: list[str] = field(default_factory=list)
    byzantine_path: str = ""
    western_path: str = ""
    page_url: str = ""

    def pair_key(self) -> str:
        return f"{self.source}|{self.byzantine_url}|{self.western_url}"


def slug(text: str, *, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return text.strip("-")[:max_len] or "untitled"


def url_hash(url: str, n: int = 10) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:n]


def fetch(url: str, *, timeout: int = 60) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def download_url(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=90, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return True


def download_pair(pair: ScorePair, corpus_dir: Path = CORPUS_DIR) -> ScorePair:
    sub = corpus_dir / pair.source.replace("_", "-")
    sub.mkdir(parents=True, exist_ok=True)

    byz_dest = sub / f"{pair.id}_byz.pdf"
    west_dest = sub / f"{pair.id}_west.pdf"
    if download_url(pair.byzantine_url, byz_dest):
        pair.byzantine_path = str(byz_dest.relative_to(ROOT))
    if download_url(pair.western_url, west_dest):
        pair.western_path = str(west_dest.relative_to(ROOT))
    return pair


def write_manifest(pairs: list[ScorePair], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def load_manifest(path: Path = MANIFEST_PATH) -> list[ScorePair]:
    if not path.exists():
        return []
    pairs: list[ScorePair] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            pairs.append(ScorePair(**raw))
    return pairs


def dedupe_pairs(pairs: list[ScorePair]) -> list[ScorePair]:
    by_key: dict[str, ScorePair] = {}
    for pair in pairs:
        by_key[pair.pair_key()] = pair
    return list(by_key.values())


def row_to_sft(
    *,
    row_id: str,
    direction: str,
    user_content: str,
    assistant_content: str,
    meta: dict,
) -> dict:
    return {
        "id": row_id,
        "direction": direction,
        "status": "raw",
        **meta,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def staff_variants(stem: str) -> list[str]:
    """Common New Byzantium / DCS staff suffix variants for a PDF stem."""
    base = stem
    for suffix in ("_en_byz", "_byz", "_en_byzantine"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    candidates = [
        f"{base}_en_staff.pdf",
        f"{base}_gr_staff.pdf",
        f"{base}_gr_en_staff.pdf",
        f"{base}_en_gr_staff.pdf",
        f"{base}_staff.pdf",
    ]
    return candidates


def pdf_stem(url: str) -> str:
    return Path(urlparse(url).path).name
