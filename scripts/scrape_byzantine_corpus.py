#!/usr/bin/env python3
"""Discover and optionally download Western ↔ Byzantine parallel score pairs.

Sources:
  - Cappella Romana Divine Liturgy (explicit Byz/Western PDF pairs)
  - New Byzantium selected hymns (GS/ES/EB triplet pages)

Outputs:
  - data/byzantine/manifest.jsonl   — one record per aligned pair
  - data/byzantine/corpus/          — downloaded PDFs (with --download)

Usage:
  python scripts/scrape_byzantine_corpus.py discover
  python scripts/scrape_byzantine_corpus.py discover --download
  python scripts/build_unseen_scenario_bank.py --all   # held-out from New Byzantium + Apolytikia
  python scripts/scrape_byzantine_corpus.py export-scenarios --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "byzantine"
CORPUS_DIR = DATA_DIR / "corpus"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"

CAPPELLA_LITURGY_URL = "https://cappellaromana.org/divine-liturgy-music/"
NEW_BYZ_HYMNS_URL = "https://newbyz.weebly.com/selected-hymns.html"

BYZ_MARKERS = ("-Byz", "-Byzantine", "-Byz_")
WEST_MARKERS = ("-Western", "-Staff", "-West", "Western-print", "Western-Print")


@dataclass
class ScorePair:
    id: str
    title: str
    source: str
    byzantine_url: str
    western_url: str
    echos: str = ""
    tags: list[str] | None = None
    byzantine_path: str = ""
    western_path: str = ""

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return text.strip("-")[:80] or "untitled"


def _fetch(url: str) -> str:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "From-Scratch-LLM-corpus/1.0"})
    resp.raise_for_status()
    return resp.text


def _is_byz_pdf(url: str) -> bool:
    u = url.lower()
    return any(m.lower() in u for m in BYZ_MARKERS) or "/byz" in u


def _is_west_pdf(url: str) -> bool:
    u = url.lower()
    return any(m.lower() in u for m in WEST_MARKERS)


def _normalize_key(url: str) -> str:
    name = Path(urlparse(url).path).stem
    for marker in BYZ_MARKERS + WEST_MARKERS + ("_Divine-Liturgy-Music_Cappella-Romana",):
        name = name.replace(marker, "")
    name = re.sub(r"(?i)(byz|byzantine|western|staff|print|thyateira|pl-iv|plagal-iv)", "", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return _slug(name)


def _fuzzy_pair_keys(byz: dict[str, tuple[str, str]], west: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    """Match keys that share a significant token (e.g. prokeimenon, cherubic)."""
    pairs: list[tuple[str, str]] = []
    used_west: set[str] = set()
    for bk in byz:
        b_tokens = set(bk.split("-")) - {"", "cappella", "byz", "western", "print", "communion"}
        best_wk = ""
        best_overlap = 0
        for wk in west:
            if wk in used_west:
                continue
            w_tokens = set(wk.split("-")) - {"", "cappella", "byz", "western", "print", "communion"}
            overlap = len(b_tokens & w_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_wk = wk
        if best_overlap >= 1 and best_wk:
            used_west.add(best_wk)
            pairs.append((bk, best_wk))
    return pairs


def scrape_cappella_romana() -> list[ScorePair]:
    html = _fetch(CAPPELLA_LITURGY_URL)
    soup = BeautifulSoup(html, "html.parser")
    byz: dict[str, tuple[str, str]] = {}
    west: dict[str, tuple[str, str]] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        full = urljoin(CAPPELLA_LITURGY_URL, href)
        title = a.get_text(strip=True) or Path(urlparse(full).path).stem
        key = _normalize_key(full)
        if _is_byz_pdf(full):
            byz[key] = (full, title)
        elif _is_west_pdf(full):
            west[key] = (full, title)

    pairs: list[ScorePair] = []
    matched_west: set[str] = set()
    for key in sorted(set(byz) & set(west)):
        b_url, b_title = byz[key]
        w_url, w_title = west[key]
        title = b_title.replace("Byzantine Notation:", "").replace("Staff Notation:", "").strip()
        matched_west.add(key)
        pairs.append(
            ScorePair(
                id=f"cappella_{key}",
                title=title or key,
                source="cappella_romana",
                byzantine_url=b_url,
                western_url=w_url,
                tags=["liturgy", "pdf", "bi-notational"],
            )
        )

    # Fuzzy match remaining (Cherubic, Anaphora, Responsorial Psalm, etc.)
    remaining_byz = {k: v for k, v in byz.items() if k not in set(byz) & set(west)}
    remaining_west = {k: v for k, v in west.items() if k not in matched_west}
    for bk, wk in _fuzzy_pair_keys(remaining_byz, remaining_west):
        if wk in matched_west:
            continue
        b_url, b_title = byz[bk]
        w_url, w_title = west[wk]
        title = b_title.replace("Byzantine Notation:", "").strip() or bk
        matched_west.add(wk)
        pairs.append(
            ScorePair(
                id=f"cappella_{bk}",
                title=title or bk,
                source="cappella_romana",
                byzantine_url=b_url,
                western_url=w_url,
                tags=["liturgy", "pdf", "bi-notational", "fuzzy_match"],
            )
        )
    return pairs


def scrape_new_byzantium() -> list[ScorePair]:
    """Parse New Byzantium pages for GS (staff) + EB (Byzantine) links on same hymn block."""
    html = _fetch(NEW_BYZ_HYMNS_URL)
    soup = BeautifulSoup(html, "html.parser")
    pairs: list[ScorePair] = []

    # Weebly pages embed hymn blocks as paragraphs with GS/ES/EB links nearby.
    blocks = soup.find_all(["p", "li", "div"])
    for block in blocks:
        text = block.get_text(" ", strip=True)
        if not text or len(text) < 10:
            continue
        gs_url = eb_url = es_url = ""
        for a in block.find_all("a", href=True):
            label = a.get_text(strip=True).upper()
            href = urljoin(NEW_BYZ_HYMNS_URL, a["href"])
            if not href.lower().endswith(".pdf"):
                continue
            if label == "GS":
                gs_url = href
            elif label == "EB":
                eb_url = href
            elif label == "ES":
                es_url = href

        staff_url = gs_url or es_url
        if staff_url and eb_url:
            hymn_title = re.sub(r"\s*GS\s*ES.*$", "", text, flags=re.I).strip()[:120]
            key = _slug(hymn_title)
            pairs.append(
                ScorePair(
                    id=f"newbyz_{key}",
                    title=hymn_title or key,
                    source="new_byzantium",
                    byzantine_url=eb_url,
                    western_url=staff_url,
                    tags=["hymn", "pdf", "triplet"],
                )
            )

    # Deduplicate by byzantine_url
    seen: set[str] = set()
    unique: list[ScorePair] = []
    for p in pairs:
        if p.byzantine_url in seen:
            continue
        seen.add(p.byzantine_url)
        unique.append(p)
    return unique


def discover_pairs() -> list[ScorePair]:
    pairs = scrape_cappella_romana() + scrape_new_byzantium()
    # Final dedupe by pair id
    by_id: dict[str, ScorePair] = {}
    for p in pairs:
        by_id[p.id] = p
    return list(by_id.values())


def download_pair(pair: ScorePair, corpus_dir: Path) -> ScorePair:
    corpus_dir.mkdir(parents=True, exist_ok=True)

    def _dl(url: str, suffix: str) -> str:
        name = f"{pair.id}_{suffix}.pdf"
        dest = corpus_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            return str(dest.relative_to(ROOT))
        resp = requests.get(url, timeout=60, headers={"User-Agent": "From-Scratch-LLM-corpus/1.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return str(dest.relative_to(ROOT))

    pair.byzantine_path = _dl(pair.byzantine_url, "byz")
    pair.western_path = _dl(pair.western_url, "west")
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
            pairs.append(ScorePair(**json.loads(line)))
    return pairs


def cmd_discover(args: argparse.Namespace) -> None:
    pairs = discover_pairs()
    print(f"Discovered {len(pairs)} parallel pairs "
          f"({sum(1 for p in pairs if p.source == 'cappella_romana')} Cappella, "
          f"{sum(1 for p in pairs if p.source == 'new_byzantium')} New Byzantium)")

    if args.download:
        for i, pair in enumerate(pairs, 1):
            print(f"  [{i}/{len(pairs)}] {pair.title[:60]}...")
            download_pair(pair, CORPUS_DIR)

    write_manifest(pairs)
    print(f"Wrote manifest → {MANIFEST_PATH}")


def cmd_export_scenarios(args: argparse.Namespace) -> None:
    """Export manifest entries as harness scenario stubs (PDF refs; text TBD via OCR)."""
    pairs = load_manifest()
    if not pairs:
        print("No manifest found. Run: python scripts/scrape_byzantine_corpus.py discover", file=sys.stderr)
        sys.exit(1)

    out = ROOT / "scenarios" / "byzantine_transcription_corpus_stubs.yaml"
    lines = [
        "# Auto-generated stubs from parallel PDF corpus. "
        "Replace input/context after OCR or manual transcription.\n"
    ]
    for pair in pairs[: args.limit]:
        lines.append(f"- id: {pair.id}")
        lines.append(f"  tags: [{pair.source}, pdf_stub, corpus]")
        lines.append("  direction: byz_to_west")
        lines.append(f"  source_url: {pair.byzantine_url}")
        lines.append(f"  reference_pdf_west: {pair.western_url}")
        if pair.byzantine_path:
            lines.append(f"  byzantine_pdf: {pair.byzantine_path}")
            lines.append(f"  western_pdf: {pair.western_path}")
        lines.append(f"  input: |")
        lines.append(f"    [CORPUS STUB] Transcribe: {pair.title}")
        lines.append(f"    Byzantine PDF: {pair.byzantine_url}")
        lines.append(f"  context: |")
        lines.append(f"    Gold reference (Western staff PDF): {pair.western_url}")
        lines.append(f"    Title: {pair.title}. Judge melodic equivalence, mode (echos),")
        lines.append(f"    martyria, ison, and microtonal spelling vs reference transcription school.")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {min(len(pairs), args.limit)} stubs → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Byzantine ↔ Western parallel corpus tools")
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="Find parallel PDF pairs and write manifest")
    discover.add_argument("--download", action="store_true", help="Download PDFs to data/byzantine/corpus/")
    discover.set_defaults(func=cmd_discover)

    export = sub.add_parser("export-scenarios", help="Export manifest as scenario stubs")
    export.add_argument("--limit", type=int, default=20)
    export.set_defaults(func=cmd_export_scenarios)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
