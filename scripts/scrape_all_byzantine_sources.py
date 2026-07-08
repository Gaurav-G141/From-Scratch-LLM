#!/usr/bin/env python3
"""Discover parallel Byzantine ↔ Western PDF pairs from reputable online sources.

Sources:
  - Cappella Romana Divine Liturgy (separate + bi-notational PDF)
  - New Byzantium (all catalog pages)
  - GOA Digital Chant Stand (Dedes / AGES — /b/ ↔ /w/ URL pairs)
  - St. Anthony's Monastery Divine Music Project (legacy index)

Usage:
  python scripts/scrape_all_byzantine_sources.py discover
  python scripts/scrape_all_byzantine_sources.py discover --download
  python scripts/scrape_all_byzantine_sources.py discover --sources cappella,newbyz,dcs
  python scripts/scrape_all_byzantine_sources.py stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.corpus_common import (  # noqa: E402
    CORPUS_DIR,
    DATA_DIR,
    MANIFEST_PATH,
    ScorePair,
    dedupe_pairs,
    download_pair,
    fetch,
    load_manifest,
    pdf_stem,
    slug,
    staff_variants,
    url_hash,
    write_manifest,
)

CAPPELLA_LITURGY_URL = "https://cappellaromana.org/divine-liturgy-music/"
CAPPELLA_BINOTATIONAL_URL = (
    "https://cappellaromana.org/wp-content/uploads/2014/04/"
    "Divine-Liturgy-in-English-binotationalscores_optimized_Divine-Liturgy-Music_Cappella-Romana.pdf"
)
NEW_BYZ_ROOT = "https://newbyz.weebly.com/"
DCS_INDEX = "https://dcs.goarch.org/goa/dcs/booksindex.html"
SAM_INDEX_BYZ = "https://music.samonastery.org/IndexBM.html"
SAM_INDEX_WEST = "https://music.samonastery.org/IndexM.html"

BYZ_MARKERS = ("-Byz", "-Byzantine", "-Byz_")
WEST_MARKERS = ("-Western", "-Staff", "-West", "Western-print", "Western-Print")

NEW_BYZ_PAGES = [
    "january.html", "february.html", "march.html", "april.html", "may.html", "june.html",
    "july.html", "august.html", "september.html", "october.html", "november.html", "december.html",
    "lenten-sundays.html", "holy-week.html", "holy-pascha.html", "palm-sunday-weekend.html",
    "ascensionpentecostall-saints.html", "triodion.html", "pentecostarion.html",
    "selected-hymns.html", "about-chant.html",
    # Extra catalog pages (discovered from site index)
    "divine-liturgy.html", "standard-liturgy.html", "hierarchical.html", "orthros-matins.html",
    "menaion.html", "other-hymns.html", "other-services.html", "sacraments.html",
    "lamentations.html", "friday-salutations.html", "pre-lenten-sundays.html",
    "pascha-weeks-2-6.html", "hymns.html", "hymns-472859.html", "hymns-163619.html",
    "carols--songs.html", "byzantine-hymnal.html", "hymnal.html", "psalms.html",
]


def _is_byz_pdf(url: str) -> bool:
    u = url.lower()
    return any(m.lower() in u for m in BYZ_MARKERS) or "/byz" in u or "_en_byz" in u or "/b/" in u


def _is_west_pdf(url: str) -> bool:
    u = url.lower()
    return any(m.lower() in u for m in WEST_MARKERS) or "_staff" in u or "/w/" in u


def _normalize_cappella_key(url: str) -> str:
    name = Path(urlparse(url).path).stem
    for marker in BYZ_MARKERS + WEST_MARKERS + ("_Divine-Liturgy-Music_Cappella-Romana",):
        name = name.replace(marker, "")
    name = re.sub(r"(?i)(byz|byzantine|western|staff|print|thyateira|pl-iv|plagal-iv)", "", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return slug(name)


def scrape_cappella_romana() -> list[ScorePair]:
    html = fetch(CAPPELLA_LITURGY_URL)
    soup = BeautifulSoup(html, "html.parser")
    byz: dict[str, tuple[str, str]] = {}
    west: dict[str, tuple[str, str]] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        full = urljoin(CAPPELLA_LITURGY_URL, href)
        title = a.get_text(strip=True) or Path(urlparse(full).path).stem
        key = _normalize_cappella_key(full)
        if _is_byz_pdf(full):
            byz[key] = (full, title)
        elif _is_west_pdf(full):
            west[key] = (full, title)

    pairs: list[ScorePair] = []
    matched_west: set[str] = set()
    for key in sorted(set(byz) & set(west)):
        b_url, b_title = byz[key]
        w_url, _ = west[key]
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
                page_url=CAPPELLA_LITURGY_URL,
            )
        )

    # Bi-notational combined PDF (same content, side-by-side layout)
    pairs.append(
        ScorePair(
            id="cappella_divine-liturgy-binotational",
            title="Divine Liturgy in English (bi-notational combined PDF)",
            source="cappella_romana",
            byzantine_url=CAPPELLA_BINOTATIONAL_URL,
            western_url=CAPPELLA_BINOTATIONAL_URL,
            tags=["liturgy", "pdf", "side_by_side", "binotational_single"],
            page_url=CAPPELLA_LITURGY_URL,
        )
    )
    return pairs


def _clean_nb_title(text: str) -> str:
    text = re.sub(r"\s*GS\s*ES.*$", "", text, flags=re.I)
    text = re.sub(r"To view PDF files.*$", "", text, flags=re.I)
    return text.strip()[:120]


def scrape_new_byzantium() -> list[ScorePair]:
    pairs: list[ScorePair] = []
    all_pdfs: dict[str, str] = {}  # url -> title hint

    for page_name in NEW_BYZ_PAGES:
        page_url = urljoin(NEW_BYZ_ROOT, page_name)
        try:
            html = fetch(page_url, timeout=45)
        except Exception as exc:
            print(f"  skip NB page {page_name}: {exc}", file=sys.stderr)
            continue
        soup = BeautifulSoup(html, "html.parser")

        # Block-level GS/ES + EB triplets
        for block in soup.find_all(["p", "li", "div", "td"]):
            text = block.get_text(" ", strip=True)
            if len(text) < 8:
                continue
            gs_url = eb_url = es_url = ""
            for a in block.find_all("a", href=True):
                label = a.get_text(strip=True).upper()
                href = urljoin(page_url, a["href"])
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
                title = _clean_nb_title(text) or pdf_stem(eb_url)
                pid = f"newbyz_{slug(title) or url_hash(eb_url)}"
                pairs.append(
                    ScorePair(
                        id=pid,
                        title=title,
                        source="new_byzantium",
                        byzantine_url=eb_url,
                        western_url=staff_url,
                        tags=["hymn", "pdf", "triplet"],
                        page_url=page_url,
                    )
                )

        # Collect all PDFs on page for stem pairing
        for a in soup.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            if href.lower().endswith(".pdf"):
                all_pdfs[href] = _clean_nb_title(a.get_text(strip=True)) or pdf_stem(href)

    pdf_set = set(all_pdfs)
    for url, hint in all_pdfs.items():
        if not ("_en_byz" in url.lower() or url.lower().endswith("_byz.pdf")):
            continue
        stem = pdf_stem(url)
        base = stem
        for suffix in ("_en_byz.pdf", "_byz.pdf"):
            if base.lower().endswith(suffix):
                base = base[: -len(suffix)]
                break
        west_url = ""
        for cand in staff_variants(base):
            full = urljoin(NEW_BYZ_ROOT, f"uploads/1/4/7/1/147110798/{cand}")
            if full in pdf_set:
                west_url = full
                break
            # also try same directory as byz url
            dir_url = url.rsplit("/", 1)[0] + "/" + cand
            if dir_url in pdf_set:
                west_url = dir_url
                break
        if not west_url:
            continue
        title = hint or base.replace("_", " ")
        pairs.append(
            ScorePair(
                id=f"newbyz_{url_hash(url)}",
                title=title,
                source="new_byzantium",
                byzantine_url=url,
                western_url=west_url,
                tags=["hymn", "pdf", "stem_pair"],
                page_url=NEW_BYZ_ROOT,
            )
        )
    return pairs


DCS_PDF_RE = re.compile(
    r"https?://dcs\.goarch\.org/media/[^\s\"']+?/(?:b|w)/[^\s\"']+\.pdf"
)


def scrape_dcs() -> list[ScorePair]:
    html = fetch(DCS_INDEX, timeout=120)
    all_urls = set(DCS_PDF_RE.findall(html))
    west_urls = {u for u in all_urls if "/w/" in u}
    byz_urls = sorted(u for u in all_urls if "/b/" in u)
    pairs: list[ScorePair] = []
    for b_url in byz_urls:
        w_url = b_url.replace("/b/", "/w/")
        if w_url not in west_urls:
            continue
        stem = Path(urlparse(b_url).path).stem
        title = stem.replace("_", " ").replace("-", " ")
        pairs.append(
            ScorePair(
                id=f"dcs_{slug(stem) or url_hash(b_url)}",
                title=title,
                source="goa_dcs",
                byzantine_url=b_url,
                western_url=w_url,
                tags=["liturgy", "pdf", "dedes", "dcs"],
                page_url=DCS_INDEX,
            )
        )
    return pairs


def _sam_normalize(name: str) -> str:
    name = re.sub(r"(?i)^bm", "m", name)
    name = re.sub(r"(?i)finale\s*2003\s*-\s*\[?", "", name)
    name = re.sub(r"\]?$", "", name)
    name = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return name.strip("-")


def _sam_is_byz_stem(stem: str) -> bool:
    low = stem.lower()
    return low.startswith("bm") or bool(re.match(r"^b\d", low))


def _sam_is_west_stem(stem: str) -> bool:
    low = stem.lower()
    return "finale 2003" in low or bool(re.match(r"^m\d", low))


def _sam_resolve_pdf(href: str, page_url: str) -> tuple[str, str] | tuple[None, None]:
    """Return (pdf_url, stem) for direct .pdf links or download.php?file=… handlers."""
    if "download.php" in href and "file=" in href:
        full = urljoin(page_url, href)
        qs = parse_qs(urlparse(full).query)
        if "file" not in qs:
            return None, None
        rel = unquote(qs["file"][0])
        if not rel.lower().endswith(".pdf"):
            return None, None
        return urljoin(SAM_INDEX_BYZ, rel), Path(rel).stem
    full = urljoin(page_url, href)
    if not full.lower().endswith(".pdf"):
        return None, None
    return full, unquote(Path(urlparse(full).path).stem)


def _sam_crawl_pdfs(*, max_pages: int = 400) -> tuple[dict[str, str], dict[str, str]]:
    """BFS from both indexes; classify PDFs by filename into byzantine vs western buckets."""
    base = f"{urlparse(SAM_INDEX_BYZ).scheme}://{urlparse(SAM_INDEX_BYZ).netloc}/"
    seen: set[str] = set()
    queue: list[str] = ["IndexBM.html", "IndexM.html"]
    byz: dict[str, str] = {}
    west: dict[str, str] = {}

    while queue and len(seen) < max_pages:
        page = queue.pop(0)
        if page in seen:
            continue
        seen.add(page)
        page_url = urljoin(base, page)
        try:
            html = fetch(page_url, timeout=45)
        except Exception:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0]
            if not href or href.startswith("mailto:"):
                continue
            pdf_url, stem = _sam_resolve_pdf(href, page_url)
            if pdf_url is not None:
                key = _sam_normalize(stem)
                if key:
                    if _sam_is_byz_stem(stem):
                        byz[key] = pdf_url
                    elif _sam_is_west_stem(stem):
                        west[key] = pdf_url
                continue
            full = urljoin(page_url, href)
            if urlparse(full).netloc != urlparse(base).netloc:
                continue
            rel = full.split("music.samonastery.org/")[-1]
            if rel.endswith((".htm", ".html")) and rel not in seen:
                queue.append(rel)
    return byz, west


def scrape_st_anthonys() -> list[ScorePair]:
    pairs: list[ScorePair] = []
    try:
        byz, west = _sam_crawl_pdfs()
    except Exception as exc:
        print(f"  skip St Anthony's: {exc}", file=sys.stderr)
        return pairs

    for key in sorted(set(byz) & set(west)):
        b_url, w_url = byz[key], west[key]
        if b_url == w_url:
            continue
        pairs.append(
            ScorePair(
                id=f"sam_{key[:60]}",
                title=key.replace("-", " "),
                source="st_anthonys",
                byzantine_url=b_url,
                western_url=w_url,
                tags=["hymn", "pdf", "chrys"],
                page_url=SAM_INDEX_BYZ,
            )
        )
    return pairs


SCRAPERS = {
    "cappella": scrape_cappella_romana,
    "newbyz": scrape_new_byzantium,
    "dcs": scrape_dcs,
    "sam": scrape_st_anthonys,
}


def discover(sources: list[str]) -> list[ScorePair]:
    pairs: list[ScorePair] = []
    for name in sources:
        fn = SCRAPERS.get(name)
        if not fn:
            continue
        print(f"Scraping {name}...", file=sys.stderr)
        found = fn()
        print(f"  → {len(found)} pairs", file=sys.stderr)
        pairs.extend(found)
    return dedupe_pairs(pairs)


def cmd_discover(args: argparse.Namespace) -> None:
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    pairs = discover(sources)
    by_source: dict[str, int] = {}
    for p in pairs:
        by_source[p.source] = by_source.get(p.source, 0) + 1

    print(f"Discovered {len(pairs)} unique parallel pairs:")
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}")

    if args.download:
        ok = 0
        for i, pair in enumerate(pairs, 1):
            print(f"  [{i}/{len(pairs)}] {pair.title[:50]}...", file=sys.stderr)
            download_pair(pair, CORPUS_DIR)
            if pair.byzantine_path and pair.western_path:
                ok += 1
        print(f"Downloaded {ok}/{len(pairs)} complete pairs")

    write_manifest(pairs, Path(args.out))
    stats = {"total": len(pairs), "by_source": by_source, "out": str(args.out)}
    stats_path = DATA_DIR / "manifest_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote manifest → {args.out}")
    print(f"Stats → {stats_path}")


def cmd_stats(args: argparse.Namespace) -> None:
    pairs = load_manifest(Path(args.manifest))
    by_source: dict[str, int] = {}
    for p in pairs:
        by_source[p.source] = by_source.get(p.source, 0) + 1
    print(json.dumps({"total": len(pairs), "by_source": by_source}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Byzantine ↔ Western parallel score corpus")
    sub = parser.add_subparsers(dest="command", required=True)

    disc = sub.add_parser("discover", help="Discover pairs from all sources")
    disc.add_argument(
        "--sources",
        default="cappella,newbyz,dcs,sam",
        help="Comma-separated: cappella,newbyz,dcs,sam",
    )
    disc.add_argument("--download", action="store_true")
    disc.add_argument("--out", default=str(MANIFEST_PATH))
    disc.set_defaults(func=cmd_discover)

    stats = sub.add_parser("stats", help="Summarize manifest.jsonl")
    stats.add_argument("--manifest", default=str(MANIFEST_PATH))
    stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
