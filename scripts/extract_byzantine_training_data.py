#!/usr/bin/env python3
"""Extract SFT training rows from parallel PDF corpus via vision LLM.

Reads manifest.jsonl, renders PDF pages to PNG, extracts text-encoded fragments
in both directions, writes raw JSONL for human/LLM pruning.

Usage:
  python scripts/extract_byzantine_training_data.py --limit-pairs 20
  python scripts/extract_byzantine_training_data.py --all --fragments-per-page 2
  python scripts/extract_byzantine_training_data.py --resume

Output:
  data/byzantine/sft_raw.jsonl       — training rows (status=raw)
  data/byzantine/extract_log.jsonl   — per-attempt log for pruning/debug
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.corpus_common import (  # noqa: E402
    CORPUS_DIR,
    DATA_DIR,
    MANIFEST_PATH,
    ScorePair,
    download_pair,
    load_manifest,
    row_to_sft,
)

OUT_RAW = DATA_DIR / "sft_raw.jsonl"
OUT_LOG = DATA_DIR / "extract_log.jsonl"
PNG_DIR = DATA_DIR / "corpus" / "png"

EXTRACT_PROMPT = """You extract Byzantine ↔ Western transcription TRAINING pairs from score images.

You will see TWO images from the same hymn: Byzantine neume notation and Western staff notation.

Extract up to {max_fragments} SHORT fragments (4–8 neumes each) from melodic passages visible on this page.
Prefer: openings after ison/martyria, formula phrases, fthora passages, ison shifts.

Return ONLY valid JSON:
{{
  "fragments": [
    {{
      "fragment_id": "opening",
      "echos": "Mode label from header",
      "byzantine_text": "multi-line Byzantine side for eval input",
      "western_text": "multi-line Western staff transcription",
      "notes": "diesis/fthora/gorgon if any"
    }}
  ]
}}

Rules for byzantine_text (when used as byz→west input):
- Start with: Direction: Byzantine → Western
- Include mode header [Mode ..., Ni = ...], (Ni) ison line, neume chain with | separators
- English neume names: oligon, petastē, apostrophos, kentēma, ison, gorgon, argon, ypsilē pnevma, elaphron

Rules for western_text:
- Mode line, Ison: pitch, pitch sequence (D4 E4 F#4...)
- Mark microtones with ↑ ↓ when visible
- Match the staff image exactly for those neumes

If the page is title text, lyrics only, or illegible, return {{"fragments": []}}.
Do not invent pitches not visible in the images.
"""

WEST_TO_BYZ_PREFIX = "Direction: Western → Byzantine\n"
BYZ_TO_WEST_PREFIX = "Direction: Byzantine → Western\n"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def pdf_page_png(pdf_path: Path, page_idx: int, out: Path, zoom: float = 2.0) -> bool:
    if out.exists() and out.stat().st_size > 0:
        return True
    if not pdf_path.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    if page_idx >= len(doc):
        doc.close()
        return False
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    pix.save(out)
    doc.close()
    return True


def load_done_ids(log_path: Path) -> set[str]:
    done: set[str] = set()
    if not log_path.exists():
        return done
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add(rec.get("job_id", ""))
    return done


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_page_fragments(
    client: OpenAI,
    *,
    pair: ScorePair,
    page_idx: int,
    byz_png: Path,
    west_png: Path,
    model: str,
    max_fragments: int,
) -> list[dict]:
    side_by_side = pair.byzantine_url == pair.western_url
    prompt = EXTRACT_PROMPT.format(max_fragments=max_fragments)
    content: list[dict] = [
        {"type": "text", "text": f"Hymn: {pair.title} | source: {pair.source} | page: {page_idx}"},
    ]
    if side_by_side:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(byz_png)}"}},
        )
        content.append({"type": "text", "text": "Bi-notational PDF: Byzantine neumes and Western staff on same page."})
    else:
        content.extend(
            [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(byz_png)}"}},
                {"type": "text", "text": "Image 1: Byzantine neume notation"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(west_png)}"}},
                {"type": "text", "text": "Image 2: Western staff notation"},
            ]
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=2000,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    data = json.loads(m.group())
    frags = data.get("fragments") or []
    return frags if isinstance(frags, list) else []


def fragments_to_rows(
    pair: ScorePair,
    page_idx: int,
    fragments: list[dict],
    *,
    model: str,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    logs: list[dict] = []
    base_meta = {
        "pair_id": pair.id,
        "source": pair.source,
        "title": pair.title,
        "page_idx": page_idx,
        "byzantine_url": pair.byzantine_url,
        "western_url": pair.western_url,
        "extraction_model": model,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    for i, frag in enumerate(fragments):
        if not isinstance(frag, dict):
            continue
        byz = str(frag.get("byzantine_text") or "").strip()
        west = str(frag.get("western_text") or "").strip()
        if not byz or not west:
            continue
        fid = str(frag.get("fragment_id") or f"f{i}")

        if not byz.startswith("Direction:"):
            byz = BYZ_TO_WEST_PREFIX + byz
        if not west.startswith("Direction:"):
            west = west  # western reference stays as notation block

        row_id = f"{pair.id}_p{page_idx}_{fid}"
        meta = {**base_meta, "fragment_id": fid, "echos": frag.get("echos", ""), "notes": frag.get("notes", "")}

        # byz → west
        rows.append(
            row_to_sft(
                row_id=f"{row_id}_b2w",
                direction="byz_to_west",
                user_content=byz,
                assistant_content=west,
                meta=meta,
            )
        )
        # west → byz (swap)
        west_input = west if west.startswith("Direction:") else WEST_TO_BYZ_PREFIX + west
        byz_out = byz.replace(BYZ_TO_WEST_PREFIX, "").strip()
        if byz_out.startswith("Direction:"):
            byz_out = re.sub(r"^Direction:[^\n]*\n", "", byz_out).strip()
        rows.append(
            row_to_sft(
                row_id=f"{row_id}_w2b",
                direction="west_to_byz",
                user_content=west_input,
                assistant_content=byz_out if byz_out.startswith("[") else f"[Mode ...]\n{byz_out}",
                meta=meta,
            )
        )

    return rows, logs


def process_pair(
    client: OpenAI,
    pair: ScorePair,
    *,
    model: str,
    max_pages: int,
    max_fragments: int,
    zoom: float,
    download: bool,
    done_pages: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    if download:
        download_pair(pair, CORPUS_DIR)

    side_by_side = pair.byzantine_url == pair.western_url
    byz_pdf = ROOT / pair.byzantine_path if pair.byzantine_path else None
    west_pdf = ROOT / pair.western_path if pair.western_path else None

    if not byz_pdf or not byz_pdf.exists():
        return [], [{"job_id": pair.id, "status": "skip", "reason": "missing_byz_pdf"}]

    if side_by_side:
        west_pdf = byz_pdf
    elif not west_pdf or not west_pdf.exists():
        return [], [{"job_id": pair.id, "status": "skip", "reason": "missing_west_pdf"}]

    all_rows: list[dict] = []
    all_logs: list[dict] = []

    doc = fitz.open(byz_pdf)
    n_pages = min(len(doc), max_pages)
    doc.close()

    for page_idx in range(n_pages):
        job_id = f"{pair.id}_p{page_idx}"
        if done_pages and job_id in done_pages:
            continue
        byz_png = PNG_DIR / pair.source / f"{pair.id}_p{page_idx}_byz.png"
        west_png = PNG_DIR / pair.source / f"{pair.id}_p{page_idx}_west.png"

        if not pdf_page_png(byz_pdf, page_idx, byz_png, zoom=zoom):
            all_logs.append({"job_id": job_id, "status": "skip", "reason": "png_byz_failed"})
            continue
        if side_by_side:
            west_png = byz_png
        elif not pdf_page_png(west_pdf, page_idx, west_png, zoom=zoom):
            all_logs.append({"job_id": job_id, "status": "skip", "reason": "png_west_failed"})
            continue

        try:
            fragments = extract_page_fragments(
                client,
                pair=pair,
                page_idx=page_idx,
                byz_png=byz_png,
                west_png=west_png,
                model=model,
                max_fragments=max_fragments,
            )
        except Exception as exc:
            all_logs.append({"job_id": job_id, "status": "error", "reason": str(exc)})
            continue

        rows, _ = fragments_to_rows(pair, page_idx, fragments, model=model)
        all_rows.extend(rows)
        all_logs.append(
            {
                "job_id": job_id,
                "status": "ok" if rows else "empty",
                "fragments": len(fragments),
                "rows": len(rows),
                "pair_id": pair.id,
                "page_idx": page_idx,
            }
        )

    return all_rows, all_logs


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--out", default=str(OUT_RAW))
    parser.add_argument("--log", default=str(OUT_LOG))
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--limit-pairs", type=int, default=0, help="0 = all")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N pairs after filters")
    parser.add_argument("--shard", default="", help="Parallel shard I/N, e.g. 0/6")
    parser.add_argument("--max-pages", type=int, default=2, help="Pages per PDF to scan")
    parser.add_argument("--fragments-per-page", type=int, default=2)
    parser.add_argument("--source", default="", help="Filter manifest by source, e.g. cappella_romana")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip pairs already logged ok")
    parser.add_argument("--fresh", action="store_true", help="Truncate output files before run")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY required for vision extraction")

    out_path = Path(args.out)
    log_path = Path(args.log)
    if args.fresh:
        out_path.write_text("", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")

    pairs = load_manifest(Path(args.manifest))
    if args.source:
        pairs = [p for p in pairs if p.source == args.source]
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/", 1))
        if shard_i < 0 or shard_n < 1 or shard_i >= shard_n:
            raise SystemExit(f"Invalid --shard {args.shard!r}; use e.g. 0/4")
        pairs = pairs[shard_i::shard_n]
    else:
        if args.offset > 0:
            pairs = pairs[args.offset :]
        if args.limit_pairs > 0:
            pairs = pairs[: args.limit_pairs]

    done = load_done_ids(log_path) if args.resume else set()
    client = OpenAI()

    total_rows = 0
    for i, pair in enumerate(pairs, 1):
        # Skip if all pages for this pair already done
        if args.resume and all(f"{pair.id}_p{p}" in done for p in range(args.max_pages)):
            print(f"[{i}/{len(pairs)}] skip {pair.id} (done)", file=sys.stderr)
            continue

        print(f"[{i}/{len(pairs)}] extract {pair.id} — {pair.title[:50]}", file=sys.stderr)
        rows, logs = process_pair(
            client,
            pair,
            model=args.model,
            max_pages=args.max_pages,
            max_fragments=args.fragments_per_page,
            zoom=2.0,
            download=args.download,
            done_pages=done if args.resume else None,
        )
        for row in rows:
            append_jsonl(out_path, row)
        for log in logs:
            append_jsonl(log_path, log)
        total_rows += len(rows)
        print(f"  +{len(rows)} rows ({total_rows} total)", file=sys.stderr)

    print(f"\nWrote {total_rows} new rows → {out_path}")
    print(f"Log → {log_path}")


if __name__ == "__main__":
    main()
