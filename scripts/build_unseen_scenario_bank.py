#!/usr/bin/env python3
"""Build held-out Byzantine scenarios from unseen corpus PDFs via vision extraction.

Usage:
  python scripts/build_unseen_scenario_bank.py --download
  python scripts/build_unseen_scenario_bank.py --extract
  python scripts/build_unseen_scenario_bank.py --all
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

import fitz
import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "byzantine" / "corpus" / "unseen"
PNG = CORPUS / "png"
OUT = ROOT / "scenarios" / "byzantine_transcription_unseen.yaml"
META = ROOT / "data" / "byzantine" / "unseen_pairs.json"

# Pairs never referenced in scenarios/*.yaml (verified by filename stem)
UNSEEN_PAIRS = [
    {
        "id": "unseen_apolytikia_resurrection",
        "title": "Apolytikia of the Resurrection (Cappella Romana)",
        "source": "cappella_romana",
        "byz_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Apolytikia-Byz_Divine-Liturgy-Music_Cappella-Romana.pdf",
        "west_url": "https://cappellaromana.org/wp-content/uploads/2014/04/Apolytikia-Western-Print_Divine-Liturgy-Music_Cappella-Romana.pdf",
    },
    {
        "id": "unseen_transfiguration_apolytikion",
        "title": "Apolytikion of the Transfiguration (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/08-06_transfiguration_apolytikion_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/08-06_transfiguration_apolytikion_en_staff.pdf",
    },
    {
        "id": "unseen_pentecost_apolytikion",
        "title": "Apolytikion of Pentecost (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/pentecost_apolytikion_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/pentecost_apolytikion_en_staff.pdf",
    },
    {
        "id": "unseen_annunciation_apolytikion",
        "title": "Apolytikion of the Annunciation (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/03-25_annunciation_apolytikion_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/03-25_annunciation_apolytikion_en_staff.pdf",
    },
    {
        "id": "unseen_circumcision_apolytikion",
        "title": "Apolytikion of the Circumcision (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/01-01_circumcision_apolytikion_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/01-01_circumcision_apolytikion_en_staff.pdf",
    },
    {
        "id": "unseen_theophany_apolytikion",
        "title": "Apolytikion of Theophany (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/01-06_theophany_apolytikion_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/01-06_theophany_apolytikion_en_staff.pdf",
    },
    {
        "id": "unseen_ascension_apolytikion",
        "title": "Apolytikion of the Ascension (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/ascension_apolytikion_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/ascension_apolytikion_en_staff.pdf",
    },
    {
        "id": "unseen_pentecost_great_prokimenon",
        "title": "Great Prokeimenon of Pentecost (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/pentecost_great_prokimenon_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/pentecost_great_prokimenon_gr_en_staff.pdf",
    },
    {
        "id": "unseen_theophany_communion",
        "title": "Theophany Communion Hymn (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/01-06_theophany_communion_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/01-06_theophany_communion_gr_en.pdf",
    },
    {
        "id": "unseen_dormition_lamentations",
        "title": "Dormition Lamentations fragment (New Byzantium)",
        "source": "new_byzantium",
        "byz_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/08-15_dormition_lamentations_en_byz.pdf",
        "west_url": "https://newbyz.weebly.com/uploads/1/4/7/1/147110798/08-15_dormition_lamentations_en_staff.pdf",
    },
]

EXTRACT_PROMPT = """You are building eval scenarios for Byzantine ↔ Western transcription.

You will see TWO images from the same hymn: Byzantine neume notation (left/first) and Western staff notation (second).

Extract ONE short eval fragment (5–7 neumes max) from the OPENING of the melody after the ison/martyria setup.

Return ONLY valid JSON:
{
  "direction": "byz_to_west",
  "echos": "Mode name from score header (e.g. Pl IV, I, III, Grave)",
  "ni_greek": "Greek Ni letter if visible (Πα, Γα, Κε, etc.)",
  "ni_western": "Western Ni pitch e.g. D4",
  "byzantine_input": "multi-line string: Direction line, mode header, (Ni) ison, neume chain with | separators",
  "western_reference": "multi-line string: Mode line, Ison line, pitch sequence",
  "notes": "any fthora/gorgon/diesis in fragment"
}

Rules:
- Use English neume names: oligon, petastē, apostrophos, kentēma, ison, gorgon, argon, ypsilē pnevma, elaphron
- Mark (diesis) or fthora if present on neumes
- Western pitches must match the staff image exactly for those neumes
- Ni = martyria/ison anchor pitch
- Do not include lyrics prose in output
"""


def download_pairs() -> list[dict]:
    CORPUS.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "From-Scratch-LLM-corpus/1.0"}
    out = []
    for pair in UNSEEN_PAIRS:
        sid = pair["id"]
        rec = dict(pair)
        for kind, url in [("byz", pair["byz_url"]), ("west", pair["west_url"])]:
            dest = CORPUS / f"{sid}_{kind}.pdf"
            if not dest.exists() or dest.stat().st_size == 0:
                print(f"download {sid} {kind}")
                r = requests.get(url, timeout=60, headers=headers)
                if r.status_code == 404:
                    print(f"  SKIP {sid} {kind}: 404 {url}")
                    rec["skipped"] = True
                    break
                r.raise_for_status()
                dest.write_bytes(r.content)
            rec[f"{kind}_path"] = str(dest.relative_to(ROOT))
            png = PNG / f"{sid}_{kind}_p0.png"
            if not png.exists():
                doc = fitz.open(dest)
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
                pix.save(png)
            rec[f"{kind}_png"] = str(png.relative_to(ROOT))
        if not rec.get("skipped"):
            out.append(rec)
    META.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Downloaded {len(out)} pairs → {META}")
    return out


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def extract_scenarios(pairs: list[dict], model: str = "gpt-4.1") -> list[dict]:
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    scenarios = []
    for pair in pairs:
        print(f"extract {pair['id']}...")
        byz_png = ROOT / pair["byz_png"]
        west_png = ROOT / pair["west_png"]
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Hymn: {pair['title']}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(byz_png)}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(west_png)}"}},
                    ],
                },
            ],
            max_tokens=800,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            print(f"  WARN: no JSON for {pair['id']}: {raw[:200]}")
            continue
        data = json.loads(m.group())
        scenarios.append(
            {
                "id": pair["id"],
                "direction": data.get("direction", "byz_to_west"),
                "echos": data.get("echos", ""),
                "tags": [pair["source"], "unseen", "corpus", "vision_extract"],
                "source_url": pair["byz_url"],
                "reference_pdf_west": pair["west_url"],
                "input": data["byzantine_input"].strip(),
                "reference_output": data["western_reference"].strip(),
                "context": f"Unseen held-out from {pair['title']}. Vision-extracted opening fragment. {data.get('notes','')}".strip(),
            }
        )
        print(f"  ok: {scenarios[-1]['input'][:80]}...")
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args()

    if args.all:
        args.download = args.extract = True

    pairs = json.loads(META.read_text()) if META.exists() and not args.download else []
    if args.download:
        pairs = download_pairs()
    if args.extract:
        if not pairs:
            pairs = download_pairs()
        scenarios = extract_scenarios(pairs, model=args.model)
        OUT.write_text(yaml.dump(scenarios, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"Wrote {len(scenarios)} scenarios → {OUT}")


if __name__ == "__main__":
    main()
