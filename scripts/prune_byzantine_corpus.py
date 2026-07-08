#!/usr/bin/env python3
"""Prune raw SFT corpus rows with rules + optional LLM judge.

Usage:
  python scripts/prune_byzantine_corpus.py
  python scripts/prune_byzantine_corpus.py --judge --in data/byzantine/sft_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_harness.judge.byzantine_checks import BYZ_MARKERS, WEST_MARKERS, check_byzantine_transcription

DEFAULT_IN = ROOT / "data" / "byzantine" / "sft_raw.jsonl"
DEFAULT_OUT = ROOT / "data" / "byzantine" / "sft_raw.jsonl"
REPORT = ROOT / "data" / "byzantine" / "prune_report.json"

PITCH_RE = re.compile(r"\b[A-G](?:[#b♭♯]|(?:↑|↓))?\d\b")
NEUME_RE = re.compile(
    r"\b(oligon|petast[eē]|apostrophos|kent[eē]ma|ison|gorgon|argon|elaphron|ypsil[eē]\s*pnevma)\b",
    re.I,
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _msg(row: dict, role: str) -> str:
    for m in row.get("messages") or []:
        if m.get("role") == role:
            return str(m.get("content") or "").strip()
    return ""


def rule_check(row: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    direction = row.get("direction", "byz_to_west")
    user = _msg(row, "user")
    assistant = _msg(row, "assistant")

    if len(user) < 20 or len(assistant) < 10:
        reasons.append("too_short")

    failures = check_byzantine_transcription(
        model_output=assistant,
        direction=direction,
        forbidden_extra=[
            r"\b(the answer is|note that|keep in mind|I transcribed|here is the conversion)\b",
        ],
    )
    reasons.extend(failures)

    if direction == "byz_to_west":
        if not NEUME_RE.search(user):
            reasons.append("user lacks neume markers")
        pitch_count = len(PITCH_RE.findall(assistant))
        if pitch_count < 2:
            reasons.append(f"assistant fewer than 2 pitches ({pitch_count})")
        if not re.search(r"(?i)(mode|ison|ni)", assistant):
            reasons.append("assistant missing mode/ison header")
    else:
        pitch_count = len(PITCH_RE.findall(user))
        if pitch_count < 2:
            reasons.append(f"user fewer than 2 pitches ({pitch_count})")
        if not NEUME_RE.search(assistant):
            reasons.append("assistant lacks neume markers")

    # Reject obvious prose-only rows
    if re.search(r"(?i)\b(copyright|download|viewer|install)\b", user + assistant):
        reasons.append("page boilerplate")

    return len(reasons) == 0, reasons


def pair_key(row: dict) -> str:
    rid = row.get("id", "")
    base = re.sub(r"_(b2w|w2b)$", "", rid)
    return base


def dedupe_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    seen_assistant: set[str] = set()
    kept: list[dict] = []
    rejected_ids: list[str] = []
    for row in rows:
        ast = _msg(row, "assistant")
        key = ast[:200]
        if key in seen_assistant:
            rejected_ids.append(row["id"])
            continue
        seen_assistant.add(key)
        kept.append(row)
    return kept, rejected_ids


def llm_judge_rows(rows: list[dict], model: str = "gpt-4o-mini") -> dict[str, float]:
    from openai import OpenAI

    client = OpenAI()
    scores: dict[str, float] = {}
    for row in rows:
        user = _msg(row, "user")
        assistant = _msg(row, "assistant")
        direction = row.get("direction", "byz_to_west")
        prompt = f"""Score this Byzantine notation transcription training example (0, 1, or 2).
Direction: {direction}
0 = wrong/unusable, 1 = partially correct, 2 = good training pair

User:
{user}

Assistant (gold):
{assistant}

Return ONLY JSON: {{"score": 0|1|2, "reason": "..."}}"""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        scores[row["id"]] = float(parsed.get("score", 0))
    return scores


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--judge", action="store_true", help="LLM score rows (requires OPENAI_API_KEY)")
    parser.add_argument("--rules-only", action="store_true", help="Skip LLM judge; accept rule-passing rows")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--min-judge-score", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_rows(Path(args.in_path))
    for i, row in enumerate(rows):
        if row.get("status") in ("rejected", "raw"):
            rows[i] = {**row, "status": "raw"}

    report = {"total": len(rows), "accepted": [], "rejected": []}
    rule_ok: list[dict] = []
    for row in rows:
        ok, reasons = rule_check(row)
        if ok:
            rule_ok.append(row)
        else:
            report["rejected"].append({"id": row["id"], "pass": "rules", "reasons": reasons})

    # Pass 2: dedupe by assistant content
    deduped, dup_ids = dedupe_rows(rule_ok)
    for rid in dup_ids:
        report["rejected"].append({"id": rid, "pass": "dedupe", "reasons": ["duplicate assistant"]})

    # Pass 3: optional LLM judge
    accepted = deduped
    judge_scores: dict[str, float] = {}
    if args.judge and not args.rules_only:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY required for --judge")
        judge_scores = llm_judge_rows(deduped, model=args.judge_model)
        accepted = []
        for row in deduped:
            score = judge_scores.get(row["id"], 0)
            if score >= args.min_judge_score:
                accepted.append(row)
            else:
                report["rejected"].append(
                    {"id": row["id"], "pass": "judge", "reasons": [f"score={score}"]}
                )

    # Write status back to all rows
    accepted_ids = {r["id"] for r in accepted}
    out_rows = []
    for row in rows:
        copy = dict(row)
        if copy["id"] in accepted_ids:
            copy["status"] = "accepted"
        elif copy.get("status") == "raw":
            copy["status"] = "rejected"
        out_rows.append(copy)

    out_path = Path(args.out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report["accepted"] = [r["id"] for r in accepted]
    report["accepted_count"] = len(accepted)
    report["rejected_count"] = len(report["rejected"])
    report["judge_scores"] = judge_scores
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Pruned {len(rows)} → {len(accepted)} accepted")
    print(f"  rejected: {report['rejected_count']}")
    print(f"Wrote {out_path}")
    print(f"Report → {args.report}")


if __name__ == "__main__":
    main()
