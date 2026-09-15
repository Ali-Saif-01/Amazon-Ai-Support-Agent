#!/usr/bin/env python3
"""
Validate the AmazonHelp golden evaluation set CSV.

Checks structural integrity and that every row is a real AmazonHelp
conversation-opening customer tweet from twcs.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


LOCKED_INTENTS = {
    "Delivery Delay / Service Issue",
    "Marked Delivered But Not Received",
    "Order Cancellation & Refund",
    "Wrong or Defective Item Received",
    "Customer Service Quality Complaint",
    "Prime Membership, Billing & Video",
    "Payment, Gift Card & Billing Issue",
    "Account Security",
    "Device & App Technical Issue",
    "Shipping Speed / Prime SLA Not Met",
    "Packaging Feedback",
    "General Help Request / Greeting",
    "Other / Unclassified",
}

VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}
VALID_CONFIDENCE = {"", "Low", "Medium", "High"}
EXPECTED_ROWS = 200
BRAND = "AmazonHelp"
REQUIRED_COLUMNS = [
    "id",
    "tweet",
    "final_intent",
    "difficulty",
    "annotator_confidence",
    "annotation_reason",
    "source_tweet_id",
]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_golden(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("Golden set CSV has no header")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Golden set missing columns: {missing}")
        return list(reader)


def lookup_source_tweets(
    twcs_path: Path, tweet_ids: set[str]
) -> tuple[dict[str, dict[str, str]], set[str], UnionFind]:
    """Return row map, AmazonHelp conversation roots, and union-find graph."""
    uf = UnionFind()
    amazon_ids: set[str] = set()
    found: dict[str, dict[str, str]] = {}

    with twcs_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["tweet_id"]
            uf.parent.setdefault(tid, tid)
            if row["author_id"] == BRAND:
                amazon_ids.add(tid)
            parent_id = (row.get("in_response_to_tweet_id") or "").strip()
            if parent_id:
                uf.union(tid, parent_id)
            if tid in tweet_ids:
                found[tid] = {
                    "tweet_id": tid,
                    "author_id": row["author_id"],
                    "inbound": row["inbound"],
                    "text": row["text"] or "",
                    "in_response_to_tweet_id": (
                        row.get("in_response_to_tweet_id") or ""
                    ).strip(),
                }

    amazon_roots = {uf.find(tid) for tid in amazon_ids}
    return found, amazon_roots, uf


def validate(
    golden_path: Path, twcs_path: Path
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    rows = load_golden(golden_path)

    if len(rows) != EXPECTED_ROWS:
        errors.append(f"Expected exactly {EXPECTED_ROWS} rows, found {len(rows)}")

    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate golden id values found")

    source_ids = [r["source_tweet_id"].strip() for r in rows]
    if any(not s for s in source_ids):
        errors.append("One or more rows have missing source_tweet_id")
    if len(source_ids) != len(set(source_ids)):
        errors.append("Duplicate source_tweet_id values found")

    tweets = [r["tweet"] for r in rows]
    if any(not (t or "").strip() for t in tweets):
        errors.append("One or more rows have missing tweet text")
    # Exact duplicate tweet bodies
    seen_text: dict[str, str] = {}
    for r in rows:
        t = r["tweet"]
        if t in seen_text:
            errors.append(
                f"Duplicate tweet text between {seen_text[t]} and {r['id']}"
            )
        else:
            seen_text[t] = r["id"]

    for r in rows:
        diff = (r.get("difficulty") or "").strip()
        if diff not in VALID_DIFFICULTIES:
            errors.append(
                f"{r['id']}: invalid difficulty '{diff}' "
                f"(expected one of {sorted(VALID_DIFFICULTIES)})"
            )

        conf = (r.get("annotator_confidence") or "").strip()
        if conf not in VALID_CONFIDENCE:
            errors.append(
                f"{r['id']}: invalid annotator_confidence '{conf}' "
                f"(expected blank or one of Low/Medium/High)"
            )

        intent = (r.get("final_intent") or "").strip()
        if intent and intent not in LOCKED_INTENTS:
            errors.append(
                f"{r['id']}: final_intent '{intent}' is not in the locked taxonomy"
            )

    # Dataset-backed checks
    found, amazon_roots, uf = lookup_source_tweets(twcs_path, set(source_ids))

    missing_in_dataset = [sid for sid in source_ids if sid not in found]
    if missing_in_dataset:
        errors.append(
            f"{len(missing_in_dataset)} source_tweet_id(s) not found in twcs.csv "
            f"(e.g. {missing_in_dataset[:3]})"
        )

    non_opening = 0
    non_inbound = 0
    text_mismatch = 0
    not_amazon_convo = 0
    roots_seen: set[str] = set()

    for r in rows:
        sid = r["source_tweet_id"].strip()
        src = found.get(sid)
        if not src:
            continue

        if src["inbound"] != "True":
            non_inbound += 1
            errors.append(f"{r['id']}: source tweet is not inbound")

        if src["in_response_to_tweet_id"]:
            non_opening += 1
            errors.append(
                f"{r['id']}: source tweet is not a conversation opener "
                f"(in_response_to_tweet_id={src['in_response_to_tweet_id']})"
            )

        root = uf.find(sid)
        if root not in amazon_roots:
            not_amazon_convo += 1
            errors.append(
                f"{r['id']}: source tweet is not in an {BRAND} conversation"
            )
        if root in roots_seen:
            errors.append(
                f"{r['id']}: duplicate conversation (root={root}); "
                "golden examples must come from different conversations"
            )
        roots_seen.add(root)

        if src["text"] != r["tweet"]:
            text_mismatch += 1
            errors.append(
                f"{r['id']}: tweet text does not match twcs.csv for source_tweet_id={sid}"
            )

    # Annotation progress (informational)
    labeled = sum(1 for r in rows if (r.get("final_intent") or "").strip())
    if labeled == 0:
        warnings.append(
            "All final_intent values are blank (expected before manual annotation)"
        )
    elif labeled < EXPECTED_ROWS:
        warnings.append(f"Annotation in progress: {labeled}/{EXPECTED_ROWS} labeled")

    conf_filled = sum(1 for r in rows if (r.get("annotator_confidence") or "").strip())
    if conf_filled == 0:
        warnings.append("All annotator_confidence values are blank")

    return errors, warnings


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--golden",
        type=Path,
        default=root / "data" / "golden_set.csv",
    )
    p.add_argument(
        "--twcs",
        type=Path,
        default=root / "twcs.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.golden.exists():
        raise SystemExit(f"Golden set not found: {args.golden}")
    if not args.twcs.exists():
        raise SystemExit(f"Dataset not found: {args.twcs}")

    print(f"Validating {args.golden} against {args.twcs} ...")
    errors, warnings = validate(args.golden, args.twcs)

    for w in warnings:
        print(f"WARNING: {w}")

    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("PASSED")
    print(f"  rows: {EXPECTED_ROWS}")
    print("  unique source_tweet_ids: yes")
    print("  unique tweet texts: yes")
    print("  all AmazonHelp conversation openings: yes")
    print("  no missing tweet text: yes")
    print("  difficulty values valid: yes")
    print("  confidence values valid: yes")
    print("  final_intent blank or locked taxonomy: yes")


if __name__ == "__main__":
    main()
