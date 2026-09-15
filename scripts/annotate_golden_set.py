#!/usr/bin/env python3
"""
Interactive annotator for data/golden_set.csv.

- Shows one tweet at a time
- Collects final_intent, annotator_confidence, annotation_reason
- Skips completed rows by default (resume-friendly)
- Saves safely after each annotation
- Stdlib only
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Locked project constants (must match the golden-set taxonomy)
# ---------------------------------------------------------------------------

LOCKED_INTENTS = [
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
]

CONFIDENCE_OPTIONS = ["High", "Medium", "Low"]

COLUMNS = [
    "id",
    "tweet",
    "final_intent",
    "difficulty",
    "annotator_confidence",
    "annotation_reason",
    "source_tweet_id",
]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "golden_set.csv"
EXPECTED_N = 200


def is_complete(row: dict[str, str]) -> bool:
    """A row is complete when final_intent is non-empty."""
    return bool((row.get("final_intent") or "").strip())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {path}")
        missing = [c for c in COLUMNS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing columns: {missing}")
        rows = list(reader)
    if len(rows) != EXPECTED_N:
        print(f"Warning: expected {EXPECTED_N} rows, found {len(rows)}")
    return rows


def save_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write CSV safely: temp file in the same folder, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="golden_set_",
        suffix=".csv.tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in COLUMNS})
        os.replace(tmp_name, path)
    except Exception:
        # Clean up temp file if something went wrong
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def progress_counts(rows: list[dict[str, str]]) -> tuple[int, int]:
    completed = sum(1 for r in rows if is_complete(r))
    remaining = len(rows) - completed
    return completed, remaining


def print_progress(rows: list[dict[str, str]]) -> None:
    completed, remaining = progress_counts(rows)
    total = len(rows)
    print(f"Completed: {completed}/{total}")
    print(f"Remaining: {remaining}/{total}")


def print_intent_menu() -> None:
    print("\nLocked intents:")
    for i, intent in enumerate(LOCKED_INTENTS, start=1):
        print(f"  {i:2d}. {intent}")


def ask_intent() -> str | None:
    """
    Ask for final_intent.
    Returns the intent string, or None if the user typed skip/quit.
    Special return values are handled by the caller via sentinel strings
    stored in a small protocol: we return ('__skip__' / '__quit__') as str.
    """
    print_intent_menu()
    while True:
        raw = input(
            "\nEnter final_intent number (1-13), "
            "or 's' to skip, 'q' to quit: "
        ).strip().lower()
        if raw in {"s", "skip"}:
            return "__skip__"
        if raw in {"q", "quit"}:
            return "__quit__"
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(LOCKED_INTENTS):
                return LOCKED_INTENTS[n - 1]
        print("Invalid choice. Enter 1-13, s, or q.")


def ask_confidence() -> str | None:
    print("\nConfidence options:")
    for i, opt in enumerate(CONFIDENCE_OPTIONS, start=1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(
            "Enter confidence (1-3 / High / Medium / Low), "
            "or 's' to skip, 'q' to quit: "
        ).strip()
        low = raw.lower()
        if low in {"s", "skip"}:
            return "__skip__"
        if low in {"q", "quit"}:
            return "__quit__"
        if raw.isdigit() and 1 <= int(raw) <= 3:
            return CONFIDENCE_OPTIONS[int(raw) - 1]
        # Allow typing the label directly (case-insensitive)
        for opt in CONFIDENCE_OPTIONS:
            if low == opt.lower():
                return opt
        print("Invalid choice. Enter 1-3, High/Medium/Low, s, or q.")


def ask_reason() -> str | None:
    while True:
        raw = input(
            "Enter annotation_reason "
            "(short note), or 's' to skip, 'q' to quit: "
        ).strip()
        low = raw.lower()
        if low in {"s", "skip"}:
            return "__skip__"
        if low in {"q", "quit"}:
            return "__quit__"
        if raw:
            return raw
        print("Reason cannot be empty. Please enter a short note, s, or q.")


def show_tweet(row: dict[str, str], index: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"Tweet {index}/{total}")
    print("=" * 72)
    print(f"id:              {row.get('id', '')}")
    print(f"source_tweet_id: {row.get('source_tweet_id', '')}")
    print(f"difficulty:      {row.get('difficulty', '')}")
    print("-" * 72)
    print("tweet:")
    print(row.get("tweet", ""))
    print("-" * 72)
    if is_complete(row):
        print(f"current final_intent:         {row.get('final_intent', '')}")
        print(f"current annotator_confidence: {row.get('annotator_confidence', '')}")
        print(f"current annotation_reason:    {row.get('annotation_reason', '')}")
        print("-" * 72)


def annotate_one(row: dict[str, str]) -> str:
    """
    Annotate a single row.
    Returns: 'saved', 'skipped', or 'quit'.
    """
    intent = ask_intent()
    if intent == "__skip__":
        return "skipped"
    if intent == "__quit__":
        return "quit"

    confidence = ask_confidence()
    if confidence == "__skip__":
        return "skipped"
    if confidence == "__quit__":
        return "quit"

    reason = ask_reason()
    if reason == "__skip__":
        return "skipped"
    if reason == "__quit__":
        return "quit"

    print("\nPlease confirm:")
    print(f"  final_intent:         {intent}")
    print(f"  annotator_confidence: {confidence}")
    print(f"  annotation_reason:    {reason}")
    confirm = input("Save this annotation? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Not saved.")
        return "skipped"

    row["final_intent"] = intent
    row["annotator_confidence"] = confidence
    row["annotation_reason"] = reason
    return "saved"


def choose_mode(rows: list[dict[str, str]]) -> str:
    """
    Ask whether to annotate remaining only, or also offer edits.
    Returns: 'remaining' or 'edit'.
    """
    completed, remaining = progress_counts(rows)
    print("\nModes:")
    print("  1. Annotate remaining incomplete tweets only (recommended)")
    if completed > 0:
        print("  2. Edit an already-completed annotation (explicit)")
    print("  q. Quit")

    while True:
        raw = input("Choose mode: ").strip().lower()
        if raw in {"1", "r", "remaining"}:
            return "remaining"
        if raw in {"2", "e", "edit"} and completed > 0:
            return "edit"
        if raw in {"q", "quit"}:
            return "quit"
        print("Invalid choice.")


def pick_completed_row(rows: list[dict[str, str]]) -> int | None:
    """Let the user pick a completed row index to edit. None = cancel."""
    completed = [(i, r) for i, r in enumerate(rows) if is_complete(r)]
    if not completed:
        print("No completed annotations to edit.")
        return None

    print("\nCompleted annotations:")
    for i, row in completed:
        intent = row.get("final_intent", "")
        print(f"  {row.get('id')}: {intent}")

    raw = input(
        "Enter golden id to edit (e.g. golden_003), or 'c' to cancel: "
    ).strip()
    if raw.lower() in {"c", "cancel", ""}:
        return None

    for i, row in enumerate(rows):
        if row.get("id") == raw and is_complete(row):
            return i
    print(f"No completed row found with id '{raw}'.")
    return None


def main() -> None:
    csv_path = DEFAULT_CSV
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")

    print("AmazonHelp golden-set annotator")
    print(f"Loading: {csv_path}")
    rows = load_rows(csv_path)

    print()
    print_progress(rows)

    saved_count = 0
    skipped_count = 0

    while True:
        completed, remaining = progress_counts(rows)
        if remaining == 0:
            print("\nAll 200 examples are annotated.")
            mode = input(
                "Edit an existing annotation anyway? [y/N]: "
            ).strip().lower()
            if mode in {"y", "yes"}:
                chosen = pick_completed_row(rows)
                if chosen is None:
                    break
                show_tweet(rows[chosen], chosen + 1, len(rows))
                result = annotate_one(rows[chosen])
                if result == "saved":
                    save_rows(csv_path, rows)
                    saved_count += 1
                    print(f"Saved -> {csv_path}")
                elif result == "quit":
                    break
                else:
                    skipped_count += 1
                continue
            break

        mode = choose_mode(rows)
        if mode == "quit":
            break

        if mode == "edit":
            chosen = pick_completed_row(rows)
            if chosen is None:
                continue
            show_tweet(rows[chosen], chosen + 1, len(rows))
            result = annotate_one(rows[chosen])
            if result == "saved":
                save_rows(csv_path, rows)
                saved_count += 1
                print(f"Saved -> {csv_path}")
                print_progress(rows)
            elif result == "quit":
                break
            else:
                skipped_count += 1
            continue

        # Annotate remaining incomplete tweets in order
        for i, row in enumerate(rows):
            if is_complete(row):
                continue

            show_tweet(row, i + 1, len(rows))
            print("Commands during prompts: s = skip this tweet, q = quit/resume later")
            result = annotate_one(row)

            if result == "saved":
                save_rows(csv_path, rows)
                saved_count += 1
                print(f"Saved -> {csv_path}")
                print_progress(rows)
            elif result == "skipped":
                skipped_count += 1
                print("Skipped (existing fields left unchanged).")
            elif result == "quit":
                print("\nQuitting. Progress is saved; you can resume later.")
                # Final summary below
                completed, remaining = progress_counts(rows)
                print()
                print("=" * 40)
                print("Session summary")
                print("=" * 40)
                print(f"Saved this session:  {saved_count}")
                print(f"Skipped this session:{skipped_count}")
                print(f"Completed: {completed}/{len(rows)}")
                print(f"Remaining: {remaining}/{len(rows)}")
                return

        # Finished a full pass over remaining items; loop to show mode again
        # in case user skipped some.
        if progress_counts(rows)[1] == 0:
            break
        print("\nFinished one pass over remaining tweets.")
        print_progress(rows)
        again = input("Start another pass over remaining? [y/N]: ").strip().lower()
        if again not in {"y", "yes"}:
            break

    completed, remaining = progress_counts(rows)
    print()
    print("=" * 40)
    print("Session summary")
    print("=" * 40)
    print(f"Saved this session:   {saved_count}")
    print(f"Skipped this session: {skipped_count}")
    print(f"Completed: {completed}/{len(rows)}")
    print(f"Remaining: {remaining}/{len(rows)}")
    print(f"File: {csv_path}")


if __name__ == "__main__":
    main()
