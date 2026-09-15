import csv
from pathlib import Path


INPUT_FILE = Path("data/reply_human_review.csv")
OUTPUT_FILE = Path("data/reply_human_review_scored.csv")


REQUIRED_COLUMNS = [
    "id",
    "tweet",
    "intent",
    "retrieved_customer",
    "retrieved_reply",
    "generated_reply",
    "relevance",
    "groundedness",
    "helpfulness",
    "professional_tone",
    "unsupported_claims",
    "overall_quality",
    "notes",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    missing = [col for col in REQUIRED_COLUMNS if col not in columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Rows copied for human review: {len(rows)}")
    print()
    print("Human scoring columns:")
    print("- relevance: 1 relevant, 0 not relevant")
    print("- groundedness: 1 supported by evidence, 0 unsupported")
    print("- helpfulness: 1 useful, 0 not useful")
    print("- professional_tone: 1 professional, 0 not professional")
    print("- unsupported_claims: 1 contains unsupported claim, 0 no unsupported claim")
    print("- overall_quality: reviewer judgment of overall reply quality")


if __name__ == "__main__":
    main()