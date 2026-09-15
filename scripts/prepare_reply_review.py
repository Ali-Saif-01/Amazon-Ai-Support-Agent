import csv
import random


INPUT_FILE = "data/agent_results.csv"
OUTPUT_FILE = "data/reply_human_review.csv"

SAMPLE_SIZE = 50
SEED = 42


def main():
    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    random.seed(SEED)

    sample = random.sample(
        rows,
        min(SAMPLE_SIZE, len(rows))
    )

    review_rows = []

    for row in sample:
        review_rows.append({
            "id": row["id"],
            "tweet": row["tweet"],
            "intent": row["intent"],
            "retrieved_customer": row["historical_customer"],
            "retrieved_reply": row["historical_reply"],
            "generated_reply": row["reply"],

            # Human reviewer fills these:
            "relevance": "",
            "groundedness": "",
            "helpfulness": "",
            "professional_tone": "",
            "unsupported_claims": "",
            "overall_quality": "",
            "notes": ""
        })

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        fieldnames = [
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
            "notes"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(review_rows)

    print("Reply human review set created.")
    print(f"Examples sampled: {len(review_rows)}")
    print(f"Random seed: {SEED}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()