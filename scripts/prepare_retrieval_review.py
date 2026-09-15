import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]

RETRIEVER = ROOT / "data" / "retriever.pkl"
GOLDEN = ROOT / "data" / "golden_set.csv"
OUTPUT = ROOT / "data" / "retrieval_review.csv"


def main():
    print("Loading retriever...")

    with RETRIEVER.open("rb") as f:
        retriever = pickle.load(f)

    vectorizer = retriever["vectorizer"]
    matrix = retriever["matrix"]
    customer_texts = retriever["customer_texts"]
    replies = retriever["replies"]

    print("Loading golden set...")

    with GOLDEN.open(newline="", encoding="utf-8") as f:
        golden_rows = list(csv.DictReader(f))

    print(f"Golden examples: {len(golden_rows)}")

    review_rows = []

    for row in golden_rows:
        query = row["tweet"].strip()

        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, matrix).flatten()

        top_indices = np.argsort(similarities)[-3:][::-1]

        for rank, index in enumerate(top_indices, start=1):
            review_rows.append({
                "golden_id": row["id"],
                "golden_tweet": query,
                "golden_intent": row["final_intent"],
                "rank": rank,
                "similarity": round(float(similarities[index]), 4),
                "retrieved_customer": customer_texts[index],
                "historical_reply": replies[index],
                "relevant": ""
            })

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "golden_id",
            "golden_tweet",
            "golden_intent",
            "rank",
            "similarity",
            "retrieved_customer",
            "historical_reply",
            "relevant"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    print()
    print("=" * 60)
    print("RETRIEVAL REVIEW CREATED")
    print("=" * 60)
    print(f"Rows written: {len(review_rows)}")
    print(f"File: {OUTPUT}")


if __name__ == "__main__":
    main()