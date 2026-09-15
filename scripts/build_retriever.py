import csv
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "historical_resolutions_clean.csv"
OUTPUT = ROOT / "data" / "retriever.pkl"


def main():
    print("Loading historical resolutions...")

    customer_texts = []
    replies = []

    with INPUT.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            customer_text = row["customer_text"].strip()
            amazon_reply = row["amazon_reply"].strip()

            if not customer_text or not amazon_reply:
                continue

            customer_texts.append(customer_text)
            replies.append(amazon_reply)

    print(f"Historical resolutions: {len(customer_texts):,}")

    print("\nBuilding TF-IDF index...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=150000,
    )

    matrix = vectorizer.fit_transform(customer_texts)

    print(f"TF-IDF matrix shape: {matrix.shape}")

    retriever = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "customer_texts": customer_texts,
        "replies": replies,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("wb") as f:
        pickle.dump(retriever, f)

    print("\nRetriever built successfully!")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()