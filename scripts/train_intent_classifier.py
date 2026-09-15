import csv
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "weak_labeled_resolutions.csv"
OUTPUT = ROOT / "data" / "intent_classifier.pkl"


def main():
    print("Loading weak-labeled training data...")

    texts = []
    labels = []

    with INPUT.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = row["customer_text"].strip()
            label = row["weak_intent"].strip()

            if not text or not label:
                continue

            texts.append(text)
            labels.append(label)

    print(f"Training examples: {len(texts):,}")
    print(f"Intent classes: {len(set(labels))}")

    print("\nTraining TF-IDF + Logistic Regression...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(texts, labels)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("wb") as f:
        pickle.dump(model, f)

    print("\nTraining complete!")
    print(f"Model saved to: {OUTPUT}")


if __name__ == "__main__":
    main()