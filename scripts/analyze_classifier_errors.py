import csv
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "data" / "intent_classifier.pkl"
GOLDEN = ROOT / "data" / "golden_set.csv"
OUTPUT = ROOT / "data" / "classifier_errors.csv"


def main():
    print("Loading classifier...")

    with MODEL.open("rb") as f:
        model = pickle.load(f)

    errors = []

    with GOLDEN.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            tweet = row["tweet"].strip()
            actual = row["final_intent"].strip()

            predicted = model.predict([tweet])[0]

            if actual != predicted:
                errors.append({
                    "tweet": tweet,
                    "actual_intent": actual,
                    "predicted_intent": predicted,
                })

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tweet",
                "actual_intent",
                "predicted_intent",
            ],
        )

        writer.writeheader()
        writer.writerows(errors)

    print("\nError analysis complete.")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()