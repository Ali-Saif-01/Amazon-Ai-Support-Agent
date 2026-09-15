import csv
import json
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score


GOLDEN_FILE = "data/golden_set.csv"
OUTPUT_FILE = "data/baseline_results.json"


def load_golden():
    with open(GOLDEN_FILE, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def majority_baseline(rows):
    labels = [row["final_intent"] for row in rows]
    majority_label = Counter(labels).most_common(1)[0][0]
    predictions = [majority_label] * len(rows)
    return majority_label, predictions


def keyword_baseline(text):
    text = text.lower()

    if (
        ("delivered" in text or "delivery" in text)
        and (
            "not received" in text
            or "didn't receive" in text
            or "did not receive" in text
            or "never received" in text
            or "not got" in text
        )
    ):
        return "Marked Delivered But Not Received"

    if (
        ("next day" in text or "next-day" in text or "2-day" in text
         or "two day" in text or "guaranteed" in text)
        and ("delivery" in text or "arrive" in text or "arrived" in text)
    ):
        return "Shipping Speed / Prime SLA Not Met"

    if (
        "hack" in text
        or "hacked" in text
        or "stolen account" in text
        or "someone accessed" in text
        or "unauthorized access" in text
        or "don't recognize" in text
        or "didn't order" in text
        or "never ordered" in text
    ):
        return "Account Security"

    if (
        "broken" in text
        or "defective" in text
        or "damaged" in text
        or "wrong item" in text
        or "wrong product" in text
    ):
        return "Wrong or Defective Item Received"

    if (
        "refund" in text
        or "cancel my order" in text
        or "cancel the order" in text
        or "cancel order" in text
    ):
        return "Order Cancellation & Refund"

    if (
        "gift card" in text
        or "charged" in text
        or "charge" in text
        or "payment" in text
        or "credit card" in text
        or "debit card" in text
    ):
        return "Payment, Gift Card & Billing Issue"

    if (
        "prime membership" in text
        or "prime subscription" in text
        or "prime video" in text
        or "prime charge" in text
    ):
        return "Prime Membership, Billing & Video"

    if (
        "app" in text
        or "application" in text
        or "device" in text
        or "kindle" in text
        or "fire tv" in text
    ):
        return "Device & App Technical Issue"

    if (
        "packaging" in text
        or "package" in text
    ):
        return "Packaging Feedback"

    if (
        "agent" in text
        or "customer service" in text
        or "support" in text
        or "representative" in text
    ):
        return "Customer Service Quality Complaint"

    if (
        "late" in text
        or "delayed" in text
        or "still waiting" in text
        or "where is my order" in text
    ):
        return "Delivery Delay / Service Issue"

    if (
        "help" in text
        or "hello" in text
        or "hi " in text
        or text.startswith("hi")
    ):
        return "General Help Request / Greeting"

    return "Other / Unclassified"


def evaluate(name, y_true, y_pred):
    return {
        "baseline": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }


def main():
    rows = load_golden()
    y_true = [row["final_intent"] for row in rows]

    majority_label, majority_predictions = majority_baseline(rows)

    keyword_predictions = [
        keyword_baseline(row["tweet"])
        for row in rows
    ]

    results = [
        evaluate(
            "Majority-class baseline",
            y_true,
            majority_predictions
        ),
        evaluate(
            "Keyword/rule baseline",
            y_true,
            keyword_predictions
        )
    ]

    output = {
        "golden_examples": len(rows),
        "majority_class": majority_label,
        "our_system": {
            "accuracy": 0.6250,
            "macro_f1": 0.6272
        },
        "baselines": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("========================================")
    print("HIVER BASELINE EVALUATION")
    print("========================================")
    print(f"Golden examples : {len(rows)}")
    print()
    print(f"Majority class  : {majority_label}")
    print()

    for result in results:
        print(result["baseline"])
        print(f"  Accuracy : {result['accuracy']:.4f}")
        print(f"  Macro F1 : {result['macro_f1']:.4f}")
        print()

    print("========================================")
    print("COMPARISON")
    print("========================================")
    print("Our system       : accuracy=0.6250, macro_f1=0.6272")

    for result in results:
        print(
            f"{result['baseline']:<20}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}"
        )

    print()
    print(f"Machine-readable results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
