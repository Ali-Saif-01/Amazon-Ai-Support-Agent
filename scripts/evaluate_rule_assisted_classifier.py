import csv
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "data" / "intent_classifier.pkl"
GOLDEN = ROOT / "data" / "golden_set.csv"


def contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def apply_rules(text, ml_prediction):
    t = text.lower().strip()

    # Rule 1: Explicit delivery promise + problem
    # Overrides Prime Membership when the actual issue is delivery speed.
    sla = contains_any(t, [
        "2-day", "2 day", "two day",
        "next day", "one-day", "one day",
        "same day",
        "guaranteed delivery",
        "guaranteed shipping",
        "prime delivery",
        "prime shipping",
        "prime promised",
        "prime promise",
    ])

    delivery_problem = contains_any(t, [
        "late", "delayed",
        "not arrived", "hasn't arrived",
        "has not arrived",
        "didn't arrive", "did not arrive",
        "still waiting",
        "not here",
        "where is",
        "overdue",
    ])

    if sla and delivery_problem:
        return "Shipping Speed / Prime SLA Not Met"

    # Rule 2: Tracking says delivered + customer says not received.
    delivered = contains_any(t, [
        "says delivered",
        "said delivered",
        "marked delivered",
        "shows delivered",
        "shown as delivered",
        "tracking says delivered",
        "tracking shows delivered",
        "delivered but",
        "delivered, but",
        "delivery confirmation",
    ])

    not_received = contains_any(t, [
        "not received",
        "was not received",
        "did not receive",
        "didn't receive",
        "never received",
        "not here",
        "wasn't delivered",
        "was not delivered",
        "didn't get it",
        "did not get it",
    ])

    if delivered and not_received:
        return "Marked Delivered But Not Received"

    # Rule 3: Security signals should override generic order/payment language.
    security = contains_any(t, [
        "hacked",
        "hack",
        "compromised",
        "someone accessed",
        "someone has access",
        "unauthorized access",
        "unauthorised access",
        "account stolen",
        "account was stolen",
        "password changed",
        "password was changed",
        "can't login",
        "cannot login",
        "can't log in",
        "cannot log in",
        "someone logged in",
        "unknown login",
        "suspicious login",
        "didn't order",
        "did not order",
        "never ordered",
        "order i didn't place",
        "order i did not place",
    ])

    if security:
        return "Account Security"

    # Otherwise keep the ML prediction.
    return ml_prediction


def main():
    print("Loading baseline classifier...")

    with MODEL.open("rb") as f:
        model = pickle.load(f)

    print("Loading golden evaluation set...")

    texts = []
    actual = []

    with GOLDEN.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            texts.append(row["tweet"].strip())
            actual.append(row["final_intent"].strip())

    print(f"Golden examples: {len(texts)}")

    print("\nRunning ML predictions...")

    ml_predictions = model.predict(texts)

    final_predictions = []

    rule_changes = 0

    for text, ml_prediction in zip(texts, ml_predictions):
        final_prediction = apply_rules(text, ml_prediction)

        if final_prediction != ml_prediction:
            rule_changes += 1

        final_predictions.append(final_prediction)

    from sklearn.metrics import accuracy_score, classification_report, f1_score

    accuracy = accuracy_score(actual, final_predictions)

    macro_f1 = f1_score(
        actual,
        final_predictions,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("RULE-ASSISTED CLASSIFIER RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Rule changes: {rule_changes}")

    print("\nPer-intent results:")

    print(
        classification_report(
            actual,
            final_predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()