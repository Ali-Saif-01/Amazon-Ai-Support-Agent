import csv
import pickle
import re
import sys
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_reply import generate_reply


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def load_classifier():
    with open(ROOT / "data" / "intent_classifier.pkl", "rb") as f:
        return pickle.load(f)


def load_retriever():
    with open(ROOT / "data" / "retriever.pkl", "rb") as f:
        return pickle.load(f)


def contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def apply_rules(text, ml_prediction):
    t = text.lower().strip()

    # Rule 1: Explicit delivery promise + problem
    sla = contains_any(t, [
        "2-day", "2 day", "two day",
        "next day", "one-day", "oneday",
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
        "not here", "where is", "overdue",
    ])

    if sla and delivery_problem:
        return "Shipping Speed / Prime SLA Not Met"

    # Rule 2: Tracking says delivered + customer says not received
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

    # Rule 3: Security signals override generic order/payment language
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

    return ml_prediction


def predict_intent(text, classifier):
    ml_prediction = classifier.predict([text])[0]
    return apply_rules(text, ml_prediction)


def retrieve_resolution(text, retriever):
    vectorizer = retriever["vectorizer"]
    matrix = retriever["matrix"]
    customer_texts = retriever["customer_texts"]
    replies = retriever["replies"]

    query_vector = vectorizer.transform([text])
    scores = cosine_similarity(query_vector, matrix).ravel()
    best_index = scores.argmax()

    return {
        "customer_text": customer_texts[best_index],
        "reply": replies[best_index],
        "similarity": float(scores[best_index]),
    }


def decide(intent, similarity):
    if intent in {
        "Account Security",
        "Payment, Gift Card & Billing Issue",
        "Wrong or Defective Item Received",
    }:
        return (
            "Escalate",
            "Sensitive or high-risk issue requires human review.",
        )

    if similarity < 0.20:
        return (
            "Escalate",
            "Retrieved historical evidence is too weak.",
        )

    return (
        "Auto-handle",
        "Supported intent with sufficiently relevant historical evidence.",
    )


def run_agent(customer_text, classifier, retriever):
    customer_text = clean_text(customer_text)

    intent = predict_intent(
        customer_text,
        classifier,
    )

    evidence = retrieve_resolution(
        customer_text,
        retriever,
    )

    reply = generate_reply(
        customer_text,
        intent,
        evidence["reply"],
    )

    decision, reason = decide(
        intent,
        evidence["similarity"],
    )

    return {
        "customer_text": customer_text,
        "intent": intent,
        "historical_customer": evidence["customer_text"],
        "historical_reply": evidence["reply"],
        "similarity": round(evidence["similarity"], 4),
        "reply": reply,
        "decision": decision,
        "decision_reason": reason,
    }


def main():
    input_file = ROOT / "data" / "golden_set.csv"
    output_file = ROOT / "data" / "agent_results.csv"

    classifier = load_classifier()
    retriever = load_retriever()

    with input_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    results = []

    for row in rows:
        result = run_agent(
            row["tweet"],
            classifier,
            retriever,
        )

        results.append({
            "id": row["id"],
            "tweet": row["tweet"],
            "gold_intent": row["final_intent"],
            **result,
        })

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    auto = sum(
        r["decision"] == "Auto-handle"
        for r in results
    )

    escalate = sum(
        r["decision"] == "Escalate"
        for r in results
    )

    print("========================================")
    print("HIVER SUPPORT AGENT")
    print("========================================")
    print(f"Examples processed : {len(results)}")
    print(f"Auto-handle        : {auto}")
    print(f"Escalate           : {escalate}")
    print(f"Output             : {output_file}")
    print("========================================")


if __name__ == "__main__":
    main()
