import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "historical_resolutions_clean.csv"
OUTPUT = ROOT / "data" / "weak_labeled_resolutions.csv"


INTENTS = [
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
]


def contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def classify(text):
    t = text.lower().strip()

    # ---------------------------------------------------------
    # 1. Marked Delivered But Not Received
    # ---------------------------------------------------------
    delivered = contains_any(t, [
        "says delivered",
        "said delivered",
        "marked delivered",
        "shows delivered",
        "shown as delivered",
        "tracking says delivered",
        "tracking shows delivered",
        "delivered but",
        "delivered and i did not",
        "delivered and i didn't",
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

    # ---------------------------------------------------------
    # 2. Shipping Speed / Prime SLA Not Met
    # ---------------------------------------------------------
    sla = contains_any(t, [
        "2-day",
        "2 day",
        "two day",
        "next day",
        "one-day",
        "one day",
        "same day",
        "guaranteed delivery",
        "guaranteed shipping",
        "prime delivery",
        "prime shipping",
        "prime promised",
        "prime promise",
        "shipping promise",
    ])

    speed_problem = contains_any(t, [
        "late",
        "delayed",
        "not arrived",
        "hasn't arrived",
        "has not arrived",
        "didn't arrive",
        "did not arrive",
        "still waiting",
        "not here",
        "where is",
    ])

    if sla and speed_problem:
        return "Shipping Speed / Prime SLA Not Met"

    # ---------------------------------------------------------
    # 3. Account Security
    # ---------------------------------------------------------
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
    ])

    if security:
        return "Account Security"

    # ---------------------------------------------------------
    # 4. Device & App Technical Issue
    # ---------------------------------------------------------
    device = contains_any(t, [
        "kindle",
        "fire tablet",
        "fire tv",
        "fire stick",
        "echo",
        "alexa",
        "device",
        "tablet",
        "app",
    ])

    technical = contains_any(t, [
        "not working",
        "doesn't work",
        "doesnt work",
        "won't work",
        "wont work",
        "broken",
        "crash",
        "crashing",
        "error",
        "freeze",
        "frozen",
        "can't open",
        "cannot open",
        "not responding",
        "problem with",
        "issue with",
    ])

    if device and technical:
        return "Device & App Technical Issue"

    # ---------------------------------------------------------
    # 5. Wrong or Defective Item
    # ---------------------------------------------------------
    wrong_item = contains_any(t, [
        "wrong item",
        "wrong product",
        "wrong order",
        "incorrect item",
        "incorrect product",
        "different item",
        "received the wrong",
        "sent the wrong",
    ])

    defective = contains_any(t, [
        "defective",
        "faulty",
        "damaged",
        "broken item",
        "arrived broken",
        "arrived damaged",
    ])

    received = contains_any(t, [
        "received",
        "arrived",
        "got",
    ])

    if (wrong_item or defective) and received:
        return "Wrong or Defective Item Received"

    # ---------------------------------------------------------
    # 6. Packaging Feedback
    # ---------------------------------------------------------
    packaging = contains_any(t, [
        "packaging",
        "packaged",
        "poorly packed",
        "poor packaging",
        "bad packaging",
        "packing",
        "box was damaged",
        "box is damaged",
        "box was",
        "box is",
    ])

    if packaging:
        return "Packaging Feedback"

    # ---------------------------------------------------------
    # 7. Order Cancellation & Refund
    # ---------------------------------------------------------
    cancellation = contains_any(t, [
        "cancel my order",
        "cancel the order",
        "cancel order",
        "cancelled my order",
        "canceled my order",
        "cancelled order",
        "canceled order",
        "want to cancel",
        "need to cancel",
        "order cancellation",
        "cancel this order",
    ])

    refund = contains_any(t, [
        "refund",
        "money back",
        "give my money back",
        "get my money back",
        "refunded",
        "refund me",
    ])

    order_context = contains_any(t, [
        "order",
        "purchase",
    ])

    if cancellation or (refund and order_context):
        return "Order Cancellation & Refund"

    # ---------------------------------------------------------
    # 8. Prime Membership, Billing & Video
    # ---------------------------------------------------------
    prime = contains_any(t, [
        "prime membership",
        "prime member",
        "amazon prime",
        "prime subscription",
        "prime charge",
        "prime fee",
        "prime billing",
        "prime video",
        "prime account",
        "prime trial",
        "prime renewal",
    ])

    if prime:
        return "Prime Membership, Billing & Video"

    # ---------------------------------------------------------
    # 9. Payment, Gift Card & Billing
    # ---------------------------------------------------------
    payment = contains_any(t, [
        "payment",
        "payments",
        "gift card",
        "giftcard",
        "charged",
        "charge",
        "credit card",
        "debit card",
        "card charged",
        "billing",
        "cashback",
        "emi",
        "account closed",
    ])

    if payment:
        return "Payment, Gift Card & Billing Issue"

    # ---------------------------------------------------------
    # 10. Customer Service Quality Complaint
    # ---------------------------------------------------------
    service_complaint = contains_any(t, [
        "customer service",
        "customer support",
        "terrible service",
        "bad service",
        "poor service",
        "worst service",
        "horrible service",
        "awful service",
        "terrible support",
        "bad support",
        "poor support",
        "no help",
        "nobody helped",
        "no one helped",
        "different answers",
        "no response",
        "nobody responded",
    ])

    if service_complaint:
        return "Customer Service Quality Complaint"

    # ---------------------------------------------------------
    # 11. Delivery Delay / Service Issue
    # ---------------------------------------------------------
    delivery = contains_any(t, [
        "delivery",
        "delivered",
        "shipping",
        "shipment",
        "courier",
        "package",
        "parcel",
        "arrive",
        "arrived",
    ])

    delay = contains_any(t, [
        "late",
        "delay",
        "delayed",
        "still waiting",
        "where is my",
        "not arrived",
        "hasn't arrived",
        "has not arrived",
        "didn't arrive",
        "did not arrive",
        "taking too long",
        "overdue",
        "not here",
    ])

    if delivery and delay:
        return "Delivery Delay / Service Issue"

    # ---------------------------------------------------------
    # 12. General Help Request / Greeting
    # ---------------------------------------------------------
    general_help = contains_any(t, [
        "hello amazon",
        "hi amazon",
        "hey amazon",
        "hello amazonhelp",
        "hi amazonhelp",
        "hey amazonhelp",
        "need assistance",
        "can someone assist",
        "can you assist",
        "i need help",
        "please assist",
    ])

    if general_help:
        return "General Help Request / Greeting"

    # ---------------------------------------------------------
    # No high-confidence rule matched
    # ---------------------------------------------------------
    return None


def main():
    print("Loading clean historical resolutions...")

    rows = []

    with INPUT.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = classify(row["customer_text"])

            if label is None:
                continue

            row["weak_intent"] = label
            rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "customer_tweet_id",
            "amazon_reply_id",
            "customer_text",
            "amazon_reply",
            "weak_intent",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nWeak labeling complete.")
    print(f"Labeled rows: {len(rows):,}")
    print(f"Output: {OUTPUT}")

    counts = {}

    for row in rows:
        intent = row["weak_intent"]
        counts[intent] = counts.get(intent, 0) + 1

    print("\nWeak-label distribution:")

    for intent, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{count:6,}  {intent}")


if __name__ == "__main__":
    main()