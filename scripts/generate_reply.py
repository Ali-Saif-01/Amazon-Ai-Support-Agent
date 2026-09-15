import csv
import pickle
import re

from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\^\w+", "", text)

    # Repair common spacing artifacts found in TWCS text.
    replacements = {
        "theaccount": "the account",
        "pleasemention": "please mention",
        "did notarrive": "did not arrive",
        "deliverystatus": "delivery status",
        "orderid": "order id",
        "customerservice": "customer service",
        "contactyou": "contact you",
        "withthe": "with the",
        "inthe": "in the",
        "onthe": "on the",
        "forthe": "for the",
        "fromthe": "from the",
        "tothe": "to the",
        "ofthe": "of the",
        "yourdetails": "your details",
        "shareyour": "share your",
        "ifyou": "if you",
        "wecan": "we can",
        "you've": "you've",
    }

    for old, new in replacements.items():
        text = re.sub(rf"(?i)\b{re.escape(old)}\b", new, text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_retriever():
    with open("data/retriever.pkl", "rb") as f:
        return pickle.load(f)


def retrieve_resolution(customer_text, retriever, top_k=1):
    vectorizer = retriever["vectorizer"]
    matrix = retriever["matrix"]
    customer_texts = retriever["customer_texts"]
    replies = retriever["replies"]

    query_vector = vectorizer.transform([customer_text])
    scores = cosine_similarity(query_vector, matrix).ravel()

    best_indices = scores.argsort()[-top_k:][::-1]

    results = []

    for index in best_indices:
        results.append({
            "customer_text": customer_texts[index],
            "reply": replies[index],
            "similarity": float(scores[index])
        })

    return results


def generate_reply(customer_text, intent, historical_reply):
    historical_reply = clean_text(historical_reply)

    # Historical replies may contain handles, names, URLs, signatures,
    # and formatting artifacts. They are evidence, not customer-facing text.
    evidence_available = bool(historical_reply)

    if not evidence_available:
        return (
            "Sorry you're having trouble. "
            "Please share a few more details about the issue so we can help."
        )

    if intent == "Delivery Delay / Service Issue":
        return (
            "I'm sorry your delivery has been delayed. "
            "Please check the latest tracking information and confirm whether "
            "the expected delivery date has passed. If it has, please contact "
            "support so they can look into the delivery."
        )

    if intent == "Shipping Speed / Prime SLA Not Met":
        return (
            "I'm sorry the delivery did not arrive within the expected time. "
            "Please check the latest tracking information and confirm whether "
            "the promised delivery window has passed. If it has, please "
            "contact support so they can investigate."
        )

    if intent == "Marked Delivered But Not Received":
        return (
            "I'm sorry you haven't received your package even though it "
            "shows as delivered. Please check the delivery details and "
            "contact support so they can help locate the package."
        )

    if intent == "Order Cancellation & Refund":
        return (
            "Sorry for the trouble with your order. Please contact support "
            "so the team can check the cancellation or refund status and "
            "help resolve the issue."
        )

    if intent == "Wrong or Defective Item Received":
        return (
            "I'm sorry the item you received wasn't as expected. "
            "Please use the return or replacement process so support can "
            "help resolve the issue."
        )

    if intent == "Payment, Gift Card & Billing Issue":
        return (
            "Sorry for the trouble with your payment or billing. "
            "Please contact support through the secure support channel so "
            "the team can review the transaction and help resolve the issue."
        )

    if intent == "Account Security":
        return (
            "Sorry you're having trouble with your account. "
            "Please use the account-support process to verify and secure "
            "your account. If you don't recognize an order or activity, "
            "please mention that when contacting support."
        )

    if intent == "Device & App Technical Issue":
        return (
            "Sorry you're having trouble with your device or the Amazon app. "
            "Please share the specific problem you're experiencing so "
            "support can help troubleshoot it."
        )

    if intent == "Prime Membership, Billing & Video":
        return (
            "Sorry you're having trouble with Prime or Prime Video. "
            "Please share the specific issue so support can check the "
            "relevant account or content details and help you."
        )

    if intent == "Customer Service Quality Complaint":
        return (
            "I'm sorry about your customer-service experience. "
            "Please share the details of what happened so the support team "
            "can review the issue and assist you."
        )

    if intent == "Packaging Feedback":
        return (
            "Thank you for your feedback about the packaging. "
            "We're sorry the packaging wasn't as expected. "
            "Please share the details with support so the feedback can be "
            "reviewed by the appropriate team."
        )

    if intent == "General Help Request / Greeting":
        return (
            "Hi! We'd be happy to help. "
            "Please tell us a little more about what you need assistance with."
        )

    return (
        "Sorry you're having trouble. "
        "Please share a few more details about the issue so we can help."
    )

def main():
    input_file = "data/golden_set.csv"
    output_file = "data/generated_replies.csv"

    retriever = load_retriever()

    with open(input_file, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    output_rows = []

    for row in rows:
        customer_text = clean_text(row["tweet"])

        retrieved = retrieve_resolution(
            customer_text,
            retriever,
            top_k=1
        )[0]

        reply = generate_reply(
            customer_text,
            row["final_intent"],
            retrieved["reply"]
        )

        output_rows.append({
            "id": row["id"],
            "tweet": row["tweet"],
            "intent": row["final_intent"],
            "retrieved_customer": retrieved["customer_text"],
            "retrieved_reply": retrieved["reply"],
            "similarity": round(retrieved["similarity"], 4),
            "generated_reply": reply
        })

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "tweet",
                "intent",
                "retrieved_customer",
                "retrieved_reply",
                "similarity",
                "generated_reply"
            ]
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("Reply generation complete.")
    print(f"Examples generated: {len(output_rows)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()