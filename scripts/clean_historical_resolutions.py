import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "historical_resolutions.csv"
GOLDEN = ROOT / "data" / "golden_set.csv"
OUTPUT = ROOT / "data" / "historical_resolutions_clean.csv"


SUPPORT_TERMS = [
    "order", "delivery", "delivered", "package", "parcel",
    "refund", "return", "replacement", "replace",
    "shipping", "shipment", "account", "payment", "charged",
    "charge", "card", "gift", "prime", "membership",
    "app", "device", "fire", "kindle", "alexa",
    "customer service", "help", "problem", "issue",
    "broken", "damaged", "missing", "late", "cancel",
    "cancellation", "seller", "courier", "tracking",
    "arrive", "arrived", "received", "wrong", "faulty",
    "login", "password", "security", "billing"
]


def is_english_like(text):
    """
    Lightweight heuristic.
    We don't need perfect language detection;
    we mainly want to remove obvious non-English text.
    """

    if not text:
        return False

    # Reject obvious non-Latin scripts.
    non_latin = re.findall(
        r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff"
        r"\u0400-\u04ff\u0900-\u097f]",
        text
    )

    if len(non_latin) > 2:
        return False

    # Common English words.
    english_words = {
        "the", "and", "you", "your", "my", "is", "are",
        "was", "were", "have", "has", "can", "please",
        "what", "why", "how", "where", "when", "not",
        "this", "that", "for", "with", "from", "amazon",
        "order", "help", "need", "want", "get"
    }

    words = re.findall(r"[a-zA-Z]+", text.lower())

    if not words:
        return False

    english_hits = sum(word in english_words for word in words)

    return english_hits >= 1


def looks_support_related(text):
    text_lower = text.lower()

    return any(term in text_lower for term in SUPPORT_TERMS)


def is_social_or_chatter(text):
    """
    Remove obvious non-support social chatter.
    Keep this conservative so genuine complaints aren't removed.
    """

    t = text.lower()

    social_patterns = [
        "#halloween",
        "#giveaway",
        "#contest",
        "#quiz",
        "anyone else",
        "thanks for the style",
        "look ...i think",
        "got my #",
        "congratulations",
        "follow me",
        "giveaway",
        "retweet",
    ]

    return any(pattern in t for pattern in social_patterns)


def main():

    print("Loading golden set...")

    golden_ids = set()

    with GOLDEN.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            golden_ids.add(row["source_tweet_id"])

    print(f"Golden IDs: {len(golden_ids)}")

    print("Loading historical resolutions...")

    kept = []
    removed_golden = 0
    removed_language = 0
    removed_support = 0
    removed_social = 0
    removed_short = 0

    with INPUT.open(newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            customer_id = row["customer_tweet_id"]
            customer_text = row["customer_text"].strip()
            amazon_reply = row["amazon_reply"].strip()

            # 1. Remove golden-set leakage.
            if customer_id in golden_ids:
                removed_golden += 1
                continue

            # 2. Basic quality check.
            if len(customer_text) < 15 or len(amazon_reply) < 10:
                removed_short += 1
                continue

            # 3. English-like filter.
            if not is_english_like(customer_text):
                removed_language += 1
                continue

            # 4. Support relevance.
            if not looks_support_related(customer_text):
                removed_support += 1
                continue

            # 5. Obvious social chatter.
            if is_social_or_chatter(customer_text):
                removed_social += 1
                continue

            kept.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "customer_tweet_id",
                "amazon_reply_id",
                "customer_text",
                "amazon_reply"
            ]
        )

        writer.writeheader()
        writer.writerows(kept)

    print("\nFinished.")
    print(f"Kept:              {len(kept):,}")
    print(f"Removed golden:    {removed_golden:,}")
    print(f"Removed language:  {removed_language:,}")
    print(f"Removed support:   {removed_support:,}")
    print(f"Removed social:    {removed_social:,}")
    print(f"Removed too short: {removed_short:,}")
    print(f"\nOutput: {OUTPUT}")


if __name__ == "__main__":
    main()
