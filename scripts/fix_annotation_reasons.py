import csv
import os
import tempfile

CSV_PATH = "data/golden_set.csv"

REASONS = {
    "golden_005": "Customer reports the delivery driver refused to travel an additional distance, causing them to collect the package themselves.",
    "golden_007": "Customer cannot reset their password because the new password is still rejected during login.",
    "golden_012": "Customer is simply enjoying their Sunday and receiving a delivery; there is no support issue or request.",
    "golden_089": "Customer wants to know whether unwanted concert tickets can be returned or resold.",
    "golden_125": "Customer complains that the delivery driver left the package inside a food bin, potentially affecting the item's condition.",
    "golden_126": "Customer complains that the packaging is excessive.",
    "golden_127": "Package was supposedly left in the building lobby, but the customer never received it.",
    "golden_129": "Customer reports that their Firestick is stuck in a reboot loop.",
    "golden_130": "Customer says the package was left outside the door but was incorrectly recorded as handed directly to them.",
    "golden_131": "Customer wants a direct customer service phone number instead of information about contacting support.",
    "golden_132": "Customer reports receiving a fake product from the same seller for the second time.",
    "golden_134": "Customer says multiple agents promised callbacks, while items were not delivered despite being marked as delivered.",
    "golden_135": "Customer wants confirmation that a gift card can be delivered by a specific date.",
    "golden_136": "Customer reports having to wait 18 days after payment to receive the requested item.",
    "golden_139": "Customer cannot access Amazon Help and wants to chat about an order.",
    "golden_140": "Customer complains about Amazon's service and warns others against buying imported items.",
    "golden_141": "Customer reports that their parcel has been delayed until December 12.",
    "golden_145": "Customer asks about checks on Amazon delivery personnel because they have seen reports of delivery-related fraud.",
    "golden_146": "Customer suspects their account may have been hacked after receiving a suspicious email and cannot log in.",
    "golden_147": "Customer says someone is using their email and they can no longer access the account.",
    "golden_148": "Customer complains that the security code arrives too late and expires before they can use it.",
    "golden_154": "Customer complains that Prime is no longer providing next-day delivery and that parcels are being left on footpaths.",
    "golden_155": "Customer paid for one-day shipping for a gift but the delivery was delayed.",
    "golden_156": "Customer is still waiting for customer support to help and complains about poor service.",
    "golden_159": "Customer received a damaged product and has been waiting a week for a replacement without updates.",
    "golden_162": "Customer ordered an item more than three weeks ago and still has not received it.",
    "golden_163": "Customer says their Amazon Pay cashback has been delayed since October 10.",
    "golden_164": "Customer reports that an Amazon Pay refund has still not appeared after 10 days.",
    "golden_171": "Customer says they were charged for Prime despite not switching to Prime and wants the money refunded.",
    "golden_174": "Customer says no delivery attempt was made and the order was simply marked undeliverable despite being a Prime order.",
    "golden_179": "Customer received a damaged product and wants either a replacement or a refund.",
    "golden_180": "Customer returned the product five days ago but has still not received the refund.",
    "golden_181": "Customer complains that Prime delivery took 3–4 days instead of the promised two-day shipping.",
    "golden_184": "Customer says they were charged twice for two bundled items and wants the extra money refunded.",
    "golden_185": "Customer's order was never delered and was automatically returned, forcing them to place the order again.",
    "golden_190": "Customer reports that two different deliveries expected that day were both delayed.",
    "golden_191": "Customer was promised a customer service callback within 24 hours but is still waiting.",
    "golden_192": "Customer received a book with damaged packaging and says they no longer have enough time to return and replace it.",
    "golden_193": "Customer says their last two Prime purchases took a week despite paying for two-day shipping.",
    "golden_194": "Customer cancelled an order but did not receive a refund for the credit card rewards used to purchase it.",
    "golden_195": "Customer bought a Fire Tablet at full price and asks whether Amazon can refund the difference after the price dropped.",
    "golden_196": "Customer reports multiple Prime orders arriving late, including one that took a week.",
    "golden_200": "Customer complains that Prime two-day delivery is taking five days instead of the promised timeframe.",
}

with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

if len(rows) != 200:
    raise RuntimeError(f"Expected 200 rows, found {len(rows)}")

if fieldnames != [
    "id",
    "tweet",
    "final_intent",
    "difficulty",
    "annotator_confidence",
    "annotation_reason",
    "source_tweet_id",
]:
    raise RuntimeError(f"Unexpected columns: {fieldnames}")

# Make sure the first four annotations are untouched.
for row in rows[:4]:
    if row["id"] not in {
        "golden_001",
        "golden_002",
        "golden_003",
        "golden_004",
    }:
        raise RuntimeError("Unexpected first four IDs")

changed = 0

for row in rows:
    row_id = row["id"]

    if row_id in REASONS:
        row["annotation_reason"] = REASONS[row_id]
        changed += 1

# Atomic replacement: write a temporary file first, then replace the CSV.
directory = os.path.dirname(CSV_PATH) or "."
fd, temp_path = tempfile.mkstemp(
    prefix="golden_set_",
    suffix=".csv",
    dir=directory,
    text=True,
)

try:
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_path, CSV_PATH)
except Exception:
    if os.path.exists(temp_path):
        os.remove(temp_path)
    raise

print(f"Updated annotation reasons: {changed}")
print(f"Rows preserved: {len(rows)}")
print("Tweets/intents/difficulty/confidence/source IDs were not modified.")
