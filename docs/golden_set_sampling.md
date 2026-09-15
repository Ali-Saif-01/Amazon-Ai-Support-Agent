# Golden Evaluation Set — Sampling & Annotation Guide

This document describes how the 200-example AmazonHelp golden evaluation set
is constructed from `twcs.csv`. It covers population definition, filtering,
sampling, leakage prevention, and annotation rules.

**Scope:** golden-set construction only. No classifier, retrieval system,
reply generator, escalation logic, or evaluation harness is built in this step.

---

## 1. Source population

| Item | Value |
|------|--------|
| Dataset | `twcs.csv` (Customer Support on Twitter) |
| Brand | `AmazonHelp` |
| Conversation definition | Connected components linked by `tweet_id → in_response_to_tweet_id` |
| Unit of sampling | Conversation-opening **customer** tweets |

### Opening tweet rule (locked)

A tweet is a conversation opener if and only if:

- `inbound == True`
- `in_response_to_tweet_id` is null/empty
- The tweet’s connected component contains at least one `AmazonHelp` tweet

### Empirically observed population (this repo’s `twcs.csv`)

| Metric | Observed | Locked expectation |
|--------|----------|--------------------|
| AmazonHelp conversations | **82,534** | ≈ 82,556 |
| Opening customer tweets | **81,347** | ≈ 81,347 |
| Non-English filtered (heuristic) | **20,657** | — |
| English opening candidates | **60,690** | — |

The conversation count differs by 22 from the approximate locked figure. The
opening-tweet count matches exactly. Sampling proceeds on the observed
population; we do **not** fabricate rows to force 82,556.

Some AmazonHelp components (~1,187) have no customer opening tweet under the
locked opener rule (e.g., brand-initiated or incomplete threads). Those
components are excluded because we only sample opening customer tweets.

---

## 2. Filtering rules

1. **Brand / conversation filter** — keep only openings in AmazonHelp components.
2. **English filter** — remove likely non-English openings.

### English heuristic (defined in this project)

No prior English-filter implementation existed in this repository. The pipeline
defines the following heuristic in `scripts/build_golden_set.py`:

- Reject tweets with substantial non-Latin scripts (CJK, Cyrillic, Arabic, Hangul, Devanagari, etc.).
- Reject low ASCII-letter ratios among alphabetic characters.
- Score English vs. common DE/ES/FR/PT/IT/NL function-word lexicons.
- Reject foreign-dominant text; require at least one English support/anchor token for short/medium tweets.

This is a **heuristic**, not a certified language ID model. Some bilingual or
code-switched tweets may be misclassified. False negatives (English removed)
and false positives (non-English kept) are both possible.

URLs, `@mentions`, and `#hashtags` are stripped before language scoring.

---

## 3. Sampling method

Sampling is **stratified** and **difficulty-aware**, not a simple uniform draw.

### Design principles

1. Every example is a real dataset tweet (no fabrication).
2. Exactly **200** examples.
3. Fixed random seed for reproducibility.
4. One example per conversation (no duplicate conversations).
5. Cover Easy / Medium / Hard cases.
6. Cover minority intents via proxy keyword strata.
7. Explicitly include known overlap / borderline cases in the Hard pool.
8. Include an **Other / Unclassified** sampling stratum so that class is not
   empty before annotation.
9. Proxy keyword matches are **sampling aids only**. They are **not** final labels.

### Random seed

```
RANDOM_SEED = 42
```

Defined in `scripts/build_golden_set.py` and recorded in
`data/golden_set_build_meta.json`.

### Difficulty assignment (sampling estimate)

Before drawing, each English opening is scored with keyword/regex proxies:

| Difficulty | Rule of thumb |
|------------|----------------|
| **Easy** | Exactly one proxy intent with ≥2 keyword hits; no overlap flags |
| **Medium** | Single weaker proxy signal, or residual non-overlap cases |
| **Hard** | Overlap flags, multi-intent signals, or other borderline ambiguity |

Annotators may override `difficulty` if the heuristic estimate is wrong.

### Hard / overlap selection

Hard examples are preferentially drawn from these overlap detectors:

- Delivery Delay **vs** Shipping Speed / Prime SLA
- Order Cancellation & Refund **vs** Payment / Gift Card / Billing
- Delivery Delay **vs** Marked Delivered But Not Received
- Customer Service Quality Complaint **vs** an actionable issue
- General Help Request **vs** a substantive issue
- Multi-signal / borderline residual hard cases

These detectors use co-occurring proxy keywords. They identify *candidates that
are likely hard to label*; they do not decide `final_intent`.

---

## 4. Target allocation (exactly 200)

### By difficulty

| Difficulty | Target |
|------------|--------|
| Easy | 70 |
| Medium | 80 |
| Hard | 50 |
| **Total** | **200** |

### Easy (70) — by proxy stratum

| Proxy stratum | n |
|---------------|---|
| Delivery Delay / Service Issue | 8 |
| Marked Delivered But Not Received | 6 |
| Order Cancellation & Refund | 7 |
| Wrong or Defective Item Received | 6 |
| Customer Service Quality Complaint | 4 |
| Prime Membership, Billing & Video | 5 |
| Payment, Gift Card & Billing Issue | 6 |
| Account Security | 4 |
| Device & App Technical Issue | 6 |
| Shipping Speed / Prime SLA Not Met | 5 |
| Packaging Feedback | 4 |
| General Help Request / Greeting | 5 |
| Other / Unclassified | 4 |

### Medium (80) — by proxy stratum

| Proxy stratum | n |
|---------------|---|
| Delivery Delay / Service Issue | 9 |
| Marked Delivered But Not Received | 6 |
| Order Cancellation & Refund | 8 |
| Wrong or Defective Item Received | 6 |
| Customer Service Quality Complaint | 5 |
| Prime Membership, Billing & Video | 6 |
| Payment, Gift Card & Billing Issue | 7 |
| Account Security | 5 |
| Device & App Technical Issue | 7 |
| Shipping Speed / Prime SLA Not Met | 6 |
| Packaging Feedback | 4 |
| General Help Request / Greeting | 5 |
| Other / Unclassified | 6 |

### Hard (50) — overlap / borderline strata

| Hard stratum | n |
|--------------|---|
| overlap:Delivery vs Shipping Speed | 10 |
| overlap:Refund vs Payment/Billing | 8 |
| overlap:Delivery vs Marked Delivered | 8 |
| overlap:CS Quality vs actionable | 10 |
| overlap:General Help vs substantive | 8 |
| ambiguous:multi-signal / borderline | 6 |

If a stratum has fewer eligible tweets than requested, the sampler backfills
from related pools (documented in `golden_set_build_meta.json` shortfalls) so
the final file still contains exactly 200 unique conversations.

---

## 5. Output files

| Path | Purpose |
|------|---------|
| `data/golden_set.csv` | Annotation worksheet (200 rows) |
| `data/golden_set_reserved_ids.txt` | Tweet IDs reserved from future train/dev |
| `data/golden_set_build_meta.json` | Population stats + sampling report |

### Annotation CSV columns

```
id,tweet,final_intent,difficulty,annotator_confidence,annotation_reason,source_tweet_id
```

For the initial build:

- `final_intent` is **blank**
- `annotator_confidence` is **blank**
- `annotation_reason` is **blank**
- `difficulty` is the sampling estimate (`Easy` / `Medium` / `Hard`)
- `source_tweet_id` is the real `twcs.csv` tweet id

---

## 6. Leakage prevention

1. Golden examples are reserved in `data/golden_set_reserved_ids.txt`.
2. **Do not** use golden-set tweets/conversations to train or tune a classifier.
3. Future train/dev splits must exclude these `source_tweet_id`s (and, preferably,
   their full conversation components).
4. Do not iteratively re-sample the golden set after seeing model errors on it
   (that would overfit the eval set).

---

## 7. Annotation rules (locked taxonomy)

Annotate with **exactly one** of these 13 intents:

1. Delivery Delay / Service Issue  
2. Marked Delivered But Not Received  
3. Order Cancellation & Refund  
4. Wrong or Defective Item Received  
5. Customer Service Quality Complaint  
6. Prime Membership, Billing & Video  
7. Payment, Gift Card & Billing Issue  
8. Account Security  
9. Device & App Technical Issue  
10. Shipping Speed / Prime SLA Not Met  
11. Packaging Feedback  
12. General Help Request / Greeting  
13. Other / Unclassified  

### Labeling policy

- **Strict single-label** classification.
- **Customer Service Quality Complaint is a last-resort substantive label.**  
  If the message contains an actionable issue *and* a CS complaint, choose the
  actionable issue.  
  Example: *“My order is late and your customer service is pathetic.”*  
  → `Delivery Delay / Service Issue`
- Pure complaints about inability to get help → `Customer Service Quality Complaint`.
- **Other / Unclassified must not become a dumping ground.** Use it only when
  the tweet is genuinely outside the other 12 classes (or is not a support ask).
- Prefer the customer’s primary actionable request when multiple issues appear;
  note secondary issues in `annotation_reason`.
- Set `annotator_confidence` to `Low`, `Medium`, or `High`.
- Adjust `difficulty` if the sampling estimate disagrees with your judgment.

---

## 8. Limitations of single-annotator labeling

- No inter-annotator agreement (IAA) measurement.
- Subjective borderline cases (especially Hard overlaps) may be unstable.
- Keyword-based difficulty/stratum assignment can mis-bucket examples before
  annotation; annotators should not trust proxy strata as labels.
- English filtering errors may under-represent non-English-looking informal
  English or over-include foreign tweets with English tokens.
- Opening-tweet-only evaluation does not measure multi-turn context quality.

Recommended mitigation if time allows: double-annotate a 40–50 example subset
and resolve disagreements.

---

## 9. How to run

```bash
# Build (reproducible; seed=42)
python3 scripts/build_golden_set.py

# Validate
python3 scripts/validate_golden_set.py
```

Requires Python 3 stdlib only (no third-party packages).

---

## 10. What this step does *not* include

- Intent classifier training
- LLM reply generation
- Retrieval / RAG
- Escalation logic
- Full evaluation harness beyond golden-set validation
