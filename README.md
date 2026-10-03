# AmazonHelp AI Support Agent

End-to-end AI support-agent pipeline built on the Twitter Customer Support dataset.
## 1. Project Overview

This project builds an AI-assisted customer-support agent for AmazonHelp using the Twitter Customer Support dataset (TWCS).

For each incoming customer message, the pipeline:
- classifies the customer issue into a small support-intent taxonomy
- retrieves a similar historical customer-support case
- produces a concise customer-facing response
- decides whether the case should be auto-handled or escalated

The system is evaluated on a fixed 200-example golden evaluation set and compared against a majority-class baseline and a keyword/rule baseline.

## 2. Dataset and Intent Taxonomy

The project uses the Twitter Customer Support (TWCS) dataset and focuses on customer conversations involving AmazonHelp.

The historical corpus contains customer-to-AmazonHelp resolution pairs. After language, support-quality, social-content, length, and golden-set leakage filtering, 114,141 clean historical resolution pairs were retained for the support pipeline.

The evaluation set contains 200 fixed, manually annotated AmazonHelp customer messages across the following 13 intents:

- Delivery Delay / Service Issue
- Marked Delivered But Not Received
- Order Cancellation & Refund
- Wrong or Defective Item Received
- Customer Service Quality Complaint
- Prime Membership, Billing & Video
- Payment, Gift Card & Billing Issue
- Account Security
- Device & App Technical Issue
- Shipping Speed / Prime SLA Not Met
- Packaging Feedback
- General Help Request / Greeting
- Other / Unclassified

The golden set is kept separate from the historical training/retrieval corpus to reduce evaluation leakage.

## 3. System Architecture

The pipeline follows a simple, reproducible sequence:

```text
Incoming customer message
        |
        v
Text cleaning
        |
        v
Intent classification
  TF-IDF + Logistic Regression
  + high-precision rule overrides
        |
        v
Historical case retrieval
  TF-IDF cosine similarity
        |
        v
Deterministic response generation
  intent-specific support templates
        |
        v
Escalation decision
  sensitive intents + low retrieval confidence
        |
        v
Final response + auto-handle / escalate decision
```

The implementation deliberately separates classification, retrieval, response generation, and escalation so that each component can be evaluated independently.

## 4. Model and Evaluation Results

The main classifier uses word-level TF-IDF features with Logistic Regression and class balancing. A small set of high-precision rule overrides handles important support distinctions such as delivered-but-not-received orders, explicit delivery promises, and account-security signals.

The final rule-assisted classifier was evaluated on the fixed 200-example golden set.

### Intent Classification

- Intent accuracy: 62.50%
- Intent macro F1: 0.6272
- Plain TF-IDF + Logistic Regression accuracy: 61.00%
- Plain TF-IDF + Logistic Regression macro F1: 0.6137

The rule-assisted classifier therefore improved accuracy by 1.50 percentage points and macro F1 by 0.0135 over the plain classifier.

### Retrieval

- Average top-1 retrieval similarity: 0.2962
- Human-reviewed top-1 relevance: 71% (100 reviewed examples)
- LLM-judged top-1 relevance: 80% (100 reviewed examples)
- Human/LLM agreement: 75%
- Cohen's kappa: 0.332

The retrieval similarity score is a lexical TF-IDF similarity measure, not a semantic relevance score. Human and LLM retrieval judgments are reported separately; the LLM judgments are not treated as ground truth.

### Escalation

- Auto-handle decisions: 126 / 200 (63.0%)
- Escalations: 74 / 200 (37.0%)

Escalation is intentionally conservative and is based on the predicted intent and the available evidence rather than attempting to answer every customer message automatically.

## 5. Baseline Comparison

Two simple baselines were evaluated on the same 200-example golden set:

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 16.00% | 0.0230 |
| Keyword/rule baseline | 42.00% | 0.4470 |
| Rule-assisted TF-IDF + Logistic Regression | 62.50% | 0.6272 |

The majority baseline always predicts the most frequent intent in the golden set. The keyword/rule baseline uses deterministic keyword and pattern matching. The final system improves substantially over both baselines on the fixed evaluation set.

## 6. Response Quality Evaluation

A fixed 50-example response review was evaluated using a structured rubric covering relevance, groundedness, helpfulness, professional tone, and unsupported claims.

Results:

- Relevance: 84.0%
- Groundedness: 98.0%
- Helpfulness: 72.0%
- Professional tone: 100.0%
- Unsupported claims: 0.0%
- Average quality score: 3.44 / 5

The same 50 reviewed responses were also scored by an LLM judge using the
documented response-quality rubric. The recorded judge scores had an average
of 3.38 / 5 and matched the human overall-quality score exactly on 94% of
examples (Cohen's kappa = 0.887).

- Human average quality: 3.44 / 5
- LLM-judge average quality: 3.38 / 5
- Human/LLM exact agreement: 94%
- Cohen's kappa: 0.887

The LLM-judge scores were produced in the ChatGPT environment because no
external LLM API key was available locally. They are therefore supporting
evaluation evidence rather than independent proof of evaluator reliability.
The rubric and recorded judge outputs are included in the repository.

The strongest aspect is safety and grounding: the evaluated responses avoided unsupported claims and maintained a professional tone. The main weakness is helpfulness, because deterministic intent templates can be too generic when a customer asks for a specific action or detail.

## 7. Top 5 Failure Modes

### 1. Delivery Delay vs Cancellation / Refund
Example: golden_001 is a delivery-delay complaint but the plain TF-IDF + Logistic Regression classifier predicted Order Cancellation & Refund.

Hypothesis: short delivery complaints often contain words such as order, wait, refund, or support that overlap with cancellation language.

### 2. Delivery Delay vs Shipping SLA
Example: golden_012 contains positive delivery feedback but was predicted as Shipping Speed / Prime SLA Not Met.

Hypothesis: delivery-related vocabulary is highly overlapping, while distinguishing ordinary lateness, explicit SLA violations, and positive delivery feedback requires context.

### 3. Payment / Billing vs Cancellation / Refund
Example: golden_063 is a return-related request but was classified as Payment, Gift Card & Billing Issue, leading to a generic payment-oriented response.

Hypothesis: refund and payment vocabulary appears across both order-resolution and money-movement cases, making the boundary difficult for lexical models.

### 4. Wrong / Defective Item vs Other Operational Issues
Example: golden_071 describes a wrong or defective-item situation but the classifier predicted Device & App Technical Issue.

Hypothesis: sparse customer messages can contain ambiguous product or device terminology, and weak supervision does not always capture the intended support action.

### 5. Generic templates reduce helpfulness
Examples from the response review include golden_007, golden_008, and golden_057, where the response was directionally relevant but did not address the customer's specific request in enough detail.

Hypothesis: deterministic intent templates improve consistency and reduce unsupported claims, but they trade away specificity when useful details from historical cases are not incorporated into the final wording.

## 8. What Is Misleading About My Headline Number?

The 62.50% intent accuracy is useful, but it should not be interpreted as an estimate of production-level support-agent accuracy.

The evaluation uses a fixed 200-example golden set, so the sample is relatively small and performance can vary with a different evaluation sample. In addition, the golden set was constructed from selected AmazonHelp customer messages and covers a custom intent taxonomy, rather than representing the full distribution of real production traffic.

The response-quality review also shows that correct intent classification does not automatically produce a highly helpful response. While the reviewed responses achieved 84.0% relevance and 98.0% groundedness, helpfulness was lower at 72.0%. This indicates that the system can identify the general support category correctly while still producing an overly generic response.

Finally, retrieval similarity is lexical TF-IDF similarity rather than a semantic relevance score, and the escalation policy has not been evaluated against real operational escalation outcomes.

Therefore, the 62.50% accuracy should be viewed as an offline benchmark for this specific dataset, taxonomy, and evaluation set—not as a production success metric.

## 9. One-Week Next Steps

If I had one additional week, I would prioritize the following improvements:

1. Improve intent classification with stronger semantic representations and targeted training examples for the highest-confusion intent pairs.

2. Replace lexical TF-IDF retrieval with a semantic embedding-based retriever and evaluate retrieval relevance separately from classifier accuracy.

3. Improve response generation by incorporating useful details from retrieved historical resolutions instead of relying primarily on deterministic intent templates.

4. Build a larger, stratified human-labeled evaluation set with more difficult and multi-intent examples, especially around delivery, refunds, billing, and account security.

5. Calibrate the escalation policy using human-reviewed outcomes so that sensitive cases are escalated reliably while low-risk cases can be safely auto-handled.

6. Add automated regression tests for classification rules, retrieval behavior, response safety, and escalation decisions so future changes can be evaluated consistently.

## 10. Reproducibility / Quick Start

### Requirements

- Python 3.10+ recommended
- Install dependencies with `python3 -m pip install -r requirements.txt`
- The repository includes the cleaned historical-resolution and weak-label datasets needed to reproduce the main pipeline.
- The original TWCS dataset (`twcs.csv`) is **not committed** because of its size. It is only required for the optional golden-set provenance validation step.

### Run the main pipeline

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/train_intent_classifier.py
python3 scripts/build_retriever.py
python3 scripts/run_agent.py
python3 scripts/evaluate_agent.py
python3 scripts/evaluate_baselines.py
```
## 11. Repository Structure

```text
hiver-sde-assignment/
├── README.md
├── requirements.txt
├── data/
│   ├── golden_set.csv
│   ├── golden_set_build_meta.json
│   ├── golden_set_for_review.txt
│   ├── golden_set_reserved_ids.txt
│   ├── historical_resolutions_clean.csv
│   ├── weak_labeled_resolutions.csv
│   ├── agent_results.csv
│   ├── evaluation_results.json
│   ├── baseline_results.json
│   ├── classifier_errors.csv
│   ├── generated_replies.csv
│   ├── reply_human_review.csv
│   ├── reply_human_review_scored.csv
│   ├── retrieval_review.csv
│   ├── retrieval_human_review.csv
│   └── retrieval_llm_judge.csv
├── docs/
│   └── golden_set_sampling.md
└── scripts/
    ├── analyze_classifier_errors.py
    ├── annotate_golden_set.py
    ├── build_golden_set.py
    ├── build_retriever.py
    ├── clean_historical_resolutions.py
    ├── evaluate_agent.py
    ├── evaluate_baselines.py
    ├── evaluate_rule_assisted_classifier.py
    ├── fix_annotation_reasons.py
    ├── generate_reply.py
    ├── prepare_reply_review.py
    ├── prepare_retrieval_review.py
    ├── run_agent.py
    ├── score_reply_review.py
    ├── train_intent_classifier.py
    ├── validate_golden_set.py
    └── weak_label_resolutions.py
```
## 12. Evaluation Methodology

### Intent Classification

The primary classifier uses word-level TF-IDF features with Logistic Regression and class balancing.

A small set of high-precision rule overrides handles important operational distinctions such as:

- delivered-but-not-received orders
- explicit delivery-speed promises
- account-security signals

The classifier is evaluated against the fixed 200-example golden set.

### Retrieval

Historical customer-to-AmazonHelp resolution pairs are indexed using TF-IDF.

For each incoming customer message, the system retrieves the highest-scoring historical customer case using cosine similarity.

The retrieval score is a lexical similarity measure, not a semantic relevance probability. Therefore, retrieval similarity should not be interpreted as a direct measure of answer quality.

### Response Generation

The final customer-facing response uses deterministic intent-specific templates.

Retrieved historical responses are retained as supporting evidence, but the final wording does not directly copy historical responses.

This design improves consistency and reduces the risk of carrying usernames, URLs, signatures, or unsupported details into customer-facing responses.

### Escalation

The escalation policy is deliberately conservative.

Cases are escalated when they involve sensitive support categories or when retrieval confidence is low.

The current policy gives additional caution to:

- Account Security
- Payment, Gift Card & Billing Issue
- Wrong or Defective Item Received
- low-similarity retrieval cases

### Baselines

Two simple baselines are evaluated using exactly the same 200-example golden set:

1. **Majority-class baseline** — always predicts the most frequent intent.
2. **Keyword/rule baseline** — uses deterministic keyword and pattern matching.

Using the same evaluation set makes the comparison reproducible and directly comparable.

## 13. Response-Quality Rubric

A structured review was conducted on a 50-example response sample.

Each response was assessed using the following criteria:

| Criterion | Definition |
|---|---|
| Relevance | Does the response address the customer's actual issue? |
| Groundedness | Is the guidance supported by the available case context and evidence? |
| Helpfulness | Does the response provide a useful next step rather than only acknowledging the issue? |
| Professional tone | Is the response concise, respectful, and appropriate for customer support? |
| Unsupported claims | Does the response invent facts, guarantees, policies, or actions not supported by the available information? |

Results:

- Relevance: **84.0%**
- Groundedness: **98.0%**
- Helpfulness: **72.0%**
- Professional tone: **100.0%**
- Unsupported claims: **0.0%**
- Average quality score: **3.44 / 5**

The review shows that the response layer is generally relevant, grounded, and professional. The main weakness is helpfulness: deterministic intent templates can be too generic when a customer asks for a specific action or detail.

These results are reported separately from the 200-example intent-classification benchmark.

## 14. Known Limitations

The current system is intentionally lightweight and has several known limitations:

- **Weak supervision:** The intent classifier is trained primarily from high-precision rule-based labels rather than fully human-labeled historical data. This introduces label noise and limits coverage of ambiguous cases.

- **Lexical retrieval:** Historical-case retrieval uses TF-IDF similarity. It works well for overlapping terminology but can miss semantically similar cases expressed with different wording.

- **Deterministic response templates:** The final customer-facing reply is generated from intent-specific templates rather than fully synthesizing the retrieved historical response. This improves safety and consistency but can reduce specificity.

- **Taxonomy ambiguity:** Some customer messages contain multiple issues or unclear intent boundaries. The fixed 13-intent taxonomy cannot represent every nuance of real support conversations.

- **Small evaluation set:** The final benchmark contains 200 hand-labeled examples. This is useful for development and comparison, but it is not large enough to establish production-level performance.

- **Escalation policy:** The escalation layer uses deterministic rules and a retrieval-similarity threshold. It is designed to be conservative, but the threshold has not been calibrated against real business escalation outcomes.

These limitations mean the reported benchmark results should be interpreted as evidence of a promising prototype rather than production-readiness.

## 15. Decision Log

1. **Focused on AmazonHelp rather than the full TWCS dataset.**  
   This matches the assignment requirement and makes the support taxonomy more coherent.

2. **Used customer-to-AmazonHelp resolution pairs for the historical corpus.**  
   This keeps retrieved evidence tied to actual customer-support interactions rather than unrelated tweets.

3. **Removed non-English and low-quality historical cases.**  
   This reduces noisy retrieval evidence and keeps the corpus focused on usable support conversations.

4. **Removed historical cases overlapping with the golden evaluation set.**  
   This reduces direct evaluation leakage between the historical corpus and the held-out evaluation examples.

5. **Used a small 13-intent taxonomy.**  
   A compact taxonomy makes the classifier and escalation policy easier to interpret and evaluate.

6. **Used weak supervision to create training labels.**  
   Manually labeling more than 100,000 historical cases was impractical, so high-precision rules were used to create a training signal.

7. **Chose TF-IDF + Logistic Regression as the primary classifier.**  
   It is lightweight, reproducible, fast to train, and provides a strong interpretable baseline for this dataset.

8. **Added only high-precision rule overrides.**  
   Rules were limited to important distinctions such as delivered-but-not-received orders, explicit delivery promises, and account-security signals rather than creating a large hand-written rule system.

9. **Rejected the character-TF-IDF classifier variant.**  
   It did not improve the fixed golden-set result, so the additional complexity was not justified.

10. **Rejected hard intent-filtered retrieval.**  
    Restricting retrieval to the predicted intent reduced average top-1 lexical similarity, so the original unrestricted retriever was retained.

11. **Rejected the soft intent-aware retrieval reranker.**  
    Although the combined score increased because of the artificial intent bonus, the underlying lexical similarity of the selected result decreased. The change was therefore not treated as a genuine retrieval improvement.

12. **Used deterministic response templates for the final customer-facing wording.**  
    This improves consistency and reduces the risk of unsupported claims, while retrieved historical responses are retained as evidence.

13. **Escalated sensitive intents and low-confidence retrieval cases.**  
    Account security, payment/billing, wrong or defective items, and weak retrieval confidence are treated more conservatively.

14. **Evaluated both intent accuracy and response quality.**  
    A support agent can classify an issue correctly while still producing an unhelpful response, so classification alone was not considered sufficient.

15. **Kept the golden set fixed for all comparisons.**  
    Using the same 200 examples for the final system and both baselines makes the offline comparison reproducible and directly comparable.

    
