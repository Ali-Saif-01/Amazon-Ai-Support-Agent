import csv
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score


BASE = Path(__file__).resolve().parent.parent

AGENT_FILE = BASE / "data" / "agent_results.csv"
REPLY_REVIEW_FILE = BASE / "data" / "reply_human_review_scored.csv"
RETRIEVAL_REVIEW_FILE = BASE / "data" / "retrieval_human_review.csv"
LLM_RETRIEVAL_REVIEW_FILE = BASE / "data" / "retrieval_llm_judge.csv"
REPLY_LLM_JUDGE_FILE = BASE / "data" / "reply_llm_judge.csv"
OUTPUT_FILE = BASE / "data" / "evaluation_results.json"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    agent_rows = load_csv(AGENT_FILE)

    # ------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------
    y_true = [r["gold_intent"] for r in agent_rows]
    y_pred = [r["intent"] for r in agent_rows]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # ------------------------------------------------------------
    # Escalation / auto-handle
    # ------------------------------------------------------------
    auto_count = sum(
        r["decision"].strip().lower() == "auto-handle"
        for r in agent_rows
    )

    escalate_count = sum(
        r["decision"].strip().lower() == "escalate"
        for r in agent_rows
    )

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------
    similarities = [
        float(r["similarity"])
        for r in agent_rows
        if r["similarity"].strip()
    ]

    avg_similarity = (
        sum(similarities) / len(similarities)
        if similarities else 0.0
    )

    # ------------------------------------------------------------
    # Reply quality review
    # ------------------------------------------------------------
    reply_metrics = {}

    if REPLY_REVIEW_FILE.exists():
        reply_rows = load_csv(REPLY_REVIEW_FILE)

        def avg_numeric(column):
            values = []

            for row in reply_rows:
                value = row.get(column, "").strip()

                if value:
                    try:
                        values.append(float(value))
                    except ValueError:
                        pass

            return sum(values) / len(values) if values else None

        reply_metrics = {
            "examples_reviewed": len(reply_rows),
            "relevance": avg_numeric("relevance"),
            "groundedness": avg_numeric("groundedness"),
            "helpfulness": avg_numeric("helpfulness"),
            "professional_tone": avg_numeric("professional_tone"),
            "unsupported_claims": avg_numeric("unsupported_claims"),
            "overall_quality": avg_numeric("overall_quality"),
        }

    # ------------------------------------------------------------
    # Retrieval human review
    # ------------------------------------------------------------
    retrieval_metrics = {
        "examples_reviewed": 0,
        "scored_examples": 0,
        "top1_relevance": None,
        "status": "pending_human_labels",
    }

    if RETRIEVAL_REVIEW_FILE.exists():
        retrieval_rows = load_csv(RETRIEVAL_REVIEW_FILE)

        scored = []

        for row in retrieval_rows:
            value = row.get("relevant", "").strip().lower()

            if value in {"1", "true", "yes", "y"}:
                scored.append(1)

            elif value in {"0", "false", "no", "n"}:
                scored.append(0)

        retrieval_metrics["examples_reviewed"] = len(retrieval_rows)
        retrieval_metrics["scored_examples"] = len(scored)

        if scored:
            retrieval_metrics["top1_relevance"] = (
                sum(scored) / len(scored)
            )
            retrieval_metrics["status"] = "scored"

    # ------------------------------------------------------------
    # Retrieval human vs LLM judge agreement
    # ------------------------------------------------------------
    retrieval_agreement = {
        "compared_examples": 0,
        "human_relevance": None,
        "llm_relevance": None,
        "agreement": None,
        "cohens_kappa": None,
    }

    if (
        RETRIEVAL_REVIEW_FILE.exists()
        and LLM_RETRIEVAL_REVIEW_FILE.exists()
    ):
        human_rows = load_csv(RETRIEVAL_REVIEW_FILE)
        llm_rows = load_csv(LLM_RETRIEVAL_REVIEW_FILE)

        human_labels = {
            row["golden_id"]: int(row["relevant"])
            for row in human_rows
            if row.get("relevant", "").strip() in {"0", "1"}
        }

        llm_labels = {
            row["golden_id"]: int(float(row["llm_relevant"]))
            for row in llm_rows
            if row.get("llm_relevant", "").strip()
        }

        common_ids = sorted(
            set(human_labels) & set(llm_labels)
        )

        if common_ids:
            human_values = [human_labels[i] for i in common_ids]
            llm_values = [llm_labels[i] for i in common_ids]

            agreement_count = sum(
                h == l
                for h, l in zip(human_values, llm_values)
            )

            total = len(common_ids)

            po = agreement_count / total

            human_positive = sum(human_values)
            llm_positive = sum(llm_values)

            pe = (
                (human_positive / total)
                * (llm_positive / total)
                +
                ((total - human_positive) / total)
                * ((total - llm_positive) / total)
            )

            kappa = (
                (po - pe) / (1 - pe)
                if pe != 1
                else 1.0
            )

            retrieval_agreement = {
                "compared_examples": total,
                "human_relevance": sum(human_values) / total,
                "llm_relevance": sum(llm_values) / total,
                "agreement": agreement_count / total,
                "cohens_kappa": kappa,
            }

    # ------------------------------------------------------------
    # Reply human vs LLM judge agreement
    # ------------------------------------------------------------
    reply_agreement = {
        "compared_examples": 0,
        "human_average_quality": None,
        "llm_average_quality": None,
        "exact_agreement": None,
        "cohens_kappa": None,
    }

    if (
        REPLY_REVIEW_FILE.exists()
        and REPLY_LLM_JUDGE_FILE.exists()
    ):
        human_rows = load_csv(REPLY_REVIEW_FILE)
        llm_rows = load_csv(REPLY_LLM_JUDGE_FILE)

        human_labels = {
            row["id"]: int(float(row["overall_quality"]))
            for row in human_rows
            if row.get("overall_quality", "").strip()
        }

        llm_labels = {
            row["id"]: int(float(row["llm_overall_quality"]))
            for row in llm_rows
            if row.get("llm_overall_quality", "").strip()
        }

        common_ids = sorted(
            set(human_labels) & set(llm_labels)
        )

        if common_ids:
            human_values = [human_labels[i] for i in common_ids]
            llm_values = [llm_labels[i] for i in common_ids]

            exact_matches = sum(
                h == l
                for h, l in zip(human_values, llm_values)
            )

            total = len(common_ids)
            agreement = exact_matches / total

            categories = [1, 2, 3, 4, 5]

            po = agreement

            pe = sum(
                (
                    sum(h == category for h in human_values) / total
                ) * (
                    sum(l == category for l in llm_values) / total
                )
                for category in categories
            )

            kappa = (
                (po - pe) / (1 - pe)
                if pe != 1
                else 1.0
            )

            reply_agreement = {
                "compared_examples": total,
                "human_average_quality": (
                    sum(human_values) / total
                ),
                "llm_average_quality": (
                    sum(llm_values) / total
                ),
                "exact_agreement": agreement,
                "cohens_kappa": kappa,
            }

    # ------------------------------------------------------------
    # Per-intent results
    # ------------------------------------------------------------
    per_intent = {}

    for intent in sorted(set(y_true)):
        tp = sum(
            1 for true_label, pred_label in zip(y_true, y_pred)
            if true_label == intent and pred_label == intent
        )
        fp = sum(
            1 for true_label, pred_label in zip(y_true, y_pred)
            if true_label != intent and pred_label == intent
        )
        fn = sum(
            1 for true_label, pred_label in zip(y_true, y_pred)
            if true_label == intent and pred_label != intent
        )

        count = sum(1 for true_label in y_true if true_label == intent)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_intent[intent] = {
            "count": count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ------------------------------------------------------------
    # Final machine-readable results
    # ------------------------------------------------------------
    results = {
        "examples_evaluated": len(agent_rows),

        "intent_classification": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        },

        "decisioning": {
            "auto_handle": auto_count,
            "escalate": escalate_count,
            "auto_handle_rate": auto_count / len(agent_rows),
        },

        "retrieval": {
            "average_top1_similarity": avg_similarity,
            "human_review": retrieval_metrics,
            "human_llm_agreement": retrieval_agreement,
        },

        "reply_quality": {
            "human_review": reply_metrics,
            "human_llm_agreement": reply_agreement,
        },

        "per_intent": per_intent,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------
    print("=" * 60)
    print("HIVER AGENT EVALUATION")
    print("=" * 60)

    print(f"Examples evaluated : {len(agent_rows)}")
    print(f"Intent accuracy    : {accuracy:.4f}")
    print(f"Intent macro F1    : {macro_f1:.4f}")
    print(f"Auto-handle        : {auto_count}")
    print(f"Escalate           : {escalate_count}")
    print(f"Auto-handle rate   : {auto_count / len(agent_rows):.4f}")
    print(f"Avg retrieval sim  : {avg_similarity:.4f}")

    print()
    print("REPLY QUALITY REVIEW")

    if reply_metrics:
        print(
            f"Examples reviewed  : "
            f"{reply_metrics['examples_reviewed']}"
        )
        print(
            f"Relevance          : "
            f"{reply_metrics['relevance']:.1%}"
        )
        print(
            f"Groundedness       : "
            f"{reply_metrics['groundedness']:.1%}"
        )
        print(
            f"Helpfulness        : "
            f"{reply_metrics['helpfulness']:.1%}"
        )
        print(
            f"Professional tone  : "
            f"{reply_metrics['professional_tone']:.1%}"
        )
        print(
            f"Unsupported claims : "
            f"{reply_metrics['unsupported_claims']:.1%}"
        )
        print(
            f"Overall quality    : "
            f"{reply_metrics['overall_quality']:.2f}/5"
        )
    else:
        print("No reply review data found.")

    print()
    print("RETRIEVAL HUMAN REVIEW")

    print(
        f"Rows prepared      : "
        f"{retrieval_metrics['examples_reviewed']}"
    )

    print(
        f"Rows scored        : "
        f"{retrieval_metrics['scored_examples']}"
    )

    if retrieval_metrics["top1_relevance"] is not None:
        print(
            f"Top-1 relevance    : "
            f"{retrieval_metrics['top1_relevance']:.1%}"
        )
    else:
        print("Top-1 relevance    : pending human labels")

    print()
    print("PER-INTENT RESULTS")

    for intent, metrics in per_intent.items():
        print(
            f"{intent}: "
            f"n={metrics['count']} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    print()
    print(f"Machine-readable results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
