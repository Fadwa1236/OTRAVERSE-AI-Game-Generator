"""Quantitative retrieval evaluation: Precision@5, Recall@5, MRR@5, coverage."""
import json
from pathlib import Path
from statistics import mean

from app.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
K = 5


def metrics(retrieved_ids: list[str], expected_ids: list[str]) -> dict:
    """Only for supported cases with non-empty relevance labels."""
    if not expected_ids:
        raise ValueError("Unsupported requests need separate rejection evaluation")
    ids = retrieved_ids[:K]
    expected = set(expected_ids)
    hits = len(set(ids) & expected)
    first_relevant = next((i for i, value in enumerate(ids, 1) if value in expected), None)
    return {
        "precision_at_5": hits / K,
        "recall_at_5": hits / len(expected),
        "mrr_at_5": 1 / first_relevant if first_relevant else 0.0,
        "complete_coverage_at_5": float(expected.issubset(ids)),
        "missed": sorted(expected - set(ids)),
    }


def run_evaluation(retriever=None):
    with open(ROOT / "data" / "retrieval_eval.json", encoding="utf-8") as f:
        cases = json.load(f)
    retriever = retriever or Retriever()
    detail = []
    for case in cases:
        if case["group"] == "unsupported":
            continue  # Retrieval alone cannot prove the game is unsupported.
        q = case["query"]
        for strategy, docs in (
            ("full_prompt", retriever.full_prompt(q, k=K)),
            ("category_aware", retriever.category_aware(q, evaluation=True)),
        ):
            doc_ids = [d["id"] for d in docs]
            detail.append({
                "query_id": case["id"], "group": case["group"], "strategy": strategy,
                "retrieved": doc_ids, **metrics(doc_ids, case["relevant_ids"]),
            })
    summary = {}
    for strategy in ("full_prompt", "category_aware"):
        rows = [row for row in detail if row["strategy"] == strategy]
        summary[strategy] = {
            key: round(mean(r[key] for r in rows), 4)
            for key in ("precision_at_5", "recall_at_5", "mrr_at_5", "complete_coverage_at_5")
        }
        summary[strategy]["supported_queries"] = len(rows)
    output = {"summary": summary, "details": detail, "unsupported_queries_excluded": 6}
    path = ROOT / "examples" / "retrieval_report.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved detailed report: {path}")
    return output


if __name__ == "__main__":
    run_evaluation()
