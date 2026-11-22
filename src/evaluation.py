"""
evaluation.py
-------------
Information Retrieval metrics from Proposal Section 3.5:
Precision@K, Mean Average Precision (MAP), NDCG@K, plus paired
significance testing between two systems.

Input conventions
-----------------
run   : dict[query_id -> list[doc_id]]   ranked best-first
qrels : dict[query_id -> set[doc_id]]    relevant docs (binary relevance)
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked[:k]
    return sum(1 for d in top if d in relevant) / k


def average_precision(ranked: Sequence[str], relevant: set[str],
                      k: int | None = None) -> float:
    """AP@k (binary relevance): sum of Precision@i at each relevant rank i <= k,
    divided by |R_q| (total relevant documents for the query)."""
    if not relevant:
        return 0.0
    if k is not None:
        ranked = ranked[:k]
    hits, score = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, d in enumerate(ranked[:k], start=1)
        if d in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def per_query_scores(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Per-query metric values (needed for paired significance tests).
    Queries without any relevant document are skipped."""
    out: dict[str, dict[str, float]] = {}
    for qid, relevant in qrels.items():
        if not relevant or qid not in run:
            continue
        ranked = run[qid]
        out[qid] = {
            f"P@{k}": precision_at_k(ranked, relevant, k),
            "AP": average_precision(ranked, relevant, k),
            f"NDCG@{k}": ndcg_at_k(ranked, relevant, k),
        }
    return out


def summarise(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries (MAP = mean of AP)."""
    if not per_query:
        return {}
    metrics = next(iter(per_query.values())).keys()
    summary = {m: float(np.mean([q[m] for q in per_query.values()])) for m in metrics}
    summary["MAP"] = summary.pop("AP")
    summary["n_queries"] = len(per_query)
    return summary


def paired_significance(
    scores_a: Mapping[str, Mapping[str, float]],
    scores_b: Mapping[str, Mapping[str, float]],
    metric: str,
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test on per-query scores of system A vs B.
    Returns mean difference (A - B), statistic and p-value."""
    from scipy.stats import wilcoxon

    common = sorted(set(scores_a) & set(scores_b))
    a = np.array([scores_a[q][metric] for q in common])
    b = np.array([scores_b[q][metric] for q in common])
    diff = a - b
    result = {"n_pairs": len(common), "mean_diff": float(diff.mean())}
    if len(common) < 6 or np.allclose(diff, 0):
        result.update(statistic=float("nan"), p_value=float("nan"))
        return result
    stat, p = wilcoxon(a, b)
    result.update(statistic=float(stat), p_value=float(p))
    return result
