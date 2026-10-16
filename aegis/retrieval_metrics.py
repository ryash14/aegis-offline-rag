"""Retrieval relevance metrics used by regression tests."""

import math


def metrics(ranking, relevant):
    ranking = list(dict.fromkeys(ranking))
    relevant = {key: value for key, value in relevant.items() if value > 0}
    out = {}
    for k in (5, 10):
        hits = [relevant.get(doc, 0) for doc in ranking[:k]]
        out[f"recall@{k}"] = (
            sum(value > 0 for value in hits) / len(relevant) if relevant else 0
        )
        dcg = sum((2**value - 1) / math.log2(i + 2) for i, value in enumerate(hits))
        ideal = sum(
            (2**value - 1) / math.log2(i + 2)
            for i, value in enumerate(sorted(relevant.values(), reverse=True)[:k])
        )
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0
    out["mrr@10"] = next(
        (1 / (i + 1) for i, d in enumerate(ranking[:10]) if d in relevant), 0
    )
    return out
