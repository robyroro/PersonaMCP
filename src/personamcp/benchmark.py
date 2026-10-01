from __future__ import annotations

import statistics
from typing import Any, Protocol

from personamcp.analysis import analyze_messages, profile_rows, words
from personamcp.embeddings import EmbeddingProvider
from personamcp.retrieval import cosine, find_similar
from personamcp.storage import Store


class CandidateResponseProvider(Protocol):
    """Future optional generators can implement this; no built-in cloud integration."""

    def generate(self, incoming: str, writing_context: dict[str, Any]) -> str: ...


def style_similarity(candidate: str, actual: str) -> dict[str, float]:
    a, b = set(words(candidate)), set(words(actual))
    length = 1 - abs(len(candidate) - len(actual)) / max(len(candidate), len(actual), 1)
    vocabulary = len(a & b) / max(len(a | b), 1)
    punct = ".,?!:;"
    punctuation = 1 - sum(abs(candidate.count(p) - actual.count(p)) for p in punct) / max(
        sum(candidate.count(p) + actual.count(p) for p in punct), 1
    )
    capitalization = float(candidate[:1].isupper() == actual[:1].isupper())
    return {
        "response_length_similarity": round(length, 4),
        "vocabulary_overlap": round(vocabulary, 4),
        "punctuation_similarity": round(punctuation, 4),
        "capitalization_similarity": capitalization,
    }


def benchmark(
    store: Store,
    provider: EmbeddingProvider | None = None,
    limit: int = 30,
    candidate_provider: CandidateResponseProvider | None = None,
) -> dict[str, Any]:
    if not 1 <= limit <= 200:
        raise ValueError("Benchmark limit must be between 1 and 200")
    interactions = store.rows("SELECT * FROM interactions ORDER BY timestamp,id")
    if len(interactions) < 10:
        raise ValueError("Benchmark requires at least 10 historical interactions")
    split = max(1, int(len(interactions) * 0.8))
    cutoff = interactions[split]["timestamp"]
    training = [r for r in interactions if r["timestamp"] < cutoff]
    holdout = [r for r in interactions if r["timestamp"] >= cutoff][:limit]
    if not training:
        raise ValueError("Temporal holdout needs messages at different timestamps")
    scores, retrieved_count, modes = [], 0, set()
    for actual in holdout:
        found = find_similar(
            store,
            actual["incoming"],
            provider=provider,
            person=actual["person"],
            platform=actual["platform"],
            before=cutoff,
            limit=3,
        )
        modes.add(found["retrieval_mode"])
        if not found["results"]:
            continue
        retrieved_count += 1
        # Style profiles are recomputed using training-only text; the hidden reply cannot leak in.
        profile = analyze_messages(profile_rows(store, person=actual["person"], before=cutoff))
        if candidate_provider:
            candidate = candidate_provider.generate(
                actual["incoming"],
                {
                    "style_profile": profile,
                    "examples": found["results"],
                    "trust_notice": found["trust_notice"],
                },
            )
        else:
            candidate = found["results"][0]["historical_quote"]["user_reply"]
        metrics = style_similarity(candidate, actual["user_reply"])
        metrics["style_similarity_score"] = round(statistics.mean(metrics.values()), 4)
        if provider:
            vectors = provider.encode([candidate, actual["user_reply"]])
            metrics["embedding_similarity"] = round(cosine(*vectors), 4)
        scores.append(metrics)
    averages = (
        {k: round(statistics.mean(s[k] for s in scores), 4) for k in scores[0]} if scores else {}
    )
    return {
        "method": "temporal_80_20_holdout",
        "cutoff": cutoff,
        "training_interactions": len(training),
        "holdout_interactions": len(holdout),
        "retrieved_cases": retrieved_count,
        "retrieval_coverage": round(retrieved_count / max(len(holdout), 1), 4),
        "retrieval_modes": sorted(modes),
        "metrics": averages,
        "candidate_source": "optional_provider" if candidate_provider else "retrieved_reply",
        "limitations": "Coverage and surface style similarity of retrieved past replies. "
        "No proof of imitation, authorship, relevance, identity, or LLM generation quality. "
        "Scores are averaged over retrieved cases; missing cases reduce coverage. "
        "Profiles/retrieval use only pre-cutoff history; raw replies are never exported.",
    }
