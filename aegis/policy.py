"""Explicit, model-bound selective-answer policies. Scores are not probabilities."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


def wilson(successes, total, z=1.96):
    if not total:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0, center - half), min(1, center + half)]


def context_fingerprint(
    encoder_sha256, mode="hybrid", k=5, candidates=40, corpus_sha256=None
):
    config = {
        "encoder_sha256": encoder_sha256,
        "corpus_sha256": corpus_sha256,
        "mode": mode,
        "k": k,
        "candidates": candidates,
        "chunk_words": 220,
        "overlap_words": 35,
        "reader_version": "batched-spans-v2",
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class AnswerPolicy:
    name: str
    threshold: float
    reader_sha256: str
    context_sha256: str
    validation_passed: bool
    artifact_sha256: str
    verifier_sha256: str | None = None
    require_agreement: bool = False

    @classmethod
    def load(cls, path, reader_sha256, context_sha256):
        payload = Path(path).read_bytes()
        data = json.loads(payload)
        if (
            data["reader_sha256"] != reader_sha256
            or data["context_sha256"] != context_sha256
        ):
            raise ValueError(
                "Policy does not match the current reader/retrieval configuration"
            )
        if data["validation_passed"] is not True:
            raise ValueError("Policy did not pass its independent validation gate")
        threshold = float(data["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("Invalid policy threshold")
        return cls(
            data["name"],
            threshold,
            reader_sha256,
            context_sha256,
            True,
            hashlib.sha256(payload).hexdigest(),
            data.get("verifier_sha256"),
            data.get("require_agreement", False),
        )
