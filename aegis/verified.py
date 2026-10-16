"""Selective answering: two task-trained readers must agree on a source span."""

import time
from .qa_metrics import normalize
from .reader import Reader


class VerifiedReader:
    def __init__(self, primary=None, verifier=None):
        self.primary = primary or Reader()
        self.verifier = verifier or Reader("models/verifier")

    def answer(self, question, passages, policy):
        if (
            not policy.require_agreement
            or policy.reader_sha256 != self.primary.fingerprint
        ):
            raise ValueError("A validated two-reader policy is required")
        if (
            policy.verifier_sha256 != self.verifier.fingerprint
            or not policy.validation_passed
        ):
            raise ValueError("Verifier fingerprint mismatch or rejected policy")
        started = time.perf_counter()
        candidate, _ = self.primary.predict(question, passages)
        agreement = False
        verified = None
        if candidate and candidate["margin"] > policy.threshold:
            verified, _ = self.verifier.predict(question, [candidate["passage"]])
            agreement = bool(
                verified
                and verified["margin"] > 0
                and normalize(candidate["text"]) == normalize(verified["text"])
            )
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        accepted = candidate if agreement else None
        result = Reader.render(accepted, elapsed, policy.threshold, policy)
        result["reader_margin"] = candidate["margin"] if candidate else None
        result["verification"] = {
            "agreement": agreement,
            "verifier_sha256": self.verifier.fingerprint,
            "verifier_margin": verified["margin"] if verified else None,
            "reason": "readers agree"
            if agreement
            else ("below calibrated margin" if not verified else "reader disagreement"),
        }
        return result
