"""One query path for CLI/UI: snapshot consistency, policy binding, bounded cache, audit."""

from collections import OrderedDict
from copy import deepcopy
import threading
import time
import uuid
from dataclasses import asdict
from .policy import context_fingerprint


class Engine:
    def __init__(self, store, reader, verified_reader=None, cache_size=64):
        if not 0 <= cache_size <= 1024:
            raise ValueError("Cache size must be 0–1024")
        self.store = store
        self.reader = reader
        self.verified_reader = verified_reader
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def query(
        self,
        question,
        *,
        mode="hybrid",
        k=5,
        candidates=40,
        policy=None,
        use_cache=True,
        audit=True,
    ):
        if not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain 1–4000 characters")
        started = time.perf_counter()
        with self.lock:
            with self.store._lock, self.store.read_snapshot():
                revision = self.store.revision()
                key = (
                    question,
                    mode,
                    k,
                    candidates,
                    revision,
                    policy.artifact_sha256 if policy else None,
                )
                if use_cache and key in self.cache:
                    result = deepcopy(self.cache[key])
                    self.cache.move_to_end(key)
                    result.update(
                        cache_hit=True,
                        retrieval_ms=0.0,
                        generation_ms=0.0,
                        reader_ms=0.0,
                    )
                else:
                    if policy:
                        fingerprint = context_fingerprint(
                            self.store.encoder.fingerprint
                            if self.store.encoder
                            else None,
                            mode,
                            k,
                            candidates,
                            corpus_sha256=self.store.fingerprint(),
                        )
                        if fingerprint != policy.context_sha256:
                            raise ValueError(
                                "Calibrated policy is not valid for this corpus/retrieval profile; recalibration required"
                            )
                    retrieval_started = time.perf_counter()
                    hits = self.store.retrieve(question, mode, k, candidates)
                    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
                    if policy and policy.require_agreement:
                        if not self.verified_reader:
                            raise ValueError(
                                "Verified reader is required by this policy"
                            )
                        result = self.verified_reader.answer(question, hits, policy)
                    else:
                        result = self.reader.answer(question, hits, policy=policy)
                    result.update(
                        retrieval_ms=round(retrieval_ms, 2),
                        index_revision=revision,
                        cache_hit=False,
                    )
                    result["retrieved_passages"] = [p.id for p in hits]
                    result["evidence"] = [asdict(p) for p in hits]
                    if self.cache_size and use_cache:
                        self.cache[key] = deepcopy(result)
                        while len(self.cache) > self.cache_size:
                            self.cache.popitem(last=False)
            result["request_id"] = str(uuid.uuid4())
            if audit:
                self.store.record_query(question, result)
            result["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result
