from dataclasses import asdict
from aegis.core import Store
from aegis.engine import Engine
from aegis.policy import AnswerPolicy, context_fingerprint, wilson
import json
import pytest


class Reader:
    fingerprint = "fixture-reader"
    calls = 0

    def answer(self, question, passages, policy=None):
        self.calls += 1
        return {
            "answer": passages[0].text,
            "citations": [asdict(passages[0])],
            "abstained": False,
            "generation_ms": 1,
            "reader_ms": 1,
        }


def test_cache_copies_values_and_invalidates_on_external_ingestion(tmp_path):
    path = tmp_path / "index.sqlite"
    store = Store(path)
    store.ingest("a", "Manual", [(1, "valve weekly")])
    reader = Reader()
    engine = Engine(store, reader, cache_size=2)
    first = engine.query("valve", mode="bm25")
    second = engine.query("valve", mode="bm25")
    assert not first["cache_hit"] and second["cache_hit"]
    assert reader.calls == 1 and first["request_id"] != second["request_id"]
    second["citations"][0]["text"] = "tampered by caller"
    assert engine.query("valve", mode="bm25")["citations"][0]["text"] == "valve weekly"
    writer = Store(path)
    writer.ingest("a", "Manual", [(1, "valve monthly")])
    result = engine.query("valve", mode="bm25")
    assert not result["cache_hit"] and result["answer"] == "valve monthly"
    assert result["index_revision"] > first["index_revision"]
    assert reader.calls == 2
    writer.close()
    store.close()


def test_policy_rejects_changed_corpus_and_wrong_profile(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    store.ingest("a", "Manual", [(1, "valve weekly")])
    policy = AnswerPolicy(
        "fixture",
        0,
        "fixture-reader",
        context_fingerprint(None, "bm25", corpus_sha256=store.fingerprint()),
        True,
        "fixture-policy",
    )
    engine = Engine(store, Reader())
    assert engine.query("valve", mode="bm25", policy=policy)
    store.ingest("b", "New", [(1, "radio frequency")])
    with pytest.raises(ValueError, match="recalibration required"):
        engine.query("valve", mode="bm25", policy=policy)
    store.close()


def test_policy_loader_rejects_nan_rejected_gate_and_fingerprint_mismatch(tmp_path):
    path = tmp_path / "policy.json"
    data = {
        "name": "fixture",
        "threshold": 0,
        "reader_sha256": "reader",
        "context_sha256": "ctx",
        "validation_passed": False,
    }
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="validation gate"):
        AnswerPolicy.load(path, "reader", "ctx")
    data["validation_passed"] = True
    data["threshold"] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="threshold"):
        AnswerPolicy.load(path, "reader", "ctx")
    with pytest.raises(ValueError, match="configuration"):
        AnswerPolicy.load(path, "other", "ctx")
    assert wilson(0, 0) == [0, 1]
    assert wilson(95, 100)[0] < 0.95 < wilson(95, 100)[1]
