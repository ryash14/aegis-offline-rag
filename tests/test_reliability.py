import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from aegis.core import Store


class Encoder:
    fingerprint = "test-vectors"

    def encode(self, texts):
        out = []
        for text in texts:
            vector = np.array(
                [text.count("valve"), text.count("radio"), 0.01], dtype=np.float32
            )
            out.append(vector / np.linalg.norm(vector))
        return np.stack(out)


def test_live_dense_reader_refreshes_external_writes(tmp_path):
    path = tmp_path / "index.sqlite"
    a = Store(path, Encoder())
    a.ingest("a", "valve", [(1, "valve inspection")])
    assert a.retrieve("valve", "dense")[0].doc_id == "a"
    b = Store(path, Encoder())
    b.ingest("b", "radio", [(1, "radio channel")])
    assert a.retrieve("radio", "dense")[0].doc_id == "b"
    b.delete("b")
    assert all(p.doc_id != "b" for p in a.retrieve("radio", "dense"))
    assert a.integrity()["ok"]
    a.close()
    b.close()


def test_writers_replace_same_document_without_fts_orphans(tmp_path):
    path = tmp_path / "index.sqlite"
    Store(path).close()

    def write(worker):
        store = Store(path)
        for i in range(12):
            store.ingest(
                "shared",
                "Manual",
                [(1, f"worker {worker} version {i} valve maintenance")],
            )
        store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, range(2)))
    store = Store(path)
    assert store.integrity()["ok"]
    assert store.db.execute("SELECT count(*) FROM search").fetchone()[0] == 1
    assert store.verify_audit()["events"] == 24
    store.close()


def test_shared_connection_queries_are_serialized(tmp_path):
    store = Store(tmp_path / "index.sqlite", Encoder())
    store.ingest("a", "valve", [(1, "valve inspection")])

    def query(_):
        return store.retrieve("valve")[0].doc_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(query, range(20))) == ["a"] * 20
    store.close()


def test_rollback_and_recovery_backup(tmp_path):
    path = tmp_path / "index.sqlite"
    store = Store(path)
    store.ingest("a", "Original", [(1, "original valve procedure")])
    # Abort after deletion and before replacement; the old index must survive.
    store.db.execute(
        "CREATE TRIGGER reject_update BEFORE INSERT ON passages BEGIN SELECT RAISE(ABORT,'injected failure'); END"
    )
    store.db.commit()
    with pytest.raises(Exception, match="injected failure"):
        store.ingest("a", "Changed", [(1, "updated radio procedure")])
    assert store.retrieve("original", "bm25")
    assert store.integrity()["ok"]
    destination = tmp_path / "backup.sqlite"
    store.backup(destination)
    with pytest.raises(FileExistsError):
        store.backup(destination)
    recovered = Store(destination)
    assert recovered.retrieve("original", "bm25")
    assert recovered.fingerprint() == store.fingerprint()
    assert recovered.verify_audit()["ok"]
    recovered.close()
    store.close()
    assert path.stat().st_mode & 0o077 == 0


def test_audit_detects_tampering_and_does_not_store_queries(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    store.ingest("a", "Doc", [(1, "valve procedure")])
    question = "private secret query"
    store.record_query(
        question, {"answer": "private output", "abstained": False, "citations": []}
    )
    checkpoint = store.verify_audit()
    assert checkpoint["ok"] and checkpoint["events"] == 2
    payload = "".join(r[0] for r in store.db.execute("SELECT record FROM audit"))
    assert question not in payload and "private output" not in payload
    assert hashlib.sha256(question.encode()).hexdigest() in payload
    with store.db:
        store.db.execute("UPDATE audit SET record='{}' WHERE seq=1")
    assert not store.verify_audit(checkpoint["head"])["ok"]
    store.close()
