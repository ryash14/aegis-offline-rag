import json
import socket
from unittest.mock import patch
import pytest
from aegis.core import Store, chunk
from aegis.retrieval_metrics import metrics
from aegis.generation import answer


def test_chunk_bounds_and_coverage():
    words = [f"w{i}" for i in range(1000)]
    parts = chunk(" ".join(words), 50, 10)
    assert all(len(p.split()) <= 50 for p in parts)
    assert set(" ".join(parts).split()) == set(words)
    with pytest.raises(ValueError):
        chunk("abc", 10, 10)


def test_update_delete_and_fts_safety(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    assert (
        store.ingest(
            "a", "Engine", [(2, "The fuel valve requires inspection every seven days.")]
        )
        == 1
    )
    assert (
        store.ingest(
            "a", "Engine", [(2, "The fuel valve requires inspection every seven days.")]
        )
        == 0
    )
    assert store.retrieve("fuel valve", "bm25")[0].page == 2
    assert store.retrieve('" OR * NOT fuel', "bm25")
    assert store.retrieve("!!!", "bm25") == []
    store.ingest(
        "a", "Engine", [(3, "The hydraulic actuator requires annual inspection.")]
    )
    assert store.retrieve("fuel", "bm25") == []
    assert store.retrieve("hydraulic", "bm25")[0].page == 3
    store.delete("a")
    assert store.retrieve("hydraulic", "bm25") == []
    store.close()


def test_metrics_deduplicate_and_graded():
    values = metrics(["a", "a", "b"], {"b": 1})
    assert values["mrr@10"] == 0.5
    assert values["recall@5"] == 1
    assert values["ndcg@10"] == pytest.approx(1 / __import__("math").log2(3))
    assert metrics(["x"], {})["ndcg@10"] == 0


def test_offline_bm25(tmp_path):
    with patch.object(
        socket.socket, "connect", side_effect=AssertionError("Network forbidden")
    ):
        store = Store(tmp_path / "index.sqlite")
        store.ingest(
            "a", "Offline", [(1, "Local evidence never requires network access.")]
        )
        assert store.retrieve("evidence", "bm25")
        assert answer("Question", [])["abstained"]
        store.close()


def test_fail_closed_citations(tmp_path):
    store = Store(tmp_path / "index.sqlite")
    store.ingest("a", "Manual", [(1, "Inspect valve weekly.")])
    passages = store.retrieve("valve", "bm25")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "response": json.dumps(
                        {
                            "answer": "Inspect daily [S99]",
                            "citations": ["S99"],
                            "abstained": False,
                        }
                    )
                }
            ).encode()

    with patch("urllib.request.OpenerDirector.open", return_value=Response()):
        result = answer("When to inspect?", passages)
    assert result["abstained"]
    assert result["citation_check"] == "rejected invalid output"
    store.close()


def test_dense_persistence_and_model_integrity(tmp_path):
    import numpy as np

    class Encoder:
        fingerprint = "fixed-test-model"

        def encode(self, texts):
            vectors = []
            for text in texts:
                vector = np.array(
                    [text.lower().count("valve"), text.lower().count("radio"), 0.01],
                    dtype=np.float32,
                )
                vectors.append(vector / np.linalg.norm(vector))
            return np.stack(vectors)

    path = tmp_path / "dense.sqlite"
    store = Store(path, Encoder())
    store.ingest("valve", "Valve", [(1, "Valve inspection weekly.")])
    store.ingest("radio", "Radio", [(1, "Radio frequency allocation.")])
    assert store.retrieve("valve", "dense")[0].doc_id == "valve"
    store.close()
    store = Store(path, Encoder())
    assert store.retrieve("radio", "hybrid")[0].doc_id == "radio"
    store.delete("radio")
    assert all(p.doc_id != "radio" for p in store.retrieve("radio", "dense"))
    store.close()

    class Changed(Encoder):
        fingerprint = "different-model"

    with pytest.raises(ValueError, match="model changed"):
        Store(path, Changed())
    lexical = Store(path)
    with pytest.raises(ValueError, match="requires its encoder"):
        lexical.ingest("new", "New", [(1, "New evidence")])
    lexical.close()


def test_pdf_provenance(tmp_path):
    import pymupdf
    from aegis.core import ingest_folder

    folder = tmp_path / "docs"
    folder.mkdir()
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "First page radio manual.")
    doc.new_page().insert_text((72, 72), "Second page hydraulic inspection.")
    doc.save(folder / "manual.pdf")
    doc.close()
    store = Store(tmp_path / "index.sqlite")
    assert ingest_folder(store, folder) == 2
    result = store.retrieve("hydraulic", "bm25")[0]
    assert result.page == 2
    assert result.source == "manual.pdf"
    assert ingest_folder(store, folder) == 0
    store.close()


def test_context_packing_keeps_original_source_labels(tmp_path):
    from aegis.core import Passage

    passages = [
        Passage("long", "a", "Long", "x" * 1000),
        Passage("short", "b", "Short", "Inspect weekly."),
    ]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "response": json.dumps(
                        {
                            "answer": "Inspect weekly [S2]",
                            "citations": ["S2"],
                            "abstained": False,
                        }
                    )
                }
            ).encode()

    with patch("urllib.request.OpenerDirector.open", return_value=Response()):
        result = answer("When?", passages, context_chars=100)
    assert not result["abstained"]
    assert result["citations"][0]["label"] == "S2"
    assert result["citations"][0]["id"] == "short"


def test_qa_metrics():
    from aegis.qa_metrics import score

    assert score("The Eiffel Tower.", ["Eiffel Tower"]) == {
        "exact_match": 1.0,
        "token_f1": 1.0,
    }
    assert score("", []) == {"exact_match": 1.0, "token_f1": 1.0}
    assert score("Paris", ["London"])["token_f1"] == 0
    assert score("red red blue", ["red blue"])["token_f1"] == pytest.approx(0.8)


def test_structured_citation_labels_rendered_when_inline_missing(tmp_path):
    from aegis.core import Passage

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "response": json.dumps(
                        {"answer": "Paris", "citations": ["S1"], "abstained": False}
                    )
                }
            ).encode()

    with patch("urllib.request.OpenerDirector.open", return_value=Response()):
        result = answer(
            "Capital?", [Passage("p", "a", "Title", "The capital is Paris.")]
        )
    assert not result["abstained"]
    assert result["answer"] == "Paris [S1]"


def test_explicit_refusal_normalized_to_abstention():
    from aegis.core import Passage

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "response": json.dumps(
                        {
                            "answer": "Insufficient evidence. [S1]",
                            "citations": ["S1"],
                            "abstained": False,
                        }
                    )
                }
            ).encode()

    with patch("urllib.request.OpenerDirector.open", return_value=Response()):
        result = answer(
            "Weather now?",
            [Passage("p", "a", "Manual", "Valve maintenance instructions.")],
        )
    assert result["abstained"]
    assert result["citations"] == []
