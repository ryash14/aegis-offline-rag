import json
import shutil
import pytest
from aegis.core import Store
from aegis.ingestion import extract_document, ingest_folder_report
from scripts.build_fixtures import build


@pytest.fixture
def documents(tmp_path):
    return build(tmp_path / "documents")


def test_native_pages_and_table_associations(documents):
    pages, meta = extract_document(documents / "native.pdf")
    assert len(pages) == 2 and "ORBIT-73" in pages[1][1]
    assert not any(p["ocr"] for p in meta["pages"])
    pages, meta = extract_document(documents / "table.pdf", tables=True)
    assert "Equipment: Beacon; Interval: 17 days; Code: ORBIT-41" in pages[0][1]
    assert meta["pages"][0]["tables"][0]["columns"] == 3
    assert len(meta["pages"][0]["tables"][0]["bbox"]) == 4


def test_scanned_pdf_requires_explicit_ocr(documents):
    with pytest.raises(ValueError, match="enable local OCR"):
        extract_document(documents / "scanned.pdf")
    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is required for OCR integration")
    pages, meta = extract_document(documents / "scanned.pdf", ocr=True)
    assert "MARINER" in pages[0][1] and "17 days" in pages[0][1]
    assert meta["pages"][0]["ocr"]


def test_bad_and_encrypted_files_are_reported_without_losing_good_documents(
    documents, tmp_path
):
    store = Store(tmp_path / "index.sqlite")
    report = ingest_folder_report(store, documents)
    assert len(report["errors"]) == 3  # scanned, encrypted, corrupt
    assert report["changed_chunks"] >= 4
    assert store.retrieve("lighthouse", "bm25")
    assert store.integrity()["ok"]
    second = ingest_folder_report(store, documents)
    assert second["changed_chunks"] == 0
    assert second["unchanged_files"] == 3
    store.close()


def test_failed_changed_document_preserves_previous_index_and_sync_is_scoped(tmp_path):
    folder = tmp_path / "first"
    folder.mkdir()
    path = folder / "manual.txt"
    path.write_text("Original valve inspection procedure.")
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("Independent radio procedure.")
    store = Store(tmp_path / "index.sqlite")
    ingest_folder_report(store, folder)
    ingest_folder_report(store, other)
    path.write_bytes(b"\xff\xfe invalid UTF8")
    report = ingest_folder_report(store, folder, sync=True)
    assert report["errors"] and store.retrieve("valve", "bm25")
    path.unlink()
    report = ingest_folder_report(store, folder, sync=True)
    assert report["deleted_documents"] == 1
    assert not store.retrieve("valve", "bm25") and store.retrieve("radio", "bm25")
    store.close()


def test_changed_source_byte_hash_updates_even_if_extracted_text_matches(
    documents, tmp_path
):
    store = Store(tmp_path / "index.sqlite")
    report = ingest_folder_report(store, documents)
    source = next(r for r in report["files"] if r["source"] == "native.pdf")
    metadata = json.loads(
        store.db.execute(
            "SELECT metadata FROM ingestion_sources WHERE doc_id=?", (source["doc_id"],)
        ).fetchone()[0]
    )
    assert metadata["source_sha256"] == source["source_sha256"]
    assert metadata["root"] == str(documents.resolve())
    store.close()
