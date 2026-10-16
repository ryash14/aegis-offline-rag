"""Bounded PDF parsing, optional local OCR, table rows, and resumable folder ingestion."""

import hashlib
import json
import math
import time
from pathlib import Path

PARSER_VERSION = "pdf-v2"
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_PAGES = 500


def extract_document(path, content=None, *, ocr=False, tables=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Symbolic-link documents are not accepted")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Document exceeds 50 MiB limit")
    content = path.read_bytes() if content is None else content
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Document exceeds 50 MiB limit")
    metadata = {
        "parser": PARSER_VERSION,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "source_bytes": len(content),
        "ocr_enabled": ocr,
        "tables_enabled": tables,
        "pages": [],
    }
    if path.suffix.lower() != ".pdf":
        text = content.decode("utf-8")
        if not text.strip():
            raise ValueError("Document contains no text")
        metadata["pages"] = [
            {"page": 1, "ocr": False, "tables": [], "characters": len(text)}
        ]
        return [(1, text)], metadata
    import pymupdf

    pages = []
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("Encrypted PDF requires decryption before ingestion")
        if len(pdf) > MAX_PAGES:
            raise ValueError("PDF exceeds 500-page limit")
        for i, page in enumerate(pdf):
            text = page.get_text(sort=True)
            page_meta = {
                "page": i + 1,
                "ocr": False,
                "tables": [],
                "characters": len(text),
            }
            scanned = len(text.strip()) < 20 and bool(page.get_images())
            if scanned:
                if not ocr:
                    raise ValueError(
                        f"Page {i + 1} appears scanned; enable local OCR to avoid partial ingestion"
                    )
                pixels = math.ceil(page.rect.width * 200 / 72) * math.ceil(
                    page.rect.height * 200 / 72
                )
                if pixels > 25_000_000:
                    raise ValueError(f"Page {i + 1} exceeds OCR pixel budget")
                textpage = page.get_textpage_ocr(language="eng", dpi=200, full=True)
                text = page.get_text(textpage=textpage, sort=True)
                page_meta.update(ocr=True, characters=len(text))
                if not text.strip():
                    raise ValueError(f"OCR found no text on page {i + 1}")
            if tables and text.strip() and not scanned:
                for number, table in enumerate(page.find_tables().tables, 1):
                    rows = table.extract()
                    if not rows:
                        continue
                    headers = [
                        " ".join(str(v or "").split()) or f"Column {j + 1}"
                        for j, v in enumerate(rows[0])
                    ]
                    rendered = []
                    for row in rows[1:]:
                        cells = [" ".join(str(v or "").split()) for v in row]
                        rendered.append(
                            "; ".join(
                                f"{header}: {value}"
                                for header, value in zip(headers, cells)
                            )
                        )
                    if rendered:
                        text += "\n\n" + f"Table {number} rows:\n" + "\n".join(rendered)
                    page_meta["tables"].append(
                        {
                            "number": number,
                            "bbox": list(table.bbox),
                            "rows": len(rows),
                            "columns": len(headers),
                            "headers": headers,
                        }
                    )
            pages.append((i + 1, text))
            metadata["pages"].append(page_meta)
    if not any(text.strip() for _, text in pages):
        raise ValueError("No text in PDF; OCR required for image-only documents")
    return pages, metadata


def ingest_folder_report(store, folder, *, ocr=False, tables=False, sync=False):
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError("Document folder does not exist")
    started = time.perf_counter()
    report = {
        "root": str(root),
        "parser": PARSER_VERSION,
        "files": [],
        "errors": [],
        "changed_chunks": 0,
        "unchanged_files": 0,
        "deleted_documents": 0,
    }
    seen = set()
    for path in sorted(root.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in {".txt", ".md", ".pdf"}
        ):
            continue
        relative = str(path.relative_to(root))
        doc_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
        seen.add(doc_id)
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Document escaped the ingestion root")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Document exceeds 50 MiB limit")
            content = path.read_bytes()
            if len(content) > MAX_FILE_BYTES:
                raise ValueError("Document exceeds 50 MiB limit")
            digest = hashlib.sha256(content).hexdigest()
            previous = store.db.execute(
                "SELECT metadata FROM ingestion_sources WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if previous:
                meta = json.loads(previous[0])
                if (
                    meta["source_sha256"] == digest
                    and meta["parser"] == PARSER_VERSION
                    and meta["ocr_enabled"] == ocr
                    and meta["tables_enabled"] == tables
                ):
                    report["unchanged_files"] += 1
                    report["files"].append(
                        {"source": relative, "doc_id": doc_id, "status": "unchanged"}
                    )
                    continue
            pages, meta = extract_document(path, content, ocr=ocr, tables=tables)
            meta["root"] = str(root)
            meta["relative_source"] = relative
            count = store.ingest(doc_id, path.stem, pages, relative, metadata=meta)
            report["changed_chunks"] += count
            report["files"].append(
                {
                    "source": relative,
                    "doc_id": doc_id,
                    "status": "indexed",
                    "chunks": count,
                    "pages": len(pages),
                    "ocr_pages": sum(p["ocr"] for p in meta["pages"]),
                    "tables": sum(len(p["tables"]) for p in meta["pages"]),
                    "source_sha256": digest,
                }
            )
        except (ValueError, RuntimeError, OSError, UnicodeError) as exc:
            report["errors"].append({"source": relative, "error": str(exc)})
    # Pruning is explicit and disabled on partial/error runs.
    if sync and not report["errors"]:
        previous = store.db.execute(
            "SELECT doc_id,metadata FROM ingestion_sources"
        ).fetchall()
        for doc_id, payload in previous:
            if json.loads(payload).get("root") == str(root) and doc_id not in seen:
                store.delete(doc_id)
                report["deleted_documents"] += 1
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report
