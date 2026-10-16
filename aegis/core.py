from __future__ import annotations
import hashlib
import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from dataclasses import dataclass
from pathlib import Path

# Set before importing model libraries. Setup downloads live in scripts/prepare.py.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def synchronized(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return run


def snapshot(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._lock, self.read_snapshot():
            return method(self, *args, **kwargs)

    return run


@dataclass
class Passage:
    id: str
    doc_id: str
    title: str
    text: str
    source: str = ""
    page: int = 1
    score: float = 0.0
    index_revision: int = 0


def chunk(text: str, size: int = 220, overlap: int = 35) -> list[str]:
    """Sentence-aware bounded word windows; oversized sentences split safely.

    This is structural chunking, not embedding-based semantic segmentation.
    """
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > overlap >= 0")
    words = text.split()
    result, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        if end < len(words):
            for boundary in range(end, start + size // 2, -1):
                if re.search(r"[.!?]$", words[boundary - 1]):
                    end = boundary
                    break
        result.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return result


class Encoder:
    def __init__(self, path="models/encoder", threads=4):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"Model missing: {path}; run scripts/prepare.py during setup"
            )
        self.model = SentenceTransformer(path, device="cpu", local_files_only=True)
        self.fingerprint = model_hash(path)

    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def model_hash(path):
    h = hashlib.sha256()
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            h.update(str(file.relative_to(path)).encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
    return h.hexdigest()


class Store:
    def __init__(self, path="data/index.sqlite", encoder=None):
        self._lock = threading.RLock()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_database = not self.path.exists()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        if new_database:
            self.path.chmod(0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY, digest TEXT);
        CREATE TABLE IF NOT EXISTS passages(id TEXT PRIMARY KEY, doc_id TEXT, title TEXT,
            text TEXT, source TEXT, page INTEGER, vector BLOB);
        CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED, title, text,
            tokenize='porter unicode61');
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY AUTOINCREMENT, previous TEXT NOT NULL, record TEXT NOT NULL, digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS ingestion_sources(doc_id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS passages_doc_id ON passages(doc_id);
        INSERT OR IGNORE INTO metadata VALUES ('revision','0');
        """)
        self.encoder = encoder
        self._vectors = None
        self._ids = []
        self._vectors_revision = None
        if encoder:
            saved = self.db.execute(
                "SELECT value FROM metadata WHERE key='encoder'"
            ).fetchone()
            if saved and saved[0] != encoder.fingerprint:
                self.db.close()
                raise ValueError(
                    "Embedding model changed; rebuild index in a new database"
                )
            if (
                not saved
                and self.db.execute("SELECT count(*) FROM passages").fetchone()[0]
            ):
                self.db.close()
                raise ValueError(
                    "Lexical-only index: rebuild with encoder to enable dense retrieval"
                )
            if not saved:
                self.db.execute(
                    "INSERT INTO metadata VALUES ('encoder', ?)", (encoder.fingerprint,)
                )
                self.db.commit()

    @synchronized
    def close(self):
        self.db.close()

    @contextmanager
    def read_snapshot(self):
        owned = not self.db.in_transaction
        if owned:
            self.db.execute("BEGIN")
        try:
            yield
        finally:
            if owned:
                self.db.rollback()

    def revision(self):
        return int(
            self.db.execute(
                "SELECT value FROM metadata WHERE key='revision'"
            ).fetchone()[0]
        )

    @snapshot
    def fingerprint(self):
        digest = hashlib.sha256()
        for doc_id, value in self.db.execute(
            "SELECT doc_id,digest FROM documents ORDER BY doc_id"
        ):
            digest.update(json.dumps([doc_id, value], ensure_ascii=False).encode())
        return digest.hexdigest()

    def _bump_revision(self):
        self.db.execute(
            "UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'"
        )

    def _audit(self, event, **fields):
        previous = self.db.execute(
            "SELECT digest FROM audit ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        previous = previous[0] if previous else "0" * 64
        record = json.dumps(
            {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **fields},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256((previous + record).encode()).hexdigest()
        self.db.execute(
            "INSERT INTO audit(previous,record,digest) VALUES (?,?,?)",
            (previous, record, digest),
        )

    @snapshot
    def verify_audit(self, expected_head=None):
        previous = "0" * 64
        count = 0
        valid = True
        for prior, record, digest in self.db.execute(
            "SELECT previous,record,digest FROM audit ORDER BY seq"
        ):
            valid = (
                valid
                and prior == previous
                and hashlib.sha256((prior + record).encode()).hexdigest() == digest
            )
            previous = digest
            count += 1
        return {
            "ok": valid and (expected_head is None or expected_head == previous),
            "events": count,
            "head": previous,
            "anchored": expected_head is not None,
        }

    @synchronized
    def record_query(self, question, result):
        # Avoid storing raw questions, answers, or source text in the audit trail.
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self._audit(
                "query",
                question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                answer_sha256=hashlib.sha256(
                    result.get("answer", "").encode()
                ).hexdigest(),
                abstained=result.get("abstained"),
                policy_sha256=result.get("policy_sha256"),
                request_id=result.get("request_id"),
                evidence_revision=result.get("index_revision"),
                source_passages=[c["id"] for c in result.get("citations", [])],
            )

    @synchronized
    def integrity(self):
        with self.read_snapshot():
            sqlite_ok = self.db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            missing = self.db.execute(
                "SELECT count(*) FROM passages p LEFT JOIN search s ON p.id=s.id WHERE s.id IS NULL"
            ).fetchone()[0]
            orphan = self.db.execute(
                "SELECT count(*) FROM search s LEFT JOIN passages p ON p.id=s.id WHERE p.id IS NULL"
            ).fetchone()[0]
            duplicates = self.db.execute(
                "SELECT count(*) FROM (SELECT id FROM search GROUP BY id HAVING count(*)>1)"
            ).fetchone()[0]
            stale = self.db.execute(
                "SELECT count(*) FROM search s JOIN passages p ON s.id=p.id WHERE s.title!=p.title OR s.text!=p.text"
            ).fetchone()[0]
            return {
                "ok": sqlite_ok and not (missing or orphan or duplicates or stale),
                "sqlite_ok": sqlite_ok,
                "missing_fts": missing,
                "orphan_fts": orphan,
                "duplicate_fts": duplicates,
                "stale_fts": stale,
                "revision": self.revision(),
            }

    @synchronized
    def backup(self, destination):
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL prevents accidentally replacing an existing backup or index.
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        connection = sqlite3.connect(target)
        try:
            self.db.backup(connection)
        finally:
            connection.close()
        return str(target)

    @synchronized
    def ingest(
        self, doc_id, title, pages, source="", size=220, overlap=35, metadata=None
    ):
        doc_id = str(doc_id)
        pages = list(pages)
        digest_input = [title, pages, source, size, overlap]
        if metadata is not None:
            digest_input.append(metadata)
        digest = hashlib.sha256(
            json.dumps(digest_input, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        previous = self.db.execute(
            "SELECT digest FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if previous and previous[0] == digest:
            return 0
        saved = self.db.execute(
            "SELECT value FROM metadata WHERE key='encoder'"
        ).fetchone()
        if saved and not self.encoder:
            raise ValueError("Dense index requires its encoder for ingestion")
        items = []
        for page, text in pages:
            for i, part in enumerate(chunk(text, size, overlap)):
                pid = hashlib.sha256(
                    f"{doc_id}:{page}:{i}:{digest}".encode()
                ).hexdigest()[:24]
                items.append(Passage(pid, str(doc_id), title, part, source, page))
        vectors = (
            self.encoder.encode([p.title + ". " + p.text for p in items])
            if self.encoder and items
            else None
        )
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute(
                "SELECT id FROM passages WHERE doc_id=?", (str(doc_id),)
            ).fetchall()
            self.db.executemany("DELETE FROM search WHERE id=?", old)
            self.db.execute("DELETE FROM passages WHERE doc_id=?", (str(doc_id),))
            for i, p in enumerate(items):
                vector = (
                    vectors[i].astype("float32").tobytes()
                    if vectors is not None
                    else None
                )
                self.db.execute(
                    "INSERT INTO passages VALUES (?,?,?,?,?,?,?)",
                    (p.id, p.doc_id, p.title, p.text, p.source, p.page, vector),
                )
                self.db.execute(
                    "INSERT INTO search VALUES (?,?,?)", (p.id, p.title, p.text)
                )
            self.db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?)", (str(doc_id), digest)
            )
            if metadata is not None:
                self.db.execute(
                    "INSERT OR REPLACE INTO ingestion_sources VALUES (?,?)",
                    (doc_id, json.dumps(metadata, sort_keys=True)),
                )
            else:
                self.db.execute(
                    "DELETE FROM ingestion_sources WHERE doc_id=?", (doc_id,)
                )
            self._bump_revision()
            self._audit(
                "ingest",
                doc_id=doc_id,
                document_digest=digest,
                revision=self.revision(),
            )
        self._vectors = None
        return len(items)

    @synchronized
    def delete(self, doc_id):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute(
                "DELETE FROM search WHERE id IN (SELECT id FROM passages WHERE doc_id=?)",
                (doc_id,),
            )
            self.db.execute("DELETE FROM passages WHERE doc_id=?", (doc_id,))
            self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self.db.execute("DELETE FROM ingestion_sources WHERE doc_id=?", (doc_id,))
            self._bump_revision()
            self._audit("delete", doc_id=doc_id, revision=self.revision())
        self._vectors = None

    @snapshot
    def lexical(self, query, k):
        terms = re.findall(r"\w+", query, re.UNICODE)[:64]
        if not terms:
            return []
        expr = " OR ".join('"' + t + '"' for t in terms)
        return [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM search WHERE search MATCH ? ORDER BY bm25(search,0,2,1), id LIMIT ?",
                (expr, k),
            )
        ]

    @snapshot
    def dense(self, query, k):
        if not self.encoder:
            raise ValueError("Dense retrieval requires a local encoder")
        import numpy as np

        revision = self.revision()
        if self._vectors is None or self._vectors_revision != revision:
            rows = self.db.execute(
                "SELECT id,vector FROM passages ORDER BY id"
            ).fetchall()
            if any(v is None for _, v in rows):
                raise ValueError("Index has missing vectors; rebuild it")
            self._ids = [r[0] for r in rows]
            self._vectors = (
                np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                if rows
                else np.empty((0, 0))
            )
            self._vectors_revision = revision
        if not self._ids:
            return []
        q = self.encoder.encode([query])[0]
        scores = self._vectors @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = sorted(top, key=lambda i: (-float(scores[i]), self._ids[i]))
        return [self._ids[i] for i in top]

    @snapshot
    def retrieve(self, query, mode="hybrid", k=5, candidates=40, reranker=None):
        if mode not in {"bm25", "dense", "hybrid"} or not 1 <= k <= candidates <= 200:
            raise ValueError("Invalid mode or require 1 <= k <= candidates <= 200")
        lists = []
        if mode in ("bm25", "hybrid"):
            lists.append(self.lexical(query, candidates))
        if mode in ("dense", "hybrid"):
            lists.append(self.dense(query, candidates))
        scores = {}
        for ids in lists:
            for rank, pid in enumerate(ids, 1):
                scores[pid] = scores.get(pid, 0) + 1 / (60 + rank)
        ids = sorted(scores, key=lambda x: (-scores[x], x))[:candidates]
        result = []
        for pid in ids:
            row = self.db.execute(
                "SELECT id,doc_id,title,text,source,page FROM passages WHERE id=?",
                (pid,),
            ).fetchone()
            result.append(
                Passage(*row, score=scores[pid], index_revision=self.revision())
            )
        if reranker and result:
            values = reranker.predict(
                [(query, p.title + ". " + p.text) for p in result]
            )
            for p, value in zip(result, values):
                p.score = float(value)
            result.sort(key=lambda p: (-p.score, p.id))
        return result[:k]


def load_reranker(path="models/reranker"):
    import torch

    torch.set_num_threads(4)
    from sentence_transformers import CrossEncoder

    if not Path(path).is_dir():
        raise FileNotFoundError(path)
    return CrossEncoder(path, device="cpu", local_files_only=True)


def ingest_folder(store, folder, *, ocr=False, tables=False):
    """Compatibility wrapper; strict errors prevent silently incomplete ingestion."""
    from .ingestion import ingest_folder_report

    report = ingest_folder_report(store, folder, ocr=ocr, tables=tables)
    if report["errors"]:
        raise ValueError(
            "; ".join(f"{e['source']}: {e['error']}" for e in report["errors"])
        )
    return report["changed_chunks"]
