import argparse
import json
import time
from dataclasses import asdict
from .core import Encoder, Store, load_reranker


def main():
    parser = argparse.ArgumentParser(
        description="AEGIS: local evidence retrieval and selective answering"
    )
    parser.add_argument("--db", default="data/index.sqlite")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("folder")
    for flag in ["lexical-only", "ocr", "tables", "sync"]:
        ingest.add_argument("--" + flag, action="store_true")
    search = commands.add_parser("search")
    search.add_argument("question")
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    for flag in ["rerank", "answer", "cpu", "reader"]:
        search.add_argument("--" + flag, action="store_true")
    search.add_argument(
        "--policy", help="Validated policy JSON, bound to corpus/retrieval/models"
    )
    search.add_argument("--model", default="qwen2.5:3b")
    search.add_argument("-k", type=int, default=5)
    delete = commands.add_parser("delete")
    delete.add_argument("doc_id")
    for name in ["stats", "doctor", "audit"]:
        commands.add_parser(name)
    backup = commands.add_parser("backup")
    backup.add_argument("destination")
    args = parser.parse_args()
    if args.command == "search" and args.policy and (not args.reader or args.rerank):
        parser.error("--policy requires --reader and does not cover reranked retrieval")
    needs_encoder = (args.command == "ingest" and not args.lexical_only) or (
        args.command == "search" and args.mode != "bm25"
    )
    store = Store(args.db, Encoder() if needs_encoder else None)
    try:
        if args.command == "ingest":
            from .ingestion import ingest_folder_report

            result = ingest_folder_report(
                store, args.folder, ocr=args.ocr, tables=args.tables, sync=args.sync
            )
            print(json.dumps(result, indent=2))
            if result["errors"]:
                raise SystemExit(2)
        elif args.command == "search":
            if args.reader and not args.rerank:
                from .reader import Reader
                from .engine import Engine
                from .policy import AnswerPolicy, context_fingerprint

                reader = Reader()
                policy = None
                verified = None
                if args.policy:
                    policy = AnswerPolicy.load(
                        args.policy,
                        reader.fingerprint,
                        context_fingerprint(
                            store.encoder.fingerprint if store.encoder else None,
                            args.mode,
                            args.k,
                            40,
                            corpus_sha256=store.fingerprint(),
                        ),
                    )
                    if policy.require_agreement:
                        from .verified import VerifiedReader

                        verified = VerifiedReader(reader)
                result = Engine(store, reader, verified).query(
                    args.question,
                    mode=args.mode,
                    k=args.k,
                    policy=policy,
                    use_cache=False,
                )
            else:
                reranker = load_reranker() if args.rerank else None
                start = time.perf_counter()
                hits = store.retrieve(
                    args.question, args.mode, args.k, reranker=reranker
                )
                result = {
                    "retrieval_ms": round((time.perf_counter() - start) * 1000, 2),
                    "passages": [asdict(p) for p in hits],
                    "index_revision": store.revision(),
                }
                if args.reader:
                    from .reader import Reader

                    result.update(Reader().answer(args.question, hits))
                elif args.answer:
                    from .generation import answer

                    result.update(answer(args.question, hits, args.model, cpu=args.cpu))
                store.record_query(args.question, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "doctor":
            result = store.integrity()
            result["audit"] = store.verify_audit()
            print(json.dumps(result, indent=2))
            if not result["ok"] or not result["audit"]["ok"]:
                raise SystemExit(1)
        elif args.command == "audit":
            print(json.dumps(store.verify_audit(), indent=2))
        elif args.command == "backup":
            print(json.dumps({"backup": store.backup(args.destination)}))
        elif args.command == "delete":
            store.delete(args.doc_id)
            print("Deleted")
        else:
            print(
                json.dumps(
                    {
                        "documents": store.db.execute(
                            "SELECT count(*) FROM documents"
                        ).fetchone()[0],
                        "chunks": store.db.execute(
                            "SELECT count(*) FROM passages"
                        ).fetchone()[0],
                        "revision": store.revision(),
                        "corpus_sha256": store.fingerprint(),
                    }
                )
            )
    finally:
        store.close()


if __name__ == "__main__":
    main()
