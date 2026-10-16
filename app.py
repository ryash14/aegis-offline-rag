"""Loopback-only workbench. Neural models never download assets at runtime."""

import os

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import time
from pathlib import Path
import streamlit as st
from aegis.core import Encoder, Store, Passage, load_reranker
from aegis.engine import Engine
from aegis.policy import AnswerPolicy, context_fingerprint
from aegis.generation import answer

st.set_page_config(page_title="AEGIS · Local Evidence", page_icon="◈", layout="wide")
st.title("◈ AEGIS / Local Evidence")
st.caption(
    "Local answers with inspectable evidence · CPU profiles · reproducible measurements"
)


@st.cache_resource
def encoder():
    return Encoder()


@st.cache_resource
def reader():
    from aegis.reader import Reader

    return Reader()


@st.cache_resource
def verifier():
    from aegis.reader import Reader

    return Reader("models/verifier")


@st.cache_resource
def ranker():
    return load_reranker()


@st.cache_resource
def engine(database, dense, verified):
    from aegis.verified import VerifiedReader

    pair = VerifiedReader(reader(), verifier()) if verified else None
    return Engine(Store(database, encoder() if dense else None), reader(), pair)


policy_path = Path("reports/v2/agreement/answer-policy.json")
policy_ready = (
    policy_path.exists()
    and json.loads(policy_path.read_text()).get("validation_passed") is True
)
with st.sidebar:
    st.header("Workspace")
    labels = {
        "data/squad.sqlite": "Public QA documents · SQuAD",
        "data/scifact.sqlite": "Scientific abstracts · SciFact",
        "data/documents.sqlite": "My documents",
    }
    database = st.selectbox("Index", list(labels), format_func=labels.get)
    profile = st.selectbox(
        "Retrieval", ["Hybrid", "Hybrid + reranker", "BM25", "Dense"]
    )
    generate = st.toggle("Generate cited answer", True)
    answer_style = st.selectbox(
        "Answer style", ["Fast exact span (CPU)", "Experimental explanation (LLM)"]
    )
    policies = ["Balanced"] + (["Cautious · validated"] if policy_ready else [])
    chosen_policy = st.selectbox(
        "Answer policy", policies, disabled=answer_style != "Fast exact span (CPU)"
    )
    cpu = st.toggle(
        "CPU-only generation", False, disabled=answer_style == "Fast exact span (CPU)"
    )
    st.caption(
        "Cautious mode declines more questions and requires a matching calibrated collection. Balanced mode answers more questions."
    )
    with st.expander("Index your documents"):
        folder = st.text_input("Local folder path")
        ocr = st.checkbox("OCR scanned pages locally")
        tables = st.checkbox("Preserve table row/header associations")
        if st.button("Index folder", disabled=not folder):
            try:
                from aegis.ingestion import ingest_folder_report

                with st.spinner("Parsing and indexing locally…"):
                    own = Store("data/documents.sqlite", encoder())
                    try:
                        ingestion = ingest_folder_report(
                            own, folder, ocr=ocr, tables=tables
                        )
                    finally:
                        own.close()
                if ingestion["errors"]:
                    st.warning(
                        "Some files could not be indexed; their previous versions were preserved."
                    )
                else:
                    st.success(
                        f"{ingestion['changed_chunks']} new or updated passages; {ingestion['unchanged_files']} unchanged files."
                    )
                st.json(ingestion)
            except Exception as exc:
                st.error(str(exc))
    st.caption(
        "Single-user workbench bound to this computer. Authentication and multiuser document permissions are outside this deployment scope."
    )

if not Path(database).exists():
    st.info("Index documents or prepare the public evaluation corpus.")
    st.stop()
store = Store(database, encoder() if profile != "BM25" else None)
try:
    cols = st.columns(3)
    cols[0].metric(
        "Documents", store.db.execute("SELECT count(*) FROM documents").fetchone()[0]
    )
    cols[1].metric(
        "Passages", store.db.execute("SELECT count(*) FROM passages").fetchone()[0]
    )
    cols[2].metric(
        "Answer mode",
        chosen_policy
        if answer_style == "Fast exact span (CPU)"
        else "Experimental LLM",
    )
    question = st.text_area(
        "Ask a question",
        placeholder="Who is viewed as the first modern geologist?",
        max_chars=4000,
    )
    if st.button(
        "Find evidence", key="search", type="primary", disabled=not question.strip()
    ):
        try:
            mode = {
                "Hybrid": "hybrid",
                "Hybrid + reranker": "hybrid",
                "BM25": "bm25",
                "Dense": "dense",
            }[profile]
            policy = None
            if (
                chosen_policy != "Balanced"
                and answer_style == "Fast exact span (CPU)"
                and generate
            ):
                if profile != "Hybrid":
                    raise ValueError(
                        "This calibrated policy requires Hybrid retrieval without reranking."
                    )
                policy = AnswerPolicy.load(
                    policy_path,
                    reader().fingerprint,
                    context_fingerprint(
                        encoder().fingerprint, corpus_sha256=store.fingerprint()
                    ),
                )
            result = None
            with st.spinner("Reading local evidence…"):
                if (
                    generate
                    and answer_style == "Fast exact span (CPU)"
                    and profile != "Hybrid + reranker"
                ):
                    result = engine(
                        database, profile != "BM25", policy is not None
                    ).query(question, mode=mode, policy=policy)
                    hits = [Passage(**row) for row in result["evidence"]]
                    retrieval_ms = result["retrieval_ms"]
                else:
                    start = time.perf_counter()
                    hits = store.retrieve(
                        question,
                        mode,
                        reranker=ranker() if profile == "Hybrid + reranker" else None,
                    )
                    retrieval_ms = (time.perf_counter() - start) * 1000
                    if generate:
                        result = (
                            reader().answer(question, hits)
                            if answer_style == "Fast exact span (CPU)"
                            else answer(question, hits, cpu=cpu)
                        )
                        result["index_revision"] = (
                            hits[0].index_revision if hits else store.revision()
                        )
                        store.record_query(question, result)
            st.caption(f"Retrieval: {retrieval_ms:.1f} ms · model loading excluded")
            if result:
                st.markdown(result["answer"])
                timing = f"Answer: {result['generation_ms'] / 1000:.2f} s"
                if result.get("cache_hit"):
                    timing += " · cached for this index revision"
                if result.get("tokens_per_second") is not None:
                    timing += f" · {result['tokens_per_second']} tokens/s"
                st.caption(timing)
                st.caption(result["citation_check"])
                if result.get("verification"):
                    st.caption(
                        "Two readers must agree before cautious mode returns an answer."
                    )
                st.download_button(
                    "Save answer and provenance",
                    json.dumps(result, indent=2),
                    "aegis-answer.json",
                    "application/json",
                )
            for i, hit in enumerate(hits, 1):
                with st.expander(
                    f"[S{i}] {hit.title} · page {hit.page}", expanded=i == 1
                ):
                    st.write(hit.text)
                    st.caption(
                        f"{hit.source} · Document {hit.doc_id} · Passage {hit.id} · Index revision {hit.index_revision}"
                    )
        except Exception as exc:
            st.error(f"Request failed: {exc}")
    with st.expander("Measured CPU answer benchmark"):
        report = Path("reports/v2/agreement/selective-answering.json")
        if report.exists():
            data = json.loads(report.read_text())
            st.caption(
                "600 paragraph-disjoint SQuAD dev questions; precision is among answered questions. Coverage shows how often the system answers."
            )
            st.dataframe(
                [
                    {
                        "Mode": label,
                        "Answered": values["answered"],
                        "Coverage": values["coverage"],
                        "Exact precision": values["exact_precision"],
                    }
                    for label, values in [
                        ("Balanced", data["baseline"]),
                        ("Cautious", data["cautious"]),
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
        old = Path("reports/squad-rag.json")
        if old.exists():
            st.caption(
                "Historical v1 run: 200 sampled answerable questions, no supplied gold context."
            )
            st.json(json.loads(old.read_text())["summary"])
    with st.expander("Measured retrieval benchmark"):
        report = Path("reports/scifact.json")
        if report.exists():
            data = json.loads(report.read_text())
            st.caption(
                f"BEIR SciFact · {data['query_count']} held-out queries; retrieval relevance, not answer accuracy."
            )
            st.dataframe(
                [
                    {
                        "Profile": mode,
                        "Recall@5": v["recall@5"],
                        "nDCG@10": v["ndcg@10"],
                        "p95 ms": v["latency_p95_ms"],
                    }
                    for mode, v in data["results"].items()
                ],
                hide_index=True,
                width="stretch",
            )
finally:
    store.close()
