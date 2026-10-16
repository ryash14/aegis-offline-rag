# AEGIS — Offline Document Intelligence

A local, CPU-capable RAG workbench for searching PDFs, Markdown and text files and answering questions with traceable document evidence. Setup downloads model weights; runtime uses local models without external APIs.

## Features

- Hybrid BM25 and dense retrieval combined with reciprocal rank fusion.
- Compact extractive reader with exact answer spans, page references and source offsets.
- Optional two-reader agreement with corpus-bound calibrated policies. Policy artifacts are not included; Balanced mode works without them.
- Local OCR and table extraction, incremental ingestion and per-file error reporting.
- Transactional SQLite indexing, revision-aware caches, consistent backups and hash-chained audit records.
- Streamlit interface and command-line tools; optional experimental Ollama generation.

## Setup

Use Python 3.12 and `uv`. Install Tesseract with English language data for OCR.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[ui,dev]'
mkdir -p data models reports
curl -4 -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o data/scifact.zip
.venv/bin/python scripts/prepare.py
.venv/bin/python scripts/stage_reader.py
# Optional second reader:
.venv/bin/python scripts/stage_verifier.py
```

`prepare.py` stages pinned encoder/reranker models and extracts the public SciFact archive. Model revisions are recorded in `models.lock.json`. Weights and datasets are not distributed in this repository. Observe upstream model and dataset licenses when staging or redistributing assets.

## Index and run

```bash
.venv/bin/python -m aegis.cli --db data/documents.sqlite ingest /absolute/path/to/documents
# Add --ocr --tables when needed.
./scripts/run-local.sh
# Open http://127.0.0.1:8501 and select My documents.
```

Or query from the terminal:

```bash
.venv/bin/python -m aegis.cli --db data/documents.sqlite search 'When must the valve be inspected?' --reader
.venv/bin/python -m aegis.cli --db data/documents.sqlite doctor
.venv/bin/python -m aegis.cli --db data/documents.sqlite audit
.venv/bin/python -m aegis.cli --db data/documents.sqlite backup data/backups/documents.sqlite
```

PDF, Markdown and UTF-8 text are supported. Failed replacements preserve the previous indexed document. Optional `--sync` prunes missing indexed documents within the specified folder after an error-free scan.

## Development

```bash
make test
make check
```

Tests that require unstaged neural models or public indexes skip when those assets are absent. Interview notes, evaluation reports, experiment scripts, model weights and local runtime data are excluded from this repository.

See [architecture](docs/ARCHITECTURE.md) and [deployment](docs/DEPLOYMENT.md) for design and operational details.

The supported scope is one trusted local user. Runtime offline flags and loopback binding do not enforce OS-level network isolation. Authentication, document ACLs and encryption at rest are not implemented. Exact source spans can still answer a question incorrectly. Hash-chained audit records require a trusted external checkpoint to detect full-chain replacement. The optional generative path is experimental.
