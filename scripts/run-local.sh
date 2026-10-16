#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
exec .venv/bin/streamlit run app.py --server.address 127.0.0.1
