#!/usr/bin/env bash
set -euo pipefail
export OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_MAX_QUEUE=4
# Bound llama.cpp's host prompt cache independently of its KV context size.
export LLAMA_ARG_CACHE_RAM=256
exec ollama serve
