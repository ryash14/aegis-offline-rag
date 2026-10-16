# Deployment scope and recovery

The supported deployment is a trusted user's local Linux workstation. Run `scripts/run-local.sh`; it binds Streamlit to 127.0.0.1 and disables usage telemetry. Neural assets load locally with offline flags. OCR uses preinstalled Tesseract English data. No internet access is required after staging.

For optional LLM explanations, stage the official GGUF and import it while the loopback daemon is running:

```bash
curl -4 -fL https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf -o models/qwen2.5-3b-instruct-q4_k_m.gguf
./scripts/serve-model.sh
# In another terminal:
ollama create qwen2.5:3b -f Modelfile
```

`serve-model.sh` disables cloud features, uses one loaded model and one parallel request, caps the queue at four, fixes a 4096-token context and caps the host prompt cache at 256 MiB. CPU extractive answering needs none of this service. LLM QA quality was poor in the saved experiment and remains explicitly experimental.

A disconnected deployment needs compatible wheels, Python, model files, indexes/documents and platform binaries/OCR data. `requirements.lock.txt` pins the measured Python environment; it is not a complete cross-platform installer. The asset manifest detects local changes against a trusted copy but is not signed upstream provenance.

## Index consistency and recovery

Use the CLI `doctor` command to check SQLite/FTS coherence and the audit chain. Create backups through the CLI `backup` command, which uses SQLite's online backup API, includes WAL-visible committed data, and refuses to overwrite an existing file. Restore by closing the application and opening a known-good backup as the selected index; run `doctor` before serving it. Tests verify that recovered indexes preserve corpus digests and retrieved content.

New index and backup files use owner-only permissions. Protect their containing directories, original documents, backups, swap and OS caches through host policy. Logical deletion is not secure erasure. Folder `--sync` is explicit and scoped; errors disable pruning.

## Audit limits

Document mutations and user-facing Engine queries append hash-chained records. Query/answer text is not stored in that trail; the interactive UI and explicit answer exports still contain source text by design. Hashes can reveal predictable inputs through guessing. Retain an audit head outside the database if you want to detect replacement of the full chain. No signing service or protection against a compromised host is claimed.

## Policy changes

The cautious artifact is valid only for its exact model/corpus/retrieval configuration. A changed source document or another collection requires a new labeled calibration experiment. The app rejects an incompatible policy instead of silently applying its precision claim to new data. Balanced mode remains available without such a policy.

Authentication, multiuser ACLs, encrypted storage and independent security certification require a separate deployment design. Do not expose this single-user workbench to a shared network as if those features existed.
