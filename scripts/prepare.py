"""Network-enabled staging only. Never imported by the runtime."""

import json
import os
from pathlib import Path
import hashlib
import zipfile

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from sentence_transformers import SentenceTransformer, CrossEncoder

sources = json.loads(Path("models.lock.json").read_text())
Path("models").mkdir(exist_ok=True)
for name, repo, cls in [
    ("encoder", "sentence-transformers/all-MiniLM-L6-v2", SentenceTransformer),
    ("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2", CrossEncoder),
]:
    target = Path("models") / name
    if not target.exists():
        print(f"Staging {repo}", flush=True)
        cls(repo, device="cpu", revision=sources[name]["revision"]).save_pretrained(
            str(target)
        )
if (
    hashlib.md5(Path("data/scifact.zip").read_bytes()).hexdigest()
    != "5f7d1de60b170fc8027bb7898e2efca1"
):
    raise ValueError("SciFact archive does not match the published BEIR checksum")
with zipfile.ZipFile("data/scifact.zip") as archive:
    root = Path("data").resolve()
    for name in archive.namelist():
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError("Unsafe archive path")
    archive.extractall(root)
manifest = {}
for root in [Path("models"), Path("data/scifact")]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
Path("reports/assets.json").write_text(json.dumps(manifest, indent=2))
print("Staging complete; runtime uses local model paths only.", flush=True)
