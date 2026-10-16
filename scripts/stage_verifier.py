"""Network-enabled staging only; inference always uses the saved local directory."""

import os
import json
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

sources = json.loads(Path("models.lock.json").read_text())
revision = sources["verifier"]["revision"]

repository = "deepset/tinyroberta-squad2"
AutoTokenizer.from_pretrained(repository, revision=revision).save_pretrained(
    "models/verifier"
)
AutoModelForQuestionAnswering.from_pretrained(
    repository, revision=revision
).save_pretrained("models/verifier")
print("Verifier staged")
