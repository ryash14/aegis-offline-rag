"""Connected staging step for the compact extractive reader."""

import os
import json
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

sources = json.loads(Path("models.lock.json").read_text())
revision = sources["reader"]["revision"]

name = "deepset/minilm-uncased-squad2"
AutoTokenizer.from_pretrained(name, revision=revision).save_pretrained("models/reader")
AutoModelForQuestionAnswering.from_pretrained(name, revision=revision).save_pretrained(
    "models/reader"
)
print("Reader staged locally")
