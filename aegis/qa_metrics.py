"""SQuAD-style normalized exact match and token overlap F1."""

import re
import string
from collections import Counter


def normalize(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score(prediction, references):
    predicted = normalize(prediction).split()
    exact, f1 = 0.0, 0.0
    for reference in references or [""]:
        gold = normalize(reference).split()
        exact = max(exact, float(predicted == gold))
        if not predicted or not gold:
            value = float(predicted == gold)
        else:
            common = sum((Counter(predicted) & Counter(gold)).values())
            value = 2 * common / (len(predicted) + len(gold))
        f1 = max(f1, value)
    return {"exact_match": exact, "token_f1": f1}
