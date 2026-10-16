import pytest
from aegis.core import Passage
from aegis.reader import Reader
from aegis.verified import VerifiedReader
from aegis.policy import AnswerPolicy


class Model:
    def __init__(self, name, text, margin=20):
        self.fingerprint = name
        self.text = text
        self.margin = margin
        self.calls = 0

    def predict(self, question, passages):
        self.calls += 1
        passage = passages[0]
        start = passage.text.find(self.text)
        return {
            "text": self.text,
            "passage": passage,
            "label": "S1",
            "start": start,
            "end": start + len(self.text),
            "margin": self.margin,
        }, 1


def policy():
    return AnswerPolicy(
        "fixture", 10, "primary", "ctx", True, "artifact", "verifier", True
    )


def test_agreement_and_disagreement_are_enforced():
    passages = [
        Passage("p", "doc", "Fixture", "The beacon is blue and the relay is red.")
    ]
    first = Model("primary", "blue")
    second = Model("verifier", "blue")
    result = VerifiedReader(first, second).answer(
        "What color is the beacon?", passages, policy()
    )
    assert not result["abstained"] and result["verification"]["agreement"]
    second.text = "red"
    result = VerifiedReader(first, second).answer(
        "What color is the beacon?", passages, policy()
    )
    assert result["abstained"] and result["citations"] == []
    first.margin = 5
    second.calls = 0
    result = VerifiedReader(first, second).answer("Question?", passages, policy())
    assert result["abstained"] and second.calls == 0


def test_verified_policy_cannot_bypass_verifier():
    instance = Reader.__new__(Reader)
    instance.fingerprint = "primary"
    with pytest.raises(ValueError, match="bypass rejected"):
        instance.answer("Question?", [], policy=policy())
