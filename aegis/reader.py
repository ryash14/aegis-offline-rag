"""Batched local span reading with exact source offsets and explicit answer policy."""

import time
from dataclasses import asdict
from pathlib import Path
from . import core


class Reader:
    def __init__(self, path="models/reader", threads=4):
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering

        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"Reader missing: {path}; run scripts/stage_reader.py during setup"
            )
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            path, local_files_only=True
        ).eval()
        self.fingerprint = core.model_hash(path)

    def predict(self, question, passages):
        """Return the best raw candidate before abstention; never invent answer text."""
        if not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain 1–4000 characters")
        if len(passages) > 40:
            raise ValueError("Reader accepts at most 40 passages")
        started = time.perf_counter()
        active = [(i, p) for i, p in enumerate(passages, 1) if p.text.strip()]
        if not active:
            return None, 0.0
        # Bound tokenization allocation for adversarial or accidentally oversized inputs.
        if any(len(p.text) > 100_000 for _, p in active):
            raise ValueError("Reader passages are limited to 100,000 characters")
        qids = self.tokenizer.encode(question, add_special_tokens=False)[:64]
        question = self.tokenizer.decode(qids, skip_special_tokens=True)
        tokens = self.tokenizer(
            [question] * len(active),
            [p.text for _, p in active],
            max_length=384,
            stride=128,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            k: v
            for k, v in tokens.items()
            if k in {"input_ids", "attention_mask", "token_type_ids"}
        }
        best_by_passage = {}
        for offset in range(0, len(tokens["input_ids"]), 8):
            with self.torch.inference_mode():
                output = self.model(
                    **{k: v[offset : offset + 8] for k, v in inputs.items()}
                )
            for local in range(len(output.start_logits)):
                feature = offset + local
                sample = int(tokens["overflow_to_sample_mapping"][feature])
                index, passage = active[sample]
                seq = tokens.sequence_ids(feature)
                offsets = tokens["offset_mapping"][feature].tolist()
                starts = output.start_logits[local].tolist()
                ends = output.end_logits[local].tolist()
                cls = (
                    tokens["input_ids"][feature]
                    .tolist()
                    .index(self.tokenizer.cls_token_id)
                )
                null = starts[cls] + ends[cls]
                valid = [
                    i
                    for i, s in enumerate(seq)
                    if s == 1 and offsets[i][1] > offsets[i][0]
                ]
                top_start = sorted(valid, key=lambda i: starts[i], reverse=True)[:20]
                top_end = sorted(valid, key=lambda i: ends[i], reverse=True)[:20]
                for s in top_start:
                    for e in top_end:
                        if e < s or e - s + 1 > 30:
                            continue
                        margin = starts[s] + ends[e] - null
                        a, b = offsets[s][0], offsets[e][1]
                        previous = best_by_passage.get(index)
                        if previous is None or margin > previous["margin"]:
                            text = passage.text[a:b]
                            assert text and 0 <= a < b <= len(passage.text)
                            best_by_passage[index] = {
                                "margin": margin,
                                "start": a,
                                "end": b,
                                "label": f"S{index}",
                                "text": text,
                                "passage": passage,
                            }
        ranked = sorted(
            best_by_passage.values(), key=lambda c: (-c["margin"], c["label"])
        )
        best = ranked[0] if ranked else None
        if best:
            different = next(
                (
                    c
                    for c in ranked[1:]
                    if c["text"].casefold() != best["text"].casefold()
                ),
                None,
            )
            best["alternative_gap"] = (
                best["margin"] - different["margin"] if different else None
            )
        return best, round((time.perf_counter() - started) * 1000, 2)

    def answer(self, question, passages, threshold=0.0, policy=None):
        if policy:
            if policy.require_agreement:
                raise ValueError(
                    "This policy requires VerifiedReader; single-reader bypass rejected"
                )
            if policy.reader_sha256 != self.fingerprint or not policy.validation_passed:
                raise ValueError("Reader policy mismatch or unvalidated policy")
            threshold = policy.threshold
        candidate, elapsed = self.predict(question, passages)
        return self.render(candidate, elapsed, threshold, policy)

    @staticmethod
    def render(candidate, elapsed, threshold=0.0, policy=None):
        accepted = candidate is not None and candidate["margin"] > threshold
        shared = {
            "generation_ms": elapsed,
            "reader_ms": elapsed,
            "reader_threshold": threshold,
            "reader_margin": candidate["margin"] if candidate else None,
            "alternative_gap": candidate.get("alternative_gap") if candidate else None,
            "backend": "extractive-minilm",
            "tokens_per_second": None,
            "answer_policy": policy.name if policy else "balanced-zero-threshold",
            "policy_sha256": policy.artifact_sha256 if policy else None,
        }
        if not accepted:
            return {
                **shared,
                "answer": "Insufficient evidence in the indexed documents.",
                "abstained": True,
                "citations": [],
                "citation_check": "extractive reader abstained",
            }
        passage = candidate["passage"]
        assert passage.text[candidate["start"] : candidate["end"]] == candidate["text"]
        return {
            **shared,
            "answer": f"{candidate['text']} [{candidate['label']}]",
            "abstained": False,
            "citations": [
                {
                    "label": candidate["label"],
                    **asdict(passage),
                    "char_start": candidate["start"],
                    "char_end": candidate["end"],
                }
            ],
            "citation_check": "exact substring and offsets verified; relevance NOT guaranteed",
        }
