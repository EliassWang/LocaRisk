# CorruptRAG — poisoned-document attack (from PoisonRAG).
#
# Replaces a randomly chosen retrieved segment with an adversarial document
# keyed by sample ID that steers the model toward an incorrect answer.
# Payload file and dataset split are dataset-specific — see
# configs/config.json → "corrupt_rag".
#
#   CorruptRAG(payload_path).inject(articles, sample, rng) → (target_index, incorrect_answer)
#   CorruptRAG(payload_path).asr(sample, response, incorrect_answer) → bool

import json
import random
from pathlib import Path

from src.evaluation.asr import is_corrupt_rag_asr_success

PAYLOAD_PATH = "data/payloads/corrupt_rag/hotpotqa_corrputRAG-AS.json"


class CorruptRAG:
    """
    Poisoned-document attack (CorruptRAG / PoisonRAG).

    Replaces a random retrieved segment with an adversarial document that
    contains plausible but factually incorrect information, keyed by sample ID.

    ASR: model's F1 against the incorrect answer exceeds F1 against correct answer.
    """

    def __init__(self, payload_path: str = PAYLOAD_PATH):
        self.payload_path = payload_path
        self._payloads: dict | None = None

    def load(self) -> dict:
        """Load and cache all adversarial payloads (keyed by sample ID)."""
        if self._payloads is None:
            self._payloads = json.loads(Path(self.payload_path).read_text(encoding="utf-8"))
        return self._payloads

    def valid_ids(self) -> set[str]:
        """Set of sample IDs that have an adversarial payload."""
        return set(self.load().keys())

    def get_entry(self, sample_id: str) -> dict | None:
        return self.load().get(sample_id)

    def inject(
        self, articles: list[str], sample: dict, rng: random.Random
    ) -> tuple[int | None, str | None]:
        """
        Replace a random segment with the adversarial document (in-place).
        Returns (target_index, incorrect_answer), or (None, None) if no payload.
        """
        entry = self.get_entry(sample["id"])
        if entry is None:
            return None, None
        target           = rng.randrange(len(articles))
        articles[target] = entry["adv_texts"][0]
        return target, entry["incorrect answer"]

    def asr(self, sample: dict, response: str, incorrect_answer: str | None) -> bool:
        """True if the model output the incorrect answer (attack succeeded)."""
        entry = self.get_entry(sample["id"])
        # Use the payload's full "correct answer" sentence, not sample["answer"]:
        # it shares the same template as "incorrect answer" (see
        # is_corrupt_rag_asr_success), so diffing them isolates the fabricated
        # claim's distinguishing tokens instead of leftover template boilerplate.
        correct_answer = entry["correct answer"] if entry else sample["answer"]
        return is_corrupt_rag_asr_success(correct_answer, incorrect_answer, response)
