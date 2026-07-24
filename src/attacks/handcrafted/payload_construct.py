# HandcraftedInjection -- fluent natural-language instruction injection.
#
# Same wrapper structure as ObliInjection (prefix + attacker task + a
# randomly-selected benign segment as diluter + postfix, appended in-place
# to one randomly chosen retrieved segment; exactly one contaminated
# segment per sample; same "the answer is unknown" target behavior; same
# ASR definition). The only difference is that prefix/postfix are
# hand-written, grammatical English sampled per sample from a small
# template set (data/payloads/handcrafted/templates.json), instead of a
# GCG-optimized adversarial string -- so this attack is model-agnostic
# (no --model-keyed lookup, unlike ObliInjection).
#
#   HandcraftedInjection().inject(articles, rng) -> (target_index, None)
#   HandcraftedInjection().asr(sample, response)  -> bool

import json
import random
from pathlib import Path

from src.evaluation.asr import is_asr_success

PAYLOAD_PATH = "data/payloads/handcrafted/templates.json"


class HandcraftedInjection:
    """
    Fluent-language indirect prompt injection.

    Samples one (prefix, postfix) template per attacked sample from a small
    hand-written set, wraps the attacker task around a random diluter
    segment (same structure as ObliInjection.construct), and appends it to
    a random retrieved segment.

    ASR: model refuses to answer (output contains "unknown").
    """

    def __init__(self, payload_path: str = PAYLOAD_PATH):
        self.payload_path = payload_path
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            self._data = json.loads(Path(self.payload_path).read_text(encoding="utf-8"))
        return self._data

    @property
    def task(self) -> str:
        return self._load()["task"]

    def construct(self, diluter_text: str, rng: random.Random) -> str:
        """Build the injected payload string: prefix + task + diluter + postfix,
        matching ObliInjection's wrapper structure with a sampled fluent template."""
        template = rng.choice(self._load()["templates"])
        return f"{template['prefix']} {self.task} {diluter_text}\n{template['postfix']}"

    def inject(self, articles: list[str], rng: random.Random) -> tuple[int, None]:
        """
        Append payload to a random segment in-place (matches ObliInjection:
        payload is concatenated onto an existing segment, not a new one).
        Returns (target_index, None).
        """
        diluter = rng.choice(articles)
        target  = rng.randrange(len(articles))
        articles[target] += self.construct(diluter, rng)
        return target, None

    def asr(self, sample: dict, response: str, _=None) -> bool:
        """True if the model refused to answer (attack succeeded)."""
        return is_asr_success(sample["answer"], response, self.task)
