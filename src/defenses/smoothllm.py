"""SmoothLLM defense for batched prompt perturbation."""

import random

import numpy as np
import torch

from src.defenses.probe_core.intervene import (
    get_segment_risky_counts,
    select_top_risky_segment_ids,
)
from src.defenses.probe_core.rl import rl
from src.evaluation.cost import empty_cost_summary
from src.inference.model import LLM


class SmoothLLM:
    """SmoothLLM.

    Title: SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks.
    Authors: Alexander Robey, Eric Wong, Hamed Hassani, and George J. Pappas.
    Paper: https://arxiv.org/abs/2310.03684
    """

    def __init__(
        self,
        target_model,
        pert_type: str = "swap",
        pert_pct: int = 10,
        num_copies: int = 10,
    ):
        if pert_type not in (None, "RandomSwapPerturbation", "swap"):
            raise ValueError("SmoothLLM only supports RandomSwapPerturbation.")

        self.target_model = target_model
        self.pert_type = "swap" if pert_type is None else pert_type
        self.pert_pct = pert_pct
        self.num_copies = num_copies
        self.defense_cost = empty_cost_summary()

    def _get_runtime(self):
        if isinstance(self.target_model, dict):
            return self.target_model
        runtime = getattr(self.target_model, "runtime", None)
        return runtime if isinstance(runtime, dict) else None

    def is_jailbroken(self, output: str) -> bool:
        return "unknown" in normalize_alnum_tokens(output)

    def perturb(self, text: str) -> str:
        if len(text) < 2 or self.pert_pct <= 0:
            return text

        characters = list(text)
        swap_count = min(
            len(text) - 1,
            max(1, int(len(text) * self.pert_pct / 100)),
        )
        for index in random.sample(range(len(text) - 1), swap_count):
            characters[index], characters[index + 1] = characters[index + 1], characters[index]
        return "".join(characters)

    def generate_batch(
        self,
        batch: list[str],
        *,
        task: str | None = None,
        max_new_tokens: int,
        do_sample: bool = False,
    ) -> list[str]:
        runtime = self._get_runtime()
        if runtime is not None:
            return LLM.send_prompts(
                runtime,
                batch,
                task=task,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                source="smoothllm",
                log_cost=True,
            )
        return self.target_model(batch=batch, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        *,
        task: str | None = None,
        batch_size: int = 1,
        max_new_len: int = 100,
        do_sample: bool = False,
    ) -> str:
        runtime = self._get_runtime()
        if runtime is not None:
            LLM.log_cost(runtime, reset=True)
        self.defense_cost = empty_cost_summary()

        perturbed_prompts = [self.perturb(prompt) for _ in range(self.num_copies)]
        outputs = []
        try:
            for start in range(0, self.num_copies, batch_size):
                batch = perturbed_prompts[start : start + batch_size]
                if not batch:
                    continue

                outputs.extend(
                    self.generate_batch(
                        batch,
                        task=task,
                        max_new_tokens=max_new_len,
                        do_sample=do_sample,
                    )
                )
                torch.cuda.empty_cache()

            labels = [self.is_jailbroken(output) for output in outputs]
            if not labels:
                raise ValueError("LLM did not generate any outputs.")

            majority_label = bool(np.mean(labels) > 0.5)
            if majority_label:
                return "contaminated"

            safe_outputs = [
                output
                for output, label in zip(outputs, labels)
                if label == majority_label
            ]
            return random.choice(safe_outputs)
        finally:
            if runtime is not None:
                self.defense_cost = LLM.log_cost(runtime)


class SmoothLocatedLLM(SmoothLLM):
    """Localized SmoothLLM that perturbs only RL-tagged risky segments."""

    def __init__(
        self,
        target_model,
        tokenizer,
        model_name: str,
        docs_number: int,
        tau: float,
        freq_dataset: str = "openwebtext",
        segment_top_pct: float = 20.0,
        seed: int = 42,
        pert_type: str = "swap",
        pert_pct: int = 10,
        num_copies: int = 10,
    ):
        super().__init__(
            target_model=target_model,
            pert_type=pert_type,
            pert_pct=pert_pct,
            num_copies=num_copies,
        )
        self.tokenizer = tokenizer
        self.tau = tau
        self.segment_top_pct = segment_top_pct
        self.seed = seed
        self.freq_data_path = f"data/corpus_freqs/{freq_dataset}/{model_name}_{docs_number}.json"

    def _tokenize_segments(self, segments: list[str]) -> tuple[list[list[tuple[str, int]]], set[int]]:
        tokenized_segments = []
        target_token_ids = set()

        for text in segments:
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            input_ids = encoding["input_ids"]
            segment_tokens = []

            for token_id, (start, end) in zip(input_ids, encoding["offset_mapping"]):
                token_text = text[start:end]
                segment_tokens.append((token_text, token_id))
                target_token_ids.add(token_id)

            tokenized_segments.append(segment_tokens)

        return tokenized_segments, target_token_ids

    def locate_risky_segments(
        self,
        segments: list[str],
        case_id=None,
    ) -> tuple[set[int], list[int]]:
        tokenized_segments, target_token_ids = self._tokenize_segments(segments)
        risky_ids = rl(
            freq_data_path=self.freq_data_path,
            tau=self.tau,
            target_token_ids=target_token_ids,
        )
        segment_risky_counts = get_segment_risky_counts(tokenized_segments, risky_ids)
        selected_segment_ids = select_top_risky_segment_ids(
            segment_risky_counts,
            target_percentage=self.segment_top_pct,
            rng=random.Random(f"{self.seed}:{case_id}:segment-select"),
        )
        return selected_segment_ids, segment_risky_counts

    def build_perturbed_contexts(
        self,
        segments: list[str],
        case_id=None,
    ) -> tuple[list[str], dict]:
        selected_segment_ids, segment_risky_counts = self.locate_risky_segments(
            segments,
            case_id=case_id,
        )

        if not selected_segment_ids:
            return ["\n".join(segments) for _ in range(self.num_copies)], {
                "selected_segment_ids": [],
                "selected_segment_count": 0,
                "segment_risky_counts": segment_risky_counts,
            }

        perturbed_contexts = []
        for _ in range(self.num_copies):
            perturbed_segments = list(segments)
            for segment_id in selected_segment_ids:
                perturbed_segments[segment_id] = self.perturb(perturbed_segments[segment_id])
            perturbed_contexts.append("\n".join(perturbed_segments))

        return perturbed_contexts, {
            "selected_segment_ids": sorted(selected_segment_ids),
            "selected_segment_count": len(selected_segment_ids),
            "segment_risky_counts": segment_risky_counts,
        }

    @staticmethod
    def _merge_task_and_prompt(prompt: str, task: str | None) -> str:
        return f"{task}:\n{prompt}" if task else prompt

    @torch.no_grad()
    def generate_from_segments(
        self,
        segments: list[str],
        *,
        case_id=None,
        task: str | None = None,
        batch_size: int = 1,
        max_new_len: int = 100,
        do_sample: bool = False,
    ) -> tuple[str, dict]:
        runtime = self._get_runtime()
        if runtime is not None:
            LLM.log_cost(runtime, reset=True)
        self.defense_cost = empty_cost_summary()

        perturbed_prompts, localization_details = self.build_perturbed_contexts(
            segments,
            case_id=case_id,
        )
        merged_perturbed_prompts = [
            self._merge_task_and_prompt(prompt, task)
            for prompt in perturbed_prompts
        ]
        outputs = []

        try:
            for start in range(0, len(merged_perturbed_prompts), batch_size):
                batch = merged_perturbed_prompts[start : start + batch_size]
                if not batch:
                    continue

                outputs.extend(
                    self.generate_batch(
                        batch,
                        task=None,
                        max_new_tokens=max_new_len,
                        do_sample=do_sample,
                    )
                )
                torch.cuda.empty_cache()

            labels = [self.is_jailbroken(output) for output in outputs]
            if not labels:
                raise ValueError("LLM did not generate any outputs.")

            majority_label = bool(np.mean(labels) > 0.5)
            if majority_label:
                response = "contaminated"
            else:
                safe_outputs = [
                    output
                    for output, label in zip(outputs, labels)
                    if label == majority_label
                ]
                response = random.choice(safe_outputs)

            localization_details.update({
                "num_copies": self.num_copies,
                "pert_type": self.pert_type,
                "pert_pct": self.pert_pct,
                "segment_top_pct": self.segment_top_pct,
                "majority_label_is_jailbroken": majority_label,
            })
            return response, localization_details
        finally:
            if runtime is not None:
                self.defense_cost = LLM.log_cost(runtime)

__all__ = ["SmoothLLM", "SmoothLocatedLLM"]
