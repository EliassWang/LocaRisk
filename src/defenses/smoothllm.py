"""SmoothLLM defense for batched prompt perturbation."""

import random
import sys

import numpy as np
import torch

from src.defenses.SLcore import Intervention
from src.defenses.SLcore import RL
from src.defenses.SLcore import DEFAULT_EMBED_MODEL
from src.evaluation.cost import empty_cost_summary
from src.inference.model import LLM


def _print_copy_progress(prefix: str | None, done: int, total: int) -> None:
    """Report smoothing-loop progress so long waits don't look hung.

    Overwrites one line (\r) on a live terminal; emits a plain line per
    update when stdout is redirected to a file/log, since \r doesn't
    create new lines there and would otherwise look like a stuck frozen line.
    """
    if prefix is None:
        return
    if sys.stdout.isatty():
        end = "\n" if done == total else ""
        sys.stdout.write(f"\r{prefix}\t[smooth] perturbed copy {done}/{total} generated{end}")
    else:
        sys.stdout.write(f"{prefix}\t[smooth] perturbed copy {done}/{total} generated\n")
    sys.stdout.flush()


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
        progress_prefix: str | None = None,
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
                _print_copy_progress(progress_prefix, len(outputs), self.num_copies)

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
        embed_model: str | None = None,
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
        self.locator = "multi_signal"
        self.freq_data_path = f"data/corpus_freqs/{freq_dataset}/{model_name}_{docs_number}.json"

        locator_config = {
            "freq_data_path": self.freq_data_path, "tau": tau, "tokenizer": tokenizer,
            "embed_model": embed_model or DEFAULT_EMBED_MODEL,
        }
        self._rl = RL(locator_config=locator_config)
        self._intervention = Intervention("drop", k=segment_top_pct)

    def locate_risky_segments(
        self,
        segments: list[str],
        case_id=None,
        question: str = "",
    ) -> tuple[set[int], list[float]]:
        if not question:
            raise ValueError("multi_signal locator requires a question string")
        scores = self._rl.score_all(question, segments)
        _, selected_segment_ids = self._intervention(
            segments, scores,
            random.Random(f"{self.seed}:{case_id}"),
        )
        return selected_segment_ids, scores

    def build_perturbed_contexts(
        self,
        segments: list[str],
        case_id=None,
        question: str = "",
    ) -> tuple[list[str], dict]:
        selected_segment_ids, segment_risky_counts = self.locate_risky_segments(
            segments,
            case_id=case_id,
            question=question,
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
        question: str = "",
        task: str | None = None,
        batch_size: int = 1,
        max_new_len: int = 100,
        do_sample: bool = False,
        progress_prefix: str | None = None,
    ) -> tuple[str, dict]:
        runtime = self._get_runtime()
        if runtime is not None:
            LLM.log_cost(runtime, reset=True)
        self.defense_cost = empty_cost_summary()

        perturbed_prompts, localization_details = self.build_perturbed_contexts(
            segments,
            case_id=case_id,
            question=question,
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
                _print_copy_progress(progress_prefix, len(outputs), len(merged_perturbed_prompts))

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
