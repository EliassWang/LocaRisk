import random

from src.defenses.probe import Probe
from src.defenses.probe_core.ablation import ablation as random_token_ablation
from src.defenses.probe_core.intervene import (
    build_risky_rows,
    get_segment_risky_counts,
    intervene_on_segment,
    select_top_risky_segment_ids,
    validate_intervention,
)
from src.defenses.probe_core.rl import rl


LOCATORS = ("freq", "random", "oracle")


def validate_locator(locator: str) -> str:
    if locator not in LOCATORS:
        valid = ", ".join(LOCATORS)
        raise ValueError(f"Unknown locator '{locator}'. Expected one of: {valid}")
    return locator


class Ablation(Probe):
    """Ablation runner for intervention and locator variants."""

    def __init__(
        self,
        llm: "LLM",
        model_name: str,
        freq_dataset: str,
        docs_number: int,
        tau: float,
        segment_top_pct: float,
        intervention: str = "swap",
        locator: str = "freq",
        seed: int = 42,
    ):
        super().__init__(
            llm=llm,
            model_name=model_name,
            freq_dataset=freq_dataset,
            docs_number=docs_number,
            tau=tau,
            segment_top_pct=segment_top_pct,
            seed=seed,
        )
        self.intervention = validate_intervention(intervention)
        self.locator = validate_locator(locator)

    def run(
        self,
        segments: list[str],
        case_id,
        oracle_segment_id: int | None = None,
    ) -> tuple[list[str], list[list[object]], list[int], set[int]]:
        if self.locator == "freq":
            return super().run(segments, case_id, oracle_segment_id=None)

        tokenized_segments, target_token_ids = self._tokenize_segments(segments)
        if self.locator == "random":
            risky_ids = random_token_ablation(
                target_token_ids=target_token_ids,
                tau=self.tau,
                seed=self.seed,
                freq_data_path=self.freq_data_path,
            )
            selected_segment_ids = self._select_from_risky_counts(
                tokenized_segments=tokenized_segments,
                risky_ids=risky_ids,
                case_id=case_id,
            )
        elif self.locator == "oracle":
            risky_ids = rl(
                freq_data_path=self.freq_data_path,
                tau=self.tau,
                target_token_ids=target_token_ids,
            )
            selected_segment_ids = (
                {oracle_segment_id}
                if oracle_segment_id is not None and 0 <= oracle_segment_id < len(segments)
                else set()
            )
        else:
            raise AssertionError(f"Unhandled locator: {self.locator}")

        return self._apply_intervention(
            segments=segments,
            tokenized_segments=tokenized_segments,
            risky_ids=risky_ids,
            selected_segment_ids=selected_segment_ids,
            case_id=case_id,
        )

    def _tokenize_segments(self, segments: list[str]) -> tuple[list[list[tuple[str, int]]], set[int]]:
        tokenized_segments = []
        target_token_ids = set()

        for text in segments:
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

            current_segment_tokens = []
            for token_id, (start, end) in zip(encoding["input_ids"], encoding["offset_mapping"]):
                current_segment_tokens.append((text[start:end], token_id))
                target_token_ids.add(token_id)

            tokenized_segments.append(current_segment_tokens)

        return tokenized_segments, target_token_ids

    def _select_from_risky_counts(
        self,
        tokenized_segments: list[list[tuple[str, int]]],
        risky_ids: list[int],
        case_id,
    ) -> set[int]:
        segment_risky_counts = get_segment_risky_counts(tokenized_segments, risky_ids)
        return select_top_risky_segment_ids(
            segment_risky_counts,
            target_percentage=self.segment_top_pct,
            rng=random.Random(f"{self.seed}:{case_id}:segment-select"),
        )

    def _apply_intervention(
        self,
        segments: list[str],
        tokenized_segments: list[list[tuple[str, int]]],
        risky_ids: list[int],
        selected_segment_ids: set[int],
        case_id,
    ) -> tuple[list[str], list[list[object]], list[int], set[int]]:
        segment_risky_counts = get_segment_risky_counts(tokenized_segments, risky_ids)
        rows = build_risky_rows(
            [
                segment
                for segment_id, segment in enumerate(tokenized_segments)
                if segment_id in selected_segment_ids
            ],
            risky_ids,
            case_id,
            [
                segment
                for segment_id, segment in enumerate(segments)
                if segment_id in selected_segment_ids
            ],
            segment_ids=[
                segment_id
                for segment_id in range(len(segments))
                if segment_id in selected_segment_ids
            ],
        )
        processed_segments = [
            intervene_on_segment(
                tokenized_segments[segment_id],
                self.intervention,
                random.Random(f"{self.seed}:{case_id}:{segment_id}:{self.intervention}"),
            ) if segment_id in selected_segment_ids else segment
            for segment_id, segment in enumerate(segments)
        ]

        return processed_segments, rows, segment_risky_counts, selected_segment_ids


__all__ = ["Ablation", "LOCATORS", "validate_locator"]
