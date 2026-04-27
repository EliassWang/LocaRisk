import random

from src.defenses.probe_core.intervene import (
    build_risky_rows,
    get_segment_risky_counts,
    intervene_on_segment,
    select_top_risky_segment_ids,
)
from src.defenses.probe_core.rl import rl


CORE_PROBE_INTERVENTION = "drop"


class Probe:
    def __init__(
        self,
        llm: "LLM",
        model_name: str,
        freq_dataset:str,
        docs_number:int,
        tau: float,
        segment_top_pct: float,
        seed: int = 42,
    ):
        # Access the tokenizer directly from the LLM instance
        self.tokenizer = llm.tok
        self.freq_data_path = f"data/corpus_freqs/{freq_dataset}/{model_name}_{docs_number}.json"
        self.tau = tau
        self.segment_top_pct = segment_top_pct
        self.intervention = CORE_PROBE_INTERVENTION
        self.seed = seed

    def run(
        self,
        segments: list[str],
        case_id,
        oracle_segment_id: int | None = None,
    ) -> tuple[list[str], list[list[object]], list[int], set[int]]:
        tokenized_segment = []
        target_token_ids = set()

        for text in segments:
            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True
            )

            input_ids = encoding["input_ids"]

            current_segment_tokens = []

            for token_id, (start, end) in zip(input_ids, encoding["offset_mapping"]):
                token_text = text[start:end]
                current_segment_tokens.append((token_text, token_id))
                target_token_ids.add(token_id)

            # Append the finished segment list to the main list
            tokenized_segment.append(current_segment_tokens)

        risky_ids = rl(
            freq_data_path=self.freq_data_path,
            tau=self.tau,
            target_token_ids=target_token_ids,
        )
        segment_risky_counts = get_segment_risky_counts(tokenized_segment, risky_ids)
        if oracle_segment_id is not None and 0 <= oracle_segment_id < len(segments):
            selected_segment_ids = {oracle_segment_id}
        else:
            selected_segment_ids = select_top_risky_segment_ids(
                segment_risky_counts,
                target_percentage=self.segment_top_pct,
                rng=random.Random(f"{self.seed}:{case_id}:segment-select"),
            )
        rows = build_risky_rows(
            [segment for segment_id, segment in enumerate(tokenized_segment) if segment_id in selected_segment_ids],
            risky_ids,
            case_id,
            [segment for segment_id, segment in enumerate(segments) if segment_id in selected_segment_ids],
            segment_ids=[segment_id for segment_id in range(len(segments)) if segment_id in selected_segment_ids],
        )
        processed_segments = [
            intervene_on_segment(
                tokenized_segment[segment_id],
                self.intervention,
                random.Random(f"{self.seed}:{case_id}:{segment_id}:{self.intervention}"),
            ) if segment_id in selected_segment_ids else segment
            for segment_id, segment in enumerate(segments)
        ]

        return processed_segments, rows, segment_risky_counts, selected_segment_ids


__all__ = ["Probe", "CORE_PROBE_INTERVENTION"]
