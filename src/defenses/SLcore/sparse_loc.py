# SparseLoc — Localization-aware test-time defense
#
# Pipeline:  RL scores each retrieved segment (multi-signal: freq + cosine_semantic)
#            → Intervention neutralizes top-k%
#
# Defaults are loaded from configs/defense/defenses.json; every param can be
# overridden at construction time by passing it explicitly.
#
#   run(segments, case_id, question) → (processed_segments, scores, selected_ids)

import json
import random
from pathlib import Path

from src.defenses.SLcore.intervene import Intervention
from src.defenses.SLcore.rl import RL

_CFG_PATH = Path("configs/defense/defenses.json")

LOCATOR = "multi_signal"


def _load_defaults() -> tuple[dict, dict]:
    """Return (sparseloc_cfg, multi_signal_cfg)."""
    cfg = json.loads(_CFG_PATH.read_text())
    return cfg["sparseloc"], cfg["locators"]["multi_signal"]


CORE_SPARSELOC_INTERVENTION = "drop"


class SparseLoc:
    """
    Localization-aware test-time defense.

    Uses RL (r = L(q, xi, X), a fused freq + cosine_semantic score) to rank
    each retrieved segment, then Intervention neutralizes the top-k% riskiest
    ones before generation.

    All constructor params fall back to configs/defense/defenses.json when
    not supplied explicitly.

    Set isDebug=True to print intermediate scores, selection, and
    before/after segments at each run() call.
    """

    def __init__(
        self,
        llm,
        model_name      : str,
        freq_dataset    : str   | None = None,
        docs_number     : int   | None = None,
        tau             : float | None = None,
        segment_top_pct : float | None = None,
        seed            : int          = 42,
        embed_model     : str   | None = None,
        intervention    : str   | None = None,
        isDebug         : bool         = False,
    ):
        sl_def, ms_def = _load_defaults()

        freq_dataset    = freq_dataset    or ms_def["freq_dataset"]
        docs_number     = docs_number     or ms_def["docs_number"]
        tau             = tau             if tau             is not None else ms_def["tau"]
        segment_top_pct = segment_top_pct if segment_top_pct is not None else ms_def["segment_top_pct"]
        intervention    = intervention    or sl_def["default_intervention"]
        embed_model     = embed_model     or ms_def["embed_model"]
        rank            = ms_def.get("rank", "decrease")

        self.isDebug  = isDebug
        self._tokenizer   = llm.tok
        self.freq_data_path = f"data/corpus_freqs/{freq_dataset}/{model_name}_{docs_number}.json"
        self.tau     = tau
        self.seed    = seed
        self.locator = LOCATOR
        self._rank   = rank

        _locator_config = {
            "freq_data_path": self.freq_data_path, "tau": tau, "tokenizer": self._tokenizer,
            "embed_model": embed_model,
            "rank": rank,
        }

        self._rl           = RL(locator_config=_locator_config, isDebug=isDebug)
        self._intervention = Intervention(intervention, k=segment_top_pct, isDebug=isDebug)

    def run(
        self,
        segments: list[str],
        case_id,
        question: str,
    ) -> tuple[list[str], list[float], set[int]]:
        """
        Localize and neutralize risky segments.

        Returns (processed_segments, scores, selected_ids).
          processed_segments -- X after intervention on selected segments
          scores             -- per-segment risk scores from RL, parallel to X
          selected_ids       -- indices of intervened segments
        """
        if self.isDebug:
            print(f"[SparseLoc|{self.locator}] case_id={case_id}  n_segments={len(segments)}")

        scores = self._rl.score_all(question, segments)
        rng    = random.Random(f"{self.seed}:{case_id}")
        processed, selected = self._intervention(segments, scores, rng)

        if self.isDebug:
            print(f"[SparseLoc|{self.locator}] done  selected={sorted(selected)}")

        return processed, scores, selected


__all__ = ["CORE_SPARSELOC_INTERVENTION", "SparseLoc"]
