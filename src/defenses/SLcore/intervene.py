# Intervention — neutralizes the top-k% riskiest segments selected by RL.
#
#   Intervention(X, scores, rng)       → (list[str], set[int])  select + apply
#   Intervention.apply_to(X, ids, rng) → list[str]              apply to given ids
#
# Kinds:  "swap" — shuffle words    "drop" — empty string
#
# Utilities (ablation paths):
#   get_segment_risky_counts(segments, risky_ids) → list[int]
#   print_risky_count_distribution(...)

from collections import Counter
import math
import random
import re

INTERVENTIONS = ("swap", "drop")


class Intervention:
    """
    Neutralization step applied after RL localization.

    Selects the top-k% highest-scoring segments and applies kind to each.

    kind  : "swap" or "drop"
    k     : percentage of segments to intervene on  (e.g. 20.0 = top 20%)
    """

    def __init__(self, kind: str, k: float, isDebug: bool = False):
        if kind not in INTERVENTIONS:
            raise ValueError(f"kind must be one of {INTERVENTIONS!r}, got {kind!r}")
        self.kind    = kind
        self.k       = k
        self.isDebug = isDebug

    def __call__(
        self, X: list[str], scores: list[float], rng: random.Random
    ) -> tuple[list[str], set[int]]:
        """Select top-k% by score, apply intervention. Returns (processed_X, selected_ids)."""
        ids = self._select(scores, len(X))
        if self.isDebug:
            print(f"[Intervention|{self.kind}] k={self.k}% → selected {len(ids)}/{len(X)}: {sorted(ids)}")
        return self.apply_to(X, ids, rng), ids

    def apply_to(self, X: list[str], ids: set[int], rng: random.Random) -> list[str]:
        """Apply to given segment indices, bypassing internal selection."""
        result = [self._apply(seg, rng) if i in ids else seg
                  for i, seg in enumerate(X)]
        if self.isDebug:
            for i in sorted(ids):
                after = result[i] if result[i] else "(dropped)"
                print(f"[Intervention|{self.kind}] seg[{i}]: {X[i][:60]!r} → {after[:60]!r}")
        return result

    # internals

    def _select(self, scores: list[float], n: int) -> set[int]:
        k = max(1, math.ceil(self.k / 100.0 * n)) if n else 0
        return set(sorted(range(n), key=lambda i: scores[i], reverse=True)[:k])

    def _apply(self, text: str, rng: random.Random) -> str:
        return self._swap(text, rng) if self.kind == "swap" else ""

    def _swap(self, text: str, rng: random.Random) -> str:
        parts    = re.findall(r"\S+|\s+", text)
        widx     = [i for i, p in enumerate(parts) if not p.isspace()]
        if len(widx) < 2: return ""
        words    = [parts[i] for i in widx]
        shuffled = words.copy()
        rng.shuffle(shuffled)
        if shuffled == words:
            shuffled = shuffled[1:] + shuffled[:1]
        for i, w in zip(widx, shuffled):
            parts[i] = w
        return "".join(parts)


# ---------------------------------------------------------------------------
# Utilities for ablation paths (random / oracle locators)
# ---------------------------------------------------------------------------

def get_segment_risky_counts(all_segments, risky_ids) -> list[int]:
    risky_set = set(risky_ids)
    return [
        sum(1 for text, tid in seg
            if tid in risky_set
            and not any(c.isalpha() and not c.isascii() for c in text))
        for seg in all_segments
    ]


def print_risky_count_distribution(segment_risky_counts,
                                   case_id=None, contaminated_segment_id=None):
    if not segment_risky_counts:
        print("No segments to process."); return

    dist  = Counter(segment_risky_counts)
    total = len(segment_risky_counts)
    c_cnt = (segment_risky_counts[contaminated_segment_id]
             if contaminated_segment_id is not None
             and 0 <= contaminated_segment_id < total else None)

    print(f"\n--- Risky Token Count Distribution"
          + (f" case_id={case_id}" if case_id is not None else "") + " ---")
    if c_cnt is not None:
        print(f"contaminated segment {contaminated_segment_id}: "
              f"{c_cnt} risky {'token' if c_cnt == 1 else 'tokens'} <- contaminated")
    for cnt in sorted(dist):
        n      = dist[cnt]
        label  = "token" if cnt == 1 else "tokens"
        suffix = " <- contaminated" if cnt == c_cnt else ""
        print(f"{cnt} risky {label}: {n / total * 100:.1f}% ({n} segments){suffix}")


__all__ = [
    "Intervention", "INTERVENTIONS",
    "get_segment_risky_counts", "print_risky_count_distribution",
]
