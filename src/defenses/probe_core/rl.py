import json


def get_percentile_cutoff(freqs, tau):
    if not freqs: return 0.0
    freqs = sorted(freqs)
    rank = (float(tau) / 100.0) * (len(freqs) - 1)
    low, high = int(rank), min(int(rank) + 1, len(freqs) - 1)
    return freqs[low] + (freqs[high] - freqs[low]) * (rank - low)


def rl(freq_data_path: str, tau: float, target_token_ids: set[int] | None = None) -> list[int]:
    with open(freq_data_path, "r") as f:
        payload = json.load(f)

    freq_map = {int(x["id"]): float(x["f"]) for x in payload.get("frequencies", [])}
    all_freqs = list(freq_map.values())

    cutoff = get_percentile_cutoff(all_freqs, tau)
    risky_token_ids = [
        tid for tid, f in freq_map.items()
        if f <= cutoff
    ]

    if target_token_ids is not None:
        risky_token_ids.extend(
            tid for tid in target_token_ids
            if tid not in freq_map
        )

    return risky_token_ids


__all__ = ["rl", "get_percentile_cutoff"]
