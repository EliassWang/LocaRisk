import random

from src.defenses.probe_core.rl import rl


def get_reference_risky_count(
    target_token_ids: set[int],
    tau: float,
    freq_data_path: str,
) -> int:
    risky_token_ids = set(
        rl(
            freq_data_path=freq_data_path,
            tau=tau,
            target_token_ids=target_token_ids,
        )
    )
    return len(risky_token_ids.intersection(target_token_ids))


def ablation(
    target_token_ids: set[int],
    tau: float,
    seed: int,
    freq_data_path: str,
) -> list[int]:
    token_ids = sorted(target_token_ids)
    risky_count = min(
        get_reference_risky_count(
            target_token_ids=target_token_ids,
            tau=tau,
            freq_data_path=freq_data_path,
        ),
        len(token_ids),
    )

    if risky_count == 0:
        return []

    rng = random.Random(seed)
    return rng.sample(token_ids, k=risky_count)


__all__ = ["ablation", "get_reference_risky_count"]
