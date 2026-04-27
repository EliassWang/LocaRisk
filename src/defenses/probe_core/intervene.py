from collections import Counter
import json
import math
import random
import re

INTERVENTIONS = ("swap", "drop")


def validate_intervention(intervention: str) -> str:
    if intervention not in INTERVENTIONS:
        valid = ", ".join(INTERVENTIONS)
        raise ValueError(f"Unknown intervention '{intervention}'. Expected one of: {valid}")
    return intervention


def is_taggable_risky_token(token, risky_set):
    token_text = token[0]
    if token[1] not in risky_set:
        return False
    return not any(char.isalpha() and not char.isascii() for char in token_text)


def get_risky_positions(segment, risky_ids):
    risky_set = set(risky_ids)
    return [
        token_index
        for token_index, token in enumerate(segment)
        if is_taggable_risky_token(token, risky_set)
    ]


def get_risky_tokens(segment, risky_ids):
    risky_set = set(risky_ids)
    return [
        token[0]
        for token in segment
        if is_taggable_risky_token(token, risky_set)
    ]


def build_risky_rows(all_segments, risky_ids, case_id, original_segments, segment_ids=None):
    rows = []

    if segment_ids is None:
        segment_ids = range(len(all_segments))

    for segment_id, segment, segment_text in zip(segment_ids, all_segments, original_segments):
        risky_positions = get_risky_positions(segment, risky_ids)
        if not risky_positions:
            continue

        risky_tokens = get_risky_tokens(segment, risky_ids)

        rows.append(
            [
                case_id,
                json.dumps(risky_tokens, ensure_ascii=False),
                len(risky_positions),
                segment_text,
            ]
        )

    return rows


def get_segment_risky_counts(all_segments, risky_ids):
    return [
        len(get_risky_positions(segment, risky_ids))
        for segment in all_segments
    ]


def swap_words_in_segment(segment, rng, risky_ids=None):
    text = "".join(token[0] for token in segment)

    parts = re.findall(r"\S+|\s+", text)
    word_indexes = [index for index, part in enumerate(parts) if not part.isspace()]

    if len(word_indexes) < 2:
        return ""

    original_words = [parts[index] for index in word_indexes]
    shuffled_words = original_words.copy()
    rng.shuffle(shuffled_words)

    if shuffled_words == original_words:
        shuffled_words = shuffled_words[1:] + shuffled_words[:1]

    for index, word in zip(word_indexes, shuffled_words):
        parts[index] = word

    return "".join(parts)


def drop_segment(segment, rng=None, risky_ids=None):
    return ""


def intervene_on_segment(segment, intervention, rng, risky_ids=None):
    intervention = validate_intervention(intervention)
    if intervention == "swap":
        return swap_words_in_segment(segment, rng, risky_ids=risky_ids)
    if intervention == "drop":
        return drop_segment(segment, rng=rng, risky_ids=risky_ids)
    raise AssertionError(f"Unhandled intervention: {intervention}")


def select_top_risky_segment_ids(segment_risky_counts, target_percentage=20.0, rng=None):
    total_segments = len(segment_risky_counts)
    if total_segments == 0:
        return set()

    target_percentage = max(0.0, min(float(target_percentage), 100.0))
    if target_percentage == 0.0:
        return set()

    target_count = math.ceil((target_percentage / 100.0) * total_segments)
    if target_count >= total_segments:
        return set(range(total_segments))

    rng = rng or random.Random(0)
    selected_ids = set()

    for risky_count in sorted(set(segment_risky_counts), reverse=True):
        bucket_ids = [
            segment_id
            for segment_id, count in enumerate(segment_risky_counts)
            if count == risky_count
        ]
        if not bucket_ids:
            continue

        remaining_slots = target_count - len(selected_ids)
        if remaining_slots <= 0:
            break

        if len(bucket_ids) <= remaining_slots:
            selected_ids.update(bucket_ids)
            continue

        selected_ids.update(rng.sample(bucket_ids, k=remaining_slots))
        break

    return selected_ids


def print_risky_count_distribution(
    segment_risky_counts,
    case_id=None,
    contaminated_segment_id=None,
):
    if not segment_risky_counts:
        print("No segments to process.")
        return

    distribution = Counter(segment_risky_counts)
    total_segments = len(segment_risky_counts)
    contaminated_risky_count = None
    if contaminated_segment_id is not None and 0 <= contaminated_segment_id < total_segments:
        contaminated_risky_count = segment_risky_counts[contaminated_segment_id]

    if case_id is None:
        print("\n--- Risky Token Count Distribution ---")
    else:
        print(f"\n--- Risky Token Count Distribution (case_id={case_id}) ---")
    if contaminated_risky_count is not None:
        print(
            f"contaminated segment {contaminated_segment_id}: "
            f"{contaminated_risky_count} risky "
            f"{'token' if contaminated_risky_count == 1 else 'tokens'} <- contaminated"
        )
    for risky_count in sorted(distribution):
        num_segments = distribution[risky_count]
        percentage = (num_segments / total_segments) * 100
        label = "token" if risky_count == 1 else "tokens"
        suffix = " <- contaminated" if contaminated_risky_count == risky_count else ""
        print(f"{risky_count} risky {label}: {percentage:.1f}% ({num_segments} segments){suffix}")
