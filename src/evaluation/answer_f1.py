import re
from collections import Counter


def get_answer_tokens(text: str) -> list[str]:
    # Lowercase and remove non-alphanumeric characters
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    # Remove articles: a, an, the
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return normalized.split()


def calculate_answer_f1(baseline_answer: str, model_answer: str) -> float:
    gold_tokens = get_answer_tokens(baseline_answer)
    pred_tokens = get_answer_tokens(model_answer)

    overlap = sum((Counter(gold_tokens) & Counter(pred_tokens)).values())

    return (2 * overlap) / (len(gold_tokens) + len(pred_tokens))