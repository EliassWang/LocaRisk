import re
from collections import Counter


def calculate_answer_f1(baseline_answer: str, model_answer: str) -> float:
    def get_tokens(text: str) -> list[str]:
        # Lowercase and remove non-alphanumeric characters
        normalized = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
        # Remove articles: a, an, the
        normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
        return normalized.split()
    gold_tokens = get_tokens(baseline_answer)
    pred_tokens = get_tokens(model_answer)

    overlap = sum((Counter(gold_tokens) & Counter(pred_tokens)).values())

    return (2 * overlap) / (len(gold_tokens) + len(pred_tokens))