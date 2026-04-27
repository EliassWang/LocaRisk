import torch

from src.evaluation.cost import add_input_tokens


def calculate_perplexity(prompt: str, llm: "LLM") -> float:
    text = llm.tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = llm.tok(text, return_tensors="pt").to(llm.model.device)
    add_input_tokens(llm.cost, inputs["input_ids"].shape[1])

    with torch.no_grad():
        outputs = llm.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs["input_ids"],
        )

    return torch.exp(outputs.loss).item()


def score_prompt_by_ppl(prompt: str, threshold: float, llm: "LLM") -> dict:
    perplexity = calculate_perplexity(prompt, llm)
    return {
        "perplexity": perplexity,
        "threshold": threshold,
        "is_risky": perplexity > threshold,
    }


def filter_prompts_by_ppl(prompts: str | list[str], threshold: float, llm: "LLM") -> str | list[str]:
    is_single_input = isinstance(prompts, str)
    process_list = [prompts] if is_single_input else prompts

    safe_prompts = []
    any_risky = False

    for prompt in process_list:
        prompt_score = score_prompt_by_ppl(prompt, threshold=threshold, llm=llm)
        if not prompt_score["is_risky"]:
            safe_prompts.append(prompt)
        else:
            any_risky = True

    if is_single_input:
        return prompts if not any_risky else ""

    return prompts if not any_risky else safe_prompts


def filter_segments_by_ppl(segments: list[str], threshold: float, llm: "LLM") -> tuple[list[str], list[dict]]:
    safe_segments = []
    segment_scores = []

    for segment_id, segment in enumerate(segments):
        score = score_prompt_by_ppl(segment, threshold=threshold, llm=llm)
        score["segment_id"] = segment_id
        segment_scores.append(score)
        if not score["is_risky"]:
            safe_segments.append(segment)

    return safe_segments, segment_scores


def score_windows_by_ppl(
    segments: list[str],
    threshold: float,
    llm: "LLM",
    window_size: int,
    stride: int = 1,
) -> list[dict]:
    if not segments:
        return []

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    effective_window_size = min(window_size, len(segments))
    window_scores = []

    for start in range(0, len(segments) - effective_window_size + 1, stride):
        end = start + effective_window_size
        window_text = "\n".join(segments[start:end])
        score = score_prompt_by_ppl(window_text, threshold=threshold, llm=llm)
        score.update({
            "window_start": start,
            "window_end": end - 1,
            "window_size": effective_window_size,
        })
        window_scores.append(score)

    if not window_scores:
        window_text = "\n".join(segments)
        score = score_prompt_by_ppl(window_text, threshold=threshold, llm=llm)
        score.update({
            "window_start": 0,
            "window_end": len(segments) - 1,
            "window_size": len(segments),
        })
        window_scores.append(score)

    return window_scores


def detect_windowed_contamination(
    segments: list[str],
    threshold: float,
    llm: "LLM",
    window_size: int,
    stride: int = 1,
) -> dict:
    window_scores = score_windows_by_ppl(
        segments=segments,
        threshold=threshold,
        llm=llm,
        window_size=window_size,
        stride=stride,
    )
    risky_window_count = sum(1 for score in window_scores if score["is_risky"])
    window_count = len(window_scores)
    risky_window_pct = (risky_window_count / window_count) * 100 if window_count else 0.0

    return {
        "window_scores": window_scores,
        "window_count": window_count,
        "risky_window_count": risky_window_count,
        "risky_window_pct": risky_window_pct,
        "blocks_on_any_risky_window": True,
        "is_contaminated": risky_window_count > 0,
    }
