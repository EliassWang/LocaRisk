import torch

from src.evaluation.cost import add_input_tokens


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def calculate_perplexity(prompt: str, llm) -> float:
    text   = llm.tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = llm.tok(text, return_tensors="pt").to(llm.model.device)
    add_input_tokens(llm.cost, inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = llm.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs["input_ids"],
        )
    return torch.exp(out.loss).item()


def score_prompt_by_ppl(prompt: str, threshold: float, llm) -> dict:
    ppl = calculate_perplexity(prompt, llm)
    return {"perplexity": ppl, "threshold": threshold, "is_risky": ppl > threshold}


def filter_segments_by_ppl(segments: list[str], threshold: float, llm) -> tuple[list[str], list[dict]]:
    safe, scores = [], []
    for sid, seg in enumerate(segments):
        s = score_prompt_by_ppl(seg, threshold, llm)
        s["segment_id"] = sid
        scores.append(s)
        if not s["is_risky"]:
            safe.append(seg)
    return safe, scores


def detect_windowed_contamination(segments: list[str], threshold: float,
                                   llm, window_size: int, stride: int = 1) -> dict:
    ws  = max(1, min(window_size, len(segments)))
    wsc = []
    for start in range(0, max(1, len(segments) - ws + 1), stride):
        end   = start + ws
        score = score_prompt_by_ppl("\n".join(segments[start:end]), threshold, llm)
        score.update({"window_start": start, "window_end": end - 1, "window_size": ws})
        wsc.append(score)
    risky = sum(1 for s in wsc if s["is_risky"])
    return {
        "window_scores": wsc,
        "window_count": len(wsc),
        "risky_window_count": risky,
        "risky_window_pct": risky / len(wsc) * 100 if wsc else 0.0,
        "is_contaminated": risky > 0,
    }


def window_scores_to_segment_scores(window_scores: list[dict]) -> list[tuple[int, float]]:
    best: dict[int, float] = {}
    for w in window_scores:
        for sid in range(w["window_start"], w["window_end"] + 1):
            best[sid] = max(best.get(sid, float("-inf")), w["perplexity"])
    return sorted(best.items())


def calibrate_threshold(scores: list[float], target_fpr: float) -> tuple[float, int, bool]:
    if not scores:
        raise ValueError("Cannot calibrate from empty scores.")
    n      = max(0, min(round(len(scores) * target_fpr), len(scores)))
    ordered = sorted(scores, reverse=True)
    if n == 0:               threshold = ordered[0]
    elif n == len(ordered):  threshold = ordered[-1] - 1e-12
    else:
        hi, lo    = ordered[n - 1], ordered[n]
        threshold = lo if hi == lo else (hi + lo) / 2
    flagged = sum(s > threshold for s in ordered)
    return threshold, flagged, flagged == n


# ---------------------------------------------------------------------------
# PPL defense classes
# ---------------------------------------------------------------------------

class PPLDefense:
    """
    Base class for PPL-based defenses.

    Call calibrate(subset) once before the main eval loop.
    run(segments, sample_id) → {context, selected_ids, blocked, details}
      context      : str       preprocessed context for the LLM
      selected_ids : set[int]  flagged segment indices
      blocked      : bool      True = skip LLM call entirely (windowed_ppl)
      details      : dict      for JSON output
    """
    name: str = ""

    def __init__(self, llm, target_fpr: float):
        self.llm        = llm
        self.target_fpr = target_fpr
        self.threshold  = None
        self._cache: dict = {}
        self._calib_meta: dict = {}

    def calibrate(self, subset) -> None:
        raise NotImplementedError

    def run(self, segments: list[str], sample_id: str) -> dict:
        raise NotImplementedError

    def calibration_summary(self) -> dict:
        return {
            "target_fpr":          self.target_fpr,
            "calibrated_threshold": self.threshold,
            **self._calib_meta,
        }

    @staticmethod
    def _segs(sample) -> list[str]:
        return [" ".join(s) for s in sample["context"]["sentences"]]


class PPL(PPLDefense):
    """Global context PPL — flags and empties context if above threshold."""
    name = "ppl"

    def calibrate(self, subset) -> None:
        scores = []
        for sample in subset:
            ctx = "\n".join(self._segs(sample))
            ppl = calculate_perplexity(ctx, self.llm)
            scores.append(ppl)
            self._cache[sample["id"]] = ppl
        self.threshold, flagged, exact = calibrate_threshold(scores, self.target_fpr)
        self._calib_meta = {
            "clean_flagged_cases": flagged,
            "clean_cases": len(scores),
            "exact": exact,
            "case_score": "prompt_perplexity",
        }

    def run(self, segments: list[str], sample_id: str) -> dict:
        ctx = "\n".join(segments)
        ppl = self._cache.get(sample_id) or calculate_perplexity(ctx, self.llm)
        flagged = ppl > self.threshold
        return {
            "context":      "" if flagged else ctx,
            "selected_ids": set(range(len(segments))) if flagged else set(),
            "blocked":      False,
            "details": {
                "mode": "ppl", "threshold": self.threshold,
                "target_fpr": self.target_fpr,
                "prompt_perplexity": ppl, "flagged": flagged,
            },
        }


class SegmentPPL(PPLDefense):
    """Per-segment PPL — drops individual segments above threshold."""
    name = "segment_ppl"

    def calibrate(self, subset) -> None:
        scores = []
        for sample in subset:
            segs = self._segs(sample)
            seg_scores = [
                {"segment_id": sid, "perplexity": calculate_perplexity(seg, self.llm)}
                for sid, seg in enumerate(segs)
            ]
            scores.append(max(s["perplexity"] for s in seg_scores))
            self._cache[sample["id"]] = seg_scores
        self.threshold, flagged, exact = calibrate_threshold(scores, self.target_fpr)
        self._calib_meta = {
            "clean_flagged_cases": flagged,
            "clean_cases": len(scores),
            "exact": exact,
            "case_score": "max_segment_perplexity",
        }

    def run(self, segments: list[str], sample_id: str) -> dict:
        if sample_id in self._cache:
            seg_scores = [
                {**s, "threshold": self.threshold, "is_risky": s["perplexity"] > self.threshold}
                for s in self._cache[sample_id]
            ]
            safe = [segments[s["segment_id"]] for s in seg_scores if not s["is_risky"]]
        else:
            safe, seg_scores = filter_segments_by_ppl(segments, self.threshold, self.llm)

        selected_ids = {s["segment_id"] for s in seg_scores if s["is_risky"]}
        return {
            "context":      "\n".join(safe),
            "selected_ids": selected_ids,
            "blocked":      False,
            "details": {
                "mode": "segment_ppl", "threshold": self.threshold,
                "target_fpr": self.target_fpr,
                "flagged_segment_count": len(selected_ids),
                "retained_segment_count": len(safe),
                "segments": seg_scores,
            },
        }


class WindowedPPL(PPLDefense):
    """Sliding-window PPL — blocks the entire sample if any window is risky."""
    name = "windowed_ppl"

    def __init__(self, llm, target_fpr: float, window_length: int, stride: int = 1):
        super().__init__(llm, target_fpr)
        self.window_length = window_length
        self.stride        = stride

    def calibrate(self, subset) -> None:
        scores = []
        for sample in subset:
            segs   = self._segs(sample)
            result = detect_windowed_contamination(segs, float("inf"), self.llm, self.window_length, self.stride)
            ws     = result["window_scores"]
            scores.append(max(s["perplexity"] for s in ws))
            self._cache[sample["id"]] = ws
        self.threshold, flagged, exact = calibrate_threshold(scores, self.target_fpr)
        self._calib_meta = {
            "clean_flagged_cases": flagged,
            "clean_cases": len(scores),
            "exact": exact,
            "case_score": "max_window_perplexity",
        }

    def calibration_summary(self) -> dict:
        return {**super().calibration_summary(), "window_length": self.window_length, "window_stride": self.stride}

    def run(self, segments: list[str], sample_id: str) -> dict:
        if sample_id in self._cache:
            ws = [
                {**w, "threshold": self.threshold, "is_risky": w["perplexity"] > self.threshold}
                for w in self._cache[sample_id]
            ]
        else:
            ws = detect_windowed_contamination(
                segments, self.threshold, self.llm, self.window_length, self.stride
            )["window_scores"]

        is_contaminated = any(w["is_risky"] for w in ws)
        selected_ids = {
            sid
            for w in ws if w["is_risky"]
            for sid in range(w["window_start"], w["window_end"] + 1)
        }
        return {
            "context":      "\n".join(segments),
            "selected_ids": selected_ids,
            "blocked":      is_contaminated,
            "details":      ws,
        }
