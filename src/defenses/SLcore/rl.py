# RL — Multi-Signal Segment Localizer (MultiLocalize), r = L(q, xi, X)
#
# Given query q and retrieved segments X, scores each xi for contamination
# risk by fusing two signals:
#
#   freq   : ri_freq = ci, ci = count of tokens in xi with freq(t) below
#            the tau-th percentile of a reference corpus (raw count, not
#            length-normalized; q unused). ObliInjection's payload appends a
#            prefix+task+diluter+postfix that roughly doubles the target
#            segment's length, so a length-normalized density score gets
#            diluted by the diluter's ordinary-frequency tokens; raw count
#            does not.
#   cosine : rcos = |di - davg|, di = 1 - cos(embed(q), embed(xi)),
#            davg = mean(di) over all segments in X
#            (larger deviation from the average query-distance = riskier)
#
# Fusion rule: r = max( minmax(freq), minmax(rcos) )
#   Whichever signal considers a segment riskiest wins for that segment, so
#   one signal being near-floor on a given attack family doesn't drag down a
#   segment the other signal is confident about. (A weighted-sum "linear"
#   fusion was ablated in lambda_sweep.py and found strictly worse; MAX is
#   the only fusion rule in production.)
#
#   RL.score(q, xi, X)  → float        r = L(q, xi, X) for one segment
#   RL.score_all(q, X)  → list[float]  scores parallel to X
#   RL.locate(q, X)     → list[str]    X sorted by descending r
#   RL(q, X)            → list[str]    alias for locate

import json

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_percentile_cutoff(freqs: list[float], tau: float) -> float:
    if not freqs: return 0.0
    freqs = sorted(freqs)
    rank  = (tau / 100.0) * (len(freqs) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(freqs) - 1)
    return freqs[lo] + (freqs[hi] - freqs[lo]) * (rank - lo)


def load_risky_ids(freq_data_path: str, tau: float,
                   token_ids: set[int] | None = None) -> set[int]:
    """Rare token IDs at or below the tau-th frequency percentile, plus OOV."""
    with open(freq_data_path) as f:
        payload = json.load(f)
    freq_map = {int(x["id"]): float(x["f"]) for x in payload.get("frequencies", [])}
    cutoff   = get_percentile_cutoff(list(freq_map.values()), tau)
    risky    = {tid for tid, f in freq_map.items() if f <= cutoff}
    if token_ids is not None:
        risky |= token_ids - frozenset(freq_map.keys())
    return risky


_embed_cache: dict = {}

def _load_embed_model(name: str):
    if name not in _embed_cache:
        tok    = AutoTokenizer.from_pretrained(name)
        model  = AutoModel.from_pretrained(name).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_cache[name] = (tok, model.to(device), device)
    return _embed_cache[name]

def _mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand(h.size()).float()
    return torch.sum(h * m, 1) / torch.clamp(m.sum(1), min=1e-9)

def _embed(texts: list[str], model_name: str) -> torch.Tensor:
    tok, model, device = _load_embed_model(model_name)
    enc = tok(texts, padding=True, truncation=True,
               max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    return F.normalize(_mean_pool(out.last_hidden_state, enc["attention_mask"]),
                        p=2, dim=1).cpu()


def load_freq_cache(freq_data_path: str, tau: float) -> tuple[frozenset[int], frozenset[int]]:
    """Rare token ids at or below the tau-th frequency percentile, plus the
    full set of token ids present in the frequency dict (for OOV handling)."""
    with open(freq_data_path) as f:
        payload = json.load(f)
    freq_map = {int(x["id"]): float(x["f"]) for x in payload.get("frequencies", [])}
    cutoff   = get_percentile_cutoff(list(freq_map.values()), tau)
    return (
        frozenset(tid for tid, f in freq_map.items() if f <= cutoff),
        frozenset(freq_map.keys()),
    )


def tokenize_with_ids(text: str, tokenizer) -> list[tuple[str, int]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(text[s:e], tid) for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"])]


def freq_count_scores(
    X: list[str], tokenizer, risky_ids: frozenset[int], freq_keys: frozenset[int]
) -> list[float]:
    """ri = ci -- raw count of rare/OOV tokens per segment, not length-normalized.
    OOV tokens (absent from the frequency dict) count as risky too."""
    tokenized = [tokenize_with_ids(xi, tokenizer) for xi in X]
    risky = set(risky_ids) | ({tid for seg in tokenized for _, tid in seg} - freq_keys)
    scores = []
    for seg in tokenized:
        c_i = sum(1 for text, tid in seg
                  if tid in risky
                  and not any(c.isalpha() and not c.isascii() for c in text))
        scores.append(float(c_i))
    return scores


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


# ---------------------------------------------------------------------------
# RL class
# ---------------------------------------------------------------------------

class RL:
    """
    Localization function  r = L(q, xi, X).

    Combines a rare-token frequency signal with a semantic cosine-distance-
    deviation signal into a single risk score per segment via MAX fusion
    (see module docstring).

    locator_config keys
      freq_data_path : path to the corpus frequency dict
      tau            : rare-token percentile threshold (default 0.1)
      tokenizer      : tokenizer used to score rare tokens
      embed_model    : sentence encoder for the cosine signal (optional, defaults to MiniLM)
      rank           : "decrease" (default) or "increase", for locate()
    """

    def __init__(
        self,
        locator_config: dict,
        isDebug       : bool       = False,
        query         : str        = "",
        context       : list[str] | None = None,
    ):
        self.locator_config = locator_config
        self.isDebug        = isDebug
        self.query          = query
        self.context        = context or []

        self.tau              = locator_config.get("tau", 0.1)
        self._tokenizer       = locator_config["tokenizer"]
        self._freq_data_path  = locator_config["freq_data_path"]
        self._freq_cache: tuple[frozenset[int], frozenset[int]] | None = None
        self.embed_model = locator_config.get("embed_model", DEFAULT_EMBED_MODEL)

    # public API

    def score(self, q: str, xi: str, X: list[str]) -> float:
        """r = L(q, xi, X)."""
        return self.score_all(q, X)[X.index(xi)]

    def score_all(self, q: str, X: list[str]) -> list[float]:
        """Fused risk score for every xi in X, parallel to X."""
        freq_norm = _min_max_normalize(self._freq_score(X))
        cos_norm  = _min_max_normalize(self._semantic_score(q, X))
        scores = [max(f, c) for f, c in zip(freq_norm, cos_norm)]
        if self.isDebug:
            for i, s in enumerate(scores):
                print(f"[RL|multi_signal] seg[{i}] r={s:.4f}  {X[i][:60]!r}")
        return scores

    def locate(self, q: str, X: list[str]) -> list[str]:
        """Return X reranked by risk (highest-risk first), respecting locator_config["rank"]."""
        scores  = self.score_all(q, X)
        descend = self.locator_config.get("rank", "decrease") == "decrease"
        order   = sorted(range(len(X)), key=lambda i: scores[i], reverse=descend)
        if self.isDebug:
            print(f"[RL|multi_signal] rank={self.locator_config.get('rank')} order: {order}")
        return [X[i] for i in order]

    def __call__(self, q: str, X: list[str]) -> list[str]:
        return self.locate(q, X)

    # freq internals

    def _load_freq_cache(self) -> tuple[frozenset[int], frozenset[int]]:
        if self._freq_cache is None:
            self._freq_cache = load_freq_cache(self._freq_data_path, self.tau)
        return self._freq_cache

    def _freq_score(self, X: list[str]) -> list[float]:
        """ri = ci -- raw count of rare/OOV tokens per segment, not length-normalized."""
        risky_ids, freq_keys = self._load_freq_cache()
        return freq_count_scores(X, self._tokenizer, risky_ids, freq_keys)

    # semantic internals

    def _semantic_score(self, q: str, X: list[str]) -> list[float]:
        embs  = _embed([q] + X, self.embed_model)
        q_emb = embs[0]
        dists = [1.0 - float(embs[i + 1] @ q_emb) for i in range(len(X))]
        d_avg = sum(dists) / len(dists) if dists else 0.0
        return [abs(di - d_avg) for di in dists]


__all__ = [
    "RL", "DEFAULT_EMBED_MODEL",
    "get_percentile_cutoff", "load_risky_ids",
    "load_freq_cache", "tokenize_with_ids", "freq_count_scores",
    "_load_embed_model", "_mean_pool",
]
