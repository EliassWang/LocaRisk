# LocaRisk

**LocaRisk** is the evaluation code for **Why Localization Matters: Allocation, Not Detection, for Sparse-Contamination Multi-Source RAG Defense** (Elias Wang, *AISec '26*, The Hague, Netherlands).

This repository studies test-time defense allocation for multi-source retrieval-augmented generation (RAG) under sparse contamination, where only one or a few retrieved context segments are adversarial and the rest of the retrieved context is benign. The paper's position: this is an *allocation* problem — fix a budget, rank the retrieved segments, spend the budget on the top of the ranking — not a *detection* problem that needs a calibrated, corpus-transferable threshold. The codebase evaluates that claim by applying global, segment-level, and localization-assisted defenses to two indirect-prompt-injection / knowledge-poisoning attack families, across five models and two datasets.

The repository supports:

- loading multi-source RAG context segments;
- optionally injecting an attack payload (ObliInjection or CorruptRAG) into a retrieved segment;
- applying global, segment-level, or localization-assisted defense variants;
- reporting attack success rate (ASR), clean/under-attack answer F1, token-cost statistics, and localization hit-rate details when available.

## Detection vs. Allocation

With one contaminated segment hidden among many (`|A| ≪ n`, position unknown), a **detection** framing asks an absolute question per segment — is `s(xᵢ) > θ`? — which requires a threshold calibrated to transfer across corpora. An **allocation** framing instead asks a relative question — is `xᵢ` in the top `k` by score, for `k` fixed before scoring? A *hit* (`A ∩ S_k ≠ ∅`) lowers attack success from `p0` to `p1 < p0`; a *miss* leaves it at `p0`. This gives `E[ASR_L] = p0 − h_L(p0 − p1)` with no threshold, calibration, or FPR term — any localizer benefits whenever its hit rate `h_L` beats `h_rand`. This is the framing behind the `sl` / `SL-smooth` defenses below (see [Localization Function](#localization-function)).

## Headline Results

Pooled over 5 models × 2 datasets × 2 attack families (200 attacked samples per model/dataset/attack; `--defense sl` with the default `drop` intervention, i.e. the paper's **SL-Drop**):

| Method | Avg. ASR | Avg. clean F1 | Avg. token cost | Rel. cost |
|---|---|---|---|---|
| No defense | 80.40% | 0.3696 | 1208.40 | 1.00× |
| Full-PPL (`ppl`) | 73.70% | 0.3688 | 2928.30 | 2.42× |
| Windowed-PPL (`windowed_ppl`) | 71.90% | 0.3113 | 5251.00 | 4.35× |
| Segment-PPL (`segment_ppl`) | 76.53% | 0.3559 | 2987.10 | 2.47× |
| SmoothLLM (`smoothllm`) | 39.40% | 0.1851 | 16831.30 | 13.93× |
| **SL-Drop (`sl`)** | **7.50%** | 0.3287 | **903.75** | **0.75×** |

SL-Drop is the only method cheaper than running no defense at all — it removes segments before generation instead of adding model-based scoring passes. Split by attack family, SL-Drop reaches 2.25% ASR on ObliInjection (optimization-based injection) and 12.75% on CorruptRAG (fluent poisoning), and attains the lowest ASR in 19 of 20 model/dataset/attack cells. Under CorruptRAG specifically it also *recovers* answer utility the attack destroyed rather than trading utility for robustness (HotpotQA under-attack F1: 0.180 undefended → 0.368 with SL-Drop, against a 0.410 clean ceiling), something the global baselines don't do — full-context SmoothLLM instead pushes under-attack F1 *below* the undefended level.

The two signals that make up SL-Drop's locator are individually poor detectors — each is near-perfect on one attack family and *below the random-selection floor* on the other (lexical: 98.0% hit rate on ObliInjection vs. 11.5% on CorruptRAG, random floor 18.5%; semantic: 10.5% vs. 99.0%, random floor 21.5%; Mistral-7B/HotpotQA, `segment_top_pct=20`). Neither is deployable as a standalone detector, yet fused they allocate effectively — the paper's central empirical argument for why allocation, not detection, is the right frame.

## Attack Families

Two attack families are supported, selected with `--attack`:

| Attack | `--attack` value | Supported datasets | Payload file |
|---|---|---|---|
| ObliInjection | `obli-injection` (default) | `hotpotqa`, `2wikimultihopqa` | `data/payloads/obli_injection/payloads.json` |
| CorruptRAG | `corrupt-rag` | `hotpotqa`, `2wikimultihopqa` | see `configs/config.json` → `corrupt_rag` |

**ObliInjection** injects a model-specific adversarial prefix/postfix wrapping an "answer is unknown" instruction into a retrieved segment. ASR measures whether the model refuses to answer.

**CorruptRAG** is a practical single-injection knowledge-poisoning attack (Zhang et al., *Practical Poisoning Attacks Against Retrieval-Augmented Generation*, SACMAT '26) that improves stealth over the earlier PoisonedRAG formulation (Zou et al., USENIX Security '25) while still compromising the answer. It replaces a retrieved segment with a plausible but factually incorrect adversarial document keyed by sample ID. ASR measures whether the model outputs the incorrect answer (F1 against incorrect answer exceeds F1 against correct answer). The payload file and the dataset split the sample IDs are keyed against are dataset-specific and configured per-dataset in `configs/config.json` → `corrupt_rag` (hotpotqa IDs come from its `validation` split, 2wikimultihopqa IDs come from its `train` split).

## Payload Directory Layout

```
data/payloads/
  obli_injection/
    payloads.json                      # model-specific prefix/postfix payloads (public placeholder)
  corrupt_rag/
    hotpotqa_corrputRAG-AS.json        # per-sample adversarial texts keyed by HotPotQA ID (validation split)
    2wikimultihopqa.json               # per-sample adversarial texts keyed by 2WikiMultihopQA ID (train split)
```

## Localization Function

The `sl` (SparseLoc) and `SL-smooth` defenses share a single localization function `r = L(q, x_i, X)` (`src/defenses/SLcore/rl.py`) that assigns each retrieved segment `x_i` a risk score given the question `q` and the full segment set `X`. The `segment_top_pct`% highest-scoring segments are treated as risky and passed to the intervention (`swap` or `drop`).

| Signal | Score | Notes |
|---|---|---|
| lexical (`freq`) | `r_freq_i = c_i`, the **raw count** of tokens in `x_i` with corpus frequency below the `tau`-th percentile (OOV tokens count as risky too) | deliberately *not* length-normalized: an injected payload contributes a roughly fixed number of low-frequency tokens regardless of segment length, so dividing by token count would dilute that margin; needs a prebuilt freq dict, see [Frequency Dictionaries](#frequency-dictionaries) |
| semantic (`cos`) | `r_cos_i = \|d_i - d_avg\|`, where `d_i = 1 - cos(embed(q), embed(x_i))` and `d_avg = mean_i(d_i)` | two-sided deviation from the batch's mean query-relevance — flags segments that are unusually *off-topic* (injection) or unusually *on-topic* (a poisoned passage optimized to win retrieval); `--embed_model` selects the sentence encoder |

The two are min-max normalized over the segment set, then combined via MAX fusion: `r_i = max(minmax(freq)_i, minmax(r_cos)_i)`. Whichever signal considers a segment riskiest wins for that segment, so one signal being near-floor on a given attack family doesn't drag down a segment the other signal is confident about. (A weighted-sum "linear" fusion was ablated in `lambda_sweep.py` and found strictly worse; MAX is the only fusion rule in production.)

### Frequency Dictionaries

The frequency signal scores segments against a per-model reference frequency dictionary at `data/corpus_freqs/{freq_dataset}/{model_name}_{docs_number}.json`, which must exist before running `--defense sl` or `--defense SL-smooth`. `bash setup.sh` builds the dictionaries shipped with this repo (`openwebtext`, `docs_number=5000`, one file per configured model); to rebuild or add one, use `helpers/build_freq_dict.py`.

## Setup

```bash
conda create -n LocaRisk python=3.12 -y
conda activate LocaRisk

python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy datasets transformers==4.57.1 accelerate bitsandbytes==0.48.1 sentencepiece protobuf tokenizers safetensors huggingface_hub
```

Some configured models are gated on Hugging Face. Log in before running them:

```bash
huggingface-cli login
```

Then prepare local data and frequency resources:

```bash
bash setup.sh
```

The public ObliInjection payload file is an empty JSON placeholder. Replace `data/payloads/obli_injection/payloads.json` locally before running ObliInjection attacked evaluations.

## Run Evaluations

```bash
bash run.sh Mistral-7B 2wikimultihopqa --defense sl --seed 42 --test_number 200
```

By default this runs the **ObliInjection** attack. To run **CorruptRAG**:

```bash
bash run.sh Mistral-7B hotpotqa --defense sl --attack corrupt-rag --seed 42 --test_number 200
bash run.sh Mistral-7B 2wikimultihopqa --defense sl --attack corrupt-rag --seed 42 --test_number 200
```

Available defenses (`--defense`):

- `sl` — SparseLoc: localization-aware defense using RL (r = L(q, xi, X)) to identify and neutralize risky segments
- `ppl` — global perplexity filter
- `segment_ppl` — per-segment perplexity filter
- `windowed_ppl` — sliding-window perplexity filter
- `smoothllm` — randomized smoothing over the full context
- `SL-smooth` — randomized smoothing applied only to localized risky segments
- `none` — pass-through baseline

Useful options:

- `--attack <name>` — `obli-injection` (default), `corrupt-rag`, or `none` (disables attack injection)
- `--adaptive` — use the defense-aware ObliInjection payload set instead of the default one (obli-injection only)
- `--embed_model <name>` — sentence encoder for the multi-signal localization function (sl / SL-smooth only)
- `--test_number <int>`
- `--seed <int>`

`tau`, `docs_number`, `freq_dataset`, and `segment_top_pct` are **not** CLI flags — they're set in `configs/defense/defenses.json` (see [Default Hyperparameters](#default-hyperparameters)); same for the `ppl_fpr`/`window_length` and SmoothLLM perturbation settings in their respective config files.

Show all options:

```bash
bash run.sh --help
```

## Ablations

`ablation.sh` sweeps every configured model, both datasets, and — by default — **both** attack families and **both** localization-based defenses (`sl`, `SL-smooth`):

```bash
bash ablation.sh --mode intervent
```

For `sl` this runs both `swap` and `drop` interventions; `SL-smooth` runs once per configuration (its intervention is always `drop` internally — `--intervention` doesn't apply to it). The localization function is the multi-signal score described in [Localization Function](#localization-function).

Restrict to a single attack, defense, or intervention with `--attack` / `--defense` / `--intervention`:

```bash
bash ablation.sh --mode intervent --attack corrupt-rag
bash ablation.sh --mode intervent --defense SL-smooth
bash ablation.sh --mode intervent --defense sl --intervention swap
```

Show all ablation options:

```bash
bash ablation.sh --help
```

## Default Hyperparameters

These are **not** exposed on the `run.sh` / `ablation.sh` / `eval.py` CLI — edit the listed config file to change them.

| Param | Config file → key | Default | Meaning |
|---|---|---|---|
| `tau` | `defenses.json → locators.multi_signal.tau` | `0.1` | rare-token percentile cutoff for the frequency signal |
| `docs_number` | `defenses.json → locators.multi_signal.docs_number` | `5000` | corpus documents used to build the freq dict |
| `freq_dataset` | `defenses.json → locators.multi_signal.freq_dataset` | `openwebtext` | reference corpus for the freq dict |
| `segment_top_pct` | `defenses.json → locators.multi_signal.segment_top_pct` | `20.0` | % of segments flagged as risky by `sl` / `SL-smooth` |
| `ppl_fpr` | `ppl.json → ppl.target_fpr` | `0.05` | target clean-case false-positive rate, shared by `ppl`, `segment_ppl`, and `windowed_ppl` |
| `window_length` | `ppl.json → windowed_ppl.window_length` | `3` | segments per window for `windowed_ppl` |
| `window_stride` | `ppl.json → windowed_ppl.window_stride` | `1` | stride between windows for `windowed_ppl` |
| `pert_type` | `smoothllm.json → smoothllm.pert_type` | `swap` | SmoothLLM perturbation type |
| `pert_pct` | `smoothllm.json → smoothllm.pert_pct` | `10` | % of tokens perturbed per SmoothLLM copy |
| `num_copies` | `smoothllm.json → smoothllm.num_copies` | `10` | perturbed copies per sample for SmoothLLM majority vote |

Generation defaults (`configs/models.json → _generation`): `max_new_tokens=20`, `temperature=0.1`, `do_sample=true`.

## Outputs

Main evaluations write:

```
log/<model>/<dataset>/attack_<attack>/<defense>/seed_<seed>/test_number_<n>/eval.json
```

`--adaptive` runs write under `attack_<attack>-adaptive` instead of `attack_<attack>`.

Clean runs add `/clean/eval.json`. `ablation.sh` calls `eval.py` per configuration, so ablation sweeps write to the same `log/...` layout, one `eval.json` per (model, dataset, attack, defense, intervention) combination.

`eval.json` contains run metadata, `avg_f1`, ASR for attacked runs, and per-sample answers. Localized defenses also include hit-rate and defense detail fields when available.

## Key Files

```
eval.py                              evaluator (single runs and ablation sweeps)
run.sh                               single-run wrapper
ablation.sh                          ablation sweep wrapper (calls eval.py --defense sl)

configs/
  config.json                        global defaults (default attack)
  models.json                        model aliases and generation settings
  defense/
    defenses.json                    SparseLoc and multi-signal locator defaults
    ppl.json                         PPL defense defaults
    smoothllm.json                   SmoothLLM defaults

src/
  attacks/
    obli_injection/payload_construct.py   ObliInjection class
    corrupt_rag/payload_construct.py      CorruptRAG class
  defenses/
    SLcore/
      rl.py                          RL localization function r = L(q, xi, X), fixed multi-signal (freq + cosine_semantic) scorer
      intervene.py                   Intervention class
      sparse_loc.py                  SparseLoc defense
    ppl.py                           PPL, SegmentPPL, WindowedPPL classes
    smoothllm.py                     SmoothLLM, SmoothLocatedLLM

data/payloads/
  obli_injection/payloads.json       empty public ObliInjection placeholder
  corrupt_rag/hotpotqa_corrputRAG-AS.json   CorruptRAG adversarial texts, hotpotqa
  corrupt_rag/2wikimultihopqa.json          CorruptRAG adversarial texts, 2wikimultihopqa
```

## Citation

```bibtex
@inproceedings{wang2026localization,
  title     = {Why Localization Matters: Allocation, Not Detection, for Sparse-Contamination Multi-Source {RAG} Defense},
  author    = {Wang, Elias},
  booktitle = {Proceedings of the 2026 ACM Workshop on Artificial Intelligence and Security (AISec)},
  year      = {2026},
  address   = {The Hague, Netherlands}
}
```
