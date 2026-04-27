# LocaRisk: Localization-Aware Test-Time Defense Allocation for Sparse-Contamination RAG

**LocaRisk** is the evaluation code for *Localization-Aware Test-Time Defense Allocation for Sparse-Contamination Multi-Source RAG*.

This repository studies test-time defense allocation for multi-source retrieval-augmented generation (RAG) under sparse indirect prompt injection. In this setting, only one or a few retrieved context segments may be contaminated, while the remaining retrieved context is benign. The main goal of the codebase is to evaluate how different defenses behave when intervention is applied globally, at finer granularity, or only to localized high-risk retrieved segments.

The repository supports:

- loading multi-source RAG context segments;
- optionally injecting an ObliInjection-style attack payload into a retrieved segment;
- applying global, localization-assisted, or diagnostic method variants;
- reporting attack success rate (ASR), clean answer F1, token-cost statistics, and localization details when available.

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

The public payload file is an empty JSON placeholder. Replace `data/payloads/payloads.json` locally before running attacked evaluations.

## Run Evaluations

```bash
bash run.sh Mistral-7B 2wikimultihopqa --defense probe --seed 42 --test_number 200
```

Available method:

- `no_defense`
- `probe`
- `ppl`
- `segment_ppl`
- `windowed_ppl`
- `smoothllm`
- `smooth_located`

Useful options:

- `--test_number <int>`
- `--seed <int>`
- `--no-attack`
- `--tau <float>`
- `--docs_number <int>`
- `--freq_dataset <name>`
- `--segment_top_pct <float>`
- `--ppl_fpr <float>`
- `--window_length <int>`
- `--smooth_pert_type <name>`
- `--smooth_pert_pct <int>`
- `--smooth_num_copies <int>`

Show all options:

```bash
bash run.sh --help
```

## Ablations

Run ablations over all configured models and both datasets:

```bash
bash ablation.sh --mode intervent
```

Run each mode:

```bash
bash ablation.sh --mode intervent
bash ablation.sh --mode locator
```

Show all ablation options:

```bash
bash ablation.sh --help
```

Ablation modes:

- `intervent`: compares `swap` and `drop`
- `locator`: compares `freq`, `random`, and `oracle`

The ablation uses the dataset context as provided; retrieved segment count is no longer swept.

## Outputs

Main evaluations write:

```bash
log/<model>/<dataset>/<defense>/seed_<seed>/test_number_<test_number>/eval.json
```

Clean runs add `/clean/eval.json`.

Ablations write:

```bash
ablation/<model>/<dataset>/freq_dataset_<name>/tau_<tau>/docs_number_<n>/segment_top_pct_<pct>/locator_<locator>/intervention_<kind>/seed_<seed>/test_number_<n>/eval.json
```

Each ablation run also writes `ablation.tsv` and `probe_summary.json` in the same output directory.

`eval.json` contains run metadata, `avg_f1`, ASR for attacked runs, and per-sample answers. Localized defenses also include hit-rate and defense detail fields when available.

## Key Files

- `eval.py`: main evaluator
- `run.sh`: evaluation wrapper
- `ablation_eval.py`: ablation evaluator
- `ablation.sh`: ablation sweep wrapper
- `configs/models.json`: model paths and generation settings
- `configs/datasets.json`: dataset definitions
- `configs/defenses.json`: defense defaults
- `src/defenses/`: defense implementations
- `data/payloads/payloads.json`: empty public payload placeholder
