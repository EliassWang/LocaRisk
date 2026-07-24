#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME=""
DATASET=""
DEFENSE=""
ATTACK=""
EMBED_MODEL=""
TEST_NUMBER=""
SEED=""
DEBUG=0
ADAPTIVE=0
NO_ATTACK=0
POSITIONAL_ARGS=()

die() {
  echo "$1" >&2
  usage >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash run.sh <model_name> <dataset> [options]
  bash run.sh --model_name <name> --dataset <name> [options]

Options:
  --model_name <name>  Model name from configs/models.json
  --dataset <name>     One of: hotpotqa, 2wikimultihopqa
  --defense <name>     One of: sl, ppl, segment_ppl, windowed_ppl, smoothllm, SL-smooth, none
  --attack <name>      One of: obli-injection (default), corrupt-rag, none (disables attack injection)
  --embed_model <name> Embedding model for the multi_signal locator (sl / SL-smooth defenses)
  --adaptive            Use the defense-aware ObliInjection payload set (obli-injection only)
  --no_attack           Skip payload injection but keep --attack's sample-selection logic
                        (e.g. CorruptRAG's payload-ID filter), for a sample-matched clean
                        baseline. Redundant with --attack none.
  --test_number <int>  Number of samples to test (eval.py default: 200)
  --seed <int>         Optional random seed (eval.py auto-generates one if omitted)
  --debug              Print intermediate RL scores and intervention decisions
  -h, --help           Show this help message

tau, docs_number, freq_dataset, segment_top_pct, ppl_fpr, window_length,
and the SmoothLLM perturbation settings are configured in
configs/defense/{defenses,ppl,smoothllm}.json, not on this CLI.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --defense)
      DEFENSE="$2"
      shift 2
      ;;
    --attack)
      ATTACK="$2"
      shift 2
      ;;
    --embed_model)
      EMBED_MODEL="$2"
      shift 2
      ;;
    --test_number)
      TEST_NUMBER="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --adaptive)
      ADAPTIVE=1
      shift
      ;;
    --no_attack)
      NO_ATTACK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" == -* ]]; then
        die "Unknown argument: $1"
      fi
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 && -z "$MODEL_NAME" ]]; then
  MODEL_NAME="${POSITIONAL_ARGS[0]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 1 && -z "$DATASET" ]]; then
  DATASET="${POSITIONAL_ARGS[1]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
  die "Too many positional arguments"
fi

if [[ -z "$MODEL_NAME" ]]; then
  die "model_name is required"
fi

if [[ -z "$DATASET" ]]; then
  die "dataset is required"
fi

if [[ -z "$DEFENSE" ]]; then
  die "defense is required"
fi

COMMON_ARGS=(
  --model "$MODEL_NAME"
  --dataset "$DATASET"
)

if [[ -n "$ATTACK" ]]; then
  COMMON_ARGS+=(--attack "$ATTACK")
fi

if [[ -n "$EMBED_MODEL" ]]; then
  COMMON_ARGS+=(--embed_model "$EMBED_MODEL")
fi

if [[ -n "$TEST_NUMBER" ]]; then
  COMMON_ARGS+=(
    --test_number "$TEST_NUMBER"
  )
fi

if [[ -n "$SEED" ]]; then
  COMMON_ARGS+=(
    --seed "$SEED"
  )
fi

if [[ "$DEBUG" -eq 1 ]]; then
  COMMON_ARGS+=(--debug)
fi

if [[ "$ADAPTIVE" -eq 1 ]]; then
  COMMON_ARGS+=(--adaptive)
fi

if [[ "$NO_ATTACK" -eq 1 ]]; then
  COMMON_ARGS+=(--no_attack)
fi

COMMON_ARGS+=(
  --defense "$DEFENSE"
)

exec python "./eval.py" "${COMMON_ARGS[@]}"
