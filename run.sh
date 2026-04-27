#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME=""
DATASET=""
DEFENSE=""
TEST_NUMBER=""
SEED=""
TAU=""
DOCS_NUMBER=""
FREQ_DATASET=""
PPL_FPR=""
WINDOW_LENGTH=""
SMOOTH_PERT_TYPE=""
SMOOTH_PERT_PCT=""
SMOOTH_NUM_COPIES=""
SEGMENT_TOP_PCT=""
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
  --defense <name>     One of: probe, ppl, segment_ppl, windowed_ppl, smoothllm, smooth_located, no_defense
  --test_number <int>  Number of samples to test (eval.py default: 200)
  --seed <int>         Optional random seed (eval.py auto-generates one if omitted)
  --tau <float>        Probe tau used by smooth_located/localized defenses
  --docs_number <int>  Probe docs_number used by smooth_located/localized defenses
  --freq_dataset <name> Probe freq_dataset used by smooth_located/localized defenses
  --segment_top_pct <float> Probe top-percentage segment selection used by smooth_located
  --ppl_fpr <float>    Target clean case-FPR for ppl, segment_ppl, and windowed_ppl
  --window_length <int> Window length for windowed_ppl
  --smooth_pert_type <name> SmoothLLM perturbation type (default from configs/defenses.json)
  --smooth_pert_pct <int> SmoothLLM perturbation percentage
  --smooth_num_copies <int> Number of SmoothLLM perturbed copies
  --no-attack          Disable attack injection into retrieved segments
  -h, --help           Show this help message
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
    --test_number)
      TEST_NUMBER="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --tau)
      TAU="$2"
      shift 2
      ;;
    --docs_number)
      DOCS_NUMBER="$2"
      shift 2
      ;;
    --freq_dataset)
      FREQ_DATASET="$2"
      shift 2
      ;;
    --segment_top_pct)
      SEGMENT_TOP_PCT="$2"
      shift 2
      ;;
    --ppl_fpr)
      PPL_FPR="$2"
      shift 2
      ;;
    --window_length)
      WINDOW_LENGTH="$2"
      shift 2
      ;;
    --smooth_pert_type)
      SMOOTH_PERT_TYPE="$2"
      shift 2
      ;;
    --smooth_pert_pct)
      SMOOTH_PERT_PCT="$2"
      shift 2
      ;;
    --smooth_num_copies)
      SMOOTH_NUM_COPIES="$2"
      shift 2
      ;;
    --no-attack)
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

if [[ -n "$TAU" ]]; then
  COMMON_ARGS+=(
    --tau "$TAU"
  )
fi

if [[ -n "$DOCS_NUMBER" ]]; then
  COMMON_ARGS+=(
    --docs_number "$DOCS_NUMBER"
  )
fi

if [[ -n "$FREQ_DATASET" ]]; then
  COMMON_ARGS+=(
    --freq_dataset "$FREQ_DATASET"
  )
fi

if [[ -n "$SEGMENT_TOP_PCT" ]]; then
  COMMON_ARGS+=(
    --segment_top_pct "$SEGMENT_TOP_PCT"
  )
fi

if [[ -n "$PPL_FPR" ]]; then
  COMMON_ARGS+=(
    --ppl_fpr "$PPL_FPR"
  )
fi

if [[ -n "$WINDOW_LENGTH" ]]; then
  COMMON_ARGS+=(
    --window_length "$WINDOW_LENGTH"
  )
fi

if [[ -n "$SMOOTH_PERT_TYPE" ]]; then
  COMMON_ARGS+=(
    --smooth_pert_type "$SMOOTH_PERT_TYPE"
  )
fi

if [[ -n "$SMOOTH_PERT_PCT" ]]; then
  COMMON_ARGS+=(
    --smooth_pert_pct "$SMOOTH_PERT_PCT"
  )
fi

if [[ -n "$SMOOTH_NUM_COPIES" ]]; then
  COMMON_ARGS+=(
    --smooth_num_copies "$SMOOTH_NUM_COPIES"
  )
fi

if [[ "$NO_ATTACK" -eq 1 ]]; then
  COMMON_ARGS+=(
    --no-attack
  )
fi

COMMON_ARGS+=(
  --defense "$DEFENSE"
)

exec python "./eval.py" "${COMMON_ARGS[@]}"
