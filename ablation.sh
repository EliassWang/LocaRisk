#!/usr/bin/env bash

set -u -o pipefail

MODE=""
TEST_NUMBER="200"
SEED="42"
TAU="0.1"
FREQ_DATASET="openwebtext"
DOCS_NUMBER="5000"
SEGMENT_TOP_PCT="20"

DATASETS=(
  "hotpotqa"
  "2wikimultihopqa"
)

MODELS=(
  "Qwen3-4B"
  "Gemma-2-2B"
  "Llama3-8B"
  "Mistral-7B"
  "Falcon3-7B"
)

FAILED_RUNS=()

usage() {
  cat <<'EOF'
Usage:
  bash ablation.sh --mode <mode>

Modes:
  intervent  Compare intervention methods: swap, drop
  locator    Compare locator methods: freq, random, oracle

Options:
  --mode <mode>  One of: intervent, intervention, locator
  -h, --help     Show this help message
EOF
}

die() {
  echo "$1" >&2
  usage >&2
  exit 1
}

require_arg() {
  if [[ $# -lt 2 || "${2:-}" == -* ]]; then
    die "$1 requires a value"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      require_arg "$@"
      MODE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  die "--mode is required"
fi

case "$MODE" in
  intervention)
    MODE="intervent"
    ;;
  intervent|locator)
    ;;
  *)
    die "Unknown mode: $MODE"
    ;;
esac

case "$MODE" in
  intervent)
    LOCATORS=(freq)
    INTERVENTIONS=(swap drop)
    ;;
  locator)
    LOCATORS=(freq random oracle)
    INTERVENTIONS=(swap)
    ;;
esac

print_run_banner() {
  local dataset="$1"
  local model="$2"
  local locator="$3"
  local intervention="$4"

  echo
  echo "============================================================"
  echo "Running mode=${MODE} dataset=${dataset} model=${model}"
  echo "seed=${SEED} test_number=${TEST_NUMBER} tau=${TAU} docs_number=${DOCS_NUMBER} freq_dataset=${FREQ_DATASET} segment_top_pct=${SEGMENT_TOP_PCT}"
  echo "locator=${locator}"
  echo "intervention=${intervention}"
  echo "============================================================"
}

run_one_configuration() {
  local dataset="$1"
  local model="$2"
  local locator="$3"
  local intervention="$4"
  local run_args=(
    --model "$model"
    --dataset "$dataset"
    --seed "$SEED"
    --test_number "$TEST_NUMBER"
    --tau "$TAU"
    --docs_number "$DOCS_NUMBER"
    --freq_dataset "$FREQ_DATASET"
    --segment_top_pct "$SEGMENT_TOP_PCT"
    --locator "$locator"
    --intervention "$intervention"
    --ablation_type "$MODE"
  )

  print_run_banner "$dataset" "$model" "$locator" "$intervention"

  if python ablation_eval.py "${run_args[@]}"; then
    echo "Completed: ${dataset} / ${model} / locator=${locator} / intervention=${intervention}"
  else
    echo "Failed: ${dataset} / ${model} / locator=${locator} / intervention=${intervention}"
    FAILED_RUNS+=("${dataset}:${model}:locator_${locator}:intervention_${intervention}")
  fi
}

echo
echo "Ablation schedule"
echo "mode=${MODE}"
echo "datasets=${DATASETS[*]}"
echo "models=${MODELS[*]}"
echo "locators=${LOCATORS[*]}"
echo "interventions=${INTERVENTIONS[*]}"
echo "seed=${SEED} test_number=${TEST_NUMBER} tau=${TAU} docs_number=${DOCS_NUMBER} freq_dataset=${FREQ_DATASET} segment_top_pct=${SEGMENT_TOP_PCT}"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for locator in "${LOCATORS[@]}"; do
      for intervention in "${INTERVENTIONS[@]}"; do
        run_one_configuration "$dataset" "$model" "$locator" "$intervention"
        sleep 3
      done
    done
  done
done

echo
echo "Ablation run finished."

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo "Failed runs:"
  printf '  %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo "All runs completed successfully."
