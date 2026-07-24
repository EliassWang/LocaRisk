#!/usr/bin/env bash

set -u -o pipefail

MODE=""
ATTACK_FILTER=""
DEFENSE_FILTER=""
EMBED_MODEL=""
INTERVENTION_FILTER=""
ADAPTIVE=0
TEST_NUMBER="200"
SEED="42"

DEFENSES=(
  "sl"
  "SL-smooth"
)

DATASETS=(
  "hotpotqa"
  "2wikimultihopqa"
)

ATTACKS=(
  "obli-injection"
  "corrupt-rag"
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
  intervent  For sl: compare intervention methods swap, drop. Also runs
             SL-smooth once per configuration (its intervention is
             always drop internally, --intervention does not apply to it).
             locator is always multi_signal for both.

Options:
  --mode <mode>         One of: intervent, intervention
  --attack <name>       Restrict to a single attack: obli-injection or corrupt-rag; default runs both
  --defense <name>      Restrict to a single defense: sl or SL-smooth; default runs both
  --embed_model <name>  Embedding model for the multi_signal locator (HuggingFace id or local path)
  --intervention <name> Restrict sl to a single intervention (e.g. swap, drop); default runs both
  --adaptive             Use the defense-aware ObliInjection payload set (applies only to the
                         obli-injection leg of the sweep; corrupt-rag runs are unaffected)
  -h, --help            Show this help message

tau, docs_number, freq_dataset, and segment_top_pct are configured in
configs/defense/defenses.json, not on this CLI.
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
    --attack)
      require_arg "$@"
      ATTACK_FILTER="$2"
      shift 2
      ;;
    --defense)
      require_arg "$@"
      DEFENSE_FILTER="$2"
      shift 2
      ;;
    --embed_model)
      require_arg "$@"
      EMBED_MODEL="$2"
      shift 2
      ;;
    --intervention)
      require_arg "$@"
      INTERVENTION_FILTER="$2"
      shift 2
      ;;
    --adaptive)
      ADAPTIVE=1
      shift
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
  intervent)
    ;;
  *)
    die "Unknown mode: $MODE"
    ;;
esac

INTERVENTIONS=(swap drop)

if [[ -n "$INTERVENTION_FILTER" ]]; then
  if [[ ! " ${INTERVENTIONS[*]} " == *" ${INTERVENTION_FILTER} "* ]]; then
    die "--intervention '${INTERVENTION_FILTER}' is not valid for mode '${MODE}' (valid: ${INTERVENTIONS[*]})"
  fi
  INTERVENTIONS=("$INTERVENTION_FILTER")
fi

if [[ -n "$ATTACK_FILTER" ]]; then
  if [[ ! " ${ATTACKS[*]} " == *" ${ATTACK_FILTER} "* ]]; then
    die "--attack '${ATTACK_FILTER}' is not valid (valid: ${ATTACKS[*]})"
  fi
  ATTACKS=("$ATTACK_FILTER")
fi

if [[ -n "$DEFENSE_FILTER" ]]; then
  if [[ ! " ${DEFENSES[*]} " == *" ${DEFENSE_FILTER} "* ]]; then
    die "--defense '${DEFENSE_FILTER}' is not valid (valid: ${DEFENSES[*]})"
  fi
  DEFENSES=("$DEFENSE_FILTER")
fi

print_run_banner() {
  local dataset="$1"
  local model="$2"
  local defense="$3"
  local intervention="$4"
  local attack="$5"

  echo
  echo "============================================================"
  echo "Running mode=${MODE} dataset=${dataset} model=${model} attack=${attack} defense=${defense}"
  echo "seed=${SEED} test_number=${TEST_NUMBER}"
  echo "locator=multi_signal"
  echo "intervention=${intervention}"
  echo "============================================================"
}

run_one_configuration() {
  local dataset="$1"
  local model="$2"
  local defense="$3"
  local intervention="$4"
  local attack="$5"
  local run_args=(
    --defense "$defense"
    --model "$model"
    --dataset "$dataset"
    --attack "$attack"
    --seed "$SEED"
    --test_number "$TEST_NUMBER"
  )

  if [[ "$defense" == "sl" ]]; then
    run_args+=(--intervention "$intervention")
  fi

  if [[ -n "$EMBED_MODEL" ]]; then
    run_args+=(--embed_model "$EMBED_MODEL")
  fi

  if [[ "$ADAPTIVE" -eq 1 && "$attack" == "obli-injection" ]]; then
    run_args+=(--adaptive)
  fi

  print_run_banner "$dataset" "$model" "$defense" "$intervention" "$attack"

  if python eval.py "${run_args[@]}"; then
    echo "Completed: ${dataset} / ${model} / attack=${attack} / defense=${defense} / intervention=${intervention}"
  else
    echo "Failed: ${dataset} / ${model} / attack=${attack} / defense=${defense} / intervention=${intervention}"
    FAILED_RUNS+=("${dataset}:${model}:attack_${attack}:defense_${defense}:intervention_${intervention}")
  fi
}

echo
echo "Ablation schedule"
echo "mode=${MODE} attacks=${ATTACKS[*]} defenses=${DEFENSES[*]} embed_model=${EMBED_MODEL:-default} adaptive=${ADAPTIVE}"
echo "datasets=${DATASETS[*]}"
echo "models=${MODELS[*]}"
echo "locator=multi_signal"
echo "interventions (sl only)=${INTERVENTIONS[*]}"
echo "seed=${SEED} test_number=${TEST_NUMBER}"

for dataset in "${DATASETS[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    for model in "${MODELS[@]}"; do
      for defense in "${DEFENSES[@]}"; do
        if [[ "$defense" == "sl" ]]; then
          for intervention in "${INTERVENTIONS[@]}"; do
            run_one_configuration "$dataset" "$model" "$defense" "$intervention" "$attack"
            sleep 3
          done
        else
          run_one_configuration "$dataset" "$model" "$defense" "-" "$attack"
          sleep 3
        fi
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
