#!/bin/bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: bash sweep_lgbm.sh [regression|classification|both]" >&2
    exit 2
fi

TASK="${1:-regression}"
case "$TASK" in
    regression|classification) tasks=("$TASK") ;;
    both) tasks=(regression classification) ;;
    *) echo "Invalid task: $TASK (expected regression, classification, or both)" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"
source "${CYP_VENV:-$PWD/.venv}/bin/activate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

for task in "${tasks[@]}"; do
    for split in train val; do
        [[ -f "data/$task/$split.csv" ]] || {
            echo "Missing data/$task/$split.csv; prepare the data first." >&2
            exit 2
        }
    done
done

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"

run_trial() {
    local task="$1"
    local label="$2"
    shift 2
    local output="runs/$task/morgan_rdkit_lgbm/sweep/${timestamp}_${label}"

    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite existing run: $output" >&2
        exit 2
    fi

    echo "Running $task / $label"
    python -u finetune.py \
        model=morgan_rdkit_lgbm \
        "task=$task" \
        device=cpu \
        evaluate_test=false \
        "hydra.run.dir=$output" \
        "$@"
}

run_task() {
    local task="$1"

    run_trial "$task" baseline
    run_trial "$task" learning_rate_0.01 model.params.learning_rate=0.01
    run_trial "$task" learning_rate_0.1 model.params.learning_rate=0.1
    run_trial "$task" n_estimators_200 model.early_stopping_rounds=0 model.params.n_estimators=200
    run_trial "$task" n_estimators_500 model.early_stopping_rounds=0 model.params.n_estimators=500
    run_trial "$task" n_estimators_800 model.early_stopping_rounds=0 model.params.n_estimators=800
    run_trial "$task" num_leaves_15 model.params.num_leaves=15
    run_trial "$task" num_leaves_63 model.params.num_leaves=63
    run_trial "$task" reg_lambda_0 model.params.reg_lambda=0
    run_trial "$task" reg_lambda_10 model.params.reg_lambda=10
}

for task in "${tasks[@]}"; do
    run_task "$task"
done
