
#!/bin/bash
set -euo pipefail

TASK="${1:-both}"

case "$TASK" in
    regression|classification|both) ;;
    *) echo "Invalid task: $TASK" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"
mkdir -p outputs

timestamp="$(date +%Y-%m-%d_%H-%M-%S)_$$"

run_trial() {
    local task="$1"
    local label="$2"
    shift 2

    local output="runs/$task/chemberta/sweep/${timestamp}_${label}"
    local overrides="$* evaluate_test=false hydra.run.dir=$output"
    local job_name="chemberta_${task}_${label}"
    local environment="MODEL=chemberta,TASK=$task,OVERRIDES=$overrides"

    if [[ -e "$output" ]]; then
        echo "Run already exists: $output" >&2
        exit 2
    fi

    echo "Submitting $task / $label"
    bsub -J "$job_name" -env "$environment" < finetune.sh
}

if [[ "$TASK" == regression || "$TASK" == both ]]; then
    run_trial regression mlp_hidden128 readout=mlp model.freeze_encoder=true readout.hidden_dim=128
    run_trial regression mlp_hidden256 readout=mlp model.freeze_encoder=true readout.hidden_dim=256
fi

if [[ "$TASK" == classification || "$TASK" == both ]]; then
    run_trial classification linear_lr5e-6 readout=linear model.freeze_encoder=false model.encoder_lr=5e-6
    run_trial classification linear_lr1e-5 readout=linear model.freeze_encoder=false model.encoder_lr=1e-5
fi
