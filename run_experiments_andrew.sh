#!/bin/bash

# Stop immediately if any command fails
set -e

MEMBER="andrew"
EXP_CHOICE="${1:-all}"

# Fixed across all experiments (same budget for fair comparison)
TOTAL_STEPS=100000

run_train() {
    # $1 = exp id, $2 = description, $3 lr, $4 gamma, $5 batch, $6 eps-start, $7 eps-end, $8 eps-decay
    local id="$1" desc="$2" lr="$3" gamma="$4" bs="$5" es="$6" ee="$7" ed="$8"
    local dir
    dir=$(printf "%s/exp_%02d" "$MEMBER" "$id")

    # Never overwrite existing results (train.py opens metrics.csv in "w" mode)
    if [ -e "$dir/metrics.csv" ] || [ -e "$dir/dqn_model.pth" ]; then
        echo "ERROR: $dir already contains results. Move or delete it first; refusing to overwrite."
        exit 1
    fi
    mkdir -p "$dir"

    echo "--> Running Experiment $id ($desc)..."
    local start end
    start=$(date +%s)
    uv run python train.py --member-name "$MEMBER" --exp-id "$id" \
        --lr "$lr" --gamma "$gamma" --batch-size "$bs" \
        --eps-start "$es" --eps-end "$ee" --eps-decay "$ed" --total-steps "$TOTAL_STEPS"
    end=$(date +%s)

    # train.py does not write a config or timing, so record them here
    cat > "$dir/config.json" <<EOF
{
  "member": "$MEMBER",
  "exp_id": $id,
  "description": "$desc",
  "env": "ALE/Pacman-v5",
  "lr": $lr,
  "gamma": $gamma,
  "batch_size": $bs,
  "eps_start": $es,
  "eps_end": $ee,
  "eps_decay": $ed,
  "total_steps": $TOTAL_STEPS,
  "seed": 42,
  "training_duration_seconds": $((end - start))
}
EOF
    echo "--> Experiment $id finished in $((end - start))s; saved to $dir"
}

run_experiment() {
    case "$1" in
        1) run_train 1  "Baseline (low LR)"          0.000075 0.98  32 1.0 0.02 60000 ;;
        2) run_train 2  "Medium Learning Rate"       0.00015  0.98  32 1.0 0.02 60000 ;;
        3) run_train 3  "High Learning Rate"         0.0003   0.98  32 1.0 0.02 60000 ;;
        4) run_train 4  "Lower Discount Factor"      0.000075 0.97  32 1.0 0.02 60000 ;;
        5) run_train 5  "Higher Discount Factor"     0.000075 0.995 32 1.0 0.02 60000 ;;
        6) run_train 6  "Batch Size 48"              0.000075 0.98  48 1.0 0.02 60000 ;;
        7) run_train 7  "Batch Size 96"              0.000075 0.98  96 1.0 0.02 60000 ;;
        8) run_train 8  "Lower Initial Epsilon"      0.000075 0.98  32 0.9 0.02 60000 ;;
        9) run_train 9  "Higher Minimum Epsilon"     0.000075 0.98  32 1.0 0.03 60000 ;;
        10) run_train 10 "Slower Epsilon Decay"      0.000075 0.98  32 1.0 0.02 85000 ;;
        *)
            echo "Invalid option: $1"
            echo "Usage: ./run_experiments_andrew.sh [1-10|all]"
            exit 1
            ;;
    esac
}

echo "=================================================="
echo " DQN Hyperparameter Runner - Member: ${MEMBER} "
echo "=================================================="

if [ "$EXP_CHOICE" = "all" ]; then
    echo "Running all 10 experiments sequentially..."
    for i in {1..10}; do
        run_experiment "$i"
    done
    echo "=================================================="
    echo " All 10 experiments completed successfully! "
    echo "=================================================="
else
    run_experiment "$EXP_CHOICE"
    echo "=================================================="
    echo " Experiment $EXP_CHOICE completed successfully! "
    echo "=================================================="
fi
