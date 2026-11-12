#!/bin/bash

set -e

MEMBER="heroine"
EXP_CHOICE="${1:-all}"

run_experiment() {
    case "$1" in
        1)
            echo "--> Running Experiment 1 (Baseline: batch 64, eps end 0.05)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 1 \
                --lr 1e-4 --gamma 0.99 --batch-size 64 \
                --eps-start 1.0 --eps-end 0.05 --eps-decay 50000 --total-steps 100000
            ;;
        2)
            echo "--> Running Experiment 2 (Higher Learning Rate)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 2 \
                --lr 5e-4 --gamma 0.99 --batch-size 64 \
                --eps-start 1.0 --eps-end 0.05 --eps-decay 50000 --total-steps 100000
            ;;
        3)
            echo "--> Running Experiment 3 (Very High Learning Rate)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 3 \
                --lr 1e-3 --gamma 0.99 --batch-size 64 \
                --eps-start 1.0 --eps-end 0.05 --eps-decay 50000 --total-steps 100000
            ;;
        4)
            echo "--> Running Experiment 4 (Low Discount Factor)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 4 \
                --lr 1e-4 --gamma 0.90 --batch-size 64 \
                --eps-start 1.0 --eps-end 0.05 --eps-decay 50000 --total-steps 100000
            ;;
        5)
            echo "--> Running Experiment 5 (Very High Discount Factor)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 5 \
                --lr 1e-4 --gamma 0.999 --batch-size 64 \
                --eps-start 1.0 --eps-end 0.05 --eps-decay 50000 --total-steps 100000
            ;;
        6)
            echo "--> Running Experiment 6 (Larger Batch Size)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 6 \
                --lr 1e-4 --gamma 0.99 --batch-size 128 \
                --eps-start 1.0 --eps-end 0.05 --eps-decay 50000 --total-steps 100000
            ;;
        7)
            echo "--> Running Experiment 7 (Higher Minimum Epsilon)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 7 \
                --lr 1e-4 --gamma 0.99 --batch-size 64 \
                --eps-start 1.0 --eps-end 0.10 --eps-decay 50000 --total-steps 100000
            ;;
        8)
            echo "--> Running Experiment 8 (Very Fast Epsilon Decay)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 8 \
                --lr 1e-4 --gamma 0.99 --batch-size 64 \
                --eps-start 1.0 --eps-end 0.05 --eps-decay 10000 --total-steps 100000
            ;;
        9)
            echo "--> Running Experiment 9 (Lower Initial Epsilon)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 9 \
                --lr 1e-4 --gamma 0.99 --batch-size 64 \
                --eps-start 0.5 --eps-end 0.05 --eps-decay 50000 --total-steps 100000
            ;;
        10)
            echo "--> Running Experiment 10 (Mixed Combination)..."
            uv run python train.py --member-name "$MEMBER" --exp-id 10 \
                --lr 2.5e-4 --gamma 0.98 --batch-size 128 \
                --eps-start 1.0 --eps-end 0.02 --eps-decay 80000 --total-steps 100000
            ;;
        *)
            echo "Invalid option: $1"
            echo "Usage: ./run_experiments_heroine.sh [1-10|all]"
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
