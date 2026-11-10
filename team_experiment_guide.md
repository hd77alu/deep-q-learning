# Team Setup & Hyperparameter Experimentation Guide

This guide outlines the standard workflow for running individual hyperparameter experiments for **Formative 3 (Deep Q-Learning)**. Follow these instructions so that everyone’s experiments are executed consistently, logging and model checkpoints are saved in isolated directories, and all team data can easily be compiled for Task 3.

---

## 1. Prerequisites & Environment Verification

Before running any experiments, ensure you have the required dependencies installed in your Python environment:

### Windows Git Bash

```bash
py -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that the following core files exist in the project root:
* `model.py` (Custom CNN Architecture)
* `buffer.py` (Replay Memory)
* `wrappers.py` (Atari Preprocessing & Frame Stacking)
* `train.py` (Training script with `argparse` configuration)

---

## 2. Set Up Your Personal Shell Script

Each team member must create their own shell script using their name (e.g., `run_experiments_alice.sh`, `run_experiments_bob.sh`).

1. Copy the script template below and save it as `run_experiments_<your_name>.sh`.
2. Change the `MEMBER="<your_name>"` variable at the top of the script to your name (in lowercase, e.g., `alice`).

### Shell Script Template (`run_experiments_<your_name>.sh`)

```bash
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Example code from the experiments that have already been run(don't reuse these same values)
MEMBER="your_name"

EXP_CHOICE=${1:-"all"}

run_experiment() {
    case "$1" in
        1)
            echo "--> Running Experiment 1 (Baseline)..."
            python train.py --member-name "$MEMBER" --exp-id 1 \
                --lr 1e-4 --gamma 0.99 --batch-size 32 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 50000 --total-steps 100000
            ;;
        2)
            echo "--> Running Experiment 2 (Higher Learning Rate)..."
            python train.py --member-name "$MEMBER" --exp-id 2 \
                --lr 2.5e-4 --gamma 0.99 --batch-size 32 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 50000 --total-steps 100000
            ;;
        3)
            echo "--> Running Experiment 3 (Lower Learning Rate)..."
            python train.py --member-name "$MEMBER" --exp-id 3 \
                --lr 5e-5 --gamma 0.99 --batch-size 32 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 50000 --total-steps 100000
            ;;
        4)
            echo "--> Running Experiment 4 (Larger Batch Size)..."
            python train.py --member-name "$MEMBER" --exp-id 4 \
                --lr 1e-4 --gamma 0.99 --batch-size 64 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 50000 --total-steps 100000
            ;;
        5)
            echo "--> Running Experiment 5 (Smaller Batch Size)..."
            python train.py --member-name "$MEMBER" --exp-id 5 \
                --lr 1e-4 --gamma 0.99 --batch-size 16 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 50000 --total-steps 100000
            ;;
        6)
            echo "--> Running Experiment 6 (Lower Discount Factor)..."
            python train.py --member-name "$MEMBER" --exp-id 6 \
                --lr 1e-4 --gamma 0.95 --batch-size 32 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 50000 --total-steps 100000
            ;;
        7)
            echo "--> Running Experiment 7 (Higher Discount Factor)..."
            python train.py --member-name "$MEMBER" --exp-id 7 \
                --lr 1e-4 --gamma 0.995 --batch-size 32 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 50000 --total-steps 100000
            ;;
        8)
            echo "--> Running Experiment 8 (Slow Epsilon Decay)..."
            python train.py --member-name "$MEMBER" --exp-id 8 \
                --lr 1e-4 --gamma 0.99 --batch-size 32 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 70000 --total-steps 100000
            ;;
        9)
            echo "--> Running Experiment 9 (Fast Epsilon Decay)..."
            python train.py --member-name "$MEMBER" --exp-id 9 \
                --lr 1e-4 --gamma 0.99 --batch-size 32 \
                --eps-start 1.0 --eps-end 0.01 --eps-decay 25000 --total-steps 100000
            ;;
        10)
            echo "--> Running Experiment 10 (Higher Minimum Epsilon)..."
            python train.py --member-name "$MEMBER" --exp-id 10 \
                --lr 1e-4 --gamma 0.99 --batch-size 32 \
                --eps-start 1.0 --eps-end 0.05 --eps-decay 50000 --total-steps 100000
            ;;
        *)
            echo "Invalid option: $1"
            echo "Usage: ./run_experiments_hamed.sh [1-10|all]"
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
        run_experiment $i
    done
    echo "=================================================="
    echo " All 10 experiments completed successfully! "
    echo "=================================================="
else
    run_experiment $EXP_CHOICE
    echo "=================================================="
    echo " Experiment $EXP_CHOICE completed successfully! "
    echo "=================================================="
fi
```

---

## 3. Make Your Shell Script Executable

Before executing your script for the first time, grant execute permissions in your terminal:

```bash
chmod +x run_experiments_<your_name>.sh
```

---

## 4. Running Experiments

### A. Run a Single Experiment (Recommended)

> Note: Experiment with 100,000 steps takes 20-25 minutes in A CPU

To run one specific experiment (e.g., Experiment 1):

```bash
./run_experiments_<your_name>.sh 1
```

### B. Run a Single Experiment in the Background (Using `nohup`)
To keep an experiment running in the background even if you close your terminal session, use `nohup`:

```bash
nohup ./run_experiments_<your_name>.sh 1 > exp_01.log 2>&1 &
```

### C. Run All 10 Experiments Sequentially
To execute all 10 runs one after another:

```bash
./run_experiments_<your_name>.sh all
```

---

## 5. Monitoring & Process Management

* **Watch live training logs** (when using background `nohup` execution):
  ```bash
  tail -f exp_01.log
  ```
  *(Press `Ctrl + C` to exit the log view without stopping training).*

* **Check if your training process is running**:
  ```bash
  ps aux | grep "exp-id"
  ```

* **Stop an active background experiment**:
  ```bash
  pkill -f "exp-id 1"
  ```

---

## 6. Output Directory Structure

Each experiment will automatically save output files into your dedicated member folder:

```text
<your_name>/
├── exp_01/
│   ├── dqn_model.pth    # Model checkpoint with best evaluation reward
│   └── metrics.csv      # Logged metrics (Episode, Steps, Length, Reward, Epsilon, Loss)
├── exp_02/
│   ├── dqn_model.pth
│   └── metrics.csv
...
└── exp_10/
    ├── dqn_model.pth
    └── metrics.csv
```

---

## 7. Reporting Results

After completing your 10 experiments, inspect the `metrics.csv` file in each folder to fill out your required Task 3 documentation table for the presentation:

### Aggregate experiment results

Run the aggregation script from the repository root:

```bash
python aggregate_results.py
```

When prompted, enter your member experiment folder name, for example:

```text
Enter the experiment folder path (for example, hamed): hamed
```

The script reads each `exp_*/metrics.csv` file and prints a summary table containing:

* `Avg_Final_Reward`: average reward from the final 20 episodes
* `Max_Reward`: highest reward recorded in the experiment

You can also provide the folder directly without the prompt:

```bash
python aggregate_results.py hamed
```

The results can then be copied into a Markdown report such as `hamed/all_exp_results/exp_table.md`.

| Experiment | Learning rate | Gamma ($\gamma$) | Batch size | Epsilon start ($\epsilon_0$) | Epsilon end ($\epsilon_{min}$) | Epsilon decay | Observed behavior / Performance |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Baseline performance; steady convergence |
| 2 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Fast initial gains, high variance |
| 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Slow learning, stable trajectory |
| ... | ... | ... | ... | ... | ... | ... | ... |