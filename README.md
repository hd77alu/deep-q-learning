# Deep Q-learning for Atari Pac-Man

A group formative assignment implementing a Deep Q-Network (DQN) from scratch in PyTorch and Gymnasium. The selected game is `ALE/Pacman-v5`. Existing contributors: Hamed, Andrew and Heroine Mutumwinka. The fourth member should add their exact name to this list and to their experiment report.

## Setup (uv only)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root run:

```bash
uv sync
```

`pyproject.toml` and `uv.lock` specify the environment. Do not use the older `requirements.txt` instructions.

## Architecture and training

```text
Atari RGB frames → grayscale 84×84 → four-frame stack (4×84×84)
                                  ↓
Conv 4→32, 8×8 stride 4 → ReLU
Conv 32→64, 4×4 stride 2 → ReLU
Conv 64→64, 3×3 stride 1 → ReLU
Flatten (3136) → Linear 512 → ReLU → Linear 5 Q-values
```

`wrappers.py` applies frame skip 4, resizing and frame stacking. `model.py` defines the CNN. `train.py` keeps an online network and a periodically copied target network. It explores with linear epsilon decay, stores transitions in `buffer.py`, samples random batches, and minimizes Huber loss between the chosen Q-value and `clipped_reward + gamma × max(target Q(next_state))` for nonterminal transitions. Adam updates the online network every four agent steps after 10,000 warm-up steps. Raw episode rewards, lengths and cumulative steps are logged in each `metrics.csv`.

## Ten experiments per member

Each member must run ten distinct hyperparameter combinations. The fourth member can use `run_my_experiments.sh`, which varies learning rate, gamma, batch size and the three epsilon parameters. From the repository root:

```bash
MEMBER=your_name ./run_my_experiments.sh 1
MEMBER=your_name ./run_my_experiments.sh all
```

Each run takes 100,000 agent steps and may take substantial time and memory. Run one experiment first to confirm hardware capacity, then run the others. Runs are saved under `your_name/exp_NN/` with `config.json`, `metrics.csv`, `dqn_model.pth`, `final_model.pth` and `completed.json`. A populated experiment folder is protected from accidental overwrite. The original `dqn_model.pth` files in the three existing member folders were selected using peak exploratory training reward; this can be noisy.

After the ten runs, write `your_name/YourNameHyperparameter_experiment_results.md` with ten rows containing learning rate, gamma, batch size, epsilon start, epsilon end, epsilon decay and evidence-based observed behaviour. Keep the models and logs for all ten runs.

## Evaluation, plots and gameplay

Evaluate saved checkpoints under greedy actions and identical seeds:

```bash
uv run python evaluate.py --episodes 5 --output evaluation_results.csv
```

The current 30-checkpoint screening used three episodes on shared seeds 1000–1002 (`evaluation_results.csv`). Heroine experiment 8 led. A further ten-episode comparison of the top three candidates on seeds 2000–2009 (`finalist_evaluation.csv`) gave experiment 8 a mean raw reward of 170.2, versus 84.0 for experiment 6 and 53.5 for experiment 4. This is the provisional winner until the fourth member’s ten models are evaluated. The selected checkpoint is also copied to root `dqn_model.pth`. Its reward, episode-length and step plot is `plots/heroine_exp08_training.png`. To regenerate it:

```bash
uv run python plot_training.py heroine/exp_08/metrics.csv --output plots/best_training.png
```

Display the current selected model:

```bash
uv run python play.py --model dqn_model.pth
```

Record a gameplay video:

```bash
uv run python play.py --model dqn_model.pth --video-dir videos --headless
```

The [gameplay demonstration](videos/rl-video-episode-0.mp4) was recorded from seed 2000 with greedy actions; the episode scored 202 over 906 agent steps. Re-evaluate and update the root checkpoint, plot and video after the fourth member finishes training.

## Submission checklist

- Four named experiment reports and 40 experiment model folders.
- Training plots for the winning experiment.
- `evaluation_results.csv`, `finalist_evaluation.csv`, selected model and gameplay video (update after fourth member).
- `Task allocation sheet.pdf` with the four actual member names and responsibilities.
- Confirm the project installs with `uv sync` and `play.py` runs using the chosen checkpoint.
