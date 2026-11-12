# Deep Q-learning for Atari Pac-Man

A group formative assignment implementing a Deep Q-Network (DQN) from scratch in PyTorch and Gymnasium. The selected game is `ALE/Pacman-v5`. Group members: Hamed, Andrew, Heroine Mutumwinka and Mahamat Bakhit.

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

Each of the four members completed ten distinct hyperparameter combinations. Mahamat Bakhit used `run_my_experiments.sh`, which varies learning rate, gamma, batch size and the three epsilon parameters. From the repository root:

```bash
./run_my_experiments.sh 1
./run_my_experiments.sh all
```

Each run takes 100,000 agent steps. Mahamat’s runner uses a 20,000-transition replay buffer to fit the available 8 GB RAM; earlier members used 100,000, so cross-member results are not controlled for buffer size. The runs are saved under `mahamat_bakhit/exp_NN/` with `config.json`, `metrics.csv`, `dqn_model.pth`, `final_model.pth` and `completed.json`. A populated experiment folder is protected from accidental overwrite. The original `dqn_model.pth` files in the three existing member folders were selected using peak exploratory training reward; this can be noisy.

The four required experiment tables are in the member folders. Mahamat’s table is `mahamat_bakhit/MahamatBakhitHyperparameter_experiment_results.md`; all ten runs have `completed.json` confirming 100,000 steps.

## Evaluation, plots and gameplay

Evaluate saved checkpoints under greedy actions and identical seeds:

```bash
uv run python evaluate.py --episodes 5 --output evaluation_results.csv
```

All 40 saved checkpoints were evaluated with greedy actions on the same five seeds (1000–1004); results are in `evaluation_results.csv`. The highest mean reward was **Heroine experiment 8: 140.4**, followed by Heroine experiment 6: 100.2. Mahamat’s best was experiment 3: 47.6. A ten-episode finalist comparison on seeds 2000–2009 gave Heroine experiment 8 a mean reward of **170.2**, Heroine experiment 6 **84.0**, Heroine experiment 4 **53.5**, and Mahamat experiment 3 **61.7**. The ten-episode results are in `finalist_evaluation.csv` and `mahamat_finalist_evaluation.csv`. Heroine experiment 8 is the final selected checkpoint, copied to root `dqn_model.pth`. Its reward, episode-length and step plot is `plots/heroine_exp08_training.png`. Our evaluation uses a finite set of seeds, so scores are estimates, not guarantees. To regenerate it:

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

The [gameplay demonstration](videos/rl-video-episode-0.mp4) was recorded from seed 2000 with greedy actions; the episode scored 202 over 906 agent steps. The video uses the final selected checkpoint.

## Submission checklist

- Four named experiment reports and 40 experiment model folders.
- Training plots for the winning experiment.
- `evaluation_results.csv`, finalist comparisons, selected model and gameplay video.
- `Task allocation sheet.pdf` with the four actual member names and responsibilities.
- Confirm the project installs with `uv sync` and `play.py` runs using the chosen checkpoint.
