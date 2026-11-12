# Individual Contribution — Andrew

**Task:** run 10 hyperparameter-tuning experiments for a DQN agent on Atari Pac-Man (Formative 3), reusing the group's shared training pipeline.

## What I did
- Inspected the shared pipeline (`train.py`, `team_experiment_guide.md`, `run_experiments_hamed.sh`, `aggregate_results.py`) and checked my 10 configurations against Hamed's for uniqueness and against `train.py` for accepted arguments.
- Wrote `run_experiments_andrew.sh`, which runs experiments 1–10 (or `all`) through the existing `train.py`, writes a `config.json` per run, stops on failure, and refuses to overwrite existing results.
- Ran all 10 experiments (100,000 steps each, seed 42). Results are in `andrew/exp_01` … `andrew/exp_10`, each with `dqn_model.pth`, `metrics.csv` and `config.json`.
- Wrote `andrew/experiment_analysis.md` (methodology, results table, observations, best configuration, limitations) and `andrew/baseline_comparison.md` (comparison with Hamed's results).

## What I did not change
No shared file was modified (`train.py`, `model.py`, `buffer.py`, `wrappers.py`, the guide, Hamed's script and Hamed's results are untouched). Everything I added is under `andrew/` plus `run_experiments_andrew.sh`.

## Key findings (training rewards, single seed)
- Best run by average reward over the last 20 episodes: **experiment 7** (batch size 96), 61.00. It also ranks first on last-50 and whole-run averages.
- Most other differences are within episode-to-episode noise, and the learning-rate trend I saw did not match Hamed's.
- Experiment 9 (and Hamed's experiment 6) are missing about 20,000 final steps from the CSV because `train.py` logs only when an episode ends.

## Honest limitations
Single seed, short budget, no greedy evaluation, and wall-clock durations that are unreliable (experiment 10 shows 41,975 s because the machine was idle). See the analysis file for detail.

## How to reproduce
```bash
./run_experiments_andrew.sh 1      # one experiment
./run_experiments_andrew.sh all    # all ten (refuses if results already exist)
python aggregate_results.py andrew # summary table
```
