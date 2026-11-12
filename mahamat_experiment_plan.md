# Mahamat Bakhit — experiment plan

Run from the repository root with `./run_my_experiments.sh all`. Each experiment runs 100,000 agent steps with a 20,000-transition replay buffer to fit the available 8 GB RAM. This differs from the earlier members' 100,000-transition buffer, so cross-member comparisons include that limitation. The script skips experiments with `completed.json` and protects existing folders from overwrite. The planned settings are:

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 50000 |
| 2 | 0.00005 | 0.99 | 64 | 1.0 | 0.05 | 50000 |
| 3 | 0.0002 | 0.99 | 64 | 1.0 | 0.05 | 50000 |
| 4 | 0.0001 | 0.95 | 64 | 1.0 | 0.05 | 50000 |
| 5 | 0.0001 | 0.995 | 64 | 1.0 | 0.05 | 50000 |
| 6 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50000 |
| 7 | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 50000 |
| 8 | 0.0001 | 0.99 | 64 | 0.8 | 0.05 | 50000 |
| 9 | 0.0001 | 0.99 | 64 | 1.0 | 0.02 | 50000 |
| 10 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 25000 |

After training: `uv run python make_reports.py mahamat_bakhit`, then `uv run python evaluate.py --episodes 5 --output evaluation_results.csv`. Compare against the existing winner and update the root checkpoint, training plot, and video if Mahamat's model wins.
