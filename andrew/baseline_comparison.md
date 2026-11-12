# Comparison with Hamed's Baseline Experiments

Andrew's numbers come from `andrew/exp_NN/metrics.csv`, Hamed's from `hamed/exp_NN/metrics.csv`, using the same calculations for both. Hamed's files were only read. He has no `config.json`, so his hyperparameters are taken from `run_experiments_hamed.sh`, and no training durations exist for his runs.

**All rewards here are training-episode rewards.** Neither of us ran a separate greedy evaluation, so nothing in this comparison is evaluation performance.

## Hyperparameters side by side

| Exp | Hamed (changed setting) | Andrew (changed setting) |
|---|---|---|
| 1 | Baseline: lr 1e-4, γ 0.99, batch 32, ε 1.0→0.01, decay 50k | Baseline: lr 7.5e-5, γ 0.98, batch 32, ε 1.0→0.02, decay 60k |
| 2 | lr 2.5e-4 | lr 1.5e-4 |
| 3 | lr 5e-5 | lr 3e-4 |
| 4 | batch 64 | γ 0.97 |
| 5 | batch 16 | γ 0.995 |
| 6 | γ 0.95 | batch 48 |
| 7 | γ 0.995 | batch 96 |
| 8 | decay 70k | ε start 0.9 |
| 9 | decay 25k | ε end 0.03 |
| 10 | ε end 0.05 | decay 85k |

All 20 configurations are different. Both of us used 100,000 steps and seed 42 and the same fixed settings (buffer, target update, optimizer, network, preprocessing). **The two baselines differ in four settings** (learning rate, gamma, epsilon end, decay), so a difference between our experiment sets cannot be pinned on any one hyperparameter.

## Performance (training rewards)

| Metric | Andrew (10 runs) | Hamed (10 runs) |
|---|---|---|
| Mean of last-20 averages | 40.18 | 41.05 |
| Median of last-20 averages | 36.12 | 41.90 |
| Mean of last-50 averages | 37.13 | 36.29 |
| Mean whole-run average reward | 27.91 | 27.59 |
| Mean of per-run max reward | 194.9 | 198.4 |
| Highest single-episode reward | 345 (exp 3) | 345 (exp 5) |
| Mean last-20 standard deviation | 20.32 | 25.22 |
| Best run by last-20 average | Exp 7, 61.00 (batch 96) | Exp 4, 47.90 (batch 64) |
| Best run by last-50 average | Exp 7, 56.88 | Exp 10, 45.18 (ε end 0.05) |
| Best run by whole-run average | Exp 7, 32.50 | Exp 4, 34.59 |
| Worst run by last-20 average | Exp 9, 27.70 | Exp 3, 26.65 (lr 5e-5) |

**Best configurations.** Andrew's best (experiment 7, 61.00) is higher than Hamed's best (experiment 4, 47.90) on last-20 reward, but on whole-run average Hamed's experiment 4 (34.59) is higher than Andrew's experiment 7 (32.50). The two sets agree on one thing: in each, the largest batch size tested was the top or near-top run on last-20 reward (Andrew batch 96 → 61.00; Hamed batch 64 → 47.90, against 41.95 and 41.85 for his batch 32 and 16). With one seed each, that is suggestive, not established.

**Averages and maxima.** The two sets are practically level. Mean last-20 reward differs by under 1 point (40.18 vs 41.05) and mean whole-run reward by 0.3. Both sets contain one 345-point episode, and in both it occurred in a run that was otherwise unremarkable (Andrew's experiment 3 reached it at step 35,891 while epsilon was still about 0.41, so exploration likely produced it). Maximum reward is a poor indicator of learning quality.

**Training stability.** Using the standard deviation of the last 20 episodes, Andrew's runs are slightly less variable on average (20.32 vs 25.22). This is driven by a few outliers, notably Hamed's experiment 5 (standard deviation 69.66, caused by a 345-point episode in its last 20) and Andrew's experiments 1 and 7 (about 36–37 each). Neither set shows stable, converged behaviour. Both are noisy at 100,000 steps.

**Exploration schedules.** Hamed's schedules ended at ε = 0.01 (0.05 in experiment 10) with decays of 25k, 50k and 70k steps. Andrew's ended at 0.02 (0.03 in experiment 9) with 60k (85k in experiment 10). Where the two sets overlap most closely, Hamed's higher exploration floor (ε end 0.05, experiment 10) came out second best on last-20 reward (46.75) and best on last-50 (45.18), while Andrew's smaller increase (0.03, experiment 9) was the worst of his runs (27.70), but that run is missing its final 20,761 steps from the CSV, so it is not a fair comparison. Slower decay did not help either of us: Hamed's 70k run (39.60) was below his baseline (41.95), and Andrew's 85k run (33.50) was below his baseline (45.35). Hamed's fastest decay (25k) scored 38.35, also below his baseline.

**Learning rate.** The two sets disagree. Andrew's reward fell as the rate rose (7.5e-5: 45.35, 1.5e-4: 38.30, 3e-4: 30.40), while Hamed's lowest rate (5e-5) was his worst run (26.65) and 1e-4 and 2.5e-4 were nearly tied (41.95, 41.70). This is a reminder that single-seed, single-change results do not carry over between baselines.


## Conclusion

Taken together, the two sets of ten experiments tell a consistent story: at this budget, DQN on Pac-Man learns slowly, results are dominated by episode-to-episode noise, and most hyperparameter changes move the final average reward by amounts comparable to that noise. The one pattern that appears in both sets is a possible benefit from a larger batch size, and the one that fails to replicate is the learning-rate effect. Claiming more would require multiple seeds and greedy evaluation.
