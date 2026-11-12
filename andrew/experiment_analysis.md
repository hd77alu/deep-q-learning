# DQN Hyperparameter Experiments on Atari Pac-Man — Andrew

All numbers below are computed directly from `andrew/exp_NN/metrics.csv` and `config.json`, produced by `run_experiments_andrew.sh` (which calls the shared `train.py`). Nothing is estimated.

## A. Introduction

A DQN agent learns an action-value function from its own experience, and how well it learns depends heavily on a handful of hyperparameters. These set how fast the network moves (learning rate, batch size), how far ahead the agent values reward (discount factor), and how much it explores versus exploits (the epsilon schedule). Tuning means changing these settings and measuring what happens to learning speed, final performance and stability. Deep RL is noisy, so these runs show what happened under one setup, not general rules.

## B. Experimental methodology

- **Environment:** `ALE/Pacman-v5` (5 discrete actions), 84×84 grayscale, frame-skip 4, 4-frame stack, up to 30 no-ops at reset.
- **Shared code, unchanged:** `train.py`, `model.py` (CNN), `buffer.py`, `wrappers.py`. Nothing was reimplemented.
- **Training procedure:** Adam; Huber loss; gradient clipping at 10; rewards clipped to {-1, 0, 1} for the learning target (the logged reward is the raw game score); 1 update every 4 steps after a 10,000-step warm-up.
- **Held fixed in every run:** 100,000 environment steps, replay capacity 100,000, target-network update every 1,000 steps, seed 42, optimizer, network, preprocessing.
- **Varied:** learning rate, gamma, batch size, epsilon start, epsilon end, epsilon decay. Epsilon decays *linearly* over environment steps, and the decay value is the number of steps taken to reach epsilon end.
- **Design:** experiment 1 is the baseline. Experiments 2–10 each change one setting from it, so every comparison is against experiment 1.
- **Evaluation:** all rewards are **training-episode** rewards, collected while the agent is still acting epsilon-greedily. There is no separate greedy evaluation phase. `dqn_model.pth` holds the weights at the single highest-reward training episode, which is a noisy selection criterion.

## C. Hyperparameter explanations

- **Learning rate:** the optimizer step size. Too high can make Q-values unstable, too low makes learning slow.
- **Discount factor (gamma):** how much future reward counts. Higher values favour long-term payoff but make bootstrapped targets larger and noisier. Lower values make the agent short-sighted.
- **Batch size:** transitions per update. Larger batches give lower-variance gradients at higher compute per update.
- **Initial epsilon:** the starting exploration probability. Starting below 1.0 means the agent begins acting on its (untrained) network earlier.
- **Minimum epsilon:** the permanent exploration floor. A higher floor keeps more random actions in late play, which gains exploration but lowers the reward of the policy being logged.
- **Epsilon decay:** how quickly exploration is removed. Slower decay gives more exploration data but leaves less of the run for exploiting what was learned.

## D. Results

| Exp | LR | Gamma | Batch | Eps start | Eps end | Eps decay | Episodes | Avg reward (all) | Avg last 20 | Std last 20 | Avg last 50 | Max reward | Avg ep. length | Late loss | Wall time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.5e-05 | 0.98 | 32 | 1.0 | 0.02 | 60,000 | 192 | 28.96 | 45.35 | 37.24 | 41.90 | 183 | 517 | 0.0282 | 2948 |
| 2 | 0.00015 | 0.98 | 32 | 1.0 | 0.02 | 60,000 | 196 | 26.31 | 38.30 | 9.11 | 34.88 | 67 | 507 | 0.0372 | 1771 |
| 3 | 0.0003 | 0.98 | 32 | 1.0 | 0.02 | 60,000 | 176 | 29.72 | 30.40 | 7.78 | 33.64 | 345 | 567 | 0.0416 | 1652 |
| 4 | 7.5e-05 | 0.97 | 32 | 1.0 | 0.02 | 60,000 | 179 | 29.16 | 44.50 | 19.06 | 34.64 | 203 | 555 | 0.0273 | 1852 |
| 5 | 7.5e-05 | 0.995 | 32 | 1.0 | 0.02 | 60,000 | 194 | 26.60 | 33.95 | 14.92 | 33.48 | 168 | 513 | 0.0352 | 2268 |
| 6 | 7.5e-05 | 0.98 | 48 | 1.0 | 0.02 | 60,000 | 201 | 25.02 | 33.40 | 18.48 | 32.96 | 173 | 497 | 0.0304 | 3047 |
| 7 | 7.5e-05 | 0.98 | 96 | 1.0 | 0.02 | 60,000 | 184 | 32.50 | 61.00 | 36.19 | 56.88 | 340 | 541 | 0.0329 | 2896 |
| 8 | 7.5e-05 | 0.98 | 32 | 0.9 | 0.02 | 60,000 | 198 | 27.01 | 53.70 | 28.20 | 44.86 | 123 | 504 | 0.0295 | 1837 |
| 9 | 7.5e-05 | 0.98 | 32 | 1.0 | 0.03 | 60,000 | 163 | 26.10 | 27.70 | 13.80 | 25.26 | 175 | 486 | 0.0306 | 2566 |
| 10 | 7.5e-05 | 0.98 | 32 | 1.0 | 0.02 | 85,000 | 175 | 27.68 | 33.50 | 18.38 | 32.78 | 172 | 570 | 0.0289 | 41975* |

"Late loss" is the mean of the per-episode average loss for episodes after step 50,000. Loss values come from the logged `Avg_Loss`.

\* **Wall time is not a reliable training duration.** It is wall-clock time from the script. Experiment 1 ran while other work shared the CPU, and experiment 10's 41,975 s (about 11.7 hours) is far outside the 1,650–3,050 s of the others, so the machine was almost certainly asleep or idle during part of it. Durations must not be used to compare compute cost.

Progression within each run:

| Exp | Avg reward, first half (steps ≤ 50k) | Second half | Last logged step | Unlogged tail steps | Longest gap between episodes |
|---|---|---|---|---|---|
| 1 | 22.69 | 36.22 | 99,327 | 673 | 1,198 |
| 2 | 21.94 | 31.55 | 99,437 | 563 | 949 |
| 3 | 26.49 | 34.51 | 99,749 | 251 | 9,274 |
| 4 | 28.22 | 30.37 | 99,260 | 740 | 8,146 |
| 5 | 23.25 | 30.72 | 99,451 | 549 | 901 |
| 6 | 18.91 | 32.88 | 99,939 | 61 | 1,049 |
| 7 | 18.05 | 54.47 | 99,546 | 454 | 1,176 |
| 8 | 20.08 | 36.03 | 99,722 | 278 | 1,097 |
| 9 | 26.38 | 25.60 | 79,239 | 20,761 | 892 |
| 10 | 24.73 | 32.22 | 99,755 | 245 | 11,620 |

Averages over all ten experiments: last-20 reward 40.18 (median 36.12), whole-run reward 27.91, mean max reward 194.9, mean last-20 standard deviation 20.32.

## E. Observations

All ten runs show the same broad pattern: average reward in the second half of training is higher than in the first half in nine of ten runs (all except experiment 9, where the tail of the run is not logged, see below). The gains are modest and most runs finish in a narrow band. Because every experiment is a single seed with a large spread between episodes (last-20 standard deviations of 8–37), **many of the differences below are within noise.** The standard error of a last-20 mean is the standard deviation divided by √20, i.e. up to about 8 points for the noisiest runs (experiments 1 and 7), so differences of 10–15 points between those runs are not clearly distinguishable from noise.

**Learning rate (exps 1–3: 7.5e-5, 1.5e-4, 3e-4).** Last-20 reward fell as the rate rose: 45.35 → 38.30 → 30.40, and last-50 reward fell the same way (41.90 → 34.88 → 33.64). The two higher rates also had much lower last-20 variance (9.11 and 7.78 versus 37.24) and higher late loss (0.0372 and 0.0416 versus 0.0282). That pattern fits a higher rate making training less settled, but it also reflects the baseline's own wide spread. Experiment 3 recorded a max reward of 345, reached at step 35,891 when epsilon was still about 0.41, so that peak came from a lucky exploratory episode and says nothing about the learned policy. In Hamed's runs the lowest rate (5e-5) did worst, the opposite direction, so no consistent learning-rate trend holds across both sets.

**Discount factor (exps 4, 1, 5: 0.97, 0.98, 0.995).** 0.97 and 0.98 were close on last-20 reward (44.50 vs 45.35), and 0.995 was lower at 33.95. Last-50 reward is flat across the three (34.64, 41.90, 33.48 for 0.97, 0.98, 0.995), so the evidence for any gamma effect is weak. Experiment 4 had a long stretch with no completed episode (8,146 steps) that influenced its statistics.

**Batch size (exps 1, 6, 7: 32, 48, 96).** The result is non-monotonic: 45.35, 33.40, 61.00. Batch 96 gave the highest last-20 (61.00), last-50 (56.88), whole-run (32.50) and second-half (54.47) averages of all my experiments, and it did so on every one of those measures. Batch 48 was below the baseline, which argues against a smooth batch-size effect. The first-half average for batch 96 was the lowest of all runs (18.05) and then it improved sharply, which is consistent with a larger batch giving cleaner updates once learning starts, but one seed cannot establish that.

**Initial epsilon (exp 8: 0.9).** Last-20 reward was 53.70 against 45.35, but the gap shrinks on wider windows (last-50: 44.86 vs 41.90) and second-half averages are the same (36.03 vs 36.22). The max reward was lower (123 vs 183). Starting at 0.9 instead of 1.0 changes only the very beginning of the schedule, so a large effect is not expected, and this result is best read as no clear difference.

**Minimum epsilon (exp 9: 0.03).** This had the lowest last-20 (27.70) and last-50 (25.26) rewards. It is also the only run where the final 20,761 steps are missing from the CSV: the last completed episode ended at step 79,239, and no earlier gap between episodes exceeds 892 steps. The run did finish all 100,000 steps (its `config.json` exists), so the most plausible explanation is that the agent spent the last ~20,000 steps inside one episode that had not ended when training stopped, and `train.py` only writes a row when an episode ends. So experiment 9's "final performance" is measured at an earlier point of training than the others, and comparing it directly is unfair. It also means its poor reward cannot be confidently blamed on the higher floor.

**Epsilon decay (exp 10: 85,000 steps).** Last-20 reward was 33.50 against 45.35, and it ran with more exploration for longer (epsilon only reached its floor at step 85,000). Its logged second-half reward (32.22) was below the baseline's (36.22). Experiment 10 also contains an 11,620-step gap between episodes. The slower schedule was therefore not beneficial within a 100,000-step budget here, but with one seed this is suggestive only.

**Stability.** The high-variance runs were experiments 1 and 7 (last-20 standard deviations 37.24 and 36.19), which are also the highest-scoring ones, so in these runs a higher average came with wider swings. The lowest-variance runs (experiments 2 and 3, 9.11 and 7.78) were also among the lower-scoring ones. Late loss was similar everywhere (0.027–0.042) and did not separate the runs except for the high-learning-rate pair.

**Convergence.** No run shows clear convergence. Rewards are still noisy at 100,000 steps and the agents played for 160–200 episodes each, which is short for Atari DQN. The conclusions should be treated as early-training behaviour.

## F. Best configuration

**Criterion:** average Episode_Reward over the last 20 logged episodes (the same metric `aggregate_results.py` reports), cross-checked against the last-50 and whole-run averages.

**Experiment 7 is the best on this criterion:** learning rate 7.5e-5, gamma 0.98, **batch size 96**, epsilon 1.0 → 0.02 over 60,000 steps. Last-20 average 61.00, last-50 average 56.88, whole-run average 32.50, max reward 340. It ranks first on all three averages, so the ranking does not depend on the window chosen. Second place on last-20 was experiment 8 (53.70), and on last-50 also experiment 8 (44.86).

**Caveat:** experiment 7's last-20 standard deviation is 36.19, and with one seed the margin over experiment 1 (45.35) is not large relative to that spread. The conclusion is that experiment 7 scored highest in this set, not that batch 96 is certainly better.

## G. Limitations

- **One seed per configuration (seed 42).** The same seed also controls the initial network, so run-to-run variation is untested. Repeating each setting over several seeds is the most important next step.
- **Short budget:** 100,000 steps, with 10,000 spent on warm-up, and 160–200 episodes per run. Differences may change with longer training.
- **Training rewards only.** There is no greedy evaluation, so exploration noise is mixed into every number, and a run with a higher epsilon floor is penalised in the logged reward. A separate evaluation (epsilon near 0 over many episodes, using the saved checkpoint) would be a fairer measure.
- **The saved checkpoint** is chosen by the best single training episode, which favours lucky episodes (for example experiment 3's reward of 345 during heavy exploration).
- **Unlogged tails.** Rows are written only when an episode ends, so experiment 9 is missing its final 20,761 steps. Smaller tails (61–740 steps) exist in the other runs.
- **One factor at a time** means interactions between settings were not tested.
- **Wall-clock durations** are affected by other processes on the machine and by the machine idling, as described above.
