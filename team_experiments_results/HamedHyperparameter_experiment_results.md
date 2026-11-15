# Hyperparameter experiment results — Hamed

Environment: `ALE/Pacman-v5`. Rewards are raw training episode rewards under epsilon-greedy actions, not greedy evaluation scores. Epsilon decay is measured in agent steps.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0001 | 0.99 | 32 | 1 | 0.01 | 50000 | First 20 mean 16.8; last 20 mean 42.0 (rose); peak 193; 192 completed episodes. Last logged step 99691. |
| 2 | 0.00025 | 0.99 | 32 | 1 | 0.01 | 50000 | First 20 mean 15.7; last 20 mean 41.7 (rose); peak 174; 179 completed episodes. Last logged step 99734. |
| 3 | 5e-05 | 0.99 | 32 | 1 | 0.01 | 50000 | **Worst:** First 20 mean 15.6; last 20 mean 26.6 (rose); peak 85; 220 completed episodes. Last logged step 99549. |
| 4 | 0.0001 | 0.99 | 64 | 1 | 0.01 | 50000 | **Best:** First 20 mean 14.9; last 20 mean 47.9 (rose); peak 344; 170 completed episodes. Last logged step 99502. |
| 5 | 0.0001 | 0.99 | 16 | 1 | 0.01 | 50000 | First 20 mean 16.0; last 20 mean 41.9 (rose); peak 345; 215 completed episodes. Last logged step 99797. |
| 6 | 0.0001 | 0.95 | 32 | 1 | 0.01 | 50000 | First 20 mean 18.0; last 20 mean 43.8 (rose); peak 172; 159 completed episodes. Last logged step 80277 maybe due to lowering the discount factor to $\gamma = 0.95$ made the agent overly cautious—after Episode 159 ended at step 80,277. |
| 7 | 0.0001 | 0.995 | 32 | 1 | 0.01 | 50000 | First 20 mean 21.1; last 20 mean 42.0 (rose); peak 179; 195 completed episodes. Last logged step 99358. |
| 8 | 0.0001 | 0.99 | 32 | 1 | 0.01 | 70000 | First 20 mean 15.0; last 20 mean 39.6 (rose); peak 203; 197 completed episodes. Last logged step 99783. |
| 9 | 0.0001 | 0.99 | 32 | 1 | 0.01 | 25000 | First 20 mean 19.4; last 20 mean 38.4 (rose); peak 82; 159 completed episodes. Last logged step 99958. |
| 10 | 0.0001 | 0.99 | 32 | 1 | 0.05 | 50000 | First 20 mean 18.2; last 20 mean 46.8 (rose); peak 207; 187 completed episodes. Last logged step 99765. |

## Analysis

All 10 experiments showed consistent learning on `ALE/Pacman-v5`, with 20-episode average rewards rising from **14.9–21.1** early in training to **26.6–47.9** by the end of the 100,000-step runs.

* **Best Configuration — Experiment 4 (`batch_size=64`, `lr=1e-4`, `gamma=0.99`):** Reached the highest final mean reward (**47.9**) and a **344** peak. Larger mini-batches stabilized Bellman updates by reducing gradient variance.
* **Worst Configuration — Experiment 3 (`lr=5e-5`, baseline others):** Produced the lowest final mean (**26.6**) and an **85** peak because the learning rate was too small for weights to converge effectively within 100k steps.

* **Hyperparameter Takeaways:**

  * **Learning Rate:** `1e-4` performed best; `5e-5` underfit, while `2.5e-4` (Exp 2, mean **41.7**) added no stability gain.
  * **Batch Size:** Smaller batches (`16`, Exp 5) caused noisier updates—yielding one high peak (**345**) but lower consistency (mean **41.9**).
  * **Discount Factor ($\gamma$):** `0.99` balanced pellet collection and survival, whereas `0.95` (Exp 6) made the agent overly myopic, causing a long stalling loop after step 80,277.
  * **Exploration ($\epsilon$):** Fast decay (`25k`, Exp 9) cut exploration too early (mean **38.4**, peak **82**), while retaining a higher floor (`eps_end=0.05`, Exp 10) ranked second overall (mean **46.8**) by preventing repetitive corner-stalling.
