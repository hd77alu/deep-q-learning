# Hyperparameter experiment results — Hamed

Environment: `ALE/Pacman-v5`. Rewards are raw training episode rewards under epsilon-greedy actions, not greedy evaluation scores. Epsilon decay is measured in agent steps.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0001 | 0.99 | 32 | 1 | 0.01 | 50000 | First 20 mean 16.8; last 20 mean 42.0 (rose); peak 193; 192 completed episodes. Last logged step 99691. |
| 2 | 0.00025 | 0.99 | 32 | 1 | 0.01 | 50000 | First 20 mean 15.7; last 20 mean 41.7 (rose); peak 174; 179 completed episodes. Last logged step 99734. |
| 3 | 5e-05 | 0.99 | 32 | 1 | 0.01 | 50000 | First 20 mean 15.6; last 20 mean 26.6 (rose); peak 85; 220 completed episodes. Last logged step 99549. |
| 4 | 0.0001 | 0.99 | 64 | 1 | 0.01 | 50000 | First 20 mean 14.9; last 20 mean 47.9 (rose); peak 344; 170 completed episodes. Last logged step 99502. |
| 5 | 0.0001 | 0.99 | 16 | 1 | 0.01 | 50000 | First 20 mean 16.0; last 20 mean 41.9 (rose); peak 345; 215 completed episodes. Last logged step 99797. |
| 6 | 0.0001 | 0.95 | 32 | 1 | 0.01 | 50000 | First 20 mean 18.0; last 20 mean 43.8 (rose); peak 172; 159 completed episodes. Last logged step 80277. |
| 7 | 0.0001 | 0.995 | 32 | 1 | 0.01 | 50000 | First 20 mean 21.1; last 20 mean 42.0 (rose); peak 179; 195 completed episodes. Last logged step 99358. |
| 8 | 0.0001 | 0.99 | 32 | 1 | 0.01 | 70000 | First 20 mean 15.0; last 20 mean 39.6 (rose); peak 203; 197 completed episodes. Last logged step 99783. |
| 9 | 0.0001 | 0.99 | 32 | 1 | 0.01 | 25000 | First 20 mean 19.4; last 20 mean 38.4 (rose); peak 82; 159 completed episodes. Last logged step 99958. |
| 10 | 0.0001 | 0.99 | 32 | 1 | 0.05 | 50000 | First 20 mean 18.2; last 20 mean 46.8 (rose); peak 207; 187 completed episodes. Last logged step 99765. |
