# Hyperparameter experiment results — Mahamat Bakhit

Environment: `ALE/Pacman-v5`. Rewards are raw training episode rewards under epsilon-greedy actions, not greedy evaluation scores. Epsilon decay is measured in agent steps.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 50000.0 | First 20 mean 16.9; last 20 mean 49.4 (rose); peak 138; 177 completed episodes. Last logged step 99982. |
| 2 | 5e-05 | 0.99 | 64 | 1.0 | 0.05 | 50000.0 | First 20 mean 16.9; last 20 mean 39.1 (rose); peak 114; 141 completed episodes. Last logged step 99874. |
| 3 | 0.0002 | 0.99 | 64 | 1.0 | 0.05 | 50000.0 | First 20 mean 16.9; last 20 mean 61.5 (rose); peak 214; 184 completed episodes. Last logged step 99652. |
| 4 | 0.0001 | 0.95 | 64 | 1.0 | 0.05 | 50000.0 | First 20 mean 16.9; last 20 mean 46.0 (rose); peak 109; 184 completed episodes. Last logged step 99612. |
| 5 | 0.0001 | 0.995 | 64 | 1.0 | 0.05 | 50000.0 | First 20 mean 16.9; last 20 mean 76.0 (rose); peak 370; 176 completed episodes. Last logged step 99835. |
| 6 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50000.0 | First 20 mean 16.9; last 20 mean 57.5 (rose); peak 194; 172 completed episodes. Last logged step 90227. |
| 7 | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 50000.0 | First 20 mean 16.9; last 20 mean 55.5 (rose); peak 198; 172 completed episodes. Last logged step 99736. |
| 8 | 0.0001 | 0.99 | 64 | 0.8 | 0.05 | 50000.0 | First 20 mean 24.6; last 20 mean 38.1 (rose); peak 201; 173 completed episodes. Last logged step 99679. |
| 9 | 0.0001 | 0.99 | 64 | 1.0 | 0.02 | 50000.0 | First 20 mean 13.8; last 20 mean 74.9 (rose); peak 349; 172 completed episodes. Last logged step 99884. |
| 10 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 25000.0 | First 20 mean 26.6; last 20 mean 44.0 (rose); peak 170; 173 completed episodes. Last logged step 99581. |
