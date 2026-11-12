# Hyperparameter experiment results — Andrew

Environment: `ALE/Pacman-v5`. Rewards are raw training episode rewards under epsilon-greedy actions, not greedy evaluation scores. Epsilon decay is measured in agent steps.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 7.5e-05 | 0.98 | 32 | 1.0 | 0.02 | 60000 | First 20 mean 16.7; last 20 mean 45.4 (rose); peak 183; 192 completed episodes. Last logged step 99327. |
| 2 | 0.00015 | 0.98 | 32 | 1.0 | 0.02 | 60000 | First 20 mean 15.4; last 20 mean 38.3 (rose); peak 67; 196 completed episodes. Last logged step 99437. |
| 3 | 0.0003 | 0.98 | 32 | 1.0 | 0.02 | 60000 | First 20 mean 17.2; last 20 mean 30.4 (rose); peak 345; 176 completed episodes. Last logged step 99749. |
| 4 | 7.5e-05 | 0.97 | 32 | 1.0 | 0.02 | 60000 | First 20 mean 16.9; last 20 mean 44.5 (rose); peak 203; 179 completed episodes. Last logged step 99260. |
| 5 | 7.5e-05 | 0.995 | 32 | 1.0 | 0.02 | 60000 | First 20 mean 16.2; last 20 mean 34.0 (rose); peak 168; 194 completed episodes. Last logged step 99451. |
| 6 | 7.5e-05 | 0.98 | 48 | 1.0 | 0.02 | 60000 | First 20 mean 14.6; last 20 mean 33.4 (rose); peak 173; 201 completed episodes. Last logged step 99939. |
| 7 | 7.5e-05 | 0.98 | 96 | 1.0 | 0.02 | 60000 | First 20 mean 15.1; last 20 mean 61.0 (rose); peak 340; 184 completed episodes. Last logged step 99546. |
| 8 | 7.5e-05 | 0.98 | 32 | 0.9 | 0.02 | 60000 | First 20 mean 15.6; last 20 mean 53.7 (rose); peak 123; 198 completed episodes. Last logged step 99722. |
| 9 | 7.5e-05 | 0.98 | 32 | 1.0 | 0.03 | 60000 | First 20 mean 14.8; last 20 mean 27.7 (rose); peak 175; 163 completed episodes. Last logged step 79239. |
| 10 | 7.5e-05 | 0.98 | 32 | 1.0 | 0.02 | 85000 | First 20 mean 27.3; last 20 mean 33.5 (rose); peak 172; 175 completed episodes. Last logged step 99755. |
