"""Compare saved checkpoints with identical greedy evaluation seeds."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from model import DQN
from wrappers import make_atari_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob', default='*/exp_*/dqn_model.pth')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('evaluation_results.csv'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_atari_env()
    model = DQN(in_channels=4, num_actions=env.action_space.n).to(device)
    rows = []
    for path in sorted(Path('.').glob(args.glob)):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        rewards = []
        for i in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + i)
            score = 0.0
            while True:
                state = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
                with torch.no_grad():
                    action = int(model(state).argmax(dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                score += reward
                if terminated or truncated:
                    break
            rewards.append(score)
        row = {'model': str(path), 'mean_reward': round(float(np.mean(rewards)), 2), 'std_reward': round(float(np.std(rewards)), 2), 'episodes': args.episodes, 'seed_start': args.seed}
        rows.append(row)
        print(row, flush=True)
    env.close()
    if not rows:
        parser.error('No checkpoints matched --glob')
    with args.output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('Best:', max(rows, key=lambda row: row['mean_reward'])['model'])


if __name__ == '__main__':
    main()
