"""Greedy Pac-Man playback, with optional video recording."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model import DQN
from wrappers import make_atari_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--video-dir', type=Path)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    mode = 'rgb_array' if args.video_dir or args.headless else 'human'
    env = make_atari_env(render_mode=mode)
    if args.video_dir:
        from gymnasium.wrappers import RecordVideo
        args.video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, video_folder=str(args.video_dir), episode_trigger=lambda _: True, disable_logger=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DQN(in_channels=4, num_actions=env.action_space.n).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        reward_sum = 0.0
        steps = 0
        done = False
        while not done:
            state = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(model(state).argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            steps += 1
            done = terminated or truncated
        print(f'Episode {episode + 1}: reward={reward_sum:.1f}, steps={steps}', flush=True)
    env.close()


if __name__ == '__main__':
    main()
