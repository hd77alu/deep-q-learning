import argparse
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from model import DQN
from buffer import ReplayBuffer
from wrappers import make_atari_env  # Grayscale (84x84), FrameStack (4)

def parse_args():
    parser = argparse.ArgumentParser(description="DQN Pac-Man Training")
    # Experiment metadata
    parser.add_argument("--member-name", type=str, required=True, help="Member identifier (e.g., alice)")
    parser.add_argument("--exp-id", type=int, required=True, help="Experiment number (1-10)")
    parser.add_argument("--save-dir", type=str, default=".", help="Base output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--eps-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--eps-decay", type=float, default=50000, help="Epsilon decay steps")
    
    # Execution params
    parser.add_argument("--total-steps", type=int, default=150000, help="Total training steps")
    parser.add_argument("--target-update-freq", type=int, default=1000, help="Target net update step frequency")
    parser.add_argument("--buffer-capacity", type=int, default=100000, help="Replay buffer max capacity")
    parser.add_argument("--learning-starts", type=int, default=10000, help="Steps before starting optimization")
    parser.add_argument("--train-freq", type=int, default=4, help="Steps between optimization passes")
    return parser.parse_args()

def get_epsilon(current_step, eps_start, eps_end, eps_decay):
    """5. Epsilon Decay Strategy (Linear Decay)"""
    fraction = min(1.0, current_step / eps_decay)
    return eps_start + fraction * (eps_end - eps_start)

def main():
    args = parse_args()
    
    # Create isolated experiment output folder
    exp_dir = os.path.join(args.save_dir, args.member_name, f"exp_{args.exp_id:02d}")
    os.makedirs(exp_dir, exist_ok=True)
    model_save_path = os.path.join(exp_dir, "dqn_model.pth")
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize Pacman Environment
    env = make_atari_env("ALE/Pacman-v5")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("Expected a discrete action space")
    action_dim = env.action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Networks & Optimizer
    online_net = DQN(in_channels=4, num_actions=action_dim).to(device)
    target_net = DQN(in_channels=4, num_actions=action_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()  # Huber Loss
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)
    
    # Metrics tracking CSV setup
    csv_path = os.path.join(exp_dir, "metrics.csv")
    csv_file = open(csv_path, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Episode", "Total_Steps", "Episode_Length", "Episode_Reward", "Epsilon", "Avg_Loss"])

    # Loop variables initialization
    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    recent_losses = []
    best_reward = -float("inf")

    for total_steps in range(1, args.total_steps + 1):
        # Epsilon-greedy action selection
        epsilon = get_epsilon(total_steps, args.eps_start, args.eps_end, args.eps_decay)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array(obs), dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            with torch.no_grad():
                q_values = online_net(state_tensor)
                action = q_values.argmax(dim=1).item()

        # Interact with environment & collect/store experiences
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_value = float(reward)
        
        # Reward clipping (-1, 0, 1) for stability
        clipped_reward = float(np.sign(reward_value))
        replay_buffer.push(obs, action, clipped_reward, next_obs, float(terminated))
        
        obs = next_obs
        episode_reward += reward_value
        episode_length += 1

        # Reset environment upon episode termination
        if done:
            episode_count += 1
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            writer.writerow([episode_count, total_steps, episode_length, episode_reward, f"{epsilon:.4f}", f"{avg_loss:.4f}"])
            csv_file.flush()
            
            # Save best performing weights
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(online_net.state_dict(), model_save_path)
                
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            recent_losses = []

        # Sample minibatch & perform optimization update
        if (
            total_steps > args.learning_starts
            and len(replay_buffer) >= args.batch_size
            and total_steps % args.train_freq == 0
        ):
            states, actions, rewards, next_states, dones = replay_buffer.sample(args.batch_size, device)

            # Current Q-values: Q(s, a)
            q_values = online_net(states)
            state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q-values using Bellman Equation: r + gamma * max_a Q_target(s', a)
            with torch.no_grad():
                next_q_values = target_net(next_states).max(dim=1)[0]
                expected_q_values = rewards + (args.gamma * next_q_values * (1.0 - dones))

            # Compute Loss & Optimize
            loss = loss_fn(state_action_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)  # Gradient clipping
            optimizer.step()
            
            recent_losses.append(loss.item())

        # Periodically update target network weights
        if total_steps % args.target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

    # Save final model weights if no checkpoints were saved
    if not os.path.exists(model_save_path):
        torch.save(online_net.state_dict(), model_save_path)
        
    csv_file.close()
    env.close()

if __name__ == "__main__":
    main()