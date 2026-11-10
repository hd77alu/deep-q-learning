import random
import collections
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Ensure states are stored as NumPy arrays to save memory
        state = np.array(state, dtype=np.uint8)
        next_state = np.array(next_state, dtype=np.uint8)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device: str | torch.device = "cpu"):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(dones, dtype=torch.float32, device=device)
        )

    def __len__(self):
        return len(self.buffer)