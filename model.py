import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 64 channels * 7 * 7 spatial output from the conv layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # Scale pixel values from [0, 255] to [0, 1] if passed as uint8 tensors
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
            
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)