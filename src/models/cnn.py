import torch
from torch import nn

class CNN1D(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        # x: (B,T,F) -> (B,F,T)
        return self.net(x.transpose(1,2))
