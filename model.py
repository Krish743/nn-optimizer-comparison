import torch
from torch import nn

class NN(nn.Module):
    def __init__(self, num_features = 784):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

    def forward(self, x):
        return self.model(x)
