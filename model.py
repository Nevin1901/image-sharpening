import torch
import torch.nn as nn


class Sharp(nn.Module):
    def __init__(self):
        super(Sharp, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 218),
                nn.ReLU(),
                nn.Linear(218, 48),
                )
        self.decoder = nn.Sequential(
                nn.Linear(48, 218),
                nn.ReLU(),
                nn.Linear(218, 512),
                nn.ReLU(),
                nn.Linear(512, 28*28),
                nn.Sigmoid()
                )

        def forward(self, x):
            x = self.flatten(x)
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded



