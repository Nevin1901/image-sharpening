import torch
import torch.nn as nn
import torchvision


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()
    
    def forward(self, x):
        print("got here")
        return x

class Sharp(nn.Module):
    def __init__(self):
        super(Sharp, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 36),
                nn.ReLU(),
                nn.Linear(36, 18),
                nn.ReLU(),
                nn.Linear(18, 9),
                )
        self.decoder = nn.Sequential(
                nn.Linear(9, 18),
                nn.ReLU(),
                nn.Linear(18, 36),
                nn.ReLU(),
                nn.Linear(36, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 28*28),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.flatten(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



