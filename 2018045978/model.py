import torch.nn as nn


class Q_net(nn.Module):
    def __init__(self):
        super().__init__()
        
        ######################################################
        ## TODO: Implement Your Model
        self.layers = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        ######################################################
        
    def forward(self, x):
        Q_values = self.layers(x)
        return Q_values