"""
Sample Model Definition and Learning Parameters
"""

import torch
from torch import nn

# model definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

MODEL = NeuralNetwork()

# training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 500

# save state dict every x epochs
CHECKPOINT = 25

# loss function and optimizer
LOSS_FUNCTION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)
