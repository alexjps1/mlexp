"""
Model Definition and Learning Parameters

Based on model recommendations from:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
from torch import nn
import torchvision.transforms as transforms

# training data transform definition
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# model definition
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

MODEL = ConvolutionalNeuralNetwork()

# training parameters
LEARNING_RATE = 0.0001
SGD_MOMENTUM = 0.9
BATCH_SIZE = 4
EPOCHS = 10000

# save state dict every x epochs
CHECKPOINT = 200

# loss function and optimizer
LOSS_FUNCTION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)
