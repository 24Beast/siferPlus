# Importing Libraries
import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, Flatten, Softmax, Sigmoid

# Constants
MID_SIZE = 16000

# Defining Modules
class PreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, 3)
        self.conv2 = Conv2d(32, 128, 3)
        self.conv3 = Conv2d(128, 32, 3)
        self.flatten = Flatten()
        self.pool = MaxPool2d(2,2)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(MID_SIZE, 1024)
        self.linear2 = Linear(1024, 128)
        self.linear3 = Linear(128, 64)
        self.linear4 = Linear(64, 16)
        self.linear5 = Linear(64, 4)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()
        

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.softmax(x)
        return x


class AuxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(MID_SIZE, 64)
        self.linear2 = Linear(64, 4)
        self.linear3 = Linear(64, 1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        label = self.linear2(x)
        label = self.softmax(label)
        target = self.linear3(x)
        target = self.sigmoid(target)
        return label, target
