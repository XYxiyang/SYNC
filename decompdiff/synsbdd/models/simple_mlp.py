import torch
import torch.nn as nn

class clsf(nn.Module):
    def __init__(self):
        super(clsf, self).__init__()
        self.fc1 = nn.Linear(13, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x): 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.sum(0) / x.shape[0]
        x = self.relu(x)
        x = self.fc3(x)
        return x