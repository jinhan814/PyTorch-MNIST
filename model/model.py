import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistModule(nn.Module):
    def __init__(self):
        super(MnistModule, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(7 * 7 * 64, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.layer3 = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
    
    def forward(self, x):
        ret = self.layer1(x)
        ret = self.layer2(ret)
        ret = ret.view(ret.size(0), -1)
        ret = self.layer3(ret)
        return ret
