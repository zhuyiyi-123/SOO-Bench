import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import copy
import numpy as np


class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim=2048,
                 out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
