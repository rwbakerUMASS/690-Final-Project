import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    dtype = torch.float64
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1],dtype=dtype), act()]
    return nn.Sequential(*layers)

class Net(nn.Module):

    def __init__(self,sizes) -> None:
        super().__init__()
        self.layers = mlp(sizes)
    
    def forward(self, x):
        return self.layers(x)