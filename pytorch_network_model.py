# Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Dataset (csv or pytorch)


#Create Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hiddenlayer1
        self.hl2
        ...
        self.hln
        

# Loss
criterion = nn.CrossEntropyLoss()
#some optim func using prefferably Adam

# Train

# Check accuracy

