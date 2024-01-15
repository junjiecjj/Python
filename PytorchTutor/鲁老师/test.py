#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:59:51 2023

@author: jack
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator
import os, sys
import numpy as np
import torch
from torch import nn
import random



# 初始化随机数种子
def set_random_seed(seed = 10, deterministic = False, benchmark = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
    return

set_random_seed()


class MLP(nn.Module):
    # coders
    def __init__(self, input_dim = 10, channel = 100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, channel)
        self.fc2 = nn.Linear(channel, input_dim)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # encoder
        with torch.no_grad():
            x = self.fc1(x)

        ## zhu shi
        # x = x.detach().cpu()


        ## decoder
        x = self.fc2(x)
        return x


input_dim = 10
channel = 2

mlp = MLP(input_dim = input_dim, channel = channel)

print(f"0:mlp.fc1.weight = \n  {mlp.fc1.weight}\nmlp.fc2.weight = \n  {mlp.fc2.weight}")

Loss = nn.MSELoss()

optimizer = torch.optim.SGD(mlp.parameters(), 1e-3)

epoch_sum = 1000
for epoch in range(epoch_sum):
    x = torch.randn(size = (128, input_dim))
    x_hat = mlp(x)

    loss = Loss(x_hat, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch == epoch_sum - 1:
        print(f"{epoch}:mlp.fc1.weight = \n  {mlp.fc1.weight}\nmlp.fc2.weight = \n  {mlp.fc2.weight}")
























































































































































