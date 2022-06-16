#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://www.jianshu.com/p/12a8207149b0


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import AlexNet

num_epochs = 100

#定义2分类网络
model = AlexNet(num_classes=2)
optimizer = optim.SGD(params=model.parameters(), lr=0.05)

scheduler = MultiStepLR(
    optimizer=optimizer,
    milestones=[10, 20, 40],  # 设定调整的间隔数
    gamma=0.5,  # 系数
    last_epoch=-1 
)

# train-like iteration
lrs, epochs = [], []
for epoch in range(num_epochs):
    lrs.append(scheduler.get_lr())  #.get_lr()获取当前学习率
    epochs.append(epoch)

    pass  # 在这里进行迭代训练
    #学习率更新
    scheduler.step()

# visualize
plt.figure()
plt.legend()
plt.plot(epochs, lrs, label='MultiStepLR')
plt.show()





















   
     
   
     
   
     
   
     