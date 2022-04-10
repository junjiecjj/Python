#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:14:02 2022

@author: jack
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os
class LightningMNISTClassifier(pl.LightningModule):
  def __init__(self):
    super(LightningMNISTClassifier, self).__init__()
    # MNIST 图片 (1, 28, 28) (channels, width, height) 
    self.layer_1 = torch.nn.Linear(28 * 28, 128)
    self.layer_2 = torch.nn.Linear(128, 256)
    self.layer_3 = torch.nn.Linear(256, 10)
  def forward(self, x):
      batch_size, channels, width, height = x.size()
      # (b, 1, 28, 28) -> (b, 1*28*28)
      x = x.view(batch_size, -1)
      # 第1层 (b, 1*28*28) -> (b, 128)
      x = self.layer_1(x)
      x = torch.relu(x)
      # 第2层 (b, 128) -> (b, 256)
      x = self.layer_2(x)
      x = torch.relu(x)
      # 第3层 (b, 256) -> (b, 10)
      x = self.layer_3(x)
      # 标签的概率分布
      x = torch.log_softmax(x, dim=1)
      return x
  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)
  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      logs = {'train_loss': loss}
      return {'loss': loss, 'log': logs}
  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      return {'val_loss': loss}
  def validation_epoch_end(self, outputs):
      # 在验证结束时调用
      # 输出是一个数组，包含在每个batch在验证步骤中返回的结果
      # 输出 = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
      avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      tensorboard_logs = {'val_loss': avg_loss}
      return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
  def prepare_data(self):
    # 图像变换对象
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,), (0.3081,))])
      
    # 对MNIST进行变换
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    
    self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
    
  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=64)
  def val_dataloader(self):
    return DataLoader(self.mnist_val, batch_size=64)
  def test_dataloader(self):
    return DataLoader(self,mnist_test, batch_size=64)
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
# 训练
model = LightningMNISTClassifier()
trainer = pl.Trainer()
trainer.fit(model)
