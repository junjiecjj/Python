#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:08:14 2022

@author: jack
"""




import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


# 功能：
#
class checkpoint():
    def __init__(self, args, istrain = False):
        self.args = args

    def save(self, trainer, epoch, is_best=False):
        print(f"In checkpoint epoch = {epoch}\n")
        trainer.loss.Print(f"I am in checkpoint call for trainer.loss.print \n")

    def load(self, a):
        print(f"In checkpoint a = {a}\n")