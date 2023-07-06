#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://www.jianshu.com/p/12a8207149b0

import torch
import skimage.color as sc
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np



#========================================================================
#  子类继承父类并重写父类的方法
#========================================================================
class A(object):
    def __init__(self, args, name='jack', train=True, benchmark=True):
         self.args = args
         self.name = name
         self.train = train 
         self.benchmark = benchmark
         self.print1()
    def print1(self):
         print(f"hello, {self.name}")



class Benchmark(A):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(args, name=name, train=train, benchmark=True)
    def print1(self):
         print(f"hello, Mr.Wang  {self.name}")


classA = A(args=222, name='junjie', train=True, benchmark=True)
classB = Benchmark(args=12, name='chen', train=True, benchmark=True)

# runfile('/home/jack/公共的/Python/库示例/Class.py', wdir='/home/jack/公共的/Python/库示例')
# hello, junjie
# hello, Mr.Wang









