#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:08:37 2022

@author: jack
"""

# from utility import  checkpoint
from option  import args
from loss import Loss
import utility
from Branch.m3 import DCGAN

class Trainer():
    def __init__(self, args,   my_loss, ckp):
        self.args = args
        self.scale = args.scale
        # print(f"trainer  self.scale = {self.scale} \n")
        self.ckp = ckp
        self.loss = my_loss
        
        
    def test(self):
        self.ckp.save(self,  100)
        self.ckp.load(222222)   
        

ckp = utility.checkpoint(args)
_loss = Loss(args, ckp)


tr = Trainer(args,  _loss, ckp)
tr.test()

gan =  DCGAN()