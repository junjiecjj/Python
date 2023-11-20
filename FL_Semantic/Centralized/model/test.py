#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:11:04 2022

@author: jack
"""

from model  import common
import math
import torch
import torch.nn.functional as F
import torch.nn.parallel as P
from torch import nn, Tensor
from einops import rearrange
import copy

import sys,os,datetime

sys.path.append("/home/jack/公共的/Pretrained-IPT-cjj/")
sys.path.append("..")
from option import args



class checkpoint():
    def __init__(self, args ):
        self.args = args
        self.ok = True

        self.dir = args.save
        print(f"self.dir = {self.dir} \n")
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(args.save, 'model'), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('trainlog.txt')) else 'w'
        self.log_file = open(self.get_path('trainlog.txt'), open_type)

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        with open(self.get_path('config.txt'), open_type) as f:
            f.write('#==========================================================\n')
            f.write(now + '\n')
            f.write('#==========================================================\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')


        self.psnrlog = {}
        for comprateTmp in args.CompressRateTrain:
            for snrTmp in args.SNRtrain:
                self.psnrlog["psnrlog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)] = torch.Tensor()



        if os.path.isfile(self.get_path('psnr_log.pt')):
            pass


        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''


        #  print(f"self.dir = {self.dir}")   # /home/jack/IPT-Pretrain/ipt/

        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)



ckp = checkpoint(args)