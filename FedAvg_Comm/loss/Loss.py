
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""
# 系统库
import sys,os

import numpy as np
import torch
import torch.nn as nn

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')



class myLoss(nn.modules.loss._Loss):
    def __init__(self, args, ):
        super(myLoss, self).__init__()
        # print('Preparing loss function:')
        self.cn = self.__class__.__name__
        self.samples = 0

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):  #  ['1*MSE']
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = torch.nn.MSELoss(reduction='sum')
            elif loss_type == 'L1':
                loss_function = torch.nn.L1Loss(reduction='sum')
            elif loss_type == 'BCE':
                loss_function = torch.nn.BCELoss(reduction='sum')  # reduction='sum'
            elif loss_type == 'CrossEntropy':
                    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_function} )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                # print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.losslog = torch.Tensor()

        self.loss_module.to(args.device)

        if args.precision == 'half':
            self.loss_module.half()
        return

    def addlog(self):
        #  losslog.shape = [1,len(loss)],[2,len(loss)],[2,len(loss)]...,[epoch,len(loss)]
        self.losslog = torch.cat((self.losslog, torch.zeros(1, len(self.loss))))
        self.samples = 0
        return

    #@profile
    def forward(self, sr, hr ):
        # print(f"{sr.shape}   {hr.shape}")
        self.samples += sr.size(0)
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.losslog[-1, i] += effective_loss.item()  #  *sr.shape[0]  # tensor.item()  获取tensor的数值
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.losslog[-1, -1] += loss_sum.item()
        return loss_sum

    def __getitem__(self, idx):
        return self.losslog[-1, idx]

    def reset(self):
        self.losslog = torch.Tensor()
        self.samples = 0
        return

    def avg(self, ):
        self.losslog[-1].div_( self.samples)
        return self.losslog[-1].item()

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.losslog[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c/n_samples))

        return ' '.join(log) # '[MSE: 1.0000] [L1: 2.0000] [BCE: 4.0000] [Total: 6.0000]'

    # 在每个压缩率和信噪比下，所有的epoch训练完再调用保存
    def save(self, apath, compr = '', tra_snr = 'random'):
        if compr != '' :
            basename = f"{self.cn}_compr={compr:.1f}_trainSnr={tra_snr}(dB)"
        else:
            basename = f"{self.cn}"
        torch.save(self.losslog, os.path.join(apath, f"{basename}.pt") )
        self.plot_AllLoss(apath, compr = compr, tra_snr = tra_snr)
        return



