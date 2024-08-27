#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2024/08/19

@author: Junjie Chen
"""

import  copy
import torch
import  numpy as np



class Server(object):
    def __init__(self, args, net, test_dataloader):
        self.args = args
        self.global_model = net
        self.test_loader = test_dataloader
        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error-free %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def aggregate_gradient_erf(self, mess_lst, lr, ):

        return

    def aggregate_diff_erf(self, mess_lst,):

        return

    def aggregate_model_erf(self, mess_lst,  ):

        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% Rician Fading MAC %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def aggregate_gradient_rician(self, mess_lst, lr, SNR, H):

        return

    def aggregate_diff_rician(self, mess_lst, SNR, H):

        return

    def aggregate_model_rician(self, mess_lst, SNR, H):

        return


    def model_eval(self):
        self.global_model.eval()

        sum_accu    = 0.0
        sum_loss    = 0.0
        examples    = 0
        loss_fn     = torch.nn.CrossEntropyLoss(reduction='sum')
        # 载入测试集
        for X, y in self.eval_loader:
            examples    += X.shape[0]
            X, y        = X.to(self.device), y.to(self.device)
            preds       = self.global_model(X)
            sum_loss    += loss_fn(preds, y).item()
            sum_accu    += (torch.argmax(preds, dim=1) == y).float().sum().item()
        acc     = sum_accu / examples
        avg_los = sum_loss / examples

        return acc, avg_los























































