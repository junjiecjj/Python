



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2024/08/19

@author: Junjie Chen
"""

import copy
import torch
import numpy as np

def model_stastic(param_W):
    params_float = torch.Tensor()
    for key, val in param_W.items():
        params_float = torch.cat((params_float, val.detach().cpu().flatten()))
    # std = params_float.std().item()
    var = params_float.var().item()
    # mean = params_float.mean().item()
    return var

def model_stastic_np(param_W):
    params_float = np.empty((0, 0), dtype = np.float32)
    for key, val in param_W.items():
        params_float = np.append(params_float, np.array(val.detach().cpu().clone()))
    std = np.std(params_float)
    var = np.var(params_float)
    mean = np.mean(params_float)
    return std, var, mean

#%% Base station
class Server(object):
    def __init__(self, args, net, global_weight, test_dataloader):
        self.args = args
        self.global_model  = net
        self.test_loader   = test_dataloader
        self.global_weight = global_weight
        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error-free %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def aggregate_signSGD(self, mess_lst, lr, ):
        w_avg = copy.deepcopy(mess_lst[0])
        for key in w_avg.keys():
            for i in range(1, len(mess_lst)):
                w_avg[key] += mess_lst[i][key]
        for param in w_avg.keys():
            self.global_weight[param] -= (lr * torch.sign(w_avg[param])).type(self.global_weight[param].dtype)
        return

    def aggregate_gradient_erf(self, mess_lst, lr, ):
        w_avg = copy.deepcopy(mess_lst[0])
        for key in w_avg.keys():
            for i in range(1, len(mess_lst)):
                w_avg[key] += mess_lst[i][key]
            w_avg[key] = torch.div(w_avg[key], len(mess_lst))

        for param in w_avg.keys():
            self.global_weight[param] -= (lr*w_avg[param]).type(self.global_weight[param].dtype)
        return

    def aggregate_diff_erf(self, mess_lst, ):
        w_avg = copy.deepcopy(mess_lst[0])
        for key in w_avg.keys():
            for i in range(1, len(mess_lst)):
                w_avg[key] += mess_lst[i][key]
            w_avg[key] = torch.div(w_avg[key], len(mess_lst))

        for param in self.global_weight.keys():
            self.global_weight[param] += w_avg[param].type(self.global_weight[param].dtype)
        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% Rician Fading MAC %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def aggregate_gradient_rician(self, mess_lst, lr, noise_var, P, H, device):
        h_sigma = [P * np.abs(H[i])**2/model_stastic(copy.deepcopy(mess)) for i, mess in enumerate(mess_lst)]
        eta = min(h_sigma)
        w_avg = copy.deepcopy(mess_lst[0])
        for key in w_avg.keys():
            for i in range(1, len(mess_lst)):
                w_avg[key] += mess_lst[i][key]
            w_avg[key] = torch.div(w_avg[key], len(mess_lst))
        # AWGN noise  full power transmit
        if noise_var > 0:
            for key, val in w_avg.items():
                val += torch.normal(torch.zeros_like(val), np.sqrt(noise_var/eta/len(mess_lst))).to(device);
        for param in self.global_weight:
            self.global_weight[param] -= lr * w_avg[param]
        return

    def aggregate_diff_rician(self, mess_lst, SNR, noise_var, P, H, device):
        h_sigma = [P * np.abs(H[i])**2/model_stastic(copy.deepcopy(mess)) for i, mess in enumerate(mess_lst)]
        eta = min(h_sigma)
        w_avg = copy.deepcopy(mess_lst[0])
        for key in w_avg.keys():
            for i in range(1, len(mess_lst)):
                w_avg[key] += mess_lst[i][key]
            w_avg[key] = torch.div(w_avg[key], len(mess_lst))
        # AWGN noise  full power transmit
        if noise_var > 0:
            for key, val in w_avg.items():
                val += torch.normal(torch.zeros_like(val), np.sqrt(noise_var/eta/len(mess_lst))).to(device);
        for param in self.global_weight:
            self.global_weight[param] += w_avg[param].type(self.global_weight[param].dtype)
        return


    #%% validate on test dataset
    def model_eval(self, device):
        self.global_model.load_state_dict(self.global_weight, strict=True)
        self.global_model.eval()

        sum_accu    = 0.0
        sum_loss    = 0.0
        examples    = 0
        # num_bt      = 0
        loss_fn     = torch.nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            # 载入测试集
            for X, y in self.test_loader:
                # num_bt += 1
                examples    += X.shape[0]
                X, y        = X.to(device), y.to(device)
                preds       = self.global_model(X)
                sum_loss    += loss_fn(preds, y).item()
                # _, pred_labels = torch.max(preds, 1)
                # sum_accu   += (pred_labels == y).float().sum().item()
                sum_accu    += (torch.argmax(preds, dim = 1) == y).float().sum().item()
        acc     = sum_accu / examples
        avg_los = sum_loss / examples

        return acc, avg_los































