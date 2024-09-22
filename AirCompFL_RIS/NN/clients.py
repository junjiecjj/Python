







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023/08/19
@author: Junjie Chen
"""




import numpy as np
import copy
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class Client(object):
    def __init__(self, args, data, model, client_name = "clientxx",):
        # self.args             = args
        self.device           = args.device
        self.id               = client_name
        self.trainloader      = DataLoader(data, batch_size = args.local_bs, shuffle = True)
        self.datasize         = len(data)
        self.model            = model
        self.num_local_update = args.local_up
        self.optimizer        = optim.SGD(params = self.model.parameters(), lr = args.lr)
        self.los_fn           = nn.CrossEntropyLoss()
        return

    ## 返回本地梯度
    def local_update_gradient(self, cur_weight, lr = 0.01):
        self.model.load_state_dict(cur_weight, strict=True)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        message = {}
        for batch_idx, (data, label) in enumerate(self.trainloader):
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx >= 0: break
        for key, param in self.model.named_parameters():
            message[key] = param.grad.data.detach()
        return message

    ## 返回更新前后的差值
    def local_update_diff(self, cur_weight, lr = 0.01, ):
        model_diff = copy.deepcopy(cur_weight)
        self.model.load_state_dict(cur_weight, strict=True)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()
        for batch_idx, (data, label) in enumerate(self.trainloader):
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx >= self.num_local_update - 1: break
        for param in self.model.state_dict():
            model_diff[param] = self.model.state_dict()[param] - model_diff[param]
        return model_diff

    ## 返回更新后的模型
    def local_update_model(self, cur_weight, lr = 0.01, ):
        # model_diff = copy.deepcopy(cur_weight)
        self.model.load_state_dict(cur_weight, strict=True)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()
        for batch_idx, (data, label) in enumerate(self.trainloader):
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx >= self.num_local_update - 1: break
        return copy.deepcopy(self.model.state_dict())

def GenClientsGroup(args, local_dt_dict, model):
    ClientsGroup = {}
    for clientname, dataset in local_dt_dict.items():
        someone = Client(args, dataset, copy.deepcopy(model), clientname)
        ClientsGroup[clientname] = someone
    return ClientsGroup








































