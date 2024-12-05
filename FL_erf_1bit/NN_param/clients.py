







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
        self.local_bs = args.local_bs
        self.device           = args.device
        self.id               = client_name
        self.datasize         = len(data)
        # args.local_bs         = int(self.datasize/2)
        # shuffle = True每次开始一个新的 epoch 并从 train_loader 中迭代数据时，train_loader 会自动将数据集中的数据打乱。这是一种常见的做法，用于确保模型接收到的数据顺序在每个 epoch 都是随机的，从而帮助模型更好地泛化。
        # 如果 shuffle 参数被设置为 False，则数据加载的顺序在每个 epoch 中保持不变。这种情况通常用于那些需要保持数据顺序的场合，比如时间序列数据处理。
        if args.dataset.lower() == 'mnist':
            self.trainloader      = DataLoader(data, batch_size = args.local_bs, shuffle = True)
        else:
            self.trainloader        = data
        self.data               = data
        self.model            = model
        self.num_local_update = args.local_up
        if args.optimizer == 'sgd':
            self.optimizer        = torch.optim.SGD(self.model.parameters(), lr = args.lr, momentum = 0.9 )
        elif args.optimizer == 'adam':
            self.optimizer        = torch.optim.Adam(self.model.parameters(), lr = args.lr)
        self.los_fn           = nn.CrossEntropyLoss()
        return

    ## 返回本地梯度,直接得到梯度
    def local_update_gradient(self, cur_weight, lr = 0.01):
        # self.trainloader      = DataLoader(self.data, batch_size = self.local_bs, shuffle = True)
        self.model.load_state_dict(cur_weight, strict = True)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for batch_idx, (data, label) in enumerate(self.trainloader):
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx >= 0:
                break
        message = {}
        for key, param in self.model.named_parameters():
            message[key] = copy.deepcopy(param.grad.data.detach())
        return message

    ## 返回本地梯度，模型相减除以学习率
    def local_update_gradient1(self, cur_weight, lr = 0.01):
        # self.trainloader      = DataLoader(self.data, batch_size = self.local_bs, shuffle = True)
        init_weight = copy.deepcopy(cur_weight)
        self.model.load_state_dict(cur_weight, strict = True)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for batch_idx, (data, label) in enumerate(self.trainloader):
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx >= 0:
                break
        message = {}
        copyw = copy.deepcopy(self.model.state_dict())
        for key in copyw.keys():
            message[key] = (init_weight[key] - copyw[key])/lr
        return message

    ## 返回更新前后的差值，本地多次mini-batch
    def local_update_diff_mini_batch(self, cur_weight, lr = 0.01, local_up = 1):
        # self.trainloader      = DataLoader(self.data, batch_size = self.local_bs, shuffle = True)
        init_weight = copy.deepcopy(cur_weight)
        # model_diff = copy.deepcopy(cur_weight)
        self.model.load_state_dict(cur_weight, strict = True)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for batch_idx, (data, label) in enumerate(self.trainloader):
            # print(f"   {self.id} = {batch_idx}")
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx >= local_up - 1:
                break
        message = {}
        copyw = copy.deepcopy(self.model.state_dict())
        for key in copyw.keys():
            message[key] = copyw[key]  - init_weight[key]
        return message

    ## 返回更新前后的差值，本地多次epoch，每次epoch遍历所有本地数据
    def local_update_diff_epoch(self, cur_weight, lr = 0.01, local_epoch = 3 ):
        # self.trainloader      = DataLoader(self.data, batch_size = self.local_bs, shuffle = True)
        init_weight = copy.deepcopy(cur_weight)
        # model_diff = copy.deepcopy(cur_weight)
        self.model.load_state_dict(cur_weight, strict = True)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for epoch in range(local_epoch):
            for batch_idx, (data, label) in enumerate(self.trainloader):
                data, label = data.to(self.device), label.to(self.device)
                preds = self.model(data)
                loss = self.los_fn(preds, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        message = {}
        copyw = copy.deepcopy(self.model.state_dict())
        for key in copyw.keys():
            message[key] = copyw[key] - init_weight[key]
        return message

def GenClientsGroup(args, local_dt_dict, model):
    ClientsGroup = {}
    for clientname, dataset in local_dt_dict.items():
        someone = Client(args, dataset, copy.deepcopy(model), clientname)
        ClientsGroup[clientname] = someone
    return ClientsGroup








































