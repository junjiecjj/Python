




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
        self.args             = args
        self.mu               = args.mu
        self.device           = args.device
        self.id               = client_name
        self.datasize         = len(data)
        # args.local_bs         = int(self.datasize/2)
        self.trainloader      = data  # DataLoader(data, batch_size = args.local_bs, shuffle = True)
        self.model            = model
        self.num_local_update = args.local_up
        if args.optimizer == 'sgd':
            self.optimizer    = torch.optim.SGD(self.model.parameters(), lr = args.lr, momentum = 0.9,) #
        elif args.optimizer == 'adam':
            self.optimizer    = torch.optim.Adam(self.model.parameters(), lr = args.lr)
        self.los_fn           = nn.CrossEntropyLoss()
        return

    ## 返回本地梯度,直接得到梯度
    def local_update_gradient(self, cur_weight, lr = 0.01):
        self.model.load_state_dict(cur_weight, strict = True)
        global_model = copy.deepcopy(self.model)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for batch_idx, (data, label) in enumerate(self.trainloader):
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)
            # fedprox, add proximal term
            if not self.args.IID: # self.fedprox:
                proximal_term = torch.tensor(0., device = self.device)
                for w, w_global in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += torch.pow(torch.norm(w - w_global, 2), 2)
                loss += (self.mu / 2 * proximal_term)
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
        init_weight = copy.deepcopy(cur_weight)
        self.model.load_state_dict(cur_weight, strict = True)
        # global_model = copy.deepcopy(self.model)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for batch_idx, (data, label) in enumerate(self.trainloader):
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)
            # # fedprox, add proximal term
            # if not self.args.IID: # self.fedprox:
            #     proximal_term = torch.tensor(0., device = self.device)
            #     for w, w_global in zip(self.model.parameters(), global_model.parameters()):
            #         proximal_term += torch.pow(torch.norm(w - w_global, 2), 2)
            #     loss += (self.mu / 2 * proximal_term)

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
    def local_update_diff(self, cur_weight, lr = 0.01, local_up = 1):
        init_weight = copy.deepcopy(cur_weight)

        self.model.load_state_dict(cur_weight, strict = True)
        global_model = copy.deepcopy(self.model)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for batch_idx, (data, label) in enumerate(self.trainloader):
            data, label = data.to(self.device), label.to(self.device)
            preds = self.model(data)
            loss = self.los_fn(preds, label)

            # fedprox, add proximal term
            if not self.args.IID: # self.fedprox:
                proximal_term = torch.tensor(0., device = self.device)
                for w, w_global in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += torch.norm(w - w_global, 2)
                loss += (self.mu / 2 * proximal_term)

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
    def local_update_diff1(self, cur_weight, lr = 0.01, local_epoch = 3 ):
        init_weight = copy.deepcopy(cur_weight)

        self.model.load_state_dict(cur_weight, strict = True)
        global_model = copy.deepcopy(self.model)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for epoch in range(local_epoch):
            for batch_idx, (data, label) in enumerate(self.trainloader):
                data, label = data.to(self.device), label.to(self.device)
                preds = self.model(data)
                loss = self.los_fn(preds, label)
                # fedprox, add proximal term
                if not self.args.IID: # self.fedprox:
                    proximal_term = torch.tensor(0., device = self.device)
                    for w, w_global in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += torch.pow(torch.norm(w - w_global, 2), 2)
                    loss += (self.mu / 2 * proximal_term)

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








































