#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 21:04:38 2025

@author: jack
"""


# import numpy as np
import copy
import torch
import torch.nn as nn


class Client(object):
    def __init__(self, args, data, model, client_name = "clientxx",):
        self.args             = args
        self.mu               = args.mu
        self.device           = args.device
        self.id               = client_name
        self.datasize         = data.dataset.__len__()
        self.trainloader      = data
        self.model            = model
        if args.optimizer == 'sgd':
            self.optimizer    = torch.optim.SGD(self.model.parameters(), lr = args.lr, momentum = 0.9, )
        elif args.optimizer == 'adam':
            self.optimizer    = torch.optim.Adam(self.model.parameters(), lr = args.lr, betas = (0.5, 0.999), eps = 1e-08,)
        self.los_fn           = torch.nn.MSELoss()
        return

    ## 返回更新前后的差值，本地多次epoch，每次epoch遍历所有本地数据
    def local_update_diff1(self, cur_weight, lr = 0.01, local_epoch = 3 ):
        init_weight = copy.deepcopy(cur_weight)

        self.model.load_state_dict(cur_weight, strict = True)
        global_model = copy.deepcopy(self.model)
        self.optimizer.param_groups[0]['lr'] = lr
        self.model.train()

        for epoch in range(local_epoch):
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
                X_hat = self.model(X)
                loss = self.los_fn(X_hat, X)
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
        return message, loss.item()


def GenClientsGroup(args, local_dt_dict, model):
    ClientsGroup = {}
    for clientname, dataset in local_dt_dict.items():
        someone = Client(args, dataset, copy.deepcopy(model), clientname)
        ClientsGroup[clientname] = someone
    return ClientsGroup








































