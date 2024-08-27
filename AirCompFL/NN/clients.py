







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
        self.id               = client_name
        self.trainloader      = DataLoader(data, batch_size = args.local_bs, shuffle = True)
        self.datasize         = len(data)
        self.model            = model
        self.num_local_update = args.local_up
        self.optimizer        = optim.SGD(params = self.model.parameters(), lr = args.lr)
        self.los_fn           = nn.CrossEntropyLoss()
        return

    ## 返回本地梯度
    def local_update_gradient(self, global_weight ):
        self.model.load_state_dict(global_weight, strict=True)
        self.model.train()

        gard = {}

        return

    ## 返回更新前后的差值
    def local_update_diff(self, global_weight, num_local_update = 10, local_bs = 128, lr = 0.01, ):

        return

    ## 返回更新后的模型
    def local_update_model(self, global_weight, num_local_update = 10, local_bs = 128, lr = 0.01, ):

        return

def GenClientsGroup(args, local_dt_dict, model):
    ClientsGroup = {}
    for clientname, dataset in local_dt_dict.items():
        someone = Client(args, dataset, copy.deepcopy(model), clientname)
        ClientsGroup[clientname] = someone
    return ClientsGroup








































