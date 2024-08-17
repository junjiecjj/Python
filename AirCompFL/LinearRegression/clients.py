# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""



import numpy as np
import copy
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# from model import get_model
from data.getData import GetDataSet


class client(object):
    def __init__(self, model, trainDataSet, args, client_name = "client10", datasize = 0):
        self.args             = args
        self.client_name      = client_name
        self.datasize         = datasize
        return

    def localUpdate(self,  ):

        return



class ClientsGroup(object):
    def __init__(self, args = None,  ):

        return

    def dataSetAllocation_balance(self):

        for i in range(self.num_of_clients):

            ##  创建一个客户端
            someone = client( )
            # 为每一个clients 设置一个名字
            self.clients_set[f"client{i}"] = someone
        return

    def dataSetAllocation_Unblance1(self):

        for i in range(self.num_of_clients):

            # 创建一个客户端
            someone = client( )
            # 为每一个clients 设置一个名字
            self.clients_set[f"client{i}"] = someone
        return




















