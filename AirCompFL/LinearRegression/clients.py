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
# from data.getData import GetDataSet


class Client(object):
    def __init__(self, args, trainX, trainY, theta_init, client_name = "clientxx",):
        self.args             = args
        self.client_name      = client_name
        self.X                = trainX
        self.Y                = trainY
        self.local_ds         = len(trainX)
        self.theta            = theta_init
        return

    ## 返回本地梯度
    def local_gradient(self, theta, local_bs = 128 ):
        idx = np.random.choice(self.local_ds, local_bs, replace = False)
        x = self.X[idx]
        y = self.Y[idx]
        gradient = (x.T @ (x@theta - y)) / local_bs

        return [gradient, local_bs]

    ## 返回更新前后的差值
    def model_diff(self, theta, num_local_update = 10, local_bs = 128, lr = 0.01 ):
        tmp = copy.deepcopy(theta)
        for _ in range(num_local_update):
            idx = np.random.choice(self.local_ds, local_bs, replace = False)
            x = self.X[idx]
            y = self.Y[idx]
            gradient = (x.T @ (x@theta - y)) / local_bs
            theta = theta - lr * gradient
        theta_diff = theta - tmp
        return [theta_diff, local_bs]


    ## 返回更新后的模型
    def updated_model(self, theta, num_local_update = 10, local_bs = 128, lr = 0.01):
        for _ in range(num_local_update):
            idx = np.random.choice(self.local_ds, local_bs, replace = False)
            x = self.X[idx]
            y = self.Y[idx]
            gradient = (x.T @ (x@theta - y)) / local_bs
            theta = theta - lr * gradient
        return [theta, local_bs]


    def local_loss(self, theta,):
        Fk = 0.5 * np.linalg.norm( self.X @ theta - self.Y, ord = 2)**2 / self.local_ds
        return Fk

    # def update_theta(self, theta):
    #     self.theta = theta
    #     return


def GenClientsGroup(args, X, Y, theta0, frac):
    ClientsGroup = {}
    for clientname, data in X.items():
        ##  创建一个客户端
        someone = Client(args, X[clientname], Y[clientname], theta0, clientname)
        # 为每一个clients 设置一个名字
        ClientsGroup[clientname] = someone
    return ClientsGroup



# a = np.array([1,2,3])
# def hh(a):
#     for i in range(2):
#         a  = a - i
#     return a
# aa = hh(a)
# print(f"a = {a}\naa = {aa}")

# a = np.array([1,2,3])
# def hh1(a):
#     for i in range(2):
#         a  -=  i
#     return a
# aa1 = hh1(a)
# print(f"a = {a}\naa1 = {aa1}")









