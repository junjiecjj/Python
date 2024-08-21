#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2024/08/19

@author: Junjie Chen
"""

import  copy
import  numpy as np



class Server(object):
    def __init__(self, args, theta_init ):
        self.args = args
        self.theta = theta_init
        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error-free %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def erf_aggregate_local_gradient(self, mess_lst, lr, ):
        grad_avg = np.mean(mess_lst, axis = 0)
        self.theta -= lr * grad_avg  #  这里必须用-=, 如果为self.theat = self.theat - lr * grad_avg，则调用该函数后self.theta不会变化
        return

    def erf_aggregate_model_diff(self, mess_lst,):
        self.theta += np.mean(mess_lst, axis = 0)
        return

    def erf_aggregate_updated_model(self, mess_lst,  ):
        self.theta = np.mean(mess_lst, axis = 0) # 这里可以直接用self.theta = xxx, 也可以用self.theta[:] = xxx
        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% AWGN MAC %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def awgn_aggregate_local_gradient(self, mess_lst, lr, SNR,  ):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 发送端不对发送信号归一化时，接收端的去噪因子，
        # grad_avg = np.mean(mess_lst, axis = 0)
        # grad_norm = [np.linalg.norm(mess, ord = 2) for mess in mess_lst]
        # # print(grad_norm)
        # eta = np.sqrt(self.args.D) / max(grad_norm)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(10**(-SNR/10)), size = grad_avg.shape)
        # self.theta -= lr * (grad_avg + noise/eta/len(mess_lst))

        ##2 eta = min_{k} P0/sigma_k^2, 发送端对发送信号归一化时，接收端的去噪因子。这两者是等价的，只要发送功率一样。
        grad_avg = np.mean(mess_lst, axis = 0)
        grad_var = [np.var(mess,) for mess in mess_lst]
        # print(grad_norm)
        eta = 1 / max(grad_var)
        noise = np.random.normal(loc = 0, scale = np.sqrt(10**(-SNR/10)), size = grad_avg.shape)
        self.theta -= lr * (grad_avg + noise/eta/len(mess_lst))

        return

    def awgn_aggregate_model_diff(self, mess_lst, SNR, ):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 不归一化时，接收端的去噪因子，
        # diff_avg = np.mean(mess_lst, axis = 0)
        # diff_norm = [np.linalg.norm(mess, ord = 2) for mess in mess_lst]
        # # print(grad_norm)
        # eta = np.sqrt(self.args.D) / max(diff_norm)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(10**(-SNR/10)), size = diff_avg.shape)
        # self.theta += (np.mean(mess_lst, axis = 0) + noise/eta/len(mess_lst) )

        ##2 eta = min_{k} P0/sigma_k^2, 发送端归一化时，接收端的去噪因子
        diff_avg = np.mean(mess_lst, axis = 0)
        diff_std = [np.std(mess,) for mess in mess_lst]
        # print(grad_norm)
        eta = 1 / max(diff_std)
        noise = np.random.normal(loc = 0, scale = np.sqrt(10**(-SNR/10)), size = diff_avg.shape)
        self.theta += (np.mean(mess_lst, axis = 0) + noise/eta/len(mess_lst) )

        return

    def awgn_aggregate_updated_model(self, mess_lst, SNR, ):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 不归一化时，接收端的去噪因子，
        # model_avg = np.mean(mess_lst, axis = 0)
        # model_norm = [np.linalg.norm(mess, ord = 2) for mess in mess_lst]
        # # print(grad_norm)
        # eta = np.sqrt(self.args.D) / max(model_norm)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(10**(-SNR/10)), size = model_avg.shape)
        # self.theta = np.mean(mess_lst, axis = 0) + noise/eta/len(mess_lst)

        ##2 eta = min_{k} P0/sigma_k^2, 发送端归一化时，接收端的去噪因子
        model_avg = np.mean(mess_lst, axis = 0)
        model_std = [np.std(mess,) for mess in mess_lst]
        # print(grad_norm)
        eta = 1 / max(model_std)
        noise = np.random.normal(loc = 0, scale = np.sqrt(10**(-SNR/10)), size = model_avg.shape)
        self.theta = np.mean(mess_lst, axis = 0) + noise/eta/len(mess_lst)

        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% Rician Fading Mac %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def rician_aggregate_local_gradient(self, mess_lst, lr, SNR, H):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 发送端不对发送信号归一化时，接收端的去噪因子，
        # grad_avg = np.mean(mess_lst, axis = 0)
        # grad_norm = [np.linalg.norm(mess, ord = 2) for mess in mess_lst]
        # # print(grad_norm)
        # eta = np.sqrt(self.args.D) / max(grad_norm)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(10**(-SNR/10)), size = grad_avg.shape)
        # self.theta -= lr * (grad_avg + noise/eta/len(mess_lst))

        ##2 eta = min_{k} P0/sigma_k^2, 发送端对发送信号归一化时，接收端的去噪因子。这两者是等价的，只要发送功率一样。
        grad_avg = np.mean(mess_lst, axis = 0)
        grad_var = [np.var(mess,) for mess in mess_lst]
        # print(grad_norm)
        eta = 1 / max(grad_var)
        noise = np.random.normal(loc = 0, scale = np.sqrt(10**(-SNR/10)), size = grad_avg.shape)
        self.theta -= lr * (grad_avg + noise/eta/len(mess_lst))

        return

    def rician_aggregate_model_diff(self, mess_lst, SNR, H):

        return

    def rician_aggregate_updated_model(self, mess_lst, SNR, H):

        return




class Server1(object):
    def __init__(self,  theta_init ):

        self.theta = theta_init
        return

    def erf(self, mess_lst, lr, ):
        grad_avg = np.mean(mess_lst, axis = 0)
        self.theta = grad_avg
        return

# theta0 = np.zeros((3, 1))
# print(f"theta0 = \n{theta0}\n")

# lr = 0.1
# s = Server1(theta0)
# print(f"0: s.theta = \n{s.theta}\n")
# mess_lst = []
# for i in range(3):
#     mess_lst.append(np.random.randn(3, 1) )

# sum = 0
# for i in range(3):
#     sum += mess_lst[i]
# sum /= 3

# s.erf(mess_lst, lr)
# print(f"1: s.theta = \n{s.theta}\n")





























