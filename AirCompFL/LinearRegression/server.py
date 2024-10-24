#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2024/08/19

@author: Junjie Chen
"""

# import  copy
import  numpy as np



class Server(object):
    def __init__(self, args, theta_init ):
        self.args = args
        self.theta = theta_init
        return

    def set_learning_rate(self, comm_round, lr0,  lr_decrease):
        if lr_decrease:
            lr = lr0/(0.004*comm_round + 1)
        else:
            lr = lr0
        return lr

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error-free %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def aggregate_erf_gradient(self, mess_lst, lr, ):
        grad_avg = np.mean(mess_lst, axis = 0)
        self.theta -= lr * grad_avg  #  这里必须用-=, 如果为self.theat = self.theat - lr * grad_avg，则调用该函数后self.theta不会变化
        return

    def aggregate_erf_diff(self, mess_lst,):
        self.theta += np.mean(mess_lst, axis = 0)
        return

    def aggregate_erf_model(self, mess_lst,  ):
        self.theta = np.mean(mess_lst, axis = 0) # 这里可以直接用self.theta = xxx, 也可以用self.theta[:] = xxx
        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% AWGN MAC %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def aggregate_awgn_gradient(self, mess_lst, lr, noise_var, P0,  ):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 发送端不对发送信号归一化时，接收端的去噪因子，
        # grad_avg = np.mean(mess_lst, axis = 0)
        # eta2 = [P0 / np.power(np.linalg.norm(mess, ord = 2), 2) for mess in mess_lst]
        # eta =  self.args.D * min(eta2)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = grad_avg.shape)
        # self.theta -= lr * (grad_avg + noise)

        ##2 eta = min_{k} P0/sigma_k^2, 发送端对发送信号归一化时，接收端的去噪因子。这两者是等价的，只要发送功率一样。
        grad_avg = np.mean(mess_lst, axis = 0)
        eta2 = [P0 / np.var(mess,) for mess in mess_lst]
        eta = min(eta2)
        noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = grad_avg.shape)
        self.theta -= lr * (grad_avg + noise)

        return

    def aggregate_awgn_diff(self, mess_lst, noise_var, P0, ):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 不归一化时，接收端的去噪因子，
        # diff_avg = np.mean(mess_lst, axis = 0)
        # eta2 = [P0 / np.power(np.linalg.norm(mess, ord = 2), 2) for mess in mess_lst]
        # eta =  self.args.D * min(eta2)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = diff_avg.shape)
        # self.theta += (diff_avg + noise )

        ##2 eta = min_{k} P0/sigma_k^2, 发送端归一化时，接收端的去噪因子
        diff_avg = np.mean(mess_lst, axis = 0)
        eta2 = [P0 / np.var(mess,) for mess in mess_lst]
        eta = min(eta2)
        noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = diff_avg.shape)
        self.theta += (diff_avg + noise)

        return

    def aggregate_awgn_model(self, mess_lst, noise_var, P0, ):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 不归一化时，接收端的去噪因子，
        # model_avg = np.mean(mess_lst, axis = 0)
        # eta2 = [P0 / np.power(np.linalg.norm(mess, ord = 2), 2) for mess in mess_lst]
        # eta =  self.args.D * min(eta2)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = model_avg.shape)
        # self.theta = model_avg + noise

        ##2 eta = min_{k} P0/sigma_k^2, 发送端归一化时，接收端的去噪因子
        model_avg = np.mean(mess_lst, axis = 0)
        eta2 = [P0 / np.var(mess,) for mess in mess_lst]
        eta = min(eta2)
        noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = model_avg.shape)
        self.theta = model_avg + noise
        return

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%% Rician Fading Mac %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def aggregate_rician_gradient(self, mess_lst, lr, noise_var, P0, H):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 发送端不对发送信号归一化时，接收端的去噪因子，
        # grad_avg = np.mean(mess_lst, axis = 0)
        # eta2 = [P0 * np.abs(H[i])**2 / np.power(np.linalg.norm(mess, ord = 2), 2)  for i, mess in enumerate(mess_lst)]
        # eta =  self.args.D * min(eta2)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = grad_avg.shape)
        # self.theta -= lr * (grad_avg + noise)

        ##2 eta = min_{k} P0/sigma_k^2, 发送端对发送信号归一化时，接收端的去噪因子。这两者是等价的，只要发送功率一样。
        grad_avg = np.mean(mess_lst, axis = 0)
        eta2 = [P0 * np.abs(H[i])**2 / np.var(mess,) for i, mess in enumerate(mess_lst)]
        eta = min(eta2)
        noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = grad_avg.shape)
        self.theta -= lr * (grad_avg + noise )
        return

    def aggregate_rician_diff(self, mess_lst, noise_var, P0, H):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 发送端不对发送信号归一化时，接收端的去噪因子，
        # diff_avg = np.mean(mess_lst, axis = 0)
        # eta2 = [P0 * np.abs(H[i])**2 / np.power(np.linalg.norm(mess, ord = 2), 2)  for i, mess in enumerate(mess_lst)]
        # eta =  self.args.D * min(eta2)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = diff_avg.shape)
        # self.theta += (diff_avg + noise)

        ##2 eta = min_{k} P0/sigma_k^2, 发送端对发送信号归一化时，接收端的去噪因子。这两者是等价的，只要发送功率一样。
        diff_avg = np.mean(mess_lst, axis = 0)
        eta2 = [P0 * np.abs(H[i])**2 / np.var(mess,) for i, mess in enumerate(mess_lst)]
        eta = min(eta2)
        noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = diff_avg.shape)
        self.theta += (diff_avg + noise )
        return

    def aggregate_rician_model(self, mess_lst, noise_var, P0, H):
        # ##1 eta = min_{k} d*P0/|z_k^t|^2, 发送端不对发送信号归一化时，接收端的去噪因子，
        # model_avg = np.mean(mess_lst, axis = 0)
        # eta2 = [P0 * np.abs(H[i])**2 / np.power(np.linalg.norm(mess, ord = 2), 2)  for i, mess in enumerate(mess_lst)]
        # eta =  self.args.D * min(eta2)
        # noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = model_avg.shape)
        # self.theta = model_avg + noise

        ##2 eta = min_{k} P0/sigma_k^2, 发送端对发送信号归一化时，接收端的去噪因子。这两者是等价的，只要发送功率一样。
        model_avg = np.mean(mess_lst, axis = 0)
        eta2 = [P0 * np.abs(H[i])**2 / np.var(mess,) for i, mess in enumerate(mess_lst)]
        eta = min(eta2)
        noise = np.random.normal(loc = 0, scale = np.sqrt(noise_var/eta/len(mess_lst)), size = model_avg.shape)
        self.theta = model_avg + noise
        return



# class Server1(object):
#     def __init__(self,  theta_init ):
#         self.theta = theta_init
#         return

#     def erf(self, mess_lst, lr, ):
#         grad_avg = np.mean(mess_lst, axis = 0)
#         self.theta = grad_avg
#         return

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





























