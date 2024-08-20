#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:43:42 2023

@author: jack
"""

import  copy
import  numpy as np



class Server(object):
    def __init__(self, args, theta_init ):
        self.args = args
        self.theta = theta_init
        return

    ##>>>>>>>>>>>>>>>>>>>>>>>>>>> error free >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def erf_aggregate_local_gradient(self, mess_lst, lr, ):
        grad_avg = np.mean(mess_lst, axis = 0)
        self.theta -= lr * grad_avg  #  这里必须用-=, 如果为self.theat = self.theat - lr * grad_avg，则调用该函数后self.theta不会变化
        return



    def erf_aggregate_model_diff(self, mess_lst,  ):

        return



    def erf_aggregate_updated_model(self, mess_lst,  ):

        return

    ##>>>>>>>>>>>>>>>>>>>>>>>>>>> error free >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def awgn_aggregate_local_gradient(self, mess_lst, lr, ):

        return



    def awgn_aggregate_model_diff(self, mess_lst, ):

        return



    def awgn_aggregate_updated_model(self, mess_lst, ):

        return

    ##>>>>>>>>>>>>>>>>>>>>>>>>>>> error free >>>>>>>>>>>>>>>>>>>>>>>>>>>
    def rician_aggregate_local_gradient(self, mess_lst,  lr, ):

        return



    def rician_aggregate_model_diff(self, mess_lst, ):

        return



    def rician_aggregate_updated_model(self, mess_lst, ):

        return





# class Server1(object):
#     def __init__(self,  theta_init ):

#         self.theta = theta_init
#         return

#     ##>>>>>>>>>>>>>>>>>>>>>>>>>>> error free >>>>>>>>>>>>>>>>>>>>>>>>>>>
#     def erf(self, mess_lst, lr, ):
#         grad_avg = np.mean(mess_lst, axis = 0)
#         self.theta -=  lr * grad_avg
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





























