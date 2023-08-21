#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:43:42 2023

@author: jack
"""

import copy
import   torch
# from model import get_model

class Server(object):
    def __init__(self, Args, test_dataloader, model = None, init_params = None):
        self.args         = Args
        self.device       = self.args.device
        self.global_model = model # get_model(self.args.model_name).to(self.device)
        # for name, var in self.global_model.state_dict().items():
            # print(f"{name}: {var.is_leaf}, {var.shape}, {var.requires_grad}, {var.type()}  ")
        self.eval_loader  = test_dataloader
        # 存储上次最新模型的字典
        self.last_pamas   = init_params
        return

    def model_aggregate(self, weight_accumulator, cnt = None):
        ## 取平均值，得到本次通信中Server得到的更新后的模型参数
        num_in_comm = int(max(self.args.num_of_clients * self.args.cfraction, 1))
        # print(f"cnt:   {cnt.values()}")
        for key in weight_accumulator:
            if cnt[key] > 0:
                weight_accumulator[key].div_(cnt[key])

        ## 传输的是模型差值
        if self.args.transmitted_diff:
            ## 先加载上次的模型，再加上这次的更新
            self.global_model.load_state_dict(self.last_pamas, strict=True)
            for key, val in weight_accumulator.items():
                if cnt[key] > 0:
                    self.global_model.state_dict()[key].add_(val)
            # for key, param in self.global_model.state_dict().items():
                # if key in weight_accumulator and cnt[key] > 0:
                    # param.add_(weight_accumulator[key])
        ## 传输的是模型参数，直接赋值
        elif not self.args.transmitted_diff :
            # print("传输的是模型参数")
            # self.global_model.load_state_dict(weight_accumulator, strict=True)
            for key, val in weight_accumulator.items():
                if cnt[key] > 0:
                    self.global_model.state_dict()[key].copy_(val.clone())
            # for key, param in self.global_model.state_dict().items():
                # if key in weight_accumulator and cnt[key] > 0:
                    # param.copy_(weight_accumulator[key])

        if self.args.ClientDP: ##  使用差分隐私
            pass

        # 得到当前最新的全局模型并赋值给存储上次最新模型的字典
        global_parameters = {}
        for key, var in self.global_model.state_dict().items():
            global_parameters[key] = var.clone()
            self.last_pamas[key]   = var.clone()

        return global_parameters # copy.deepcopy(self.global_model.state_dict())   #   global_parameters


    def model_eval(self):
        self.global_model.eval()
        ## 训练结束之后，我们要通过测试集来验证方法的泛化性，注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        ##  加载Server在最后得到的模型参数
        # self.global_model.load_state_dict(global_parameters, strict=True)
        sum_accu    = 0.0
        sum_loss    = 0.0
        examples    = 0
        loss_fn     = torch.nn.CrossEntropyLoss(reduction='sum')
        # 载入测试集
        for X, y in self.eval_loader:
            examples    += X.shape[0]
            X, y        = X.to(self.device), y.to(self.device)
            preds       = self.global_model(X)
            sum_loss    += loss_fn(preds, y).item()
            sum_accu    += (torch.argmax(preds, dim=1) == y).float().sum().item()
        acc     = sum_accu / examples
        avg_los = sum_loss / examples

        return acc, avg_los






































