#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:43:42 2023

@author: jack
"""


import   torch
from model import get_model

class Server(object):
    def __init__(self, Args, test_dataloader, model = None):
        self.args         = Args
        self.device       = self.args.device
        self.global_model = get_model(self.args.model_name).to(self.device)
        # for name, var in self.global_model.state_dict().items():
            # print(f"{name}: {var.is_leaf}, {var.shape}, {var.requires_grad}, {var.type()}  ")
        self.eval_loader  = test_dataloader
        return

    def model_aggregate(self, weight_accumulator):
        # 传输的是模型差值，且使用差分隐私
        if self.args.transmitted_diff and self.args.DP:
            for name, data in self.global_model.state_dict().items():
                update_per_layer = weight_accumulator[name] * self.args["lambda"]
                if self.args['dp']:
                    sigma = self.args['sigma']
                    noise = torch.normal(mean = 0, std = sigma, size = update_per_layer.shape ).to(self.device)
                    update_per_layer.add_(noise)
                if data.type() != update_per_layer.type():
                    print(f"{data.type()}, {update_per_layer.type()}")
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)
        # 传输的是模型差值，且不使用差分隐私
        elif self.args.transmitted_diff and not self.args.DP:
            for key, val in weight_accumulator.items():
                self.global_model.state_dict()[key].add_(val)
        # 传输的是模型参数，直接赋值
        elif not self.args.transmitted_diff:
            self.global_model.load_state_dict(weight_accumulator, strict=True)

        # 得到当前最新的全局模型
        global_parameters = {}
        for key, var in self.global_model.state_dict().items():
            global_parameters[key] = var.clone()

        return global_parameters # copy.deepcopy(self.global_model.state_dict())   #   global_parameters


    def model_eval(self):
        self.global_model.eval()
        ## 训练结束之后，我们要通过测试集来验证方法的泛化性，注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        ## 加载Server在最后得到的模型参数
        # self.global_model.load_state_dict(global_parameters, strict=True)
        sum_accu    = 0.0
        sum_loss    = 0.0
        examples    = 0
        loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')
        # 载入测试集
        for data, label in self.eval_loader:
            examples    += data.shape[0]
            data, label = data.to(self.device), label.to(self.device)
            preds       = self.global_model(data)
            sum_loss    += loss_fn(preds, label).item()
            preds       = torch.argmax(preds, dim=1)
            sum_accu    += (preds == label).float().sum().item()
        acc     = sum_accu / examples
        avg_los = sum_loss / examples

        return acc, avg_los











