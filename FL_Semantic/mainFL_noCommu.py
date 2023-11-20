# -*- coding: utf-8 -*-
"""
Created on 2023/06/30
@author: Junjie Chen

这是客户端和服务器共享一个全局模型，且所有的客户端共享一个优化器和损失函数，且兼容传输模型和传输模型差值两种模式的程序，
对于传输模型差值：
    因为共享一个模型，所以：
    (1) 客户端在每轮通信之前需要把最新的全局模型加载；且客户端返回的是自身训练后的模型参数减去这轮初始时的全局模型得到的差值；
    (2) 因为每个客户端在训练的时候是在当前global全局模型加载后训练的，且因为是共享的全局模型，每个客户端训练完后，全局模型都会改变，因此服务器需要在更新全局模型之前把上次的全局模型先加载一遍，在将这轮通信得到的所有客户端模型参数差值的平均值加到全局模型上；

对于直接传输模型：
    (1) 客户端在每轮通信之前需要把最新的全局模型加载；且客户端返回的是自身训练后的模型参数；
    (2) 服务端直接把客户端返回的模型参数做个平均，加载到全局模型；

本地化差分隐私中，每个用户将各自的数据进行扰动后，再上传至数据收集者处，而任意两个用户之间 并不知晓对方的数据记录，本地化差分隐私中并不存在全局敏感性的概念，因此，拉普拉斯机制和指数机 制并不适用．

"""



# import os
# import sys
from tqdm import tqdm
import numpy as np
import torch


## 以下是本项目自己编写的库
# checkpoint

# from  pipeline_serial import Quant_BitFlipping

import Utility

from clients import ClientsGroup

from server import Server

from model import get_model

# 参数
from config import args

# 损失函数
from loss.Loss import myLoss

# 优化器
import Optimizer

import MetricsLog

# from rdp_analysis import calibrating_sampled_gaussian



#==================================================  seed =====================================================
# 设置随机数种子
Utility.set_random_seed(args.seed, deterministic = True, benchmark = True)
Utility.set_printoption(5)

#  加载 checkpoint, 如日志文件, 时间戳, 指标等
ckp = Utility.checkpoint(args)

#=================================================== main =====================================================
def main():
    recorder = MetricsLog.TraRecorder(5, name = "Train", )

    ## 初始化模型， 因为后面的 server 和 ClientsGroup 都是共享一个net，所以会有一些处理比较麻烦
    net = get_model(args.model_name).to(args.device)
    data_valum = 0.0
    key_order = []
    for key, var in net.state_dict().items():
        data_valum += var.numel()
        key_order.append(key)
        # print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}  " )
    print(key_order)
    print(f"Data volume = {data_valum} (floating point number) ")
    ## 得到全局的参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    loss_func = myLoss(args)
    optim = Optimizer.make_optimizer(args, net, )

    myClients = ClientsGroup(net, args.dir_minst, args )
    testDataLoader = myClients.test_data_loader

    server = Server(args, testDataLoader, net, global_parameters)

    ##============================= 完成以上准备工作 ================================#

    num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))
    # MV_recordParam = MetricsLog.RecorderFL( int(num_in_comm * 2 * len(key_order)) + 1)
    stastic = ["Mean", "1-norm", "2-norm", "Variance"]
    MV_recordAll = MetricsLog.RecorderFL( int(num_in_comm * len(stastic)) + 1 )
    ##==================================================================================================
    ##                                核心代码
    ##==================================================================================================

    for round_idx in range(args.num_comm):
        ###  MV_recordParam.addline(round_idx)
        MV_recordAll.addline(round_idx)

        MV_client = []
        loss_func.addlog()
        lr =  optim.updatelr()
        recorder.addlog(round_idx)
        print(f"Communicate round : {round_idx + 1} / {args.num_comm} : ")
        ckp.write_log(f"Communicate round : {round_idx + 1} / {args.num_comm} : ", train = True)

        sum_parameters = {}
        for name, params in server.global_model.state_dict().items():
            sum_parameters[name] = torch.zeros(params.shape).to(args.device)

        candidates = [f"client{i}" for i in np.random.choice(range(args.num_of_clients), num_in_comm, replace = False)]
        # candidates = np.random.choice(list(myClients.clients_set.keys()), num_in_comm, replace = False)
        weight = []
        for client in  candidates:
            print(f"    {client},", end = '' )
            client_dtsize = myClients.clients_set[client].datasize
            weight.append(client_dtsize)

            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, loss_func, optim, global_parameters, )
            ##=========================================================
            MV_client.extend(Utility.MeanVarStatistic(local_parameters))
            ##=========================================================

            for key, params in server.global_model.state_dict().items():
                # sum_parameters[key].add_(local_parameters[key])
                if key in local_parameters:
                    sum_parameters[key].add_(local_parameters[key] * client_dtsize)
                    # cnt[key] += 1
        MV_recordAll.assign(MV_client)
        print("\n")
        global_parameters = server.model_aggregate(sum_parameters, weight)
        # print(f"global_parameters: {global_parameters['fc2.bias']}")
        ## 训练结束之后，Server端进行测试集来验证方法的泛化性，
        acc, evl_loss = server.model_eval()

        ##==================================================================================================
        ##                          学习率递减, loss平均, 日志
        ##==================================================================================================
        ## 优化器学习率调整
        # optim.schedule()
        epochLos = loss_func.avg()
        recorder.assign([lr, acc, epochLos, evl_loss])

        print(f"    Accuracy: {acc}, lr = {lr}, train_loss = {epochLos:.3f}, evl_loss = {evl_loss:.3f}" )
        ckp.write_log(f"    lr = {lr}, loss={epochLos:.3f}, Accuracy={acc:.3f}, evl_loss = {evl_loss:.3f}", train=True)
        recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'Accuracy', 'train loss', 'val loss'])
        ### MV_record.Clients_MeanVar_avg(num_in_comm, key_order, savepath = ckp.savedir, savename = 'Mean_VarOfClients')
        MV_recordAll.Client_mean_var_L12(num_in_comm, stastic = stastic, savepath = ckp.savedir, savename = 'MeanVarL12OfClients')
        MV_recordAll.Client_mean_var_L12_avg(num_in_comm, stastic = stastic, savepath = ckp.savedir, savename = 'MeanVarL12OfClientsAvg')
        if (round_idx + 1) % 10 == 0:
            recorder.save(ckp.savedir)
            MV_recordAll.save(ckp.savedir, "MeanVarL12OfClients.pt")
    recorder.save(ckp.savedir)
    MV_recordAll.save(ckp.savedir, "MeanVarL12OfClients.pt")
    print(key_order)
    print(f"Data volume = {data_valum} (floating point number) ")
    return


if __name__=="__main__":
    main()


















