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

"""



import os
from tqdm import tqdm
import numpy as np
import torch
import random

# 以下是本项目自己编写的库
# checkpoint
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


#==================================================== device ===================================================
# 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;
if torch.cuda.is_available():
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    args.device = torch.device("cpu")
    print("PyTorch is running on CPU.")

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
    # for key, var in net.state_dict().items():
        # print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}  " )

    ## 得到全局的参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    ## 定义损失函数
    loss_func = myLoss(args)
    ## 优化算法，随机梯度下降法, 使用 Adam 下降法
    optim = Optimizer.make_optimizer(args, net, )

    ## 创建 Clients 群
    myClients = ClientsGroup(net, args.dir_minst, args.dataset, args.isIID, args.num_of_clients, args.device, args.test_batchsize)
    testDataLoader = myClients.test_data_loader

    ## 创建 server
    server = Server(args, testDataLoader, net, global_parameters)

    ##============================= 完成以上准备工作 ================================#
    ##  选取的 Clients 数量
    num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))

    ## num_comm 表示通信次数
    for round_idx in range(args.num_comm):
        loss_func.addlog()
        lr =  optim.updatelr()
        recorder.addlog(round_idx)
        print(f"Communicate round {round_idx + 1} / {args.num_comm} : ")
        ckp.write_log(f"Communicate round {round_idx + 1} / {args.num_comm} : ", train=True)

        ##==================================================================================================
        ##                     核心代码
        ##==================================================================================================
        ## 从100个客户端随机选取10个
        candidates = ['client{}'.format(i) for i in np.random.choice(range(args.num_of_clients), num_in_comm, replace = False)]
        # candidates = np.random.choice(list(myClients.clients_set.keys()), num_in_comm, replace = False)

        sum_parameters = {}
        for name, params in server.global_model.state_dict().items():
            sum_parameters[name] = torch.zeros_like(params)

        ## 每个 Client 基于当前模型参数和自己的数据训练并更新模型, 返回每个Client更新后的参数
        for client in tqdm(candidates):
            ## 获取当前Client训练得到的参数
            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, loss_func, optim, global_parameters, args)
            ## 对所有的Client返回的参数累加（最后取平均值）
            for var in sum_parameters:
                sum_parameters[var].add_(local_parameters[var])

        ## 取平均值，得到本次通信中Server得到的更新后的模型参数
        for var in sum_parameters:
            sum_parameters[var] = (sum_parameters[var] / num_in_comm)

        global_parameters = server.model_aggregate(sum_parameters )

        ## 训练结束之后，我们要通过测试集来验证方法的泛化性，注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        acc, evl_loss = server.model_eval()

        ##==================================================================================================
        ##                          学习率递减, loss平均, 日志
        ##==================================================================================================
        ## 优化器学习率调整
        optim.schedule()
        epochLos = loss_func.avg()

        print(f"    Accuracy: {acc}, lr = {lr}, train_loss = {epochLos:.3f}, evl_loss = {evl_loss:.3f}" )
        ckp.write_log(f"    lr = {lr}, loss={epochLos:.3f}, Accuracy={acc:.3f}, evl_loss = {evl_loss:.3f}", train=True)
        recorder.assign([lr, acc, epochLos, evl_loss])

        if (round_idx + 1) % args.save_freq == 0:
            torch.save(net, os.path.join(ckp.savedir, '{}_Ncomm={}_E={}_B={}_lr={}_num_clients={}_cf={:.1f}.pt'.format(args.model_name, round_idx, args.loc_epochs, args.local_batchsize, args.learning_rate, args.num_of_clients, args.cfraction )))

        recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'Accuracy', 'train loss', 'val loss'])
    recorder.save(ckp.savedir)
    return


if __name__=="__main__":
    main()


















