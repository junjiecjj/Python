# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

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
Utility.set_printoption(3)

#  加载 checkpoint, 如日志文件, 时间戳, 指标等
ckp = Utility.checkpoint(args)

#=================================================== main =====================================================
def main():
    recorder = MetricsLog.TraRecorder(5, name = "Train", )

    ## 初始化模型
    net = get_model(args.model_name).to(args.device)

    ## 定义损失函数
    loss_func = myLoss(args)
    ## 优化算法，随机梯度下降法, 使用 Adam 下降法
    optim = Optimizer.make_optimizer(args, net, )

    ## 创建 Clients 群
    myClients = ClientsGroup(net, args.dir_minst, args.dataset, args.isIID, args.num_of_clients, args.device, args.test_batchsize)
    testDataLoader = myClients.test_data_loader

    ## 创建 server
    server = Server(args, testDataLoader, args.device, net)

    ##============================= 完成以上准备工作 ================================#
    ##  选取的 Clients 数量
    num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))

    ## 得到全局的参数
    global_parameters = {}

    ## 得到每一层中全连接层中的名称fc1.weight 以及权重weights(tenor)
    for key, var in server.global_model.state_dict().items():
        global_parameters[key] = var.clone()

    ## num_comm 表示通信次数，此处设置为1k, 通讯次数一共1000次
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
            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, loss_func, optim, global_parameters)
            ## 对所有的Client返回的参数累加（最后取平均值）
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        ## 取平均值，得到本次通信中Server得到的更新后的模型参数
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        # server.global_model.load_state_dict(global_parameters, strict=True)
        server.model_aggregate(global_parameters)

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


















