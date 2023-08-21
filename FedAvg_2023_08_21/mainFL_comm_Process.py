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



import os, sys
from tqdm import tqdm
import numpy as np
import torch
import multiprocessing
# from multiprocessing import Process, Lock


# 以下是本项目自己编写的库
# checkpoint

from pipeline_multiprocess import Quant_BPSK_AWGN_Pipe

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

from rdp_analysis import calibrating_sampled_gaussian

#==================================================== device ===================================================
# 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;
if torch.cuda.is_available():
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    args.device = torch.device("cpu")
    print("PyTorch is running on CPU.")

# if args.LDP:
#     Total_iters = int( args.num_comm * args.loc_epochs)
#     print("Calcuating Sigma \n")
#     args.sigma = calibrating_sampled_gaussian(args.q, args.eps, args.delta, Total_iters)
#     print(f"sigma = {args.sigma }")
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
    for key, var in net.state_dict().items():
        data_valum += var.numel()
    print(f"sum data volum = {data_valum}")
    ## 得到全局的参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    ## 定义损失函数
    loss_func = myLoss(args)
    ## 优化算法，随机梯度下降法, 使用 Adam 下降法
    optim = Optimizer.make_optimizer(args, net, )

    ## 创建 Clients 群
    myClients = ClientsGroup(net, args.dir_minst, args )
    testDataLoader = myClients.test_data_loader

    ## 创建 server
    server = Server(args, testDataLoader, net, global_parameters)

    ##============================= 完成以上准备工作 ================================#
    ##  选取的 Clients 数量
    num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))

    ##==================================================================================================
    ##                                核心代码
    ##==================================================================================================
    ## num_comm 表示通信次数
    for round_idx in range(args.num_comm):
        loss_func.addlog()
        lr =  optim.updatelr()
        recorder.addlog(round_idx)
        print(f"Communicate round : {round_idx + 1} / {args.num_comm} : ")
        ckp.write_log(f"Communicate round : {round_idx + 1} / {args.num_comm} : ", train = True)

        ## 从100个客户端随机选取10个
        candidates = ['client{}'.format(i) for i in np.random.choice(range(args.num_of_clients), num_in_comm, replace = False)]
        # candidates = np.random.choice(list(myClients.clients_set.keys()), num_in_comm, replace = False)

        sum_parameters = {}
        cnt = {}
        for name, params in server.global_model.state_dict().items():
            sum_parameters[name] = torch.zeros_like(params)
            cnt[name]            = 0.0

        ##================================================================================================
        ## 每个 Client 基于当前模型参数和自己的数据串行训练，并行通信传输, 返回每个Client更新后的参数
        ##================================================================================================
        m = multiprocessing.Manager()
        dict_param = m.dict()
        dict_berfer = m.dict()
        lock = multiprocessing.Lock()  # 这个一定要定义为全局
        jobs = []

        for client in  candidates:
            ## 获取当前Client训练得到的参数
            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, loss_func, optim, global_parameters,)
            for key in local_parameters:
                local_parameters[key] = np.array(local_parameters[key].detach().cpu())
            # 量化、编码、信道、解码、反量化; client = '', param_W = '', snr = 2.0 , quantBits = 8, com_round = 1,  dic_res = " ", lock = None
            ps = multiprocessing.Process(target=Quant_BPSK_AWGN_Pipe, args=(round_idx, client, local_parameters, 2.0, 6, dict_param, dict_berfer, lock))
            jobs.append(ps)
            ps.start()

        for p in jobs:
            p.join()
        for key, berfer in dict_berfer.items():
            print(f"CommRound {round_idx}: {key}, ber = {berfer['ber']:.8f}, fer = {berfer['fer']:.8f}, a_iter = {berfer['ave_iter']:.8f}")
        ##================================ 串行训练，并行通信传输结束 ========================================
        ## 对所有的Client返回的参数累加（最后取平均值）
        for client, param_client in dict_param.items():
            for key, params in server.global_model.state_dict().items():
                # sum_parameters[key].add_(local_parameters[key])
                if key in param_client[client]:
                    sum_parameters[key].add_(torch.tensor(param_client[client][key], dtype = torch.float32).to(args.device))
                    cnt[key] += 1

        global_parameters = server.model_aggregate(sum_parameters, cnt)
        ## 训练结束之后，Server端进行测试集来验证方法的泛化性，
        acc, evl_loss = server.model_eval()

        ##==================================================================================================
        ##                          学习率递减, loss平均, 日志
        ##==================================================================================================
        ## 优化器学习率调整
        optim.schedule()
        epochLos = loss_func.avg()
        recorder.assign([lr, acc, epochLos, evl_loss])

        print(f"    Accuracy: {acc}, lr = {lr}, train_loss = {epochLos:.3f}, evl_loss = {evl_loss:.3f}" )
        ckp.write_log(f"    lr = {lr}, loss={epochLos:.3f}, Accuracy={acc:.3f}, evl_loss = {evl_loss:.3f}", train=True)
        recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'Accuracy', 'train loss', 'val loss'])
        if (round_idx + 1) % 100 == 0:
            recorder.save(ckp.savedir)
    return


if __name__=="__main__":
    main()


















