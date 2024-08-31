
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
# from tqdm import tqdm
import numpy as np
import torch
import multiprocessing as mp
# import torch.multiprocessing as mp
# from multiprocessing import Process, Lock


## 以下是本项目自己编写的库
## checkpoint

from pipeline_multiprocess import Quant_LDPC_BPSK_AWGN_equa
# from pipeline_multiprocess import Quant_LDPC_BPSK_AWGN_Pipe
from pipeline_multiprocess import Quant_BbitFlipping, Quant_1bitFlipping

from pipeline_multiprocess import  acc2Qbits, Qbits2Lr,Qbits2Lr_1
from pipeline_multiprocess import acc2Qbits1, err2Qbits

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
    recorder = MetricsLog.TraRecorder(7, name = "Train", )

    ## 初始化模型， 因为后面的 server 和 ClientsGroup 都是共享一个net，所以会有一些处理比较麻烦
    net = get_model(args.model_name).to(args.device)
    data_valum = 0.0
    for key, var in net.state_dict().items():
        data_valum += var.numel()
    print(f"Data volume = {data_valum} (floating point number) ")
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
    acc = 0
    ##==================================================================================================
    ##                                      核心代码
    ##==================================================================================================
    comm_cost = 0.0
    ## num_comm 表示通信次数
    for round_idx in range(args.num_comm):
        loss_func.addlog()

        recorder.addlog(round_idx)
        print(f"Communicate round : {round_idx + 1} / {args.num_comm} : ")
        ckp.write_log(f"Communicate round : {round_idx + 1} / {args.num_comm} : ", train = True)

        sum_parameters = {}
        for name, params in server.global_model.state_dict().items():
            sum_parameters[name] = torch.zeros(params.shape).to(args.device)

        ## 从100个客户端随机选取10个
        candidates = ['client{}'.format(i) for i in np.random.choice(range(args.num_of_clients), num_in_comm, replace = False)]
        # candidates = np.random.choice(list(myClients.clients_set.keys()), num_in_comm, replace = False)
        weight = []
        weigh_dict = {}

        ## (1) 研究固定量化比特在不同的翻转概率下的性能，
        quantbits = 1
        P_bitflip = 0.5
        learn_rate = 0.003
        optim.set_lr(learn_rate)

        ## (2) 动态量化, : 根据当前比特翻转概率和性能调整量化比特数和学习率; 研究不同翻转概率下，根据当前性能调整量化比特数; 当P_bitflip = 0时, 为无错下的动态量化;
        # P_bitflip = 0.3
        # quantbits, learn_rate = err2Qbits(acc, P_bitflip)
        # optim.set_lr(learn_rate)

        ## (3) 动态量化, : 根据当前信噪比和性能调整量化比特数和学习率，研究不同信噪比下，根据当前性能调整量化比特；
        # snr = 2
        # quantbits, learn_rate = acc2Qbits(acc, snr)
        # optim.set_lr(learn_rate)

        comm_cost += data_valum * quantbits
        # print(f"   num_bits =  {quantbits}, ")
        lr =  optim.updatelr()
        ##================================================================================================
        ## 每个 Client 基于当前模型参数和自己的数据串行训练，并行通信传输, 返回每个Client更新后的参数
        ##================================================================================================
        m = mp.Manager()
        dict_param  = m.dict()
        dict_berfer = m.dict()
        # lock = mp.Lock()  ## 这个一定要定义为全局
        jobs = []

        for client in candidates:
            # rdm = np.random.RandomState()
            client_dtsize = myClients.clients_set[client].datasize
            weight.append(client_dtsize)
            weigh_dict[client] = client_dtsize

            ## 获取当前Client训练得到的参数
            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, loss_func, optim, global_parameters,)
            for key in local_parameters:
                local_parameters[key] = np.array(local_parameters[key].detach().cpu())

            # 量化、编码、信道、解码、反量化;
            # ps = mp.Process(target = Quant_LDPC_BPSK_AWGN_equa, args=(round_idx, client, local_parameters, snr, quantbits, dict_param, dict_berfer, lock))
            if quantbits > 1:
                ps = mp.Process(target = Quant_BbitFlipping, args = (local_parameters, P_bitflip, quantbits, round_idx, client, dict_param, dict_berfer,))
            elif quantbits == 1:
                ps = mp.Process(target = Quant_1bitFlipping, args = (local_parameters, P_bitflip, quantbits, round_idx, client, dict_param, dict_berfer,))

            jobs.append(ps)
            ps.start()

        for p in jobs:
            p.join()
        # print("\n")
        for clent, berfer in dict_berfer.items():
            print(f"    CommRound {round_idx}: {clent} {weigh_dict[clent]}, ber = {berfer['ber']:.10f} ")
            # print(f"CommRound {round_idx}: {clent}, ber = {berfer['ber']:.8f}, fer = {berfer['fer']:.8f}, a_iter = {berfer['ave_iter']:.8f}")
        ##================================ 串行训练，并行通信传输结束 ========================================
        ## 对所有的Client返回的参数累加（最后取平均值）
        # for key, param in dict_param.items():
            # print(f"{key} :")
        for clent, param_client in dict_param.items():
            # print(f"{clent} : {param_client.keys()}")
            for key, params in server.global_model.state_dict().items():
                # print(f"{param_client.keys()}")
                # sum_parameters[key].add_(local_parameters[key])
                if key in param_client:
                    sum_parameters[key].add_((torch.tensor(param_client[key], dtype = torch.float32) * weigh_dict[clent]).to(args.device))

        global_parameters = server.model_aggregate(sum_parameters, weight)
        ## 训练结束之后，Server端进行测试集来验证方法的泛化性，
        acc, evl_loss = server.model_eval()
        ##==================================================================================================
        ##                          学习率递减, loss平均, 日志
        ##==================================================================================================
        ## 优化器学习率调整
        # optim.schedule()
        epochLos = loss_func.avg()
        recorder.assign([lr, acc, epochLos, evl_loss, comm_cost, quantbits])

        print(f"  Accuracy: {acc}, lr = {lr}, {quantbits} bits, train_loss = {epochLos:.3f}, evl_loss = {evl_loss:.3f}" )
        # ckp.write_log(f"    lr = {lr}, loss={epochLos:.3f}, Accuracy={acc:.3f}, evl_loss = {evl_loss:.3f}", train = True)
        # recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'Accuracy', 'train loss', 'val loss'])
        if (round_idx + 1) % 5 == 0:
            recorder.save(ckp.savedir)
            recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'Accuracy', 'train loss', 'val loss', 'com cost', 'Bits'])
    recorder.save(ckp.savedir)
    print(f"Data volume = {data_valum} (floating point number) ")
    return


if __name__=="__main__":
    main()


















