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

# from pipeline_multiprocess import Quant_LDPC_BPSK_AWGN_equa
# from pipeline_multiprocess import Quant_LDPC_BPSK_AWGN_Pipe
from pipeline_multiprocess import Quant_BbitFlipping, Quant_1bitFlipping

# from pipeline_multiprocess import  acc2Qbits, Qbits2Lr,Qbits2Lr_1
# from pipeline_multiprocess import acc2Qbits1, err2Qbits

import Utility

from clients import ClientsGroup

from server import Server

# from model import get_model

# 参数
from config import args

# 损失函数
from loss.Loss import myLoss

# 优化器
import Optimizer

import MetricsLog

# from rdp_analysis import calibrating_sampled_gaussian

from models  import AutoEncoder #AED_cnn_mnist
from models import  LeNet #LeNet_3

#==================================================  seed =====================================================
# 设置随机数种子
Utility.set_random_seed(args.seed, deterministic = True, benchmark = True)
Utility.set_printoption(5)

#  加载 checkpoint, 如日志文件, 时间戳, 指标等
ckp = Utility.checkpoint(args)

#=================================================== main =====================================================
def main():
    ## 加载预训练的分类器;
    classifier =  LeNet.LeNet_3().to(args.device)
    # pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-01-22:20:58.pt"
    pretrained_classifier = f"/home/{args.user_name}/FL_semantic/LeNet_model/LeNet_Minst_classifier.pt"
    # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
    classifier.load_state_dict(torch.load(pretrained_classifier, map_location = args.device ))

    raw_dim    = 28 * 28
    comrate = 0.2
    Snr = 2
    print(f"压缩率:{comrate:.2f}, 信噪比:{Snr} dB")

    encoded_dim = int(raw_dim * comrate)
    args.learning_rate = 0.001

    recorder = MetricsLog.TraRecorder(7, name = "Train", compr = comrate, tra_snr = Snr )
    testrecoder  = MetricsLog.TesRecorder(Len = 4)
    ## 初始化模型， 因为后面的 server 和 ClientsGroup 都是共享一个net，所以会有一些处理比较麻烦
    net =  AutoEncoder.AED_cnn_mnist(encoded_space_dim = encoded_dim, snr = Snr ).to(args.device)
    data_valum = 0.0
    for key, var in net.state_dict().items():
        data_valum += var.numel()
    print(f"  sum data volum = {data_valum}")
    ## 得到全局的参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    # for key, var in global_parameters.items():
        # if key == 'encoder.encoder_cnn.3.num_batches_tracked':
            # print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} ,{var.dtype}, {var}" )
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
    P_bitflip = 0
    quantbits = 8
    ##==================================================================================================
    ##                                核心代码
    ##==================================================================================================
    ## num_comm 表示通信次数
    for round_idx in range(args.num_comm):
        loss_func.addlog()
        recorder.addlog(round_idx)
        print(f"  Communicate round : {round_idx + 1} / {args.num_comm} : ")
        # ckp.write_log(f"  Communicate round : {round_idx + 1} / {args.num_comm} : ", train = True)

        sum_parameters = {}
        for name, params in server.global_model.state_dict().items():
            sum_parameters[name] = torch.zeros(params.shape).type(params.dtype).to(args.device)
        # for key, var in sum_parameters.items():
            # print(f"2  {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} ,{var.dtype} " )
        ## 从100个客户端随机选取10个
        candidates = ['client{}'.format(i) for i in np.random.choice(range(args.num_of_clients), num_in_comm, replace = False)]
        # candidates = np.random.choice(list(myClients.clients_set.keys()), num_in_comm, replace = False)
        weight = []
        weigh_dict = {}

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
                ps = mp.Process(target = Quant_BbitFlipping, args = (round_idx, client, local_parameters, P_bitflip, quantbits, dict_param, dict_berfer, ))
            elif quantbits == 1:
                ps = mp.Process(target = Quant_1bitFlipping, args = (round_idx, client, local_parameters, P_bitflip, quantbits, dict_param, dict_berfer, ))

            jobs.append(ps)
            ps.start()

        for p in jobs:
            p.join()
        # print("\n")
        # for clent, berfer in dict_berfer.items():
            # print(f"    CommRound {round_idx}: {clent} {weigh_dict[clent]}, ber = {berfer['ber']:.10f} ")
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
                    sum_parameters[key].add_((torch.tensor(param_client[key]) * weigh_dict[clent]).to(args.device))

        global_parameters = server.model_aggregate(sum_parameters, weight)

        ## 训练结束之后，Server端进行测试集来验证方法的泛化性，
        acc, psnr01, psnr, val_mse = server.semantic_model_eval(classifier)

        ##==================================================================================================
        ##                          学习率递减, loss平均, 日志
        ##==================================================================================================
        ## 优化器学习率调整
        # optim.schedule()
        epochLos = loss_func.avg()
        recorder.assign([lr, acc, psnr01, psnr, val_mse, epochLos])

        print(f"  PSNR: {psnr01:.3f}/{psnr:.3f}, acc = {acc}, lr = {lr}, train_loss = {epochLos:.3f}, evl_loss = {val_mse:.3f}" )
        # ckp.write_log(f"  lr = {lr}, loss={epochLos:.3f}, acc = {acc:.3f}, psnr={psnr:.3f}/{psnr01:.3f}, evl_loss = {val_mse:.3f}", train=True)
        recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'acc', 'psnr01', 'psnr', 'val_mse', 'epochLos'])
        if (round_idx + 1) % 20 == 0:
            recorder.save(ckp.savedir)
    server.R_SNR_testdata(ckp, testrecoder, classifier, comrate, Snr, SNRlist = args.SNRtest )
    server.R_SNR_plotval(ckp.testResdir, classifier, trainR = comrate, tra_snr = Snr, snrlist = args.SNRtest)
    recorder.save(ckp.savedir)
    print(f"Data volume = {data_valum} (floating point number) ")
    return


#==========================================  主函数, 多压缩率和信噪比下训练并多信噪比下测试, 传输时量化, 并行  =====================================
def FL_Sem_R_SNR_m2m_Quant():
    testrecoder  = MetricsLog.TesRecorder(Len = 4)
    raw_dim    = 28 * 28
    ## 加载预训练的分类器;
    classifier =  LeNet.LeNet_3().to(args.device)
    # pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-01-22:20:58.pt"
    pretrained_classifier = f"/home/{args.user_name}/FL_semantic/LeNet_model/LeNet_Minst_classifier.pt"
    # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
    classifier.load_state_dict(torch.load(pretrained_classifier, map_location = args.device ))

    for idx_c, comrate in enumerate(args.CompRate):
        encoded_dim = int(raw_dim * comrate)
        print(f"压缩率:{comrate:.2f} ({idx_c+1}/{len(args.CompRate)})")
        for idx_s, Snr in enumerate(args.SNRtrain):
            print(f"  信噪比:{Snr} dB ({idx_s+1}/{len(args.SNRtrain)})")
            recorder = MetricsLog.TraRecorder(7, name = "Train", compr = comrate, tra_snr = Snr )
            ## 初始化模型， 因为后面的 server 和 ClientsGroup 都是共享一个net，所以会有一些处理比较麻烦
            net =  AutoEncoder.AED_cnn_mnist(encoded_space_dim = encoded_dim, snr = Snr ).to(args.device)
            data_valum = 0.0
            for key, var in net.state_dict().items():
                data_valum += var.numel()
            print(f"  sum data volum = {data_valum}")
            ## 得到全局的参数
            global_parameters = {}
            for key, var in net.state_dict().items():
                global_parameters[key] = var.clone()
            # for key, var in global_parameters.items():
                # if key == 'encoder.encoder_cnn.3.num_batches_tracked':
                    # print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} ,{var.dtype}, {var}" )
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
            P_bitflip = 0
            quantbits = 1
            ##==================================================================================================
            ##                                核心代码
            ##==================================================================================================
            ## num_comm 表示通信次数
            for round_idx in range(args.num_comm):
                loss_func.addlog()

                recorder.addlog(round_idx)
                print(f"  Communicate round : {round_idx + 1} / {args.num_comm} : ")
                # ckp.write_log(f"  Communicate round : {round_idx + 1} / {args.num_comm} : ", train = True)

                sum_parameters = {}
                for name, params in server.global_model.state_dict().items():
                    sum_parameters[name] = torch.zeros(params.shape).type(params.dtype).to(args.device)
                # for key, var in sum_parameters.items():
                    # print(f"2  {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} ,{var.dtype} " )
                ## 从100个客户端随机选取10个
                candidates = ['client{}'.format(i) for i in np.random.choice(range(args.num_of_clients), num_in_comm, replace = False)]
                # candidates = np.random.choice(list(myClients.clients_set.keys()), num_in_comm, replace = False)
                weight = []
                weigh_dict = {}

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
                        ps = mp.Process(target = Quant_BbitFlipping, args = (round_idx, client, local_parameters, P_bitflip, quantbits, dict_param, dict_berfer, ))
                    elif quantbits == 1:
                        ps = mp.Process(target = Quant_1bitFlipping, args = (round_idx, client, local_parameters, P_bitflip, quantbits, dict_param, dict_berfer, ))

                    jobs.append(ps)
                    ps.start()

                for p in jobs:
                    p.join()
                # print("\n")
                # for clent, berfer in dict_berfer.items():
                    # print(f"    CommRound {round_idx}: {clent} {weigh_dict[clent]}, ber = {berfer['ber']:.10f} ")
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
                            sum_parameters[key].add_((torch.tensor(param_client[key]) * weigh_dict[clent]).to(args.device))

                global_parameters = server.model_aggregate(sum_parameters, weight)

                ## 训练结束之后，Server端进行测试集来验证方法的泛化性，
                acc, psnr01, psnr, val_mse = server.semantic_model_eval(classifier)

                ##==================================================================================================
                ##                          学习率递减, loss平均, 日志
                ##==================================================================================================
                ## 优化器学习率调整
                # optim.schedule()
                epochLos = loss_func.avg()
                recorder.assign([lr, acc, psnr01, psnr, val_mse, epochLos])

                print(f"  PSNR: {psnr01:.3f}/{psnr:.3f}, acc = {acc}, lr = {lr}, train_loss = {epochLos:.3f}, evl_loss = {val_mse:.3f}" )
                # ckp.write_log(f"  lr = {lr}, loss={epochLos:.3f}, acc = {acc:.3f}, psnr={psnr:.3f}/{psnr01:.3f}, evl_loss = {val_mse:.3f}", train=True)
                recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'acc', 'psnr01', 'psnr', 'val_mse', 'epochLos'])
                if (round_idx + 1) % 100 == 0:
                    recorder.save(ckp.savedir)
            server.R_SNR_testdata(ckp, testrecoder, classifier, comrate, Snr, SNRlist = args.SNRtest )
            server.R_SNR_plotval(ckp.testResdir, classifier, trainR = comrate, tra_snr = Snr, snrlist = args.SNRtest)
    return



#==========================================  主函数, 多压缩率和信噪比下训练并多信噪比下测试, 传输时量化, 并行  =====================================
def FL_Sem_R_SNR_o2o_Quant():
    testrecoder  = MetricsLog.TesRecorder(Len = 4)
    raw_dim    = 28 * 28
    ## 加载预训练的分类器;
    classifier =  LeNet.LeNet_3().to(args.device)
    # pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-01-22:20:58.pt"
    pretrained_classifier = f"/home/{args.user_name}/FL_semantic/LeNet_model/LeNet_Minst_classifier.pt"
    # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
    classifier.load_state_dict(torch.load(pretrained_classifier, map_location = args.device ))

    for idx_c, (comrate, Snr) in enumerate(zip(args.CompRate, args.SNRtrain )):
        encoded_dim = int(raw_dim * comrate)
        print(f"压缩率:{comrate:.2f}, 信噪比:{Snr}(dB) ({idx_c+1}/{len(args.CompRate)})")
        # for idx_s, Snr in enumerate(args.SNRtrain):
        # print(f"  信噪比:{Snr} dB ({idx_s+1}/{len(args.SNRtrain)})")
        recorder = MetricsLog.TraRecorder(7, name = "Train", compr = comrate, tra_snr = Snr )
        ## 初始化模型， 因为后面的 server 和 ClientsGroup 都是共享一个net，所以会有一些处理比较麻烦
        net =  AutoEncoder.AED_cnn_mnist(encoded_space_dim = encoded_dim, snr = Snr ).to(args.device)
        data_valum = 0.0
        for key, var in net.state_dict().items():
            data_valum += var.numel()
        print(f"  sum data volum = {data_valum}")
        ## 得到全局的参数
        global_parameters = {}
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
        # for key, var in global_parameters.items():
            # if key == 'encoder.encoder_cnn.3.num_batches_tracked':
                # print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} ,{var.dtype}, {var}" )
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
        P_bitflip = 0
        quantbits = 8
        ##==================================================================================================
        ##                                核心代码
        ##==================================================================================================
        ## num_comm 表示通信次数
        for round_idx in range(args.num_comm):
            loss_func.addlog()

            recorder.addlog(round_idx)
            print(f"  Communicate round : {round_idx + 1} / {args.num_comm} : ")
            # ckp.write_log(f"  Communicate round : {round_idx + 1} / {args.num_comm} : ", train = True)

            sum_parameters = {}
            for name, params in server.global_model.state_dict().items():
                sum_parameters[name] = torch.zeros(params.shape).type(params.dtype).to(args.device)
            # for key, var in sum_parameters.items():
                # print(f"2  {key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} ,{var.dtype} " )
            ## 从100个客户端随机选取10个
            candidates = ['client{}'.format(i) for i in np.random.choice(range(args.num_of_clients), num_in_comm, replace = False)]
            # candidates = np.random.choice(list(myClients.clients_set.keys()), num_in_comm, replace = False)
            weight = []
            weigh_dict = {}

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
                    ps = mp.Process(target = Quant_BbitFlipping, args = (round_idx, client, local_parameters, P_bitflip, quantbits, dict_param, dict_berfer, ))
                elif quantbits == 1:
                    ps = mp.Process(target = Quant_1bitFlipping, args = (round_idx, client, local_parameters, P_bitflip, quantbits, dict_param, dict_berfer, ))
                jobs.append(ps)
                ps.start()
            for p in jobs:
                p.join()
            # print("\n")
            # for clent, berfer in dict_berfer.items():
                # print(f"    CommRound {round_idx}: {clent} {weigh_dict[clent]}, ber = {berfer['ber']:.10f} ")
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
                        sum_parameters[key].add_((torch.tensor(param_client[key]) * weigh_dict[clent]).to(args.device))
            global_parameters = server.model_aggregate(sum_parameters, weight)
            ## 训练结束之后，Server端进行测试集来验证方法的泛化性，
            acc, psnr01, psnr, val_mse = server.semantic_model_eval(classifier)
            ##==================================================================================================
            ##                          学习率递减, loss平均, 日志
            ##==================================================================================================
            ## 优化器学习率调整
            # optim.schedule()
            epochLos = loss_func.avg()
            recorder.assign([lr, acc, psnr01, psnr, val_mse, epochLos])

            print(f"  PSNR: {psnr01:.3f}/{psnr:.3f}, acc = {acc}, lr = {lr}, train_loss = {epochLos:.3f}, evl_loss = {val_mse:.3f}" )
            # ckp.write_log(f"  lr = {lr}, loss={epochLos:.3f}, acc = {acc:.3f}, psnr={psnr:.3f}/{psnr01:.3f}, evl_loss = {val_mse:.3f}", train=True)
            recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'acc', 'psnr01', 'psnr', 'val_mse', 'epochLos'])
            if (round_idx + 1) % 100 == 0:
                recorder.save(ckp.savedir)
        server.R_SNR_testdata(ckp, testrecoder, classifier, comrate, Snr, SNRlist = args.SNRtest )
        server.R_SNR_plotval(ckp.testResdir, classifier, trainR = comrate, tra_snr = Snr, snrlist = args.SNRtest)
    return


main()

# FL_Sem_R_SNR_o2o_Quant()

















