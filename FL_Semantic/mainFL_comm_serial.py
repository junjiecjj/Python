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


# 以下是本项目自己编写的库
# checkpoint

from pipeline_serial import Quant_Bbit_Pipe
from pipeline_serial import Quant_1bit_Pipe
import Utility
from clients import ClientsGroup
from server import Server

# 参数
from config import args

# 损失函数
from loss.Loss import myLoss

# 优化器
import Optimizer

import MetricsLog

from models  import AutoEncoder #AED_cnn_mnist
from models import  LeNet #LeNet_3
#==================================================  seed =====================================================

Utility.set_random_seed(args.seed, deterministic = True, benchmark = True)
Utility.set_printoption(5)


ckp = Utility.checkpoint(args)
print(f"lr = {args.learning_rate}")


#==========================================  主函数, 一个压缩率和一个信噪比下训练并多信噪比下测试, 传输时精确浮点数, 串行  ======================================
def FL_Sem_1R_1SNR_NoQuant():
    recorder = MetricsLog.TraRecorder(7, name = "Train", )
    # testrecoder  = MetricsLog.TesRecorder(Len = 4)
    raw_dim       = 28 * 28
    comrate       = 0.5
    Snr           = 10
    encoded_dim = int(raw_dim * comrate)
    ## 初始化模型， 因为后面的 server 和 ClientsGroup 都是共享一个net，所以会有一些处理比较麻烦
    net =  AutoEncoder.AED_cnn_mnist(encoded_space_dim = encoded_dim, snr = Snr ).to(args.device)
    data_valum = 0.0
    for key, var in net.state_dict().items():
        data_valum += var.numel()
        # print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}  " )
    print(f"sum data volum = {data_valum}")
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

    ## 加载预训练的分类器;
    classifier =  LeNet.LeNet_3().to(args.device)
    pretrained_model = f"/home/{args.user_name}/FL_semantic/LeNet_model/LeNet_Minst_classifier.pt"
    classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device ))

    ##============================= 完成以上准备工作 ================================#
    ##  选取的 Clients 数量
    num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))
    ##==================================================================================================
    ##                                核心代码
    ##==================================================================================================
    ## num_comm 表示通信次数
    for round_idx in range(args.num_comm):
        loss_func.addlog()

        recorder.addlog(round_idx)
        print(f"Communicate round : {round_idx + 1} / {args.num_comm} : ")
        ckp.write_log(f"Communicate round : {round_idx + 1} / {args.num_comm} : ", train = True)

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
        # if (round_idx ) % 1 == 0:
        round_cdf_pdf = os.path.join(ckp.cdf_pdf, f"round_{round_idx}")
        os.makedirs(round_cdf_pdf, exist_ok=True)
        ## 每个 Client 基于当前模型参数和自己的数据训练并更新模型, 返回每个Client更新后的参数
        for client_idx, client in  enumerate(candidates):
            client_dtsize = myClients.clients_set[client].datasize
            weight.append(client_dtsize)
            weigh_dict[client] = client_dtsize
            # print(f"1 acc_count = {acc_count}")
            ## 获取当前Client训练得到的参数
            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, loss_func, optim, global_parameters, )
            # print(f"2 acc_count = {acc_count}")
            if  client_idx == 1:
                Utility.localUpdateCDF(round_idx, client, local_parameters, round_cdf_pdf)

            ## 对所有的Client返回的参数累加（最后取平均值）
            for key, params in server.global_model.state_dict().items():
                # sum_parameters[key].add_(local_parameters[key])
                if key in local_parameters:
                    sum_parameters[key].add_((local_parameters[key].clone() * weigh_dict[client])) #.to(args.device))
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

        print(f"    PSNR: {psnr01:.3f}/{psnr:.3f}, acc = {acc:.3f}, lr = {lr}, train_loss = {epochLos:.3f}, evl_loss = {val_mse:.3f}" )
        # ckp.write_log(f"    lr = {lr}, loss={epochLos:.3f}, acc = {acc:.3f}, psnr={psnr:.3f}/{psnr01:.3f}, evl_loss = {val_mse:.3f}", train=True)
        recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'acc', 'psnr01', 'psnr', 'val_mse', 'epochLos'])
        # if (round_idx + 1) % 100 == 0:
            # recorder.save(ckp.savedir)

    # server.R_SNR_testdata(ckp, testrecoder, classifier, comrate, Snr, SNRlist = args.SNRtest )
    # server.R_SNR_plotval(ckp.testResdir, classifier, trainR = comrate, tra_snr = Snr, snrlist = args.SNRtest)

    return



##=========================================  主函数, 多压缩率和信噪比下训练并多信噪比下测试, 传输时精确浮点数, 串行 ==========================================
def FL_Sem_R_SNR_NotQuant():
    raw_dim       = 28 * 28
    testrecoder  = MetricsLog.TesRecorder(Len = 4)
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
                ## 每个 Client 基于当前模型参数和自己的数据训练并更新模型, 返回每个Client更新后的参数
                for client in  candidates:
                    client_dtsize = myClients.clients_set[client].datasize
                    weight.append(client_dtsize)
                    weigh_dict[client] = client_dtsize
                    # print(f"1 acc_count = {acc_count}")
                    ## 获取当前Client训练得到的参数
                    local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, loss_func, optim, global_parameters, )
                    # print(f"2 acc_count = {acc_count}")
                    ## 对所有的Client返回的参数累加（最后取平均值）
                    for key, params in server.global_model.state_dict().items():
                        # sum_parameters[key].add_(local_parameters[key])
                        if key in local_parameters:
                            sum_parameters[key].add_((local_parameters[key].clone() * weigh_dict[client])) #.to(args.device))

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

#==========================================  主函数, 多压缩率和信噪比下训练并多信噪比下测试, 传输时量化, 串行  =====================================
def FL_Sem_R_SNR_Quant():
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
            ## 初始化模型，
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
            ## 损失函数
            loss_func = myLoss(args)
            ## 优化算法，
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
            QBits = 4
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
                ## 每个 Client 基于当前模型参数和自己的数据训练并更新模型, 返回每个Client更新后的参数
                for client in  candidates:
                    client_dtsize = myClients.clients_set[client].datasize
                    weight.append(client_dtsize)
                    weigh_dict[client] = client_dtsize
                    # print(f"1 acc_count = {acc_count}")
                    ## 获取当前Client训练得到的参数
                    local_parameters  = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, loss_func, optim, global_parameters, )
                    # print(f"2 acc_count = {acc_count}")

                    ## 转为 np
                    for key in local_parameters:
                        local_parameters[key] = np.array(local_parameters[key].detach().cpu())

                    ## 量化、 反量化;
                    if QBits > 1:
                        local_parameters = Quant_Bbit_Pipe(com_round = 1, client = client, param_W = local_parameters, err_rate = 0, quantBits = QBits, )
                    elif QBits == 1:
                        local_parameters = Quant_1bit_Pipe(com_round = 1, client = client, param_W = local_parameters, err_rate = 0, quantBits = QBits, )
                    ## 转为 tensor
                    local_parameters_torch = {}
                    for key, val in local_parameters.items():
                        # print(f"{val.dtype}")
                        local_parameters_torch[key] = torch.tensor(val).to(args.device)
                        # print(f"{local_parameters_torch[key].dtype}")

                    # print(f"  {client}: ber = {ber}, fer = {fer}, avg_iter = {avg_iter}")
                    ##  参数聚合
                    for key, params in server.global_model.state_dict().items():
                        # sum_parameters[key].add_(local_parameters[key])
                        if key in local_parameters_torch:
                            # print(f"{local_parameters_torch[key].dtype}, {sum_parameters[key].dtype}")
                            # if params.dtype != torch.int64:
                            sum_parameters[key].add_(local_parameters_torch[key].clone() * weigh_dict[client]) #.to(args.device))
                            # else:
                                # sum_parameters[key].add_(local_parameters[key].clone())
                # for key, params in  sum_parameters.items():
                #     if params.dtype != torch.float32:
                #         print(f"      sum:  {key}, {params},         param.dtype = {params.dtype} ")
                ## 平均
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

                print(f"   PSNR: {psnr01:.3f}/{psnr:.3f}, acc = {acc}, lr = {lr}, train_loss = {epochLos:.3f}, evl_loss = {val_mse:.3f}" )
                # ckp.write_log(f"  lr = {lr}, loss={epochLos:.3f}, acc = {acc:.3f}, psnr={psnr:.3f}/{psnr01:.3f}, evl_loss = {val_mse:.3f}", train=True)
                recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'acc', 'psnr01', 'psnr', 'val_mse', 'epochLos'])
                if (round_idx + 1) % 100 == 0:
                    recorder.save(ckp.savedir)
            server.R_SNR_testdata(ckp, testrecoder, classifier, comrate, Snr, SNRlist = args.SNRtest )
            server.R_SNR_plotval(ckp.testResdir, classifier, trainR = comrate, tra_snr = Snr, snrlist = args.SNRtest)
    return


FL_Sem_1R_1SNR_NoQuant()
# FL_Sem_1R_1SNR_Quant()
# FL_Sem_R_SNR_NotQuant()
# FL_Sem_R_SNR_Quant()
















