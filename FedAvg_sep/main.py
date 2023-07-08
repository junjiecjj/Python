# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

这是客户端和服务器分别有完全独立的全局模型，且所有的客户端有独立的优化器和损失函数，且兼容传输模型和传输模型差值两种模式的程序，
对于传输模型差值：
    因为共享一个模型，所以：
    (1) 客户端在每轮通信之前需要把最新的全局模型加载；且客户端返回的是自身训练后的模型参数减去这轮初始时的全局模型得到的差值；
    (2) 因为每个客户端在训练的时候是在当前global全局模型加载后训练的，且因为是共享的全局模型，每个客户端训练完后，全局模型都会改变，因此服务器需要在更新全局模型之前把上次的全局模型先加载一遍，在将这轮通信得到的所有客户端模型参数差值的平均值加到全局模型上；

对于直接传输模型：
    (1) 客户端在每轮通信之前需要把最新的全局模型加载；且客户端返回的是自身训练后的模型参数；
    (2) 服务端直接把客户端返回的模型参数做个平均，加载到全局模型；

"""

from tqdm import tqdm
import numpy as np
import torch


## 以下是本项目自己编写的库
## checkpoint
import Utility

from clients import ClientsGroup

from server import Server

## 参数
from config import args


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
    recorder = MetricsLog.TraRecorder(3, name = "Train", )
    ## 创建 Clients 群
    myClients = ClientsGroup(args.model_name, args.dir_minst, args )
    testDataLoader = myClients.test_data_loader

    ## 创建 server
    server = Server(args, testDataLoader, )

    ##  选取的 Clients 数量
    num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))

    ## 得到全局的参数
    global_parameters = {}
    ## 得到每一层中全连接层中的名称fc1.weight 以及权重weights(tenor)
    for key, var in server.global_model.state_dict().items():
        global_parameters[key] = var.clone()

    ##==================================================================================================
    ##                     核心代码
    ##==================================================================================================
    ## num_comm 表示通信次数，
    for round_idx in range(args.num_comm):
        recorder.addlog(round_idx)
        print(f"Communicate round {round_idx + 1} / {args.num_comm} : ")
        ckp.write_log(f"Communicate round {round_idx + 1} / {args.num_comm} : ", train=True)

        ## 从100个客户端随机选取10个
        candidates = ['client{}'.format(i) for i in np.random.choice(range(args.num_of_clients), num_in_comm, replace = False)]
        # candidates = np.random.choice(list(myClients.clients_set.keys()), num_in_comm, replace = False)

        sum_parameters = {}
        for name, params in server.global_model.state_dict().items():
            sum_parameters[name] = torch.zeros_like(params)

        for client in tqdm(candidates):
            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, global_parameters, )   #lossFun = loss_func, optim = optim
            for var in sum_parameters:
                sum_parameters[var].add_(local_parameters[var])
        for var in global_parameters:
            sum_parameters[var] = (sum_parameters[var] / num_in_comm)

        global_parameters = server.model_aggregate(sum_parameters)
        acc, evl_loss = server.model_eval()
        recorder.assign([ acc, evl_loss])

        print(f"    Accuracy: {acc}, evl_loss = {evl_loss:.3f}" )
        recorder.plot_inonefig(ckp.savedir, metric_str = [ 'Accuracy', 'val loss'])
    recorder.save(ckp.savedir)
    return


if __name__=="__main__":
    main()


















