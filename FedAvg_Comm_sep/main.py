# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""

from tqdm import tqdm
import numpy as np
import torch


# 以下是本项目自己编写的库
# checkpoint
import Utility

from clients import ClientsGroup

from server import Server

# 参数
from config import args




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
    ## 创建 Clients 群
    myClients = ClientsGroup(args.model_name, args.dir_minst, args.dataset, args.isIID, args.num_of_clients, args.device, args.test_batchsize, )
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

    ## num_comm 表示通信次数，此处设置为1k, 通讯次数一共1000次
    for round_idx in range(args.num_comm):
        # loss_func.addlog()
        round_loss = 0.0

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

        for client in tqdm(candidates):
            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, global_parameters )   #lossFun = loss_func, optim = optim
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        server.model_aggregate(global_parameters)
        acc, evl_loss = server.model_eval()

        print(f"    Accuracy: {acc}, train_loss = {round_loss:.3f}, evl_loss = {evl_loss:.3f}" )

    return


if __name__=="__main__":
    main()


















