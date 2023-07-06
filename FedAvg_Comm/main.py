# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""



import os
from tqdm import tqdm
import numpy as np
import torch


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

#==================================================  seed =============================== =====================
# 设置随机数种子
Utility.set_random_seed(args.seed, deterministic = True, benchmark = True)
Utility.set_printoption(3)


#  加载 checkpoint, 如日志文件, 时间戳, 指标等
ckp = Utility.checkpoint(args)


#=================================================== main =====================================================
def main():
    recorder = MetricsLog.TraRecorder(4, name = "Train", )

    # 初始化模型
    net = get_model(args.model_name).to(args.device)

    # 定义损失函数
    loss_func = myLoss(args)
    # 优化算法，随机梯度下降法, 使用Adam下降法
    optim = Optimizer.make_optimizer(args, net, )

    ## 创建Clients群
    myClients = ClientsGroup('mnist', args.IID, args.num_of_clients, args.device)
    testDataLoader = myClients.test_data_loader
    server = Server(args, testDataLoader)

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    #  选取若干个Clients
    num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))

    # 得到全局的参数
    global_parameters = {}

    # 得到每一层中全连接层中的名称fc1.weight 以及权重weights(tenor)
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # num_comm 表示通信次数，此处设置为1k, 通讯次数一共1000次
    for round_idx in range(args.num_comm):
        loss_func.addlog()
        lr =  optim.updatelr()
        recorder.addlog(round_idx)
        print(f"Communicate round {round_idx + 1} / {args.num_comm} : ")
        ckp.write_log(f"Communicate round {round_idx + 1} / {args.num_comm} : ", train=True)

        ### 从100个客户端随机选取10个
        ## np.random.shuffle(np.arange(args.num_of_clients))
        chosen = np.random.choice(range(args.num_of_clients), num_in_comm, replace=False)
        # order = np.random.permutation(args.num_of_clients)

        # 生成10个客户端
        clients_in_comm = ['client{}'.format(i) for i in chosen]

        sum_parameters = None
        # 每个Client基于当前模型参数和自己的数据训练并更新模型, 返回每个Client更新后的参数
        for client in tqdm(clients_in_comm):
            # 获取当前Client训练得到的参数
            local_parameters = myClients.clients_set[client].localUpdate(args.loc_epochs, args.local_batchsize, net, loss_func, optim, global_parameters)
            # 对所有的Client返回的参数累加（最后取平均值）
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        # 取平均值，得到本次通信中Server得到的更新后的模型参数
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        optim.schedule()
        ## 训练结束之后，我们要通过测试集来验证方法的泛化性，注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        #  加载Server在最后得到的模型参数
        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(args.device), label.to(args.device)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().sum().item()
            num += data.shape[0]
        acc = sum_accu / num
        epochLos = loss_func.avg()
        print(f"    Accuracy: {acc}, lr = {lr}, loss={epochLos:.3f}" )
        ckp.write_log(f"    lr = {lr}, loss={epochLos:.3f}, Accuracy={acc:.3f}", train=True)
        recorder.assign([lr, epochLos, acc])

        if (round_idx + 1) % args.save_freq == 0:
            torch.save(net, os.path.join(ckp.savedir, '{}_Ncomm={}_E={}_B={}_lr={}_num_clients={}_cf={:.1f}.pt'.format(args.model_name, round_idx, args.loc_epochs, args.local_batchsize, args.learning_rate, args.num_of_clients, args.cfraction )))

        recorder.plot_inonefig(ckp.savedir, metric_str = ['lr', 'train loss', 'Accuracy'])
    recorder.save(ckp.savedir)
    return


if __name__=="__main__":
    main()


















