# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""






import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# from model import get_model
from data.getData import GetDataSet
# 优化器
# import Optimizer

class client(object):
    def __init__(self, model, trainDataSet, args, client_name = "client10"):
        self.args             = args
        self.train_ds         = trainDataSet
        self.device           = args.device
        # self.local_model      = get_model(local_modelname).to(self.device)
        self.client_name      = client_name
        self.train_dataloader = None
        self.local_parameters = None
        self.local_model      = model
        if args.Random_Mask == True:
            self.mask = {}
            # for name, param in self.local_model.state_dict().items():
            #     p = torch.ones_like(param)*args.prop
            #     if torch.is_floating_point(param):
            #         self.mask[name] = torch.bernoulli(p)
            #     else:
            #         self.mask[name] = torch.bernoulli(p).long()
        # for key, var in self.mask.items():
            # print(f"0:  {key}: {var}")
        # print(f"mask = {self.mask['conv1.bias']}")
        return

    def localUpdate(self, localEpoch, localBatchSize, lossFun, opti, global_parameters, ):
        ## localEpoch:         当前Client的迭代次数
        ## localBatchSize:      当前Client的batchsize大小
        ## Net Server:        共享的模型
        ## LossFun:          损失函数
        ## opti:             优化函数
        ## global_parmeters:   当前通讯中最全局参数
        ## return:           返回当前Client基于自己的数据训练得到的新的模型参数
        # for key, var in global_parameters.items():
            # print(f"0:  {key}: {var.min():.3f}, {var.max():.3f}, {var.mean():.3f}")
        ## 加载当前通信中最新全局参数, 传入网络模型，并加载global_parameters参数的
        ## 1
        self.local_model.load_state_dict(global_parameters, strict=True)
        ## 2
        # for name, param in global_parameters.items():
            # self.local_model.state_dict()[name].copy_(param.clone())
        self.local_model.train()

        ## local 差分隐私: Local DP-SGD
        if self.args.LDP == True:
            lossFun = torch.nn.CrossEntropyLoss(reduction='none')
            idx = np.random.choice(range(len(self.train_ds)), round(len(self.train_ds)*self.args.q),  replace=False)
            sampled_dataset = TensorDataset(self.train_ds[idx][0], self.train_ds[idx][1])
            # print(f"{self.client_name}: {len(sampled_dataset)}")
            self.train_dataloader = DataLoader(sampled_dataset, batch_size = localBatchSize, shuffle = True)
            for epoch in range(self.args.loc_epochs):
                opti.zero_grad()       ## 必须在反向传播前先清零。
                ## 初始化记录裁剪和添加噪声的容器
                clipped_grads = {}
                for key, param in self.local_model.named_parameters():
                    clipped_grads[key] = torch.zeros_like(param)

                for batch, (X, y) in enumerate(self.train_dataloader):
                    # model.zero_grad()
                    X, y =  X.to(self.device), y.to(self.device)
                    y_hat = self.local_model(X)
                    losses = lossFun(y_hat, y)
                    for los in losses:
                        los.backward(retain_graph=True)
                        # 裁剪梯度，C 为边界值，使得模型参数梯度在 [-C,C] 范围内
                        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.args.clip)
                        # 存储裁剪后的梯度
                        for key, param in self.local_model.named_parameters():
                            clipped_grads[key].add_(param.grad)
                        self.local_model.zero_grad()
                for key, param in self.local_model.named_parameters():
                    # 初始化噪声
                    noise = torch.normal(mean = 0, std = self.args.sigma * self.args.clip, size = param.shape ).to(self.device)
                    # 添加高斯噪声
                    clipped_grads[key].add_(noise)
                    param.grad = clipped_grads[key] / round(len(self.train_ds)*self.args.q)
                opti.step()
        else:
            ## 载入Client自有数据集, 加载本地数据
            self.train_dataloader = DataLoader(self.train_ds, batch_size = localBatchSize, shuffle = True)
            ## 设置迭代次数
            for epoch in range(localEpoch):
                for data, label in self.train_dataloader:
                    # 加载到GPU上
                    data, label = data.to(self.device), label.to(self.device)
                    # 模型上传入数据
                    preds = self.local_model(data)
                    # 计算损失函数
                    loss = lossFun(preds, label)
                    # 将梯度归零，初始化梯度
                    opti.zero_grad()
                    # 反向传播
                    loss.backward()
                    # 计算梯度，并更新梯度
                    opti.step()

        ## 如果传输的是模型参数差值
        if self.args.transmitted_diff:
            local_update = {}
            for key, var in self.local_model.state_dict().items():
                local_update[key] = var - global_parameters[key]
        else: ## 直接传递模型参数
            local_update = {}
            for key, var in self.local_model.state_dict().items():
                local_update[key] = var.clone()
            # 返回当前Client基于自己的数据训练得到的新的模型参数,  返回 self.local_model.state_dict() 或 local_parms都可以。

        ## 随机掩码
        if self.args.Random_Mask == True:
            # print("随机掩码")
            self.mask = {}
            for key, param in local_update.items():
                p = torch.ones_like(param) * self.args.prop
                self.mask[key] = torch.bernoulli(p)
                local_update[key].mul_(self.mask[key])
        ## 模型压缩
        if self.args.Compression == True:
            # print(f"{len(local_update)}")
            # print("模型压缩")
            local_update = sorted(local_update.items(), key = lambda item:abs(torch.mean(item[1].float())), reverse=True)
            ret_size = int(self.args.crate * len(local_update))
            # print(f"{ret_size}")
            local_update =  dict(local_update[:ret_size])

        return  local_update  # self.local_model.state_dict() #  local_parms


'''
创建Clients群100个
得到Mnist数据,一共有60000个样本
100个客户端
IID：
    我们首先将数据集打乱(GetDataSet中实现)，然后为每个Client分配 600 个样本。
Non-IID：
    我们首先根据数据标签将数据集排序(GetDataSet中实现), 即 MNIST 中的数字大小，
    然后将其划分为 200 组大小为 300 的数据切片，然后分给每个 Client 两个切片。
'''
class ClientsGroup(object):
    ##  dataSetName 数据集的名称, isIID 是否是IID, numOfClients 客户端的数量, dev 设备(GPU), clients_set 客户端
    ## data_root, dataSetName = 'MNIST',  isIID = False, numOfClients = 100, device = None, test_batsize = 128
    def __init__(self, model, data_root, args = None):
        # args.dir_minst, args.dataset, args.isIID, args.num_of_clients, args.device, args.test_batchsize
        self.local_model       = model
        self.data_root         = data_root
        self.dataset_name      = args.dataset
        self.is_iid            = args.isIID
        self.num_of_clients    = args.num_of_clients
        self.device            = args.device
        self.clients_set       = {}
        self.test_data_loader  = None
        self.test_bs           = args.test_batchsize
        self.args              = args
        self.dataSetAllocation()
        return

    def dataSetAllocation(self):
        # 得到已经被重新分配的数据
        mnistDataSet = GetDataSet(self.dataset_name, self.is_iid, self.data_root)

        test_data    =  mnistDataSet.test_data
        test_label   =  mnistDataSet.test_label

        # 加载测试数据
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size = self.test_bs, shuffle = False)

        train_data  = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        # print(f"1: {train_data.shape}, {train_label.shape}")

        ''' 然后将其划分为200组大小为300的数据切片,然后分给每个Client两个切片 '''
        # 60000 / 100 / 2 = 600/2 = 300
        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2

        # 将序列进行随机排序
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)

        # print("*" * 100)
        # print("客户端数据索引随机打乱:")
        # print(f"{shards_id}, {shards_id.shape}")
        # print("*" * 100)
        for i in range(self.num_of_clients):
            ## shards_id1, shards_id2 是所有被分得的两块数据切片
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]

            ## 将数据以及的标签分配给该客户端
            data_shards1  = train_data[shards_id1 * shard_size : (shards_id1 + 1) * shard_size ]
            data_shards2  = train_data[shards_id2 * shard_size : (shards_id2 + 1) * shard_size ]
            label_shards1 = train_label[shards_id1 * shard_size: (shards_id1 + 1) * shard_size ]
            label_shards2 = train_label[shards_id2 * shard_size: (shards_id2 + 1) * shard_size ]

            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.hstack((label_shards1, label_shards2))
            local_data, local_label = torch.cat([data_shards1, data_shards2], axis = 0 ), torch.cat([label_shards1, label_shards2], axis = 0 )

            # 创建一个客户端
            someone = client(self.local_model, TensorDataset(local_data, local_label), self.args, 'client{}'.format(i))
            # 为每一个clients 设置一个名字
            self.clients_set['client{}'.format(i)] = someone
        return

# # if __name__=="__main__":
# MyClients = ClientsGroup('mnist', False, 100, 0)

# print(MyClients.clients_set['client10'].train_ds[0:10])

# print(MyClients.clients_set['client11'].train_ds[400:500])

































