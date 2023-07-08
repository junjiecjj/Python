# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""






import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from model import get_model
from data.getData import GetDataSet


class client(object):
    def __init__(self, device, modelname, trainDataSet, client_name = "client10"):
        self.device           = device
        # self.modelname        = modelname
        self.train_ds         = trainDataSet
        self.client_name      = client_name
        self.local_model      = get_model(modelname).to(self.device)
        self.train_dataloader = None
        self.local_parameters = None
        return

    ## args.loc_epochs, args.local_batchsize,
    def localUpdate(self, localEpoch, localBatchSize, global_parameters, args):
        ## localEpoch:         当前Client的迭代次数
        ## localBatchSize:      当前Client的batchsize大小
        ## global_parmeters:   当前通讯中最全局参数
        ## return:           返回当前Client基于自己的数据训练得到的新的模型参数


        ## 加载当前通信中最新全局参数, 传入网络模型，并加载global_parameters参数的
        self.local_model.load_state_dict(global_parameters, strict = True)

        # for name, param in global_parameters.items():
            # self.local_model.state_dict()[name].copy_(param.clone())

        ## 载入Client自有数据集, 加载本地数据
        self.train_dataloader = DataLoader(self.train_ds, batch_size = localBatchSize, shuffle = True)

        ## 测试是否加载成功
        # for key, var in self.local_model.state_dict().items():
        #     diff = var - global_parameters[key]
        #     print(f"{key}: {diff.size()}, {diff.min()}, {diff.max()} " )
        lossFun = torch.nn.CrossEntropyLoss()
        optim   = torch.optim.Adam(self.local_model.parameters(), lr = 0.001, betas = (0.5, 0.999), eps = 1e-8)
        # optim  = torch.optim.SGD(self.local_model.parameters(), lr = 0.01, momentum = 0.0001 )

        self.local_model.train()
        ## 设置迭代次数
        for epoch in range(localEpoch):
            # print(f" {epoch}", end = ", ")
            for data, label in self.train_dataloader:
                data, label = data.to(self.device), label.to(self.device)
                preds = self.local_model(data)
                loss =  lossFun(preds, label)
                optim.zero_grad()
                loss.backward()
                optim.step()

        ## 如果传输的是模型参数差值
        if args.transmitted_diff:
            diff = {}
            for key, var in self.local_model.state_dict().items():
                diff[key] = var - global_parameters[key]
            # print(f"0: {global_parameters['fc2.bias']}")
            # print(f"1: {self.local_model.state_dict()['fc2.bias']}")
            # print(f"{self.client_name}: {diff['fc2.bias']}")
            # for key, var in global_parameters.items():
                # print(f"1:  {key}: {var.min():.3f}, {var.max():.3f}, {var.mean():.3f}")
            return diff
        ## 直接传递模型参数
        else:
            local_parms = {}
            for key, var in self.local_model.state_dict().items():
                local_parms[key] = var.clone()
            # 返回当前Client基于自己的数据训练得到的新的模型参数,  返回 self.local_model.state_dict() 或 local_parms都可以。
            return  local_parms  # self.local_model.state_dict() #  local_parms


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
# args.dir_minst, args.dataset, args.isIID, args.num_of_clients, args.device, args.test_batchsize, args.model_name,
class ClientsGroup(object):
    #  dataSetName 数据集的名称, isIID 是否是IID, numOfClients 客户端的数量, dev 设备(GPU), clients_set 客户端
    def __init__(self, modelname, data_root, dataSetName = 'MNIST',  isIID = False, numOfClients = 100, device = None, test_batsize = 128):
        self.modelname         = modelname
        self.data_root         = data_root
        self.dataset_name      = dataSetName
        self.is_iid            = isIID
        self.num_of_clients    = numOfClients
        self.device            = device
        self.test_bs           = test_batsize
        self.clients_set       = {}
        self.test_data_loader  = None

        self.dataSetAllocation()
        return

    def dataSetAllocation(self):
        # 得到已经被重新分配的数据
        mnistDataSet = GetDataSet(self.dataset_name, self.is_iid, self.data_root)

        test_data  =  mnistDataSet.test_data
        test_label =  mnistDataSet.test_label

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
            someone = client(self.device, self.modelname, TensorDataset(local_data, local_label), f"client{i}", )
            # 为每一个clients 设置一个名字
            self.clients_set['client{}'.format(i)] = someone
        return

# # if __name__=="__main__":
# MyClients = ClientsGroup('mnist', False, 100, 0)

# print(MyClients.clients_set['client10'].train_ds[0:10])

# print(MyClients.clients_set['client11'].train_ds[400:500])

































