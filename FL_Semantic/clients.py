# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""



import numpy as np
import copy
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# from model import get_model
from data.getData import GetDataSet
# 优化器
# import Optimizer

from LDPC.quantiation import  QuantilizeBbits_torch, SR1Bit_torch, QuantilizeMean



class client(object):
    def __init__(self, model, trainDataSet, args, client_name = "client10", datasize = 0):
        self.args             = args
        self.train_ds         = trainDataSet
        self.device           = args.device
        # self.local_model      = get_model(local_modelname).to(self.device)
        self.client_name      = client_name
        self.train_dataloader = None
        # self.local_parameters   = None
        self.datasize         = datasize
        self.local_model      = model
        # self.G                = args.G

        return

    def localUpdate(self, localEpoch, localBatchSize, lossFun, opti, global_parameters, ):
        ## 1: 加载当前通信中最新全局参数, 传入网络模型，并加载global_parameters参数的
        self.local_model.load_state_dict(global_parameters, strict=True)
        # for name, param in global_parameters.items():
            # self.local_model.state_dict()[name].copy_(copy.deepcopy(param))
        # print(f"{self.client_name} 0: {global_parameters['fc2.bias']}")
        # print(f"{self.client_name} 0: {self.local_model.state_dict()['fc2.bias']}")
        self.local_model.train()

        ## 载入Client自有数据集, 加载本地数据
        self.train_dataloader = DataLoader(self.train_ds, batch_size = localBatchSize, shuffle = True, )
        ## 设置迭代次数
        if not self.args.isBalance:
            if self.datasize >= 999:
                localEpoch = self.args.moreEpoch
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
        ## print(f"{self.client_name} 1: {global_parameters['fc2.bias']}")
        ## print(f"{self.client_name} 1: {self.local_model.state_dict()['fc2.bias']}")
        local_update = {}
        ## 如果传输的是模型参数差值
        # if self.args.transmit_diff:
        for key, var in self.local_model.state_dict().items():
            local_update[key] = var - global_parameters[key]
        # else: ## 直接传递模型参数
        #     for key, var in self.local_model.state_dict().items():
        #         local_update[key] = var.clone()
            # 返回当前Client基于自己的数据训练得到的新的模型参数,  返回 self.local_model.state_dict() 或 local_parms都可以。
        # print(f"{self.client_name} 0: {local_update['fc2.bias']}")
        ## print(f"{self.client_name} 2: {local_update['fc2.bias']}")
        # if 1:
        #     print("使用量化\n")
        #     for key in local_update:
        #         # local_update[key] = QuantilizeMean(local_update[key])
        #         local_update[key] = SR1Bit_torch(local_update[key], BG = 8) ## B = self.args.OneBitB)
        #         # local_update[key] = QuantilizeBbits_torch(local_update[key], B = 4, rounding = 'nr')
        # #### print(f"{self.client_name} 3: {local_update['fc2.bias']}")
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
    def __init__(self, model, data_root, args = None,  ):
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
        if args.isBalance:
            self.dataSetAllocation_balance()
        else:
            self.dataSetAllocation_Unblance1()
        return

    def dataSetAllocation_balance(self):
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

            ##  创建一个客户端
            someone = client(self.local_model, TensorDataset(local_data, local_label), self.args, f"client{i}", datasize = local_data.shape[0])
            # 为每一个clients 设置一个名字
            self.clients_set[f"client{i}"] = someone
        return

    def dataSetAllocation_Unblance1(self):
        # 得到已经被重新分配的数据
        mnistDataSet = GetDataSet(self.dataset_name, self.is_iid, self.data_root)

        test_data    =  mnistDataSet.test_data
        test_label   =  mnistDataSet.test_label

        # 加载测试数据
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size = self.test_bs, shuffle = False)

        train_data  = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        # print(f"1: {train_data.shape}, {train_label.shape}")

        ''' 客户端0-49：每个客户端1000个数据，第50-99个客户端每个人200个数据 '''
        avg = 2 * mnistDataSet.train_data_size // self.num_of_clients
        more = self.args.more
        less = avg - more
        idx_split = [range(i*more, (i+1)*more) if i < 50 else  range(50*more + (i-50)*less,   50*more + (i+1-50)*less) for i in range(self.num_of_clients)]
        for i in range(self.num_of_clients):
            ## 将数据以及的标签分配给该客户端
            data_shards = train_data[idx_split[i]]
            label_shards = train_label[idx_split[i]]

            # 创建一个客户端
            someone = client(self.local_model, TensorDataset(data_shards, label_shards), self.args, f"client{i}", datasize = data_shards.shape[0])
            # 为每一个clients 设置一个名字
            self.clients_set[f"client{i}"] = someone
        return

    def dataSetAllocation_Unblance2(self):
        # 得到已经被重新分配的数据
        mnistDataSet = GetDataSet(self.dataset_name, self.is_iid, self.data_root)

        test_data    =  mnistDataSet.test_data
        test_label   =  mnistDataSet.test_label

        # 加载测试数据
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size = self.test_bs, shuffle = False)

        train_data  = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        # print(f"1: {train_data.shape}, {train_label.shape}")

        ''' 客户端0-49：每个客户端 1000 个数据，第50-99个客户端每个人200个数据 '''
        ''' 然后将其划分为200组大小为300的数据切片,然后分给每个Client两个切片 '''
        base_size = 100
        # 将序列进行随机排序
        total_slice = mnistDataSet.train_data_size // base_size  # 600
        Ar = np.random.permutation(total_slice)
        avg = total_slice * 2 // self.num_of_clients
        more = 10
        less = avg - more
        idx_split = [Ar[range(i*more, (i+1)*more)] if i < 50 else  Ar[range(50*more + (i-50)*less, 50*more + (i+1-50)*less)] for i in range(self.num_of_clients)]
        Ct_idx = []
        for slic in idx_split:
            tmp = []
            for s in slic:
                tmp.extend(list(range(s*base_size, (s+1)*base_size)))
            Ct_idx.append(tmp)

        for i in range(self.num_of_clients):
            ## shards_id1, shards_id2 是所有被分得的两块数据切片
            data_shards = train_data[Ct_idx[i]]
            label_shards = train_label[Ct_idx[i]]

            # 创建一个客户端
            someone = client(self.local_model, TensorDataset(data_shards, label_shards), self.args, f"client{i}", datasize = data_shards.shape[0])
            # 为每一个clients 设置一个名字
            self.clients_set[f"client{i}"] = someone
        return




# [list(range(i*10, (i+1)*10)) if i < 5 else  list(range(50 + (i-5)*2, 50 + (i-4)*2)) for i in range(10)]

# # # if __name__=="__main__":
# MyClients = ClientsGroup(1, args.dir_minst, args = args)
# ## 打印每个客户端的数据量
# for i in range(args.num_of_clients):
#     num = MyClients.clients_set[f'client{i}'].datasize
#     print(f"client{i}: {num}")

# print(MyClients.clients_set['client10'].train_ds[0:10])

# print(MyClients.clients_set['client11'].train_ds[400:500])


# https://deepinout.com/numpy/numpy-questions/812_numpy_python_numpy_split_array_into_unequal_subarrays.html

# https://geek-docs.com/numpy/numpy-ask-answer/812_numpy_python_numpy_split_array_into_unequal_subarrays.html

# https://blog.csdn.net/exsolar_521/article/details/107958839


# https://deepinout.com/numpy/numpy-questions/812_numpy_python_numpy_split_array_into_unequal_subarrays.html























