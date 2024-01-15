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
    def __init__(self,  ):


        return

    def localUpdate(self,   ):

        return



class ClientsGroup(object):

    def __init__(self, model, data_root, args = None,  ):

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


















