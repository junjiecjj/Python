




import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet


class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        '''
            localEpoch 当前Client的迭代次数
            localBatchSize 当前Client的batchsize大小
            Net Server共享的模型
            LossFun 损失函数
            opti 优化函数
            global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        # 加载当前通信中最新全局参数
        # 传入网络模型，并加载global_parameters参数的
        Net.load_state_dict(global_parameters, strict=True)
        # 载入Client自有数据集
        # 加载本地数据
        self.train_dl = DataLoader(self.train_ds, batch_size = localBatchSize, shuffle = True)
        # 设置迭代次数
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # 计算损失函数
                loss = lossFun(preds, label)
                # 将梯度归零，初始化梯度
                opti.zero_grad()
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        return Net.state_dict()

    def local_val(self):
        return

class ClientsGroup(object):
    #  dataSetName 数据集的名称, isIID 是否是IID, numOfClients 客户端的数量, dev 设备(GPU), clients_set 客户端
    def __init__(self, dataSetName = 'MNIST', isIID = False, numOfClients = 100, dev = None):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.dataSetAllocation()
        return

    def dataSetAllocation(self):
        # 得到已经被重新分配的数据
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        # test_data  = torch.tensor(mnistDataSet.test_data)
        # test_label =  torch.tensor(mnistDataSet.test_label)

        test_data  =  mnistDataSet.test_data
        test_label =  mnistDataSet.test_label
        # print(f"0: {test_data.shape}, {test_label.shape}")
        # 加载测试数据
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size = 128, shuffle = False)

        train_data  = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        # print(f"1: {train_data.shape}, {train_label.shape}")

        ''' 然后将其划分为200组大小为300的数据切片,然后分给每个Client两个切片 '''
        # 60000 / 100 / 2 = 600/2 = 300
        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        # print("shard_size:"+str(shard_size))

        # np.random.permutation 将序列进行随机排序
        # np.random.permutation(60000//300=200)
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        # 一共200个
        print("*" * 100)
        print("客户端数据索引随机打乱:")
        print(f"{shards_id}, {shards_id.shape}")
        print("*" * 100)
        for i in range(self.num_of_clients):
            ## shards_id1, shards_id2 是所有被分得的两块数据切片
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]

            ## 例如 shard_id1 = 10, 10 * 300 : 10*300+300
            ## 将数据以及的标签分配给该客户端
            data_shards1  = train_data[shards_id1 * shard_size : (shards_id1 + 1) * shard_size ]
            data_shards2  = train_data[shards_id2 * shard_size : (shards_id2 + 1) * shard_size ]
            label_shards1 = train_label[shards_id1 * shard_size: (shards_id1 + 1) * shard_size ]
            label_shards2 = train_label[shards_id2 * shard_size: (shards_id2 + 1) * shard_size ]

            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.hstack((label_shards1, label_shards2))
            local_data, local_label = torch.cat([data_shards1, data_shards2], axis = 0 ), torch.cat([label_shards1, label_shards2], axis = 0 )
            # print(f"2: {local_data.shape}, {local_label.shape}")

            # 创建一个客户端
            # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            someone = client(TensorDataset( local_data, local_label), self.dev)
            # 为每一个clients 设置一个名字
            self.clients_set['client{}'.format(i)] = someone
        return

# # if __name__=="__main__":
# MyClients = ClientsGroup('mnist', False, 100, 0)

# print(MyClients.clients_set['client10'].train_ds[0:10])

# print(MyClients.clients_set['client11'].train_ds[400:500])

































