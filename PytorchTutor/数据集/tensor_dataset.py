#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""

class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    ## Dataset wrapping tensors.
    ## Each sample will be retrieved by indexing tensors along the first dimension.
    ## Args:
    ##     *tensors (Tensor): tensors that have the same size of the first dimension.

    tensors: Tuple[Tensor, ...]
    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

顾名思义，torch.utils.data 中的 TensorDataset 基于一系列张量构建数据集。这些张量的形状可以不尽相同，但第一个维度必须具有相同大小，这是为了保证在使用 DataLoader 时可以正常地返回一个批量的数据。

*tensors 告诉我们实例化 TensorDataset 时传入的是一系列张量，即：

dataset = TensorDataset(tensor_1, tensor_2, ..., tensor_n)

随后的 assert 是用来确保传入的这些张量中，每个张量在第一个维度的大小都等于第一个张量在第一个维度的大小，即要求所有张量在第一个维度的大小都相同。
__getitem__ 方法返回的结果等价于

return tensor_1[index], tensor_2[index], ..., tensor_n[index]
从这行代码可以看出，如果 n nn 个张量在第一个维度的大小不完全相同，则必然会有一个张量出现 IndexError。确保第一个维度大小相同也是为了之后传入 DataLoader 中能够正常地以一个批量的形式加载。
__len__ 就不用多说了，因为所有张量的第一个维度大小都相同，所以直接返回传入的第一个张量在第一个维度的大小即可。
TensorDataset 将张量的第一个维度视为数据集大小的维度，数据集在传入 DataLoader 后，该维度也是 batch_size 所在的维度


注意：TensorDataset 中的参数必须是 tensor
"""

from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader

# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
a = torch.arange(36).reshape(12, 3)

# b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
b = torch.arange(44, 44+12)
train_ids = TensorDataset(a, b)
# 切片输出
print(train_ids[0:2])
print('=' * 80)
# 循环取数据
for x_train, y_label in train_ids:
    print(x_train, y_label)
# DataLoader进行数据封装
print('=' * 80)
train_loader = DataLoader(dataset=train_ids, batch_size = 4, shuffle = True)
for e in range(2):
    for i, data in enumerate(train_loader):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
        x_data, label = data
        print(f"batch {i}: \n  x_data:{x_data}\n  label: {label}")



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader

a = torch.arange(48).reshape(12, 2, 2)
b = torch.arange(44, 44+12)
train_ids = TensorDataset(a, b)
# 切片输出
print(train_ids[0:2])
print('=' * 80)
# 循环取数据
for x_train, y_label in train_ids:
    print(x_train, y_label)
# DataLoader进行数据封装
print('=' * 80)
train_loader = DataLoader(dataset=train_ids, batch_size=4, shuffle=True)
for i, data in enumerate(train_loader):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    x_data, label = data
    print(f"batch {i}: \n  x_data:{x_data}\n  label: {label}")

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""

class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    ## Dataset wrapping tensors.
    ## Each sample will be retrieved by indexing tensors along the first dimension.
    ## Args:
    ##     *tensors (Tensor): tensors that have the same size of the first dimension.

    tensors: Tuple[Tensor, ...]
    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

顾名思义，torch.utils.data 中的 TensorDataset 基于一系列张量构建数据集。这些张量的形状可以不尽相同，但第一个维度必须具有相同大小，这是为了保证在使用 DataLoader 时可以正常地返回一个批量的数据。

*tensors 告诉我们实例化 TensorDataset 时传入的是一系列张量，即：

dataset = TensorDataset(tensor_1, tensor_2, ..., tensor_n)

随后的 assert 是用来确保传入的这些张量中，每个张量在第一个维度的大小都等于第一个张量在第一个维度的大小，即要求所有张量在第一个维度的大小都相同。
__getitem__ 方法返回的结果等价于

return tensor_1[index], tensor_2[index], ..., tensor_n[index]
从这行代码可以看出，如果 n nn 个张量在第一个维度的大小不完全相同，则必然会有一个张量出现 IndexError。确保第一个维度大小相同也是为了之后传入 DataLoader 中能够正常地以一个批量的形式加载。
__len__ 就不用多说了，因为所有张量的第一个维度大小都相同，所以直接返回传入的第一个张量在第一个维度的大小即可。
TensorDataset 将张量的第一个维度视为数据集大小的维度，数据集在传入 DataLoader 后，该维度也是 batch_size 所在的维度


注意：TensorDataset 中的参数必须是 tensor
"""

import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
# import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
# import sys
import torch

class DatasetSplit(TensorDataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def data_tf_mnist(x):
    x = transforms.ToTensor()(x)
    x = (x - 0.5) / 0.5
    x = x.reshape((-1, 28, 28))
    return x

batch_size = 25

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_set =  datasets.MNIST(root = '/home/jack/公共的/MLData/', train = True, download = True, transform = data_tf_mnist)
test_set =  datasets.MNIST(root = '/home/jack/公共的/MLData/', train = False, download = True, transform = data_tf_mnist)
test_iter = DataLoader(test_set, batch_size=batch_size, shuffle = False,  )
labels = np.array(train_set.targets)

num_clients = 100
## data size
train_data_size = len(train_set.data)  # 60000
test_data_size  = len(test_set.data)   # 10000

## (1) Non-IID
## 将60000个数据non-IID 分给100个客户端，按标签大小排序后，获取每个客户端分到的数据的索引, 存储在dict_users中
shard_per_user = 2
num_shards = num_clients * shard_per_user            # 200
shard_size = train_set.data.shape[0] // num_shards   # 300

sorted_idx = labels.argsort()
shards_idx = np.random.permutation(train_data_size // shard_size)
dict_users = {i: np.array([], dtype = int) for i in range(num_clients)}
for i in range (num_clients):
    idx_rand = shards_idx[i * shard_per_user : (i + 1)*shard_per_user]
    for r in idx_rand:
        dict_users[i] = np.hstack((dict_users[i], sorted_idx[r*shard_size:(r+1)*shard_size]))

## 两种根据每个客户端索引获取具体X和Y的方法.
## 方法一：
##>>>>>>>>>>>>>>>>>
train_data = ((train_set.data/255 - 0.5)/0.5).reshape(-1,1,28,28)
dict_local_dt = {}
for i in range (num_clients):
    dict_local_dt[i] = TensorDataset(train_data[dict_users[i]], torch.tensor(labels[dict_users[i]]))

## 方法二：
##>>>>>>>>>>>>>>>>>
local_dt_dict = {}
for user_id in range (num_clients):
    local_dt_dict[user_id] = DatasetSplit(train_set, dict_users[user_id])

#================
train_loader_dict = {}
for user_id in range (num_clients):
    train_loader_dict[user_id] =  DataLoader(dict_local_dt[user_id], batch_size = 12, shuffle=True)

for batch, (X, y) in enumerate(train_loader_dict[1]):
    X, y = X.to(device), y.to(device)
    print(f"X.shape = {X.shape}, y.shape = {y.shape}")   # X的每个元素都是 0 - 1的.
    if batch ==  0:
        break


## (2) IID
# shard_size = train_data_size // num_clients   # 300
# random_order = np.random.permutation(train_data_size)
# dict_users = {}
# for user_id in range(num_clients):
#     dict_users[user_id] = random_order[user_id * shard_size : (user_id + 1) * shard_size]







































