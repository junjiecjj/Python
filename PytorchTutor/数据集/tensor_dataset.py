#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# class TensorDataset(Dataset[Tuple[Tensor, ...]]):
#     ## Dataset wrapping tensors.
#     ## Each sample will be retrieved by indexing tensors along the first dimension.
#     ## Args:
#     ##     *tensors (Tensor): tensors that have the same size of the first dimension.

#     tensors: Tuple[Tensor, ...]
#     def __init__(self, *tensors: Tensor) -> None:
#         assert all(tensors[0].size(0) == tensor.size(0)
#                    for tensor in tensors), "Size mismatch between tensors"
#         self.tensors = tensors

#     def __getitem__(self, index):
#         return tuple(tensor[index] for tensor in self.tensors)

#     def __len__(self):
#         return self.tensors[0].size(0)

"""
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

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
a = torch.arange(36).reshape(12, 3)

b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
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


































