#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:31:15 2022

@author: jack

https://zhuanlan.zhihu.com/p/86441879

https://www.its404.com/article/m0_37192554/105246723


https://zhuanlan.zhihu.com/p/358924078

https://zhuanlan.zhihu.com/p/501510475

https://www.cvmart.net/community/detail/5250

https://blog.csdn.net/qq_38410428/article/details/119392993?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4.pc_relevant_antiscanv2&utm_relevant_index=9

https://zhuanlan.zhihu.com/p/467103734



"""
from torch.nn.parallel.data_parallel import DataParallel
from  torch.nn   import DataParallel


import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn
from datetime import timedelta


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))




def setup(global_rank, world_size):
    # 配置Master Node的信息
    os.environ['MASTER_ADDR'] = 'XXX.XXX.XXX.XXX'
    os.environ['MASTER_PORT'] = 'XXXX'

    # 初始化Process Group
    # 关于init_method, 参数详见https://pytorch.org/docs/stable/distributed.html#initialization
    dist.init_process_group("nccl", init_method='env://', rank=global_rank, world_size=world_size, timeout=timedelta(seconds=5))

def cleanup():
    dist.destroy_process_group()




def run_demo(local_rank, args):
    # 计算global_rank和world_size
    global_rank = local_rank + args.node_rank * args.nproc_per_node
    world_size = args.nnode * args.nproc_per_node
    setup(global_rank=global_rank, world_size=world_size)
    # 设置seed
    torch.manual_seed(args.seed)

    # 创建模型, 并将其移动到local_rank对应的GPU上
    model = ToyModel().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(local_rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    print([data for data in model.parameters()])

    cleanup()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--nproc_per_node', type=int)
    parser.add_argument('--nnode', type=int)
    parser.add_argument('--node_rank', type=int)
    args = parser.parse_args()

    mp.spawn(run_demo, args=(args,), nprocs=args.nproc_per_node)










