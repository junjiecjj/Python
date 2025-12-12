
# -*- coding: utf-8 -*-
"""
Created on 2024/08/15

@author: Junjie Chen

"""


import numpy as np
import torch


# 初始化随机数种子
def set_random_seed(seed = 42,):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

def set_printoption(precision = 3):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    torch.set_printoptions(
        precision = precision,    # 精度，保留小数点后几位，默认4
        threshold = 1000,
        edgeitems = 3,
        linewidth = 150,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile = None,
        sci_mode = False  # 用科学技术法显示数据，默认True
    )




def BitAcc(acc):
    if acc <= 0.8:
        B = 8
    elif 0.8 < acc <= 0.9:
        B = 6
    elif 0.9 < B <= 0.95:
        B = 4
    elif 0.95 < B <= 1:
        B = 1
    return B
































































































































































