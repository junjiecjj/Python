# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved




# # 如果ipt.py不是运行入口文件
# from .common import default_conv
# def IPTT():
#       default_conv(121113222222222222)
# IPTT()

# or

# 如果ipt.py不是运行入口文件
from . import common
def IPTT():
      common.default_conv(121113222222222222)
IPTT()

# # # 如果ipt.py是运行入口文件
# from common import default_conv
# def IPTT():
#      default_conv(121113222222222222)
# IPTT()

# # 如果ipt.py不是运行入口文件
# import common
# def IPTT():
#       common.default_conv(121113222222222222)
# IPTT()