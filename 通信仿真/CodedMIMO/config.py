#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:57:24 2023
@author: Junjie Chen
"""


import argparse
import socket, getpass , os
# import numpy as np

# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
user_home = os.path.expanduser('~')
home = os.path.expanduser('~')


Args = {
"minimum_snr" : 0,
"maximum_snr" : 20 ,
"increment_snr" : 0.5,
"maximum_error_number" : 1000,
"maximum_block_number" : 1000000,

"Nt" : 4,
"Nr" : 6,
"Ncl" : 4,
"Nray" : 6,

"M" : 16,


"P" : 1,
"d" : 2,

## others
"home" : home,
"smallprob": 1e-15
}



args = argparse.Namespace(**Args)
