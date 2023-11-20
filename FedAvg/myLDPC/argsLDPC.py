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


ldpc_args = {

"minimum_snr" : 3 ,
"maximum_snr" : 4 ,
"increment_snr" : 0.5,
"maximum_error_number" : 500,
"maximum_block_number" : 100000000,

## LDPC***0***PARAMETERS
"max_iteration" : 50,
"encoder_active" : 1,
"file_name_of_the_H" : "PEG1024regular0.5.txt",


## others
"home" : home,
"smallprob": 1e-15
}




arg = argparse.Namespace(**ldpc_args)
