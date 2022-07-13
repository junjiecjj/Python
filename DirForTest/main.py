#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 21:43:26 2022

@author: jack
"""

from importlib import import_module

# import model

# from model.common import  Upsampler

import model.common as com
com.default_conv(1233)

from model import ipt
ipt.IPTT()


# from model import common


# ipt1 = import_module('model.ipt')


import sys
sys.path.append('/home/jack/公共的/Python/DirForTest/')
import test_script
test_script.func()

CLASS = test_script.TestClass()
CLASS.func()


import test_package



test_package.pack1.mod1.pack1_func_1()


test_package.pack1.mod2.pack1_func_2()




