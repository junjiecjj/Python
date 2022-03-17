#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:30:45 2022

@author: jack
https://mp.weixin.qq.com/s?__biz=MzAxODI5ODMwOA==&mid=2666561612&idx=2&sn=502e4bd10cebaf4f5515bd07221aa156&chksm=80dc44e7b7abcdf166478f6315b800c06ebe88704e5efc88b18defe7915ae716c4cc50214e1f&mpshare=1&scene=24&srcid=0311geVY7fu2tyCixHSk2F0H&sharer_sharetime=1646957678144&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=AbQuqt%2BWwkRs373GJkPsdnM%3D&acctmode=0&pass_ticket=03qZsJ2pZlqB67Dci6UO5DtFdUL%2FyUiOT3Cv2PY3sldV3fcuNDSrhc4GyGCH9wF4&wx_header=0#rd

让 Python 起飞的 24 个骚操作！
"""

import time


# 第1式：测算代码运行时间
# 平凡方法
tic = time.time()
much_job = [x**2 for x in range(1,1000000,3)]
toc = time.time()

print("used {:.5}s".format(toc-tic))


# 快捷方法（jupyter环境）
# %%time
# much_job = [x**2 for x in range(1,1000000,3)]




# 第2式：测算代码多次运行平均时间
# 平凡方法
