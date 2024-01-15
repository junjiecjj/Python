#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:19:17 2022

https://www.jianshu.com/p/8185e125afaf

@author: jack
"""

## 原始信息，长度为k.
# oinfo = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
oinfo = [1, 0, 1, 0, 0, 0, 1, 1, 0, 1]
crc_n = 5
# 初始化生成多项式p，长度为 n-k+1.
# loc = [32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 2, 1, 0]
loc = [0,1,3,5]

## 生成多项式
n = crc_n + 1      #  6
p = [0 for i in range(n)]
for i in loc:
    p[i] = 1
times = len(oinfo)  #  10

# 左移补n-k个零，总长为k+n-k = n
info = oinfo.copy()
for i in range(crc_n):
    info.append(0)


# q为商，长除法
q = []
for i in range(times):
    if info[i] == 1:
        q.append(1)
        for j in range(n):
            info[j + i] = info[j + i] ^ p[j]
    else:
        q.append(0)

# 余数，校验位
check_code = info[-crc_n:]

# 生成编码
code = oinfo.copy()
for i in check_code:
    code.append(i)

# print(f'信息oinfo：{len(oinfo)}\t\n {oinfo}' )
# print(f'生成多项式p：{len(p)}\t\n {p}')
# print(f'商q：{len(q)}\t\n {q}')
# print(f'余数check_code：{len(check_code)}\t\n {check_code}')
# print(f'编码code：{len(code)}\t\n {code}')
# 信息oinfo：10
#  [1, 0, 1, 0, 0, 0, 1, 1, 0, 1]
# 生成多项式p：6
#  [1, 1, 0, 1, 0, 1]
# 商q：10
#  [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
# 余数check_code：5
#  [0, 1, 1, 1, 0]
# 编码code：15
#  [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0]
