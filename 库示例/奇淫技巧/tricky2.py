#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:27:53 2022

@author: jack

https://mp.weixin.qq.com/s?__biz=MzIxNjM4NDE2MA==&mid=2247497142&idx=1&sn=10bb72761b6cc9cd9634a78e053060e9&chksm=978b6279a0fceb6fa7a8bbf5e118d424fb12624b8e3da18bb4f4255a91da4eb657f166e00fe1&mpshare=1&scene=24&srcid=0425ntK15od5uYil0JIXOVIw&sharer_sharetime=1650850214648&sharer_shareid=8d8081f5c3018ad4fbee5e86ad64ec5c&exportkey=AbbDTtBvFUz3zs3nhvGQ7TY%3D&acctmode=0&pass_ticket=p716zXobtGbfw0swZspHFjoQO%2FNSfDcuD7NQpsXHDcKtjr3FZlocc12C39arlJpn&wx_header=0#rd
"""


# 1
name = 'leo'
n=37
s = '{} has {} message.'.format(name,n)
print(s)

# 2
name = 'leo'
n=37
s = '{name} has {n} message.' 
print(s.format_map(vars()))


name = 'jack'
n = 43
print(s.format_map(vars()))





# 过滤列表方法
"""
定义:
itertools.compress()
输入:
iterable对象
相应的Boolean选择器序列
输出:
iterable对象中对应选择器为True的元素
用途:
当需要用另外一个相关联的序列来过滤某个序列的时候，这个函数非常有用
eg:
两个列表如下，其元素相对应，现请根据count输出address,条件只输出count大于5的对应地址：
"""
from itertools import compress

addres = [
          '123 N apple',
          '234 N yahoo',
          '457 E google',
          '212 N ibm',
          '987 N hp',
          '653 W aliyun',
          '487 N sina',
          '109 W baidu',
          ]

counts = [0,3,10,4,1,7,6,1]

more5 = [n>5 for n in counts]


a = list(compress(addres,more5))
print(f"a = {a}")

#====================================================================
import numpy as np
a = np.array([[1, 2], [3, 4], [5, 6]])
print(f"np.compress([0, 1], a, axis=0) = {np.compress([0, 1], a, axis=0)}")

print(f"np.compress([False, True, True], a, axis=0) = {np.compress([False, True, True], a, axis=0)}")

print(f"np.compress([False, True], a, axis=1) = {np.compress([False, True], a, axis=1)}")


print(f"np.compress([False, True], a) = {np.compress([False, True], a)}")

#====================================================================
import itertools 
import operator
Codes =['C', 'C++', 'Java', 'Python'] 
selectors = [False, False, False, True] 
  
Best_Programming = itertools.compress(Codes, selectors) 
  
for each in Best_Programming:
    print(each)



#====================================================================
import itertools 
import operator 
  
  
example = itertools.compress('ABCDE', [1, 0, 1, 0, 0]) 
  
for each in example:
    print(each)


#====================================================================
# 列表搜索-堆函数
import heapq

portfolio = [
     {'name':'ali', 'shares':100, 'price':91.1},
     {'name':'baidu', 'shares':50, 'price':543.22 },
     {'name':'yahoo', 'shares':200, 'price':21.09 },
     {'name':'tencent', 'shares':35, 'price':31.75},
     {'name':'pingduoduo', 'shares':45, 'price':16.35},
     {'name':'sina', 'shares':75, 'price':115.65},
     ]


cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
expensive = heapq.nlargest(6, portfolio, key=lambda s: s['price'])

print(f"cheap = \n{cheap}")
print(f"expensive = \n{expensive}")



# 复杂列表分类-group法
rows = [
        {'city':'nanjing','date':'07/01/2012'},
        {'city':'beijing','date':'07/04/2012'},
        {'city':'shanghai','date':'07/02/2012'},
        {'city':'suzhou','date':'07/03/2012'},
        {'city':'guangzhou','date':'07/02/2012'},
        {'city':'tianjin','date':'07/02/2012'},
        {'city':'chengdu','date':'07/01/2012'},
        {'city':'wuxi','date':'07/04/2012'},
        ]

from itertools import groupby



rows.sort(key=lambda r: r['date'])

for date, items in groupby(rows, key=lambda r: r['date']):
     print(date)
     for i in items:
          print("  ",i)

























































































































































































































