#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:39:51 2022

@author: jack
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
selectors = [False, True, False, True]

Best_Programming = itertools.compress(Codes, selectors)

for each in Best_Programming:
    print(each)
# C++
# Python


#====================================================================
import itertools
import operator


example = itertools.compress('ABCDE', [1, 0, 1, 0, 0])

for each in example:
    print(each)



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





