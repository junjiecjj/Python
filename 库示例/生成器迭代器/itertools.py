#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:39:51 2022

@author: jack
"""
# itertools主要来分为三类函数，分别为无限迭代器、输入序列迭代器、组合生成器，我们下面开始具体讲解。
import itertools




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  一、 无限迭代器


x = itertools.accumulate(range(5))
print(list(x))


x = itertools.chain(range(4),[1,2],[3,4])
print(list(x))


x = [{"a":1},{"a":2},{"a":3}]
x = itertools.groupby(x,lambda y: y['a']<=2)
for name,group in x:
    print(name,group)

x = itertools.islice(range(20),0,11,3)
print(list(x))

x = itertools.count(start=10,step=2)
print(list(itertools.islice(x,0,5,1)))


x = itertools.filterfalse(lambda x: x > 5, range(10))
print(list(x))


x = itertools.dropwhile(lambda x: x > 5, [8,6,4,2])
print(list(x))


x = itertools.cycle([0,1,2])
print(list(itertools.islice(x,0,5,2)))


li =[(2, 3), (3, 1), (4, 6), (5, 3), (6, 5), (7, 2)]
list(itertools.starmap(lambda x,y: x+y, li))

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






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 二、输入序列迭代器







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 三、组合生成器
# https://blog.csdn.net/watfe/article/details/80108774
# 为了输出结果好看，加了list(map(lambda x:''.join(x)……
print(list(map(lambda x:''.join(x),itertools.combinations('ABCD', 2))))
print(list(map(lambda x:''.join(x),itertools.permutations('ABCD', 2))))
print(list(map(lambda x:''.join(x),itertools.combinations_with_replacement('ABCD', 2))))
print(list(map(lambda x:''.join(x),itertools.product('ABCD', repeat=2))))
# ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
# ['AB', 'AC', 'AD', 'BA', 'BC', 'BD', 'CA', 'CB', 'CD', 'DA', 'DB', 'DC']
# ['AA', 'AB', 'AC', 'AD', 'BB', 'BC', 'BD', 'CC', 'CD', 'DD']
# ['AA', 'AB', 'AC', 'AD', 'BA', 'BB', 'BC', 'BD', 'CA', 'CB', 'CC', 'CD', 'DA', 'DB', 'DC', 'DD']


######### python把list的所有元素生成排列和组合
##################
#product 就是产生多个列表或者迭代器的n维积。如果没有特别指定repeat默认为列表和迭代器的数量。((x,y) for x in A for y in B)
import itertools
l = [1, 2, 3]
a = list(itertools.product(l,))
print(len(a), a)

a = list(itertools.product(l, l))
print(len(a), a)

a = list(itertools.product(l, repeat=3))
print(len(a), a)

################### 计算所有可能的排列, permutations, 排列数 A(n, m)
elements = [1,2,3,4,5]

permutation  =  itertools.permutations(elements, r = 3)
# 打印结果
for p in permutation:
    print(p)

permutation  =  itertools.permutations(elements, r = 2)
# 打印结果
for p in permutation:
    print(p)

permutation  =  itertools.permutations(elements, r = 1)
# 打印结果
for p in permutation:
    print(p)

################### 组合的话可以用 itertools.combinations： 组合数 C(n, m)
a = list(itertools.combinations([1,2,3,4,5], 1))
print(len(a), a)

a = list(itertools.combinations([1,2,3,4,5], 2))
print(len(a), a)

a = list(itertools.combinations([1,2,3,4,5], 3))
print(len(a), a)

############################
# combinations_with_replacement 这个函数用来生成指定数目r的元素可重复的所有组合。然而这个函数依然要保证元素组合的unique性。
a = itertools.combinations_with_replacement([1,2,3,4,5], 1)
# 打印结果
for p in a:
    print(p)

a = itertools.combinations_with_replacement([1,2,3,4,5], 2)
# 打印结果
for p in a:
    print(p)

a = itertools.combinations_with_replacement([1,2,3,4,5], 3)
# 打印结果
for p in a:
    print(p)





