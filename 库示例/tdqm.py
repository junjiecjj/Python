#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:32:22 2022

@author: jack
"""
from tqdm import tqdm
import time
from tqdm import trange


#tqdm中的tqdm()是实现进度条美化的基本方法，在for循环体中用tqdm()包裹指定的迭代器或range()即可，下面是两个简单的例子：

text = ""
for char in tqdm(["a", "b", "c", "d"]):
    time.sleep(0.25)
    text = text + char
    
print(text)




#传入range()：
for it in tqdm(range(10)):
    time.sleep(0.5)


#trange：作为tqdm(range())的简洁替代，如下例：
for i in trange(100):
    time.sleep(0.01)


#也可以使用一些较为高级的写法使得代码更简练，如下面两种写法，得到的效果相同：
'''method 1'''
with tqdm(total=100) as pbar:
    for i in range(10):
        time.sleep(0.1)
        pbar.update(10)

'''method 2'''
pbar = tqdm(total=100)
for i in range(10):
    time.sleep(0.1)
    pbar.update(10)
pbar.close()

#===============================================================================================
# https://zhuanlan.zhihu.com/p/163613814
#===============================================================================================

# 基于迭代对象运行: tqdm(iterator)

#trange(i)是tqdm(range(i))的一种简单写法
for i in trange(100):
    time.sleep(0.01)

for i in tqdm(range(100), desc='Processing'):
    time.sleep(0.01)

dic = ['a', 'b', 'c', 'd', 'e']
pbar = tqdm(dic)
for i in pbar:
    pbar.set_description('Processing '+i)
    time.sleep(0.2)

# 手动进行更新

import time
from tqdm import tqdm
with tqdm(total=200) as pbar:
    pbar.set_description('Processing:')
    # total表示总的项目, 循环的次数20*10(每次更新数目) = 200(total)
    for i in range(20):
        # 进行动作, 这里是过0.1s
        time.sleep(0.1)
        # 进行进度更新, 这里设置10个
        pbar.update(10)

# 发呆0.5s
def action():
    time.sleep(0.5)
with tqdm(total=100000, desc='Example', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
    for i in range(10):
        # 发呆0.5秒
        action()
        # 更新发呆进度
        pbar.update(10000)


#===============================================================================================
#   https://blog.csdn.net/qq_33472765/article/details/82940843
#===============================================================================================


import time
from tqdm import tqdm, trange

 
for i in tqdm(range(100)):
    time.sleep(0.01)

alist = list('letters')
bar = tqdm(alist)
for letter in bar:
    bar.set_description(f"Now get {letter}")
    time.sleep(0.1)



pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    time.sleep(0.1)
    pbar.set_description("Processing %s" % char)