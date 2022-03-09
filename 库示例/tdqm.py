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

# 发呆0.5s
def action():
    time.sleep(0.5)
with tqdm(total=100000, desc='Example', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
    for i in range(10):
        # 发呆0.5秒
        action()
        # 更新发呆进度
        pbar.update(10000)
