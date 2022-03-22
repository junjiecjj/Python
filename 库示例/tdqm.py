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
    
#===============================================================================================
#   https://blog.csdn.net/qq_33472765/article/details/82940843
#===============================================================================================

from tqdm import tqdm
for i in tqdm(range(10000)):
    pass


import time
from tqdm import tqdm
for i in tqdm(range(100),ascii=True,desc='jack'):
    time.sleep(0.01)



from tqdm import tqdm
from time import sleep

text = ""
for char in tqdm(["a", "b", "c", "d"]):
    sleep(0.25)
    text = text + char


from tqdm import trange

for i in trange(100):
    sleep(0.01)


pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    sleep(0.25)
    pbar.set_description("Processing %s" % char)

# Manual control of tqdm() updates using a with statement:
with tqdm(total=100) as pbar:
    for i in range(10):
        sleep(0.1)
        pbar.update(10)
        
        
pbar = tqdm(total=100)
for i in range(10):
    sleep(0.1)
    pbar.update(10)
pbar.close()



from tqdm import tqdm, trange
from random import random, randint
from time import sleep

with trange(10) as t:
    for i in t:
        # Description will be displayed on the left
        t.set_description('GEN %i' % i)
        # Postfix will be displayed on the right,
        # formatted automatically based on argument's datatype
        t.set_postfix(loss=random(), gen=randint(1,999), str='h',
                      lst=[1, 2])
        sleep(0.1)

with tqdm(total=10, bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
          postfix=["Batch", dict(value=0)]) as t:
    for i in range(10):
        sleep(0.1)
        t.postfix[1]["value"] = i / 2
        t.update()



from tqdm import tqdm
class TqdmExtraFormat(tqdm):
    """Provides a `total_time` format parameter"""
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time) + " in total")
        return d

for i in TqdmExtraFormat(
      range(9), ascii=" .oO0",
      bar_format="{total_time}: {percentage:.0f}%|{bar}{r_bar}"):
    if i == 4:
        break



from time import sleep
from tqdm import trange, tqdm
from multiprocessing import Pool, RLock, freeze_support

L = list(range(9))

def progresser(n):
    interval = 0.001 / (n + 2)
    total = 5000
    text = "#{}, est. {:<04.2}s".format(n, interval * total)
    for _ in trange(total, desc=text, position=n):
        sleep(interval)

def test1():
    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention
    p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    p.map(progresser, L)

test1()



from time import sleep
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor

L = list(range(9))

def progresser(n):
    interval = 0.001 / (n + 2)
    total = 5000
    text = "#{}, est. {:<04.2}s".format(n, interval * total)
    for _ in trange(total, desc=text):
        sleep(interval)
    if n == 6:
        tqdm.write("n == 6 completed.")
        tqdm.write("`tqdm.write()` is thread-safe in py3!")

def test2():
    with ThreadPoolExecutor() as p:
        p.map(progresser, L)
test2()



from tqdm.asyncio import tqdm

async for i in tqdm(range(9)):
    if i == 2:
        break



from tqdm.asyncio import tqdm

async for i in tqdm(range(9)):
    if i == 2:
        break



import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
# (can use `tqdm.gui.tqdm`, `tqdm.notebook.tqdm`, optional kwargs, etc.)
tqdm.pandas(desc="my bar!")

# Now you can use `progress_apply` instead of `apply`
# and `progress_map` instead of `map`
df.progress_apply(lambda x: x**2)









