#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:27:10 2023

@author: jack

以下是 print() 方法的语法:

print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
    参数
    objects – 复数，表示可以一次输出多个对象。输出多个对象时，需要用 , 分隔。
    sep – 用来间隔多个对象，默认值是一个空格。
    end – 用来设定以什么结尾。默认值是换行符 \n，我们可以换成其他字符串。
    file – 要写入的文件对象。
    flush --输出是否被缓存通常决定于 file，但如果 flush 关键字参数为 True，流会被强制刷新。


本文档主要是记录 print 的各种骚操作，比如：
倒计时、转圈显示、进度条等，这在程序调试和多进程显示的时候非常有用。关键的技术如下：

\r表示将光标的位置回退到本行的开头位置
\b表示将光标的位置回退一位
\n表示到下一行前面

"""

##=========================== 1、横条加载 ==============================

import time
num = 20
for i in range(num):
    print("#", end="", flush=True)
    time.sleep(0.1)
##  ####################

def printer(text, delay = 0.1):
    for ch in text:
        print(ch, end='', flush = True)
        time.sleep(delay)

printer("玄铁重剑是金庸小说名下第一神剑，持之则天下无敌...")

##=========================== 2、倒计时显示 ==============================

import time
for i in range(5, 0, -1):
    print(f"\r 倒计时{i}秒！", end="", flush=True)
    time.sleep(1)


# 在print中，\r就可以让打印之后有重新回到本行开头的位置继续打印，相当于重新刷了一遍，但是我们不难发现，倒计时前面有个小空缺，那是因为"\r"占了一个小位置，所以我们把代码重新改造一下。

import time
for i in range(5, 0, -1):
    print(f"\r 倒计时{i}秒！",  end="", flush=True)
    time.sleep(1)
print("\r倒计时结束！")


##=========================== 3、转圈等待显示 ==============================

import time
Sum = 10         # 设置倒计时时间
timeflush = 0.25  # 设置屏幕刷新的间隔时间
for i in range(0, int(Sum/timeflush)):
    List = ["\\", "|", "/", "—"]
    index = i % 4
    print(f"\r程序正在运行 {List[index]}", end="")
    time.sleep(timeflush)

def waiting(cycle = 20, delay = 0.08):
    for i in range(cycle):
        for ch in ['-', '\\', '|', '/']:
            print('\r程序正在运行 %s ...'%ch, end = '', flush = True)
            time.sleep(delay)
    return

waiting()


def cover(cycle = 20, delay = 0.08):
    for i in range(cycle):
        s = '\r%d'%i
        print(s.ljust(3), end='', flush = True)
        time.sleep(delay)
    return
cover()



##=========================== 4、进度条显示 ==============================

import time
days = 365
for i in range(days):
    print("\r进度条百分比：{}%".format(round((i + 1) * 100 /days)), end="", flush=True)
    time.sleep(0.02)


# 进度条改进版
import time
num = 50         #设置倒计时时间
timeflush = 0.5   #设置屏幕刷新的间隔时间
for i in range(0, int(num/timeflush)+1):
    print("\r正在加载:|" + "*" * i + " "*(int(num/timeflush)+1-i) + f"|{str(i)}%", end="")
    time.sleep(timeflush)
print("\r加载完成！")







from tqdm import tqdm
for i in tqdm(range(10000)):
    pass




##=========================== 多进程进度条显示 ==============================


import time

from tqdm import tqdm
from multiprocessing.pool import ThreadPool


def fun():
    """测试函数"""
    time.sleep(0.01)


num = 1000
pbar = tqdm(total=num)
update = lambda *args: pbar.update()
pool = ThreadPool(4)

for i in range(num):
    pool.apply_async(fun, callback=update)
pool.close()
pool.join()

















































































































































































