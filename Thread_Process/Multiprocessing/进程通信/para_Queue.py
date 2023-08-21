#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:28:07 2022

@author: jack
"""

from multiprocessing import Process, Queue

import os, time, random

def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__ == "__main__":
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    pw.start()
    pr.start()
    pw.join()  # 等待pw结束
    pr.terminate()  # pr进程里是死循环，无法等待其结束，只能强行终止
    print( '所有数据都写入并且读完')

# Process to write: 3141986
# Put A to queue...
# Process to read: 3141989
# Get A from queue.
# Put B to queue...
# Get B from queue.
# Put C to queue...
# Get C from queue.
# 所有数据都写入并且读完

##========================================================================================
## 多个子进程向队列写入数据(可能是无序的)，主进程依次读取队列的元素;
def write(q, i):
    print('Process to write: %s' % os.getpid())
    # for value in ['A', 'B', 'C']:
    print('Put %d to queue...' % i)
    q.put(i)
    time.sleep(random.random())  ## sleep在后


if __name__ == "__main__":
    time_start = time.time()  # 记录开始时间
    q = Queue()
    ps_list = []
    for i in  range(10):
        pw = Process(target = write, args=(q, i,))
        # pr = Process(target=read, args=(q,))
        pw.start()
        ps_list.append(pw)

    for p in ps_list:
        p.join()  # 等待pw结束

    get = []
    while True:
        if not q.empty():
            value = q.get(True)
            print( 'Get %s from queue.' % value)
            get.append(value)
            # time.sleep(random.random())
        else:
            break

    print( '所有数据都写入并且读完')
    print(get)
    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
# Process to write: 3143168
# Put 0 to queue...
# Process to write: 3143171
# Put 1 to queue...Process to write: 3143181

# Put 2 to queue...
# Process to write: 3143190
# Put 3 to queue...
# Process to write: 3143200
# Put 4 to queue...
# Process to write: 3143210
# Put 5 to queue...
# Process to write: 3143221
# Put 6 to queue...
# Process to write: 3143231
# Put 7 to queue...
# Process to write: 3143241
# Put 8 to queue...
# Process to write: 3143251
# Put 9 to queue...
# Get 0 from queue.
# Get 1 from queue.
# Get 2 from queue.
# Get 3 from queue.
# Get 4 from queue.
# Get 5 from queue.
# Get 6 from queue.
# Get 7 from queue.
# Get 8 from queue.
# Get 9 from queue.
# 所有数据都写入并且读完
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  ## sleep在后，有序
# 结束测试
# 1.0658156871795654

### 真并行

##========================================================================================
## 多个子进程向队列写入数据(可能是无序的)，主进程依次读取队列的元素;
def write(q, i):
    print('Process to write: %s' % os.getpid())
    time.sleep(1) ## sleep在前
    # for value in ['A', 'B', 'C']:
    print('Put %d to queue...' % i)
    q.put(i)



if __name__ == "__main__":
    time_start = time.time()  # 记录开始时间
    q = Queue()
    ps_list = []
    for i in  range(10):
        pw = Process(target = write, args=(q, i,))
        # pr = Process(target=read, args=(q,))
        pw.start()
        ps_list.append(pw)

    for p in ps_list:
        p.join()  # 等待pw结束

    get = []
    while True:
        if not q.empty():
            value = q.get(True)
            print( 'Get %s from queue.' % value)
            get.append(value)
            time.sleep(random.random())
        else:
            break

    print( '所有数据都写入并且读完')
    print(get)
    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

# Process to write: 3143906
# Process to write: 3143910
# Process to write: 3143916
# Process to write: 3143921
# Process to write: 3143926
# Process to write: 3143931
# Process to write: 3143936
# Process to write: 3143941
# Process to write: 3143946
# Process to write: 3143951
# Put 0 to queue...
# Put 1 to queue...
# Put 2 to queue...
# Put 3 to queue...
# Put 4 to queue...
# Put 5 to queue...
# Put 6 to queue...
# Put 7 to queue...
# Put 8 to queue...
# Put 9 to queue...
# Get 0 from queue.
# Get 1 from queue.
# Get 2 from queue.
# Get 3 from queue.
# Get 4 from queue.
# Get 5 from queue.
# Get 6 from queue.
# Get 7 from queue.
# Get 8 from queue.
# Get 9 from queue.
# 所有数据都写入并且读完
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  ## sleep在前 且都是1s，有序
# 结束测试
# 1.0809264183044434

### 真并行

##========================================================================================
## 多个子进程向队列写入数据(可能是无序的)，主进程依次读取队列的元素;
def write(q, i):
    print('Process to write: %s' % os.getpid())
    time.sleep(1) ## sleep在前
    # for value in ['A', 'B', 'C']:
    print('Put %d to queue...' % i)
    q.put(i)



if __name__ == "__main__":
    time_start = time.time()  # 记录开始时间
    q = Queue()
    ps_list = []
    for i in  range(10):
        pw = Process(target = write, args=(q, i,))
        # pr = Process(target=read, args=(q,))
        pw.start()
        ps_list.append(pw)

    for p in ps_list:
        p.join()  # 等待pw结束

    get = []
    while True:
        if not q.empty():
            value = q.get(True)
            print( 'Get %s from queue.' % value)
            get.append(value)
            time.sleep(random.random())
        else:
            break

    print( '所有数据都写入并且读完')
    print(get)
    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

# Process to write: 3137500
# Process to write: 3137503
# Process to write: 3137508
# Process to write: 3137513
# Process to write: 3137518
# Process to write: 3137523
# Process to write: 3137528
# Process to write: 3137533
# Process to write: 3137538
# Process to write: 3137543
# Put 0 to queue...
# Put 1 to queue...
# Put 2 to queue...
# Put 3 to queue...
# Put 4 to queue...
# Put 5 to queue...
# Put 6 to queue...
# Put 7 to queue...
# Put 8 to queue...
# Put 9 to queue...
# Get 0 from queue.
# Get 1 from queue.
# Get 2 from queue.
# Get 3 from queue.
# Get 4 from queue.
# Get 5 from queue.
# Get 6 from queue.
# Get 7 from queue.
# Get 8 from queue.
# Get 9 from queue.
# 所有数据都写入并且读完
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  ## sleep在前 且都是1s，有序
# 结束测试
# 4.98452091217041

### 真并行

##========================================================================================
## 多个子进程向队列写入数据(可能是无序的)，主进程依次读取队列的元素;
def write(q, i):
    print('Process to write: %s' % os.getpid())
    time.sleep(random.random()) ## sleep在前
    # for value in ['A', 'B', 'C']:
    print('Put %d to queue...' % i)
    q.put(i)



if __name__ == "__main__":
    time_start = time.time()  # 记录开始时间
    q = Queue()
    ps_list = []
    for i in  range(10):
        pw = Process(target = write, args=(q, i,))
        # pr = Process(target=read, args=(q,))
        pw.start()
        ps_list.append(pw)

    for p in ps_list:
        p.join()  # 等待pw结束

    get = []
    while True:
        if not q.empty():
            value = q.get(True)
            print( 'Get %s from queue.' % value)
            get.append(value)
            # time.sleep(random.random())
        else:
            break

    print( '所有数据都写入并且读完')
    print(get)
    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

# Process to write: 3138077
# Process to write: 3138080
# Process to write: 3138085
# Process to write: 3138090
# Process to write: 3138095
# Process to write: 3138100
# Process to write: 3138105
# Process to write: 3138110
# Process to write: 3138115
# Process to write: 3138120Put 4 to queue...

# Put 2 to queue...
# Put 0 to queue...
# Put 1 to queue...
# Put 8 to queue...
# Put 6 to queue...
# Put 3 to queue...
# Put 9 to queue...
# Put 7 to queue...
# Put 5 to queue...
# Get 4 from queue.
# Get 2 from queue.
# Get 0 from queue.
# Get 1 from queue.
# Get 8 from queue.
# Get 6 from queue.
# Get 3 from queue.
# Get 9 from queue.
# Get 7 from queue.
# Get 5 from queue.
# 所有数据都写入并且读完
# [4, 2, 0, 1, 8, 6, 3, 9, 7, 5]  ## sleep在前 且都是random，无序
# 结束测试
# 1.0455811023712158














