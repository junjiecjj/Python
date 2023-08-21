#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:24:09 2022
https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd
@author: jack
"""

from multiprocessing import Process

import os

def run_proc(name):

    print('Run child process %s (%s)...' % (name, os.getpid()))

# if __name__=='__main__':

print('Parent process %s.' % os.getpid())
p = Process(target = run_proc, args = ('test',) )
print('Child process will start.')
p.start()
p.join()
print('Child process end.')



import random
import time
import multiprocessing

# https://blog.csdn.net/springlustre/article/details/88703947
def worker(name, q):
    t = 0
    for i in range(2):
        # print(name + " " + str(i))
        x = random.randint(1, 3)
        t += x
        time.sleep(x * 0.1)
    print(f"{name} = {t}" )
    q.put(t)

## 这种方法获得的结果是无序的；
q = multiprocessing.Queue()
jobs = []
for i in range(10):
    p = multiprocessing.Process(target=worker, args=(str(i), q))
    jobs.append(p)
    p.start()

for p in jobs:
    p.join()

results = [q.get() for j in jobs]
print(results)

