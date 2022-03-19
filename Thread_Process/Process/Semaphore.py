#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:32:01 2022
https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd
@author: jack
"""

from multiprocessing import Process, Semaphore

import time, random

def go_wc(sem, user):

    sem.acquire()

    print('%s 占到一个茅坑' % user)

    time.sleep(random.randint(0, 3))

    sem.release()

    print(user, 'OK')

if __name__ == '__main__':

    sem = Semaphore(2)

    p_l = []

    for i in range(5):

        p = Process(target=go_wc, args=(sem, 'user%s' % i,))

        p.start()

        p_l.append(p)

    for i in p_l:

        i.join()