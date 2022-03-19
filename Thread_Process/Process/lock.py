#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:28:07 2022

@author: jack
"""

from multiprocessing import Process, Lock

def l(lock, num):

    lock.acquire()

    print("Hello Num: %s" % (num))

    lock.release()

if __name__ == '__main__':

    lock = Lock()  # 这个一定要定义为全局

    for num in range(20):

        Process(target=l, args=(lock, num)).start()