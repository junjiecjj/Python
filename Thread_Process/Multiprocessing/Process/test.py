#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:55:48 2023

@author: jack
"""

#*****************************************
from multiprocessing import Process,Lock,Pool
import os,time
import random
import multiprocessing
import sys
import  json

import numpy as np
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double

from multiprocessing import Manager,Process,Lock
import os




def f(i, x, arr, l, L, d, n):
    x.value += i
    arr[i] = i
    l.append(f'Hello {i}')
    L[i] = 2*i
    d[f"{i}"] = i**2
    n.a = 10

if __name__ == '__main__':
    N = 10
    server = multiprocessing.Manager()
    x = server.Value('d', 0.0)
    arr = server.Array('i', range(N))
    l = server.list()
    L = server.list(range(10))
    d = server.dict()
    n = server.Namespace()

    process_list = []
    for i in range(N):
        p = multiprocessing.Process(target=f, args=(i, x, arr, l, L, d, n))
        p.start()
        process_list.append(p)

    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    print(f"x.value = {x.value}")
    print(f"arr = {arr}")
    print(f"arr[2] = {arr[2]}")
    print(f"np.array(arr) = {np.array(arr)}")
    print(l)
    print(d)
    print(n)











