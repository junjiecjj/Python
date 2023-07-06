#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:29:40 2022

@author: jack
"""

import multiprocessing

def f(x, arr, l, d, n):

    x.value = 3.14

    arr[0] = 5

    l.append('Hello')

    d[1] = 2

    n.a = 10

if __name__ == '__main__':

    server = multiprocessing.Manager()

    x = server.Value('d', 0.0)

    arr = server.Array('i', range(10))

    l = server.list()

    d = server.dict()

    n = server.Namespace()

    proc = multiprocessing.Process(target=f, args=(x, arr, l, d, n))

    proc.start()

    proc.join()

    print(x.value)

    print(arr)

    print(l)

    print(d)

    print(n)