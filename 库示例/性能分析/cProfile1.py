#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:08:29 2022

@author: jack
https://static.kancloud.cn/ju7ran/gaoji/1453328
"""

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def fib_seq(n):
    res = []
    if n > 0:
        res.extend(fib_seq(n-1))
    res.append(fib(n))
    return res

fib_seq(30)

import cProfile

cProfile.run('fib_seq(30)')