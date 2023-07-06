#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 21:52:09 2023

@author: jack
"""



import time



# 2
def func(n):
    print('Ready to eat.')
    for i in range(0, n):
        print(f'eat. {i}')
        # yield相当于return，下一次循环从yield的下一行开始
        arg = yield i
        print('func', arg)

# if __name__ == '__main__':
f = func(10)
for j in range(12):
    print(f'main-next: {next(f)}', )
    print(f'main-send: {f.send(100+j) }\n' )
    time.sleep(1)
