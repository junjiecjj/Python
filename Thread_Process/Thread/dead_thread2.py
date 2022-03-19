#!/usr/bin/env python
#-*-coding=utf-8-*-

import  threading

def loop1():
    x=0
    while True:
        x+=1

def loop2():
    x=0
    while True:
        x+=2

def loop3():
    x=0
    while True:
        x+=3

def loop4():
    x=0
    while True:
        x+=4

def main():
    fun_list=[loop1,loop2,loop3,loop4]
    for i in range(multiprocessing.cpu_count()):
        t=threading.Thread(target=fun_list[i])
        t.start()

if __name__=='__main__':
    main()
