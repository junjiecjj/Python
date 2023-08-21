#!/usr/bin/env python
#-*-coding=utf-8-*-
import multiprocessing
from multiprocessing import Process
import time

class dead_loop(object):
    def __init__(self):
        self.name = 'DEAD_LOOP'

    def loop1(self):
        x=0
        while True:
            x+=1

    def loop2(self):
        x=0
        while True:
            x+=2

    def loop3(self):
        x=0
        while True:
            x+=3

    def loop4(self):
        x=0
        while True:
            x+=4

def main():
    D_loop = dead_loop()
    fun_list = [D_loop.loop1,D_loop.loop2,D_loop.loop3,D_loop.loop4,D_loop.loop1,D_loop.loop2,D_loop.loop3,D_loop.loop4,D_loop.loop1,D_loop.loop2,D_loop.loop3,D_loop.loop4,D_loop.loop1,D_loop.loop2,D_loop.loop3,D_loop.loop4]
    p_list=[]
    a = multiprocessing.cpu_count()
    print("CPU count is :%d " % a)
    # time.sleep(1)
    # a = 0
    # a = a^2
    for i in range(multiprocessing.cpu_count()):
        print("dead circle %d start...at %s" % (i,time.ctime()))
        p = Process(target=fun_list[i])
        p_list.append(p)
        p.start()

    for i in p_list:
        i.join()

    print('end')

if __name__=='__main__':
    main()
"""这是测试利用Process模块能否达到真正的多进行并行程序的测试代码，结果表明可以。此代码创建了四个进程，
每个进程都是一个死循环，但是每个进程的死循环函数不一样，此电脑有4个核，最后对系统监测表明:四个核都
用了，且每个核的利用率为100%，说明是真正的多进程编程
"""




























