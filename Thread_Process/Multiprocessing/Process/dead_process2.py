#!/usr/bin/env python
#-*-coding=utf-8-*-


import multiprocessing

from multiprocessing import Process
import time

class dead_loop(object):
    def __init__(self):
        self.name = "DEAD_LOOP"

    def loop(self,i):
        print("Hello,",i)
        name = multiprocessing.current_process().name
        start = time.ctime()
        print("process %s start at %s."% (name,start))
        x=0
        while True:
            x=x^i
        end =time.ctime()
        print("process %s end at %s."% (name,end))

def main():
    D_Loop = dead_loop()
    p_list=[]
    a=multiprocessing.cpu_count()
    print("CPU count is %d ." % a)
    for i in range(multiprocessing.cpu_count()):
        print("dead circle %d start..at %s ." % (i, time.ctime()))
        p = Process(target=D_Loop.loop, name="worker"+str(i), args=(i,))
        p_list.append(p)
        p.start()
        #p.join()
    for i in p_list:
        i.join()

    print('end')

if __name__=='__main__':
    main()
"""
这是测试利用Process模块能否达到真正的多进行并行程序的测试代码，结果表明可以。此代码创建了四个进程，
每个进程都是一个死循环，但是每个进程的死循环函数一样，只是参数不同，此电脑有4个核，最后对系统监
测表明:四个核都用了，且每个核的利用率为100%，说明是真正的多进程编程
"""

