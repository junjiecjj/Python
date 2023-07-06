#!/usr/bin/env python
#-*-coding=utf-8-*-
import multiprocessing
from multiprocessing import Process
import time

class dead_loop(object):
    def __init__(self):
        self.name = "DEAD_LOOP"

    def loop(self,i):
        start = time.ctime()
        Start = time.time()
        print("circle %d start at %s."% (i,start))
        x=2
        k=0
        while k<10000000000:
            x=x^4
            k+=1
        print("last x_%d is %d."%(i,x))
        end =time.ctime()
        End = time.time()
        print("circle %d end at %s."% (i,end))
        minut = (End-Start)/60
        print("circle %d consum %f minus"%(i,minut))

def main():
    t1 = time.ctime()
    T1 = time.time()
    print("Start at %s."%t1)
    D_Loop = dead_loop()
    a=multiprocessing.cpu_count()
    print("CPU count is %d ." % a)
    for i in range(multiprocessing.cpu_count()):
        D_Loop.loop(i)
    t2 = time.ctime()
    T2 = time.time()
    minute = (T2-T1)/60
    print('end at %s, comsum time :%f minus.'%(t2,minute))

if __name__=='__main__':
    main()

