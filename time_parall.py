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
        name = multiprocessing.current_process().name
        print("process %s start at %s."% (name,start))
        x=2
        k=0
        while k<10000000000:
            x=x^4
            k+=1
        print("last x_%d is %d."%(i,x))
        end =time.ctime()
        End = time.time()
        print("process %s end at %s."% (name,end))
        minut = (End-Start)/60
        print("process %s consum %f minus"%(name,minut))


def main():
    t1 = time.ctime()
    T1 = time.time()
    print("Start at %s."%t1)
    D_Loop = dead_loop()
    a=multiprocessing.cpu_count()
    print("CPU count is %d ." % a)
    p_list = []
    for i in range(64):
        p = Process(target=D_Loop.loop,name="worker"+str(i),args=(i,))
        p_list.append(p)
        p.start()
        #p.join()
    for i in p_list:
        i.join()
    t2 = time.ctime()
    T2 = time.time()
    minute = (T2-T1)/60
    print('end at %s, comsum time :%f minus.'%(t2,minute))

if __name__=='__main__':
    main()


"""
这是和time_line.py对比的代码，time_line是八个穿行，这里是并行，
串行的结果为:
Start at Tue Aug 14 21:07:15 2018.
CPU count is 8 .
circle 0 start at Tue Aug 14 21:07:15 2018.
last x_0 is 2.
circle 0 end at Tue Aug 14 21:07:20 2018.
circle 0 consum 0.088017 minus
circle 1 start at Tue Aug 14 21:07:20 2018.
last x_1 is 2.
circle 1 end at Tue Aug 14 21:07:26 2018.
circle 1 consum 0.087942 minus
circle 2 start at Tue Aug 14 21:07:26 2018.
last x_2 is 2.
circle 2 end at Tue Aug 14 21:07:31 2018.
circle 2 consum 0.088299 minus
circle 3 start at Tue Aug 14 21:07:31 2018.
last x_3 is 2.
circle 3 end at Tue Aug 14 21:07:36 2018.
circle 3 consum 0.089411 minus
circle 4 start at Tue Aug 14 21:07:36 2018.
last x_4 is 2.
circle 4 end at Tue Aug 14 21:07:42 2018.
circle 4 consum 0.088610 minus
circle 5 start at Tue Aug 14 21:07:42 2018.
last x_5 is 2.
circle 5 end at Tue Aug 14 21:07:47 2018.
circle 5 consum 0.088369 minus
circle 6 start at Tue Aug 14 21:07:47 2018.
last x_6 is 2.
circle 6 end at Tue Aug 14 21:07:52 2018.
circle 6 consum 0.087938 minus
circle 7 start at Tue Aug 14 21:07:52 2018.
last x_7 is 2.
circle 7 end at Tue Aug 14 21:07:58 2018.
circle 7 consum 0.087976 minus
end at Tue Aug 14 21:07:58 2018, comsum time :0.706566 minus.

real	0m42.430s
user	0m42.423s
sys	0m0.004s

并行的结果为:
Start at Tue Aug 14 21:20:35 2018.
CPU count is 8 .
process worker0 start at Tue Aug 14 21:20:35 2018.
process worker1 start at Tue Aug 14 21:20:35 2018.
process worker2 start at Tue Aug 14 21:20:35 2018.
process worker3 start at Tue Aug 14 21:20:35 2018.
process worker4 start at Tue Aug 14 21:20:35 2018.
process worker5 start at Tue Aug 14 21:20:35 2018.
process worker6 start at Tue Aug 14 21:20:35 2018.
process worker7 start at Tue Aug 14 21:20:35 2018.
last x_3 is 2.
process worker3 end at Tue Aug 14 21:20:46 2018.
process worker3 consum 0.195282 minus
last x_6 is 2.
process worker6 end at Tue Aug 14 21:20:46 2018.
process worker6 consum 0.195276 minus
last x_2 is 2.
process worker2 end at Tue Aug 14 21:20:46 2018.
process worker2 consum 0.195338 minus
last x_7 is 2.
process worker7 end at Tue Aug 14 21:20:46 2018.
process worker7 consum 0.195478 minus
last x_1 is 2.
process worker1 end at Tue Aug 14 21:20:46 2018.
process worker1 consum 0.195553 minus
last x_5 is 2.
process worker5 end at Tue Aug 14 21:20:46 2018.
process worker5 consum 0.195818 minus
last x_0 is 2.
process worker0 end at Tue Aug 14 21:20:47 2018.
last x_4 is 2.
process worker0 consum 0.197868 minus
process worker4 end at Tue Aug 14 21:20:47 2018.
process worker4 consum 0.197846 minus
end at Tue Aug 14 21:20:47 2018, comsum time :0.197910 minus.

real	0m11.909s
user	1m33.837s
sys	0m0.013s


"""
