#!/usr/bin/env python
#-*-coding=utf-8-*-

import pp

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

job_server=pp.Server()
print("pp 可以工作的核心数:%d" % job_server.get_ncpus())
#fun_list=[loop1,loop2,loop3,loop4]
for i in range(job_server.get_ncpus()):
    job_server.submit(loop1)
#上述代码会使四个核都利用，且每个核的利用率为100%

job_server=pp.Server()
print("pp 可以工作的核心数:%d" % job_server.get_ncpus())
fun_list=[loop1,loop2,loop3,loop4]
for i in range(job_server.get_ncpus()):
    job_server.submit(fun_list[i])
#上述代码也可以使四个核都被利用，且利用率都为100%

"""这是测试利用pp模块能否创建真正的多线程并行程序的测试代码，此代码写了四个线程(此电脑有4个核)，
且每个线程都是一个死循环，测试结果表面4个核都跑满了，使用率都是100%，说明pp模块可以做到真正的多线程
"""


