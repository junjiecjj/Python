#!/usr/bin/env python
#-*-coding=utf-8-*-


from multiprocessing import Pool

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
        x==4

def main():
    fun_list=[loop1,loop2,loop3,loop4]
    ps = Pool(multiprocessing.cpu_count())
    for i in range(multiprocessing.cpu_count()):
        ps.apply_async(loop1)
    ps.close()
    ps.join() #这一行必须有，否则直接结束
    print("end!!!!!!!!!")

if __name__=='__main__':
    main()

"""这是测试利用进程池Pool模块能否达到真正的多进行并行程序的测试代码，结果表明可以。且用了不同的测试
函数。此代码创建了四个进程，每个进程都是一个死循环，此电脑有4个核，最后对系统监测表明:四个核都用了，且每个核的利用率为
100%，说明是真正的多进程编程"""

