#!/usr/bin/env pythonhas
#-*-coding=utf-8-*-

import threading

def loop():
    x=0
    while True:
        x+=1

def main():
    for i in range(multiprocessing.cpu_count()):
        t = threading.Thread(target=loop)
        t.start()

if __name__=='__main__':
    main()
'''
这是利用了线程的并行，这台电脑的CPU有4个核，这个程序运行时虽然每个核都用到了，但是都没有用满，
加起来接近100%，说明python里面的多线程并不是真正的多线程。这里的四个target都是loop函数，为了排除
threading不能实现真正多线程的原因可能是相同的target模块导致的，我又试了四个不同的loop函数，见
dead_thread2.py，四个核都用了，且加起来也是100%，但四个单独核也还是跑不满。,事实证明不是由于相同
的loop函数导致的，是threading模块死活也实现不了真正的多线程；
'''

#××××××××××××××××××××××××××××××××××××××××××××××××××

x=0
while True:
    x+=1

"""
这是串行的程序，在这台有四个核的CPU的电脑上，只利用了一个核，且这个核的利用率为100%，其他
三个核的利用率接近0，说明python中只要是串行程序，不管计算量有多大，也不管计算机的计算性能有多强大，
哪怕是集群，也始终只利用了一个CPU的一个核。所以在集群上用python，必须要利用并行计算，且如果要利用
并行计算也只能利用多进程的并行（from multiprocessing import Process，Pool），或者利用第三方多线程
模块pp，它可以真正的实现多线程，且可以跑满每个核。否则没必要用集群
"""
"这是测试threading模块能否创建真正的多线程并行程序的测试代码，结果说明不能"
