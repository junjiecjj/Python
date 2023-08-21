#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:32:20 2022
https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd
@author: jack

Condition（条件变量）
可以把Condition理解为一把高级的锁，它提供了比Lock, RLock更高级的功能，允许我们能够控制复杂的线程同步问题。Condition在内部维护一个锁对象（默认是RLock），可以在创建Condigtion对象的时候把琐对象作为参数传入。

Condition也提供了acquire, release方法，其含义与锁的acquire, release方法一致，其实它只是简单的调用内部锁对象的对应的方法而已。Condition还提供了其他的一些方法。

构造方法：Condition([lock/rlock])

可以传递一个Lock/RLock实例给构造方法，否则它将自己生成一个RLock实例。
实例方法：
    acquire([timeout])：首先进行acquire，然后判断一些条件。如果条件不满足则wait
    release()：释放 Lock
    wait([timeout]): 调用这个方法将使线程进入Condition的等待池等待通知，并释放锁。使用前线程必须已获得锁定，否则将抛出异常。处于wait状态的线程接到通知后会重新判断条件。
    notify(): 调用这个方法将从等待池挑选一个线程并通知，收到通知的线程将自动调用acquire()尝试获得锁定（进入锁定池）；其他线程仍然在等待池中。调用这个方法不会释放锁定。使用前线程必须已获得锁定，否则将抛出异常。
    notifyAll(): 调用这个方法将通知等待池中所有的线程，这些线程都将进入锁定池尝试获得锁定。调用这个方法不会释放锁定。使用前线程必须已获得锁定，否则将抛出异常。


"""

import multiprocessing

import time

def stage_1(cond):
    """perform first stage of work,
    then notify stage_2 to continue"""
    name = multiprocessing.current_process().name
    print('Starting', name)
    with cond:
        print('{} done and ready for stage 2'.format(name))
        cond.notify_all()

def stage_2(cond):
    """wait for the condition telling us stage_1 is done"""
    name = multiprocessing.current_process().name
    print('Starting', name)
    with cond:
        cond.wait()
        print('{} running'.format(name))

if __name__ == '__main__':
    condition = multiprocessing.Condition()
    s1 = multiprocessing.Process(name='s1',  target=stage_1, args=(condition,))
    s2_clients = [multiprocessing.Process(name='stage_2[{}]'.format(i), target=stage_2, args=(condition,),) for i in range(1, 7) ]
    for c in s2_clients:
        c.start()
        time.sleep(1)
    s1.start()
    s1.join()
    for c in s2_clients:
        c.join()



# Starting stage_2[1]
# Starting stage_2[2]
# Starting stage_2[3]
# Starting stage_2[4]
# Starting stage_2[5]
# Starting stage_2[6]
# Starting s1
# s1 done and ready for stage 2
# stage_2[1] running
# stage_2[3] running
# stage_2[5] running
# stage_2[6] running
# stage_2[4] running
# stage_2[2] running





