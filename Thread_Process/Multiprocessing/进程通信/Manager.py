#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:29:40 2022

@author: jack


这个模式支持跨进程共享所有对象，也即是想要共享 “自定义对象”，只能使用这个方式！

在server process模型中，有一个manager进程（就是那个server进程），负责管理实际的对象，真正的对象也是在manager进程的内存空间中。所有需要访问该对象的进程都需要先连接到该管理进程，然后获取到对象的一个代理对象(Proxy object)。这个模型是一个典型的RPC(远程过程调用)的模型。因为每个客户进程实际上都是在访问manager进程当中的对象，因此完全可以通过这个实现对象共享。



(二) 服务进程
    # https://www.jianshu.com/p/cf617911e0f0

    Manager()返回的manager对象控制了一个server进程，此进程包含的python对象可以被其他的进程通过proxies来访问。从而达到多进程间数据通信且安全。Manager模块常与Pool模块一起使用。
    由 Manager() 返回的管理器对象控制一个服务进程，该进程保存Python对象并允许其他进程使用代理操作它们。
    Manager() 返回的管理器支持类型： list 、 dict 、 Namespace 、 Lock 、 RLock 、 Semaphore 、 BoundedSemaphore 、 Condition 、 Event 、 Barrier 、 Queue 、            Value 和 Array 。
    管理器是独立运行的子进程，其中存在真实的对象，并以服务器的形式运行，其他进程通过使用代理访问共享对象，这些代理作为客户端运行。Manager()是BaseManager的子类，返回一个启动的SyncManager()实例，可用于创建共享对象并返回访问这些共享对象的代理。

    manager = Manager()
    return_list = manager.list()
    pool = Pool(processes=len(BOHAO))
    for host, port in BOHAO.items():
        pool.apply_async(getIpFromAdsl, args=(host, port, return_list))
    pool.close()
    pool.join()
    return_list = list(return_list)


"""
# https://zhuanlan.zhihu.com/p/64702600


from multiprocessing import Process, Manager

def fun1(dic,lis,index):

    dic[index] = 'a'
    dic['2'] = 'b'
    lis.append(index)    #[0,1,2,3,4,0,1,2,3,4,5,6,7,8,9]
    #print(l)

if __name__ == '__main__':
    with Manager() as manager:
        dic = manager.dict()#注意字典的声明方式，不能直接通过{}来定义
        l = manager.list(range(5))#[0,1,2,3,4]

        process_list = []
        for i in range(10):
            p = Process(target=fun1, args=(dic,l,i))
            p.start()
            process_list.append(p)

        for res in process_list:
            res.join()
        print(dic)
        print(l)

# {0: 'a', '2': 'b', 1: 'a', 2: 'a', 3: 'a', 4: 'a', 5: 'a', 6: 'a', 7: 'a', 8: 'a', 9: 'a'}
# [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



##===========================================================================
# https://segmentfault.com/q/1010000010403117
import multiprocessing


def worker(procnum, return_dict):
    '''worker function'''
    print(f'{procnum} represent!')
    return_dict[procnum] = procnum
    return

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print (return_dict )
# 0 represent!
# 1 represent!
# 2 represent!
# 3 represent!
# 4 represent!
# {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


# https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd
import multiprocessing
import numpy as np


def f(x, arr, l, d, n):
    x.value = 3.14
    arr[0] = 5
    l.append('Hello')
    d[1] = 2
    n.a = 10

if __name__ == '__main__':
    server = multiprocessing.Manager()
    x = server.Value('d', 0.0)
    arr = server.Array('i', range(10))
    l = server.list()
    d = server.dict()
    n = server.Namespace()
    proc = multiprocessing.Process(target=f, args=(x, arr, l, d, n))
    proc.start()
    proc.join()
    print(f"x.value = {x.value}")
    print(f"arr = {arr}")
    print(f"arr[2] = {arr[2]}")
    print(f"np.array(arr) = {np.array(arr)}")
    print(l)
    print(d)
    print(n)
# x.value = 3.14
# arr = array('i', [5, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# arr[2] = 2
# np.array(arr) = [5 1 2 3 4 5 6 7 8 9]
# ['Hello']
# {1: 2}
# Namespace(a=10)


##===========================================================================
def f(i, x, arr, l, L, d, n):
    x.value += i
    arr[i] = i
    l.append(f'Hello {i}')
    L[i] = 2*i
    d[f"{i}"] = i**2
    n.a = 10

if __name__ == '__main__':
    N = 10
    server = multiprocessing.Manager()
    x = server.Value('d', 0.0)
    arr = server.Array('i', range(N))
    l = server.list()
    L = server.list(range(10))
    d = server.dict()
    n = server.Namespace()

    process_list = []
    for i in range(N):
        p = multiprocessing.Process(target=f, args=(i, x, arr, l, L, d, n))
        p.start()
        process_list.append(p)

    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    print(f"x.value = {x.value}")
    print(f"arr = {arr}")
    print(f"arr[2] = {arr[2]}")
    print(f"np.array(arr) = {np.array(arr)}")
    print(l)
    print(d)
    print(n)

# x.value = 45.0
# arr = array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# arr[2] = 2
# np.array(arr) = [0 1 2 3 4 5 6 7 8 9]
# ['Hello 0', 'Hello 1', 'Hello 2', 'Hello 3', 'Hello 4', 'Hello 5', 'Hello 6', 'Hello 7', 'Hello 8', 'Hello 9']
# {'0': 0, '1': 1, '2': 4, '3': 9, '4': 16, '5': 25, '6': 36, '7': 49, '8': 64, '9': 81}
# Namespace(a=10)





##===========================================================================
# https://www.cnblogs.com/linhaifeng/articles/7428874.html#_label5


from multiprocessing import Manager,Process,Lock
import os
def work(d,lock):
    # with lock: #不加锁而操作共享的数据,肯定会出现数据错乱
        d['count']-=1

if __name__ == '__main__':
    lock=Lock()
    with Manager() as m:
        dic = m.dict({'count':100})
        p_l=[]
        for i in range(100):
            p=Process(target=work,args=(dic,lock))
            p_l.append(p)
            p.start()
        for p in p_l:
            p.join()
        print(dic)


# {'count': 0}























