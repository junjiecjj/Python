#!/usr/bin/env python
#-*-coding=utf-8-*-
'''
#https://www.cnblogs.com/linhaifeng/articles/7428874.html
#*****************************************
from multiprocessing import Process,Queue
import time

q=Queue(3)

q.put(3)
q.put(2)
q.put(1)

print(q.full())

print(q.get())
print(q.get())
print(q.get())

print(q.empty())


#************************************

from multiprocessing import Process,Queue
import time,random,os

def consumer(q):
    while True:
        res=q.get()
        if  res is None:#收到结束信息则结束
            break
        time.sleep(random.randint(1,3))
        print('\033[45m%s 吃 %s\033[0m'% (os.getpid(),res))

def producer(q):
    for i in range(10):
        time.sleep(random.randint(1,2))
        res='包子%s'%i
        q.put(res)
        print('\033[44m%s 生产了 %s\033[0m'%(os.getpid(),res))
    q.put(None) #发送结束信号

if __name__=='__main__':

    q=Queue()

    p1=Process(target=producer,args=(q,)) #生产者

    c1=Process(target=consumer,args=(q,)) #消费者

    p1.start()
    c1.start()
    p1.join()
    c1.join()
    print("主")

#************************************************
'''
from multiprocessing import Process,Queue
import time,random,os
def consumer(q):
    while True:
        res=q.get()
        if res is None:break #收到结束信号则结束
        time.sleep(random.randint(1,3))
        print('\033[45m%s 吃 %s\033[0m' %(os.getpid(),res))

def producer(name,q):
    for i in range(2):
        time.sleep(random.randint(1,3))
        res='%s%s' %(name,i)
        q.put(res)
        print('\033[44m%s 生产了 %s\033[0m' %(os.getpid(),res))



if __name__ == '__main__':
    q=Queue()
    #生产者们:即厨师们
    p1=Process(target=producer,args=('包子',q))
    p2=Process(target=producer,args=('骨头',q))
    p3=Process(target=producer,args=('泔水',q))

    #消费者们:即吃货们
    c1=Process(target=consumer,args=(q,))
    c2=Process(target=consumer,args=(q,))

    #开始
    p1.start()
    p2.start()
    p3.start()
    c1.start()
    c2.start()

    p1.join() #必须保证生产者全部生产完毕,才应该发送结束信号
    p2.join()
    p3.join()
    q.put(None) #有几个消费者就应该发送几次结束信号None
    q.put(None) #发送结束信号
    print('主')
#**********************************************************************




