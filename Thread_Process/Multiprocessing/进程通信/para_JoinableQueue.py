#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:28:49 2022

@author: jack
"""

from multiprocessing import Process, JoinableQueue

import time, random

def consumer(q):

    while True:

        res = q.get()

        print('消费者拿到了 %s' % res)

        q.task_done()

def producer(seq, q):

    for item in seq:

        time.sleep(random.randrange(1,2))

        q.put(item)

        print('生产者做好了 %s' % item)

    q.join()

if __name__ == "__main__":

    q = JoinableQueue()

    seq = ('产品%s' % i for i in range(5))

    p = Process(target=consumer, args=(q,))

    p.daemon = True  # 设置为守护进程，在主线程停止时p也停止，但是不用担心，producer内调用q.join保证了consumer已经处理完队列中的所有元素

    p.start()

    producer(seq, q)

    print('主线程')





#===============================================================================
# https://www.cnblogs.com/linhaifeng/articles/7428874.html
#===============================================================================


from multiprocessing import Process,JoinableQueue
import time,random,os
def consumer(q):
    while True:
        res=q.get()
        time.sleep(random.randint(1,3))
        print('\033[45m%s 吃 %s\033[0m' %(os.getpid(),res))

        q.task_done() #向q.join()发送一次信号,证明一个数据已经被取走了

def producer(name,q):
    for i in range(3):
        time.sleep(random.randint(1,3))
        res='%s%s' %(name,i)
        q.put(res)
        print('\033[44m%s 生产了 %s\033[0m' %(os.getpid(),res))
    q.join()


if __name__ == '__main__':
    q=JoinableQueue()
    #生产者们:即厨师们
    p1=Process(target=producer,args=('包子',q))
    p2=Process(target=producer,args=('骨头',q))
    p3=Process(target=producer,args=('泔水',q))

    #消费者们:即吃货们
    c1=Process(target=consumer,args=(q,))
    c2=Process(target=consumer,args=(q,))
    c1.daemon=True
    c2.daemon=True

    #开始
    p_l=[p1,p2,p3,c1,c2]
    for p in p_l:
        p.start()

    p1.join()
    p2.join()
    p3.join()
    print('主')

    #主进程等--->p1,p2,p3等---->c1,c2
    #p1,p2,p3结束了,证明c1,c2肯定全都收完了p1,p2,p3发到队列的数据

# 3126777 生产了 包子0
# 3126780 吃 包子0
# 3126779 生产了 泔水0
# 3126777 生产了 包子1
# 3126778 生产了 骨头0
# 3126780 吃 包子1
# 3126779 生产了 泔水1
# 3126777 生产了 包子2
# 3126778 生产了 骨头1
# 3126781 吃 泔水0
# 3126780 吃 骨头0
# 3126778 生产了 骨头2
# 3126780 吃 包子2
# 3126781 吃 泔水13126779 生产了 泔水2

# 3126780 吃 骨头1
# 3126781 吃 骨头2
# 3126780 吃 泔水2
# 主






























