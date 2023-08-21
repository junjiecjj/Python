#!/usr/bin/env python
#-*-coding=utf-8-*-


###**********************************************************************
###   queue 与 Process
###**********************************************************************


# https://www.cnblogs.com/linhaifeng/articles/7428874.html
#*****************************************
from multiprocessing import Process,Queue
import time

q=Queue(3) ##  3 是队列中允许最大项数，省略则无大小限制。

q.put(3)
q.put(2)
q.put(1)

print(q.full())

print(q.get())
print(q.get())
print(q.get())

print(q.empty())
# True
# 3
# 2
# 1
# True


#*****************************************
from multiprocessing import Process,Queue
import time

q=Queue(10) ##  3 是队列中允许最大项数，省略则无大小限制。

a = {'a':1, 'b':2, 'c':3}
q.put(a)
a = {'aa':1, 'bb':2, 'cc':3}
q.put(a)
a = {'aaa':1, 'bbb':2, 'ccc':3}
q.put(a)
q.put(34)
q.put(98)

print(q.full())

print(q.get())
print(q.get())
print(q.get())
print(q.get())
print(q.get())
print(q.empty())
# {'a': 1, 'b': 2, 'c': 3}
# {'aa': 1, 'bb': 2, 'cc': 3}
# {'aaa': 1, 'bbb': 2, 'ccc': 3}
# 34
# 98
# True

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

from multiprocessing import Process,Queue
import time,random,os
def consumer(q):
    while True:
        res=q.get()
        if res is None:break #收到结束信号则结束
        time.sleep(random.randint(1,3))
        print('\n\033[36m%s 吃 %s\033[0m\n' %(os.getpid(),res))

def producer(name,q):
    for i in range(2):
        time.sleep(random.randint(1,3))
        res='%s%s' %(name,i)
        q.put(res)
        print('\n\033[32m%s 生产了 %s\033[0m\n' %(os.getpid(),res))

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
    q.put(None) #有几个消费者就应该发送几次结束信号None,否则无法退出队列。
    q.put(None) #发送结束信号
    c1.join()
    c2.join()
    print('主')
#**********************************************************************

from multiprocessing import Process,JoinableQueue
import time,random,os
def consumer(q):
    while True:
        res=q.get()
        time.sleep(random.randint(1,3))
        print('\n\033[36m%s 吃 %s\033[0m\n' %(os.getpid(),res))

        q.task_done() #向q.join()发送一次信号,证明一个数据已经被取走了

def producer(name,q):
    for i in range(2):
        time.sleep(random.randint(1,3))
        res='%s%s' %(name,i)
        q.put(res)
        print('\n\033[32m%s 生产了 %s\033[0m\n' %(os.getpid(),res))
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
    #因而c1,c2也没有存在的价值了,应该随着主进程的结束而结束,所以设置成守护进程






#===============================================================================
#  https://www.jb51.net/article/170581.htm
#===============================================================================



from multiprocessing import Queue
q = Queue() # 生成一个队列对象
# put方法是往队列里面放值
q.put('Cecilia陈')
q.put('xuchen')
q.put('喜陈')

# get方法是从队列里面取值
print(q.get())
print(q.get())
print(q.get())

q.put(5)
q.put(6)
print(f"q.get() = {q.get()}")
print(f"q.get() = {q.get()}")
print(f"q.get() = {q.get()}")  # 会停在这里，因为队列空了

#**********************************************************************

from multiprocessing import Queue
q = Queue(3)
q.put('Cecilia陈')
q.put('xuchen')
q.put('喜陈')
print(q.full()) # 判断队列是否满了 返回的是True/False
q.put(2) # 当队列已经满的时候，再放值,程序会阻塞，但不会结束


#**********************************************************************
from multiprocessing import Queue
q = Queue(3)
q.put('zhao',block=True,timeout=2)
q.put('zhao',block=True,timeout=2)
q.put('zhao',block=True,timeout=2)
q.put('zhao',block=True,timeout=5) # 此时程序将对等待5秒以后报错了

#**********************************************************************
from multiprocessing import Queue
q = Queue()
q.put('Cecilia陈')
print(q.get())
q.get(block=True,timeout=2) # 此时程序会等待2秒后，报错了，队列里面没有值了



#**********************************************************************
from multiprocessing import Queue
q = Queue(2)
q.put('Cecilia陈')
q.put('喜陈')
print(q.full())
q.put('xichen',block=False) # 队列已经满了，我不等待了，直接报错

#**********************************************************************
from multiprocessing import Queue
q = Queue(2)
q.put('Cecilia陈')
q.put('喜陈')
print(q.get())
print(q.get())
print(q.get(block=False)) # 队列已经没有值了，我不等待了，直接报错


#**********************************************************************
from multiprocessing import Queue
q = Queue(2)
q.put('Cecilia陈')
q.put('喜陈')
print(q.full())

q.put_nowait('xichen') # 程序不等待，不阻塞，直接报错






#**********************************************************************


from multiprocessing import Queue
q = Queue(2)
q.put('Cecilia陈')
q.put('喜陈')
print(q.get())
print(q.get())
print(q.full())
q.get_nowait()# 再取值的时候，程序不等待，不阻塞，程序直接报错




#**********************************************************************

'''
multiprocessing模块支持进程间通信的两种主要形式:管道和队列
都是基于消息传递实现的,但是队列接口  , 单看队列的存取数据用法
'''

from multiprocessing import Queue
q=Queue(3)

#put ,get ,put_nowait,get_nowait,full,empty
q.put(3)
q.put(3)
q.put(3)
# q.put(3)  # 如果队列已经满了，程序就会停在这里，等待数据被别人取走，再将数据放入队列。
      # 如果队列中的数据一直不被取走，程序就会永远停在这里。
try:
  q.put_nowait(3) # 可以使用put_nowait，如果队列满了不会阻塞，但是会因为队列满了而报错。
except: # 因此我们可以用一个try语句来处理这个错误。这样程序不会一直阻塞下去，但是会丢掉这个消息。
  print('队列已经满了')

# 因此，我们再放入数据之前，可以先看一下队列的状态，如果已经满了，就不继续put了。
print(q.full()) #满了
print(q.get())
print(q.get())
print(q.get())
# print(q.get()) # 同put方法一样，如果队列已经空了，那么继续取就会出现阻塞。
try:
  q.get_nowait(3) # 可以使用get_nowait，如果队列满了不会阻塞，但是会因为没取到值而报错。
except: # 因此我们可以用一个try语句来处理这个错误。这样程序不会一直阻塞下去。
  print('队列已经空了')

print(q.empty()) #空了





#**********************************************************************
# 子进程向父进程发送数据
# 这是一个queue的简单应用，使用队列q对象调用get函数来取得队列中最先进入的数据。
from multiprocessing import Process, Queue
def f(q,name,age):
  q.put(name,age) #调用主函数中p进程传递过来的进程参数 put函数为向队列中添加一条数据。
if __name__ == '__main__':
  q = Queue() #创建一个Queue对象
  p = Process(target=f, args=(q,'Cecilia陈',18)) #创建一个进程
  p.start()
  print(q.get())
  p.join()




#**********************************************************************
# 基于Queue队列实现的生产者消费者模型
from multiprocessing import Queue,Process
# 生产者
def producer(q,name,food):
  for i in range(3):
    print(f'{name}生产了{food}{i}')
    res = f'{food}{i}'
    q.put(res)
# 消费者
def consumer(q,name):
  while True:
    res = q.get(timeout=5)
    print(f'{name}吃了{res}')
if __name__ == '__main__':
  q = Queue() # 为的是让生产者和消费者使用同一个队列，使用同一个队列进行通讯
  p1 = Process(target=producer,args=(q,'Cecilia陈','巧克力'))
  c1 = Process(target=consumer,args=(q,'Tom'))
  p1.start()
  c1.start()
  p1.join()
  c1.join()




#**********************************************************************

from multiprocessing import Queue,Process
def producer(q,name,food):
  for i in range(3):
    print(f'{name}生产了{food}{i}')
    res = f'{food}{i}'
    q.put(res)
  q.put(None) # 当生产者结束生产的的时候，我们再队列的最后再做一个表示，告诉消费者，生产者已经不生产了，让消费者不要再去队列里拿东西了
def consumer(q,name):
  while True:
    res = q.get(timeout=5)
    if res == None:break # 判断队列拿出的是不是生产者放的结束生产的标识，如果是则不取，直接退出，结束程序
    print(f'{name}吃了{res}')
if __name__ == '__main__':
  q = Queue() # 为的是让生产者和消费者使用同一个队列，使用同一个队列进行通讯
  p1 = Process(target=producer,args=(q,'Cecilia陈','巧克力'))
  c1 = Process(target=consumer,args=(q,'Tom'))
  p1.start()
  c1.start()






#**********************************************************************

from multiprocessing import Queue,Process
import time,random

def producer(q,name,food):
  for i in range(3):
    print(f'{name}生产了{food}{i}')
    time.sleep((random.randint(1,3)))
    res = f'{food}{i}'
    q.put(res)
  # q.put(None) # 当生产者结束生产的的时候，我们再队列的最后再做一个表示，告诉消费者，生产者已经不生产了，让消费者不要再去队列里拿东西了



def consumer(q,name):
  while True:
    res = q.get(timeout=5)
    if res == None:break # 判断队列拿出的是不是生产者放的结束生产的标识，如果是则不取，直接退出，结束程序
    time.sleep((random.randint(1, 3)))
    print(f'{name}吃了{res}')

if __name__ == '__main__':
  q = Queue() # 为的是让生产者和消费者使用同一个队列，使用同一个队列进行通讯
  # 多个生产者进程
  p1 = Process(target=producer,args=(q,'Cecilia陈','巧克力'))
  p2 = Process(target=producer,args=(q,'xichen','冰激凌'))
  p3 = Process(target=producer,args=(q,'喜陈','可乐'))
  # 多个消费者进程
  c1 = Process(target=consumer,args=(q,'Tom'))
  c2 = Process(target=consumer,args=(q,'jack'))


  # 告诉操作系统启动生产者进程
  p1.start()
  p2.start()
  p3.start()

  # 告诉操作系统启动消费者进程
  c1.start()
  c2.start()

  p1.join()
  p2.join()
  p3.join()

  q.put(None) # 几个消费者put几次
  q.put(None)






#**********************************************************************

from multiprocessing import Queue,Process,JoinableQueue
import time,random

def producer(q,name,food):
  for i in range(3):
    print(f'{name}生产了{food}{i}')
    # time.sleep((random.randint(1,3)))
    res = f'{food}{i}'
    q.put(res)
  # q.put(None) # 当生产者结束生产的的时候，我们再队列的最后再做一个表示，告诉消费者，生产者已经不生产了，让消费者不要再去队列里拿东西了
  q.join()


def consumer(q,name):
  while True:
    res = q.get(timeout=5)
    # if res == None:break # 判断队列拿出的是不是生产者放的结束生产的标识，如果是则不取，直接退出，结束程序
    # time.sleep((random.randint(1, 3)))
    print(f'{name}吃了{res}')
    q.task_done()#向q.join()发送一次信号,证明一个数据已经被取走了


if __name__ == '__main__':
  q = JoinableQueue() # 为的是让生产者和消费者使用同一个队列，使用同一个队列进行通讯
  # 多个生产者进程
  p1 = Process(target=producer,args=(q,'Cecilia陈','巧克力'))
  p2 = Process(target=producer,args=(q,'xichen','冰激凌'))
  p3 = Process(target=producer,args=(q,'喜陈','可乐'))
  # 多个消费者进程
  c1 = Process(target=consumer,args=(q,'Tom'))
  c2 = Process(target=consumer,args=(q,'jack'))


  # 告诉操作系统启动生产者进程
  p1.start()
  p2.start()
  p3.start()

  # 把生产者设为守护进程
  c1.daemon = True
  c2.daemon = True
  # 告诉操作系统启动消费者进程
  c1.start()
  c2.start()

  p1.join()
  p2.join()
  p3.join() # 等待生产者生产完毕

  print('主进程')

  ### 分析
  # 生产者生产完毕--这是主进程最后一行代码结束--q.join()消费者已经取干净了,没有存在的意义了
  # 这是主进程最后一行代码结束,消费者已经取干净了,没有存在的意义了.守护进程的概念.






#**********************************************************************


from multiprocessing import Process,Queue,JoinableQueue
q = JoinableQueue()
q.put('zhao') # 放队列里一个任务
q.put('qian')
print(q.get())
q.task_done() # 完成了一次任务
print(q.get())
q.task_done() # 完成了一次任务
q.join() #计数器不为0的时候 阻塞等待计数器为0后通过

# 想象成一个计数器 :put +1  task_done -1





#**********************************************************************
#=======================================================================
# https://www.jianshu.com/p/5781af162692
#=======================================================================

#coding=utf-8
from multiprocessing import Queue
q=Queue(3) #初始化一个Queue对象，最多可接收三条put消息
q.put("消息1")
q.put("消息2")
print(q.full())  #False
q.put("消息3")
print(q.full()) #True

#因为消息列队已满下面的try都会抛出异常，第一个try会等待2秒后再抛出异常，第二个Try会立刻抛出异常
try:
    q.put("消息4",True,2)
except:
    print("消息列队已满，现有消息数量:%s"%q.qsize())

try:
    q.put_nowait("消息4")
except:
    print("消息列队已满，现有消息数量:%s"%q.qsize())

#推荐的方式，先判断消息列队是否已满，再写入
if not q.full():
    q.put_nowait("消息4")

#读取消息时，先判断消息列队是否为空，再读取
if not q.empty():
    for i in range(q.qsize()):
        print(q.get_nowait())





#**********************************************************************

from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q):
    for value in ['A', 'B', 'C']:
        print( 'Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    while True:
        if not q.empty():
            value = q.get(True)
            print( 'Get %s from queue.' % value)
            time.sleep(random.random())
        # else:
            # break

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    pr.terminate()  # pr进程里是死循环，无法等待其结束，只能强行终止

    print( '所有数据都写入并且读完')



#**********************************************************************




#**********************************************************************
# https://juejin.cn/post/6844903450065502215
#**********************************************************************
import multiprocessing
#向队列中写入数据
N=8
def work_1(q):
       try:
           n=1
           while n<N:
               print("work_1,%d"%n)
               q.put(n)
               time.sleep(1)
               n+=1
       except BaseException:
            print("work_1 error")
       finally:
           print("work_1 end!!!")

#取出队列中的数据
def work_2(q):
       try:
           n=1
           while n<N:
                print("word_2,%d"%q.get())
                time.sleep(1)
                n+=1
       except BaseException:
           print("work_2 error")
       finally:
           print("work_2 end")

if __name__ == "__main__":
      q= multiprocessing.Queue()
      p1=multiprocessing.Process(target=work_1,args=(q,))
      p2=multiprocessing.Process(target=work_2,args=(q,))
      p1.start()
      p2.start()
      p1.join()
      p2.join()
      print("all over")



###**********************************************************************
###   queue 与 pool, 最好别联合使用 pool和queue，虽然程序不报错，但是结果不是想要的，
###**********************************************************************

#修改import中的Queue为Manager
from multiprocessing import Manager,Pool
import os,time,random

def reader(q):
    print("reader启动(%s),父进程为(%s)\n"%(os.getpid(),os.getppid()))
    for i in range(q.qsize()):
        print("reader从Queue获取到消息：%s"%q.get(True))

def writer(q):
    print("writer启动(%s),父进程为(%s)\n"%(os.getpid(),os.getppid()))
    for i in "dongGe":
        q.put(i)

if __name__=="__main__":
    print("(%s) start"%os.getpid())
    q = Manager().Queue() #使用Manager中的Queue来初始化
    po = Pool()
    #使用阻塞模式创建进程，这样就不需要在reader中使用死循环了，可以让writer完全执行完成后，再用reader去读取
    po.apply(writer,(q,))
    po.apply(reader,(q,))
    po.close()
    po.join()
    print("(%s) End"%os.getpid())


# (2081851) start
# writer启动(3107348),父进程为(2081851)
# reader启动(3107350),父进程为(2081851)
# reader从Queue获取到消息：d
# reader从Queue获取到消息：o
# reader从Queue获取到消息：n
# reader从Queue获取到消息：g
# reader从Queue获取到消息：G
# reader从Queue获取到消息：e
# (2081851) End


## 这里是对的是凑巧，正好是 apply同步且数据较少，一旦数据多就会错

#**********************************************************************

#修改import中的Queue为Manager
from multiprocessing import Manager,Pool
import os,time,random

def reader(q):
    print("reader启动(%s),父进程为(%s)\n"%(os.getpid(),os.getppid()))
    for i in range(q.qsize()):
        print("reader从Queue获取到消息：%s"%q.get(True))

def writer(q):
    print("writer启动(%s),父进程为(%s)\n"%(os.getpid(),os.getppid()))
    for i in "dongGe":
        q.put(i)

if __name__=="__main__":
    print("(%s) start"%os.getpid())
    q = Manager().Queue() #使用Manager中的Queue来初始化
    po = Pool()
    #使用阻塞模式创建进程，这样就不需要在reader中使用死循环了，可以让writer完全执行完成后，再用reader去读取
    po.apply_async(writer,(q,))
    po.apply_async(reader,(q,))
    po.close()
    po.join()
    print("(%s) End"%os.getpid())

# (4066976) start
# reader启动(4185703),父进程为(4066976)
# writer启动(4185700),父进程为(4066976)


# reader从Queue获取到消息：d
# (4066976) End

## 这里结果往往不是预期想要的


#**********************************************************************








#**********************************************************************








#**********************************************************************








#**********************************************************************








#**********************************************************************








#**********************************************************************








#**********************************************************************








#**********************************************************************








#**********************************************************************




















































































