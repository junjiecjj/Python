#/usr/bin/env python3
# -*- coding: utf-8 -*-


### 这样测试程序的运行时间
###  /usr/bin/time -p python test.py



#*****************************************************
##  进程锁
##  例一
from multiprocessing import Process,Lock
import os,time
import random

def work(lock):
    lock.acquire()
    print('%s is running' %os.getpid())
    time.sleep(1)
    print('%s is done' %os.getpid())
    lock.release()

if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    lock=Lock()
    process_list = []
    for i in range(8):
        p=Process(target=work,args=(lock,))
        p.start()
        process_list.append(p)

    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

"""需要8s跑完"""

# 525452 is running
# 525452 is done
# 525453 is running
# 525453 is done
# 525454 is running
# 525454 is done
# 525455 is running
# 525455 is done
# 525456 is running
# 525456 is done
# 525457 is running
# 525457 is done
# 525458 is running
# 525458 is done
# 525459 is running
# 525459 is done
# 结束测试
# 8.016218423843384
# real 8.07
# user 0.06
# sys 0.02



#*******************************
## 例二
from multiprocessing import Process,Lock
import os,time
import random
def work(lock, i):
    time.sleep(1)
    lock.acquire()
    print('%s is running %d' %(os.getpid(), i))
    print('%s is done' %os.getpid())
    lock.release()

if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    lock = Lock()
    ps=[]
    for i in range(8):
        p=Process(target=work,args=(lock, i))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()
    print('结束测试')
    time_end = time.time()            # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)


# ❯ /usr/bin/time -p python test.py
# 640639 is running 0
# 640639 is done
# 640640 is running 1
# 640640 is done
# 640641 is running 2
# 640641 is done
# 640643 is running 3
# 640643 is done
# 640644 is running 4
# 640644 is done
# 640645 is running 5
# 640645 is done
# 640646 is running 6
# 640646 is done
# 640647 is running 7
# 640647 is done
# 结束测试
# 1.0968592166900635
# 结束测试
# 1.0122969150543213
# real 1.08
# user 0.06
# sys 0.03

#*********************************
"""
从上述例一和例二可以看出，lock()只在lock.acquire和lock.release之间有用，例2只需1s就可以。 所以，
 lock.acquire和lock.release之间的程序是穿行的，但是锁之外的程序还是并行的
"""

#**********************************************
# 写文件举例，多个进程共享同一文件
#例一

#文件db的内容为：{"count":300}
#注意一定要用双引号，不然json无法识别
from multiprocessing import Process,Lock
import time,json,random


def search():
    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f' 剩余票数%s \n' %dic['count'])

def get():
    dic=json.load(open('/home/jack/snap/db.txt'))
    time.sleep(0.1) #模拟读数据的网络延迟
    if dic['count'] >0:
        dic['count']-=1
        time.sleep(0.2) #模拟写数据的网络延迟
        json.dump(dic,open('/home/jack/snap/db.txt','w'))
        print(f' 购票成功 \n')

def task(lock):
    search()
    get()


if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f"count:{dic['count']}")
    lock=Lock()
    process_list=[]
    for i in range(10): #模拟并发100个客户端抢票
        p=Process(target=task,args=(lock,))
        p.start()
        process_list.append(p)
    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f"count:{dic['count']}")
    print(f'结束测试')

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)


# 结果为：
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
# count:299
# 结束测试
# 0.38925743103027344
# 运行时间为0.3 s，且db.txt文件的内容在运行完上述代码后变为{"count": 299}
# 并发运行，效率高，但竞争写同一文件，数据写入错乱
## 可以看出，如果没加锁，则读取和写入会不按预期进行，读取的都是300，因为读取很快，而没加锁，所以都是读出300，最后写也没加锁，都是写入299
#**********************************************
#例二

#文件db的内容为：{"count":300}
#注意一定要用双引号，不然json无法识别
from multiprocessing import Process,Lock
import time,json,random

def search():
    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f' 剩余票数%s ' %dic['count'])

def get():
    dic=json.load(open('/home/jack/snap/db.txt'))
    time.sleep(0.1) #模拟读数据的网络延迟
    if dic['count'] >0:
        dic['count']-=1
        time.sleep(0.2) #模拟写数据的网络延迟
        json.dump(dic,open('/home/jack/snap/db.txt','w'))
        print(f' 购票成功 ')

def task(lock):
    search()
    lock.acquire()
    get()
    lock.release()


if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f"count:{dic['count']}")
    lock=Lock()
    process_list=[]
    for i in range(10): #模拟并发100个客户端抢票
        p=Process(target=task,args=(lock,))
        p.start()
        process_list.append(p)
    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f"count:{dic['count']}")
    print(f'结束测试')

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

# count:300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  剩余票数300
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
#  购票成功
# count:290
# 结束测试
# 3.102154016494751

# 运行时间为3 s，且db.txt文件的内容在运行完上述代码后变为{"count": 290}
## 虽然以上过程最后的票数290看着符合预期，但是还是很危险的，因为他们读取的都是300，这里最后的票数正确的原因是因为进程数较少，如果多很可能错，所以，正确的做法如下：


#加锁：购票行为由并发变成了串行，牺牲了运行效率，但保证了数据安全
#****************************************************
#例三

#文件db的内容为：{"count":300}
#注意一定要用双引号，不然json无法识别

from multiprocessing import Process,Lock
import time,json,random

def search():
    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f' 剩余票数%s ' %dic['count'])

def get():
    dic=json.load(open('/home/jack/snap/db.txt'))
    time.sleep(0.1) #模拟读数据的网络延迟
    if dic['count'] >0:
        dic['count']-=1
        time.sleep(0.2) #模拟写数据的网络延迟
        json.dump(dic,open('/home/jack/snap/db.txt','w'))
        print(f' 购票成功 ')

def task(lock):
    lock.acquire()
    search()
    get()
    lock.release()


if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f"count:{dic['count']}")
    lock=Lock()
    process_list=[]
    for i in range(10): #模拟并发100个客户端抢票
        p=Process(target=task,args=(lock,))
        p.start()
        process_list.append(p)
    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    dic=json.load(open('/home/jack/snap/db.txt'))
    print(f"count:{dic['count']}")
    print(f'结束测试')

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

# count:300
#  剩余票数300
#  购票成功
#  剩余票数299
#  购票成功
#  剩余票数298
#  购票成功
#  剩余票数297
#  购票成功
#  剩余票数296
#  购票成功
#  剩余票数295
#  购票成功
#  剩余票数294
#  购票成功
#  剩余票数293
#  购票成功
#  剩余票数292
#  购票成功
#  剩余票数291
#  购票成功
# count:290
# 结束测试
# 3.1872072219848633

# 运行时间为3 s，且db.txt文件的内容在运行完上述代码后变为{"count": 290}

"""
注意上述例二和例三的区别，search()是否在lock.acquire和lock.release()之间对程序的打印影响很大
例二很危险的，别用，这样读写加锁只能用例三。

"""
#************************************************

#https://www.cnblogs.com/gregoryli/p/7892222.html
#************************************

#**********************************************
#http://python.jobbole.com/82045/
import multiprocessing
import sys
import time

def worker_with(lock, f):
    print("work_with start")
    time.sleep(2)
    with lock:
        fs = open(f, 'a+')
        n = 10
        while n > 1:
            fs.write("Lockd acquired via with\n")
            n -= 1
        fs.close()
    print("work_with end")

def worker_no_with(lock, f):
    print("work_no_with start")
    time.sleep(4)
    lock.acquire()
    try:
        fs = open(f, 'a+')
        n = 10
        while n > 1:
            fs.write("Lock acquired directly\n")
            n -= 1
        fs.close()
    finally:
        lock.release()
    print("work_no_with end")

if __name__ == "__main__":
    lock = multiprocessing.Lock()
    f = '/home/jack/文档/test.txt'
    w = multiprocessing.Process(target = worker_with, args=(lock, f))
    nw = multiprocessing.Process(target = worker_no_with, args=(lock, f))
    w.start()
    nw.start()
    w.join()
    nw.join()
    print("end!!!")


# 结果为：

# work_with start
# work_no_with start
# work_with end
# work_no_with end
# end!!!

# real	0m4.123s
# user	0m0.057s
# sys	0m0.024s

# test.txt文件的内容为：

# Lockd acquired via with
# Lockd acquired via with
# Lockd acquired via with
# Lockd acquired via with
# Lockd acquired via with
# Lockd acquired via with
# Lockd acquired via with
# Lockd acquired via with
# Lockd acquired via with
# Lock acquired directly
# Lock acquired directly
# Lock acquired directly
# Lock acquired directly
# Lock acquired directly
# Lock acquired directly
# Lock acquired directly
# Lock acquired directly
# Lock acquired directly
"""
此代码的运行时间为4s,说明‘锁’的作用范围是lock.acquire()和lock.release()之间，在lock.acquire()之前的
运算还是并行的,但是前提是没有加join()，否则，在lock>acquire()之前和lock.release之后的程还是变为
穿行的；
此代码验证了：虽然有锁lock，但是写文件的时间极端，可以在写之前进行其他运算，这时其他运算还是并行的
"""
#*****************************************************
import multiprocessing
import sys
import time
import random
from multiprocessing import Process,Pool

def worker_with(lock,i):
    name = multiprocessing.current_process().name
    start = time.ctime()
    print("process %s with lock satrt"%name)
    a=i
    b=i+1
    c=i-1
    d=i*2
    time.sleep(i)
    end = time.ctime()
    lock.acquire()
    with open('/home/jack/桌面/test.txt','a') as f:
        f.write("*********分割线**************\n")
        f.write("process %s start at %s end at %s.\n" % (name,start,end))
        f.write("a is %f.\n"%a)
        f.write("b is %f.\n"%b)
        f.write("c is %f.\n"%c)
        f.write("d is %f.\n"%d)
    lock.release()
    print("process %s with lock end.."%name)

def worker_without(i):
    name = multiprocessing.current_process().name
    start = time.ctime()
    print("process %s without lock satrt"%name)
    a=i
    b=i+1
    c=i-1
    d=i*2
    #time.sleep(random.randrange(0,3))
    end = time.ctime()
    with open('/home/jack/桌面/test.txt','a') as f:
        f.write("**********分割线**************\n")
        f.write("process %s start at %s end at %s.\n" % (name,start,end))
        f.write("a is %f.\n"%a)
        f.write("b is %f.\n"%b)
        f.write("c is %f.\n"%c)
        f.write("d is %f.\n"%d)
    print("process %s without lock end.."%name)

def main():
    lock = multiprocessing.Lock()
    p_list=[]
    for i in range(20):
        # p  = multiprocessing.Process(target = worker_with, args=(lock,i,))
        p = multiprocessing.Process(target = worker_without, args=(i,))
        #p1.start()
        p.start()
        p_list.append(p)
        #p_list.append(p1)#
        #p.join()

    for i in range(20):
        p_list[i].join()

    print("end!!!")

if __name__=='__main__':
    main()



##  用 worker_with 加锁的
