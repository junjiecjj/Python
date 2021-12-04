#/usr/bin/env python3
# -*- coding: utf-8 -*-

#https://www.cnblogs.com/linhaifeng/articles/7428874.html#_label5
#***************************************************************
#例一
#并发运行,效率高,但竞争同一打印终端,带来了打印错乱

from multiprocessing import Process
import os,time
import random

def work():
    print('%s is running' %os.getpid())
    time.sleep(1)
    #注意，这里如果是sleep(1)则看不出来错乱；但是如果是sleep(random.randrange(0,2))则看得出错乱
    print('%s is done' %os.getpid())

if __name__ == '__main__':
    for i in range(8):
        p=Process(target=work)
        p.start()
        p.join()  #加了join()就变成串行了，且不会错乱，需要8s跑完；不加为并行，1s跑完
    print("end")

结果：
加join()

2841 is running
2841 is done
2842 is running
2842 is done
2843 is running
2843 is done
2844 is running
2844 is done
2847 is running
2847 is done
2848 is running
2848 is done
2850 is running
2850 is done
2851 is running
2851 is done
end

real	0m8.111s
user	0m0.068s
sys	0m0.038s

结果：不加join()


29327 is running
29328 is running
29329 is running
29330 is running
29331 is running
29332 is running
29333 is running
end
29334 is running
29327 is done
29328 is done
29329 is done
29330 is done
29331 is done
29332 is done
29333 is done
29334 is done

real	0m1.073s
user	0m0.073s
sys	0m0.012s
#***************************************************
from multiprocessing import Process
import os,time
import random

def work():
    print('%s is running' %os.getpid())
    time.sleep(1)
    #注意，这里如果是sleep(1)则看不出来错乱；但是如果是sleep(random.randrange(0,2))则看得出错乱
    print('%s is done' %os.getpid())

if __name__ == '__main__':
    ps=[]
    for i in range(8):
        p=Process(target=work)
        p.start()
        ps.append(p)

    for i in range(8):
        ps[i].join() #加了join()就变成串行了，且不会错乱，需要8s跑完；不加为并行，1s跑完
    print("end")

结果：

1096 is running
1097 is running
1098 is running
1099 is running
1100 is running
1101 is running
1102 is running
1103 is running
1097 is done
1096 is done
1098 is done
1099 is done
1100 is done
1101 is done
1102 is done
1103 is done
end

real	0m1.078s
user	0m0.082s
sys	0m0.010s
#*************************************************
from multiprocessing import Process
import os,time
import random

def work():
    time.sleep(1)
    print('%s is running' %os.getpid())
    #注意，这里如果是sleep(1)则看不出来错乱；但是如果是sleep(random.randrange(0,2))则看得出错乱
    print('%s is done' %os.getpid())

if __name__ == '__main__':
    for i in range(8):
        p=Process(target=work)
        p.start()
        p.join()  #加了join()就变成串行了，且不会错乱，需要8s跑完；不加为并行，1s跑完
    print("end")

结果为:

22009 is running
22009 is done
22017 is running
22017 is done
22018 is running
22018 is done
22019 is running
22019 is done
22020 is running
22020 is done
22021 is running
22021 is done
22022 is running
22022 is done
22023 is running
22023 is done
end

real	0m8.110s
user	0m0.072s
sys	0m0.034s

不加join()，结果为:

end
22745 is running
22745 is done
22746 is running
22746 is done
22747 is running
22747 is done
22748 is running
22748 is done
22749 is running
22749 is done
22750 is running
22750 is done
22751 is running
22751 is done
22752 is running
22752 is done

real	0m1.074s
user	0m0.064s
sys	0m0.023s
#****************************
from multiprocessing import Process
import os,time
import random

def work():
    time.sleep(1)
    print('%s is running' %os.getpid())
    #注意，这里如果是sleep(1)则看不出来错乱；但是如果是sleep(random.randrange(0,2))则看得出错乱
    print('%s is done' %os.getpid())

if __name__ == '__main__':
    ps=[]
    for i in range(8):
        p=Process(target=work)
        p.start()
        ps.append(p)

    for i in range(8):
        ps[i].join()
    print("end")

结果为:

1275 is running
1274 is running
1275 is done
1274 is done
1276 is running
1276 is done
1277 is running
1277 is done
1278 is running
1278 is done
1279 is running
1279 is done
1280 is running
1280 is done
1281 is running
1281 is done
end

real	0m1.073s
user	0m0.064s
sys	0m0.023s
#*****************************************************
#由并发变成了串行,牺牲了运行效率,但避免了竞争
#例三
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
    lock=Lock()
    for i in range(8):
        p=Process(target=work,args=(lock,))
        p.start()
        #p.join()
    print("end!!!!!")
"""需要8s跑完"""

结果：

31877 is running
end!!!!!
31877 is done
31878 is running
31878 is done
31879 is running
31879 is done
31880 is running
31880 is done
31881 is running
31881 is done
31882 is running
31882 is done
31883 is running
31883 is done
31884 is running
31884 is done

real	0m8.093s
user	0m0.082s
sys	0m0.022s


加了join()，结果为：

31962 is running
31962 is done
31963 is running
31963 is done
31964 is running
31964 is done
31965 is running
31965 is done
31966 is running
31966 is done
31967 is running
31967 is done
31968 is running
31968 is done
31969 is running
31969 is done
end!!!!!

real	0m8.115s
user	0m0.088s
sys	0m0.024s

#******************************************
#例四
from multiprocessing import Process,Lock
import os,time
import random
def work(lock):
    time.sleep(1)
    lock.acquire()
    print('%s is running' %os.getpid())
    print('%s is done' %os.getpid())
    lock.release()

if __name__ == '__main__':
    lock=Lock()
    for i in range(8):
        p=Process(target=work,args=(lock,))
        p.start()
        #p.join() #不加join()为并行，加了为穿行
    print("end!!!!!!!!!1")

"""需要1s跑完"""
结果：

end!!!!!!!!!1
32217 is running
32217 is done
32219 is running
32219 is done
32218 is running
32218 is done
32220 is running
32220 is done
32221 is running
32221 is done
32222 is running
32222 is done
32223 is running
32223 is done
32224 is running
32224 is done

real	0m1.080s
user	0m0.069s
sys	0m0.024s


加join()，结果为：

32302 is running
32302 is done
32303 is running
32303 is done
32304 is running
32304 is done
32305 is running
32305 is done
32306 is running
32306 is done
32341 is running
32341 is done
32342 is running
32342 is done
32343 is running
32343 is done
end!!!!!!!!!1

real	0m8.105s
user	0m0.071s
sys	0m0.029s

#*******************************
例五
from multiprocessing import Process,Lock
import os,time
import random
def work(lock):
    time.sleep(1)
    lock.acquire()
    print('%s is running' %os.getpid())
    print('%s is done' %os.getpid())
    lock.release()

if __name__ == '__main__':
    lock=Lock()
    ps=[]
    for i in range(8):
        p=Process(target=work,args=(lock,))
        ps.append(p)
        p.start()

    for p in ps:
        p.join() 

结果为：

30337 is running
30337 is done
30338 is running
30338 is done
30339 is running
30339 is done
30340 is running
30340 is done
30341 is running
30341 is done
30342 is running
30342 is done
30343 is running
30343 is done
30344 is running
30344 is done

real	0m1.084s
user	0m0.082s
sys	0m0.017s
#*********************************
"""
从上述例二和例、例三、例四、例五可以看出，lock()只在lock.acquire和lock.release之间有用，例四只需1s
就可以。但是例四加了join()之后还是变成了穿行，所以，结论如下：
(一):如果没有加join()，则lock.acquire和lock.release之间的程序是穿行的，但是锁之外的程序还是并行的
(二):如果加了join()，则lock.acquire和lock.release之间的程序是穿行的，且锁之外的程序还是穿行的
(三):但是最后是穿行还是并行与加join()的方式有关，看例四加join()和例五
"""

#**********************************************
写文件举例，多个进程共享同一文件
#例一

#文件db的内容为：{"count":300}
#注意一定要用双引号，不然json无法识别
from multiprocessing import Process,Lock
import time,json,random
def search():
    dic=json.load(open('/home/jack/文档/db.txt'))
    print('\033[43m剩余票数%s\033[0m' %dic['count'])

def get():
    dic=json.load(open('/home/jack/文档/db.txt'))
    time.sleep(0.1) #模拟读数据的网络延迟
    if dic['count'] >0:
        dic['count']-=1
        time.sleep(0.2) #模拟写数据的网络延迟
        json.dump(dic,open('/home/jack/文档/db.txt','w'))
        print('\033[43m购票成功\033[0m')

def task(lock):
    search()
    get()
if __name__ == '__main__':
    lock=Lock()
    for i in range(10): #模拟并发100个客户端抢票
        p=Process(target=task,args=(lock,))
        p.start()

结果为：


剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功

运行时间为0.3 s，且db.txt文件的内容在运行完上述代码后变为{"count": 299}
#并发运行，效率高，但竞争写同一文件，数据写入错乱

#**********************************************
#例二

#文件db的内容为：{"count":300}
#注意一定要用双引号，不然json无法识别
from multiprocessing import Process,Lock
import time,json,random
def search():
    dic=json.load(open('/home/jack/文档/db.txt'))
    print('\033[43m剩余票数%s\033[0m' %dic['count'])

def get():
    dic=json.load(open('/home/jack/文档/db.txt'))
    time.sleep(0.1) #模拟读数据的网络延迟
    if dic['count'] >0:
        dic['count']-=1
        time.sleep(0.2) #模拟写数据的网络延迟
        json.dump(dic,open('/home/jack/文档/db.txt','w'))
        print('\033[43m购票成功\033[0m')

def task(lock):
    search()
    lock.acquire()
    get()
    lock.release()
if __name__ == '__main__':
    lock=Lock()
    for i in range(10): #模拟并发100个客户端抢票
        p=Process(target=task,args=(lock,))
        p.start()

结果为：

剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
剩余票数300
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功
购票成功

运行时间为3 s，且db.txt文件的内容在运行完上述代码后变为{"count": 290}

#加锁：购票行为由并发变成了串行，牺牲了运行效率，但保证了数据安全
#****************************************************
#例三

#文件db的内容为：{"count":300}
#注意一定要用双引号，不然json无法识别

from multiprocessing import Process,Lock
import time,json,random
def search():
    dic=json.load(open('/home/jack/文档/db.txt'))
    print('\033[43m剩余票数%s\033[0m' %dic['count'])

def get():
    dic=json.load(open('/home/jack/文档/db.txt'))
    time.sleep(0.1) #模拟读数据的网络延迟
    if dic['count'] >0:
        dic['count']-=1
        time.sleep(0.2) #模拟写数据的网络延迟
        json.dump(dic,open('/home/jack/文档/db.txt','w'))
        print('\033[43m购票成功\033[0m')

def task(lock):
    lock.acquire()
    search()
    get()
    lock.release()
if __name__ == '__main__':
    lock=Lock()
    for i in range(10): #模拟并发100个客户端抢票
        p=Process(target=task,args=(lock,))
        p.start()

打印结果为：

剩余票数300
购票成功
剩余票数299
购票成功
剩余票数298
购票成功
剩余票数297
购票成功
剩余票数296
购票成功
剩余票数295
购票成功
剩余票数294
购票成功
剩余票数293
购票成功
剩余票数292
购票成功
剩余票数291
购票成功

运行时间为3 s，且db.txt文件的内容在运行完上述代码后变为{"count": 290}

"""注意上述例二和例三的区别，search()是否在lock.acquire和lock.release()之间对程序的打印影响很大"""
#************************************************

#https://www.cnblogs.com/gregoryli/p/7892222.html
#************************************
#第一个
from multiprocessing import Process, Lock

def f(i):
    print("hello,world,",i)

if __name__ == '__main__':
    #lock = Lock()
    for num in range(10):
        p=Process(target=f, args=(num,))
        p.start() 

结果：

hello,world, 0
hello,world, 1
hello,world, 2
hello,world, 3
hello,world, 4
hello,world, 5
hello,world, 6
hello,world, 7
hello,world, 8
hello,world, 9

real	0m0.065s
user	0m0.059s
sys	0m0.018s
#****************************************
#第二个
from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()
    for num in range(10):
        Process(target=f, args=(lock, num)).start()
结果：

hello world 0
hello world 1
hello world 2
hello world 3
hello world 4
hello world 5
hello world 6
hello world 7
hello world 8
hello world 9

real	0m0.082s
user	0m0.081s
sys	0m0.015s
#******************************************************
#第三个
from multiprocessing import Process, Lock
import random,time

def f(i):
    time.sleep(2)#如果为sleep(1)，则此代码需要1秒跑完；
    print('hello,world,',i)

if __name__ == '__main__':
    #lock = Lock()
    for num in range(10):
        p=Process(target=f, args=(num,))
        p.start()
        #p.join() #在sleep(1)的条件下，有join()需要10s,会错乱；没有join()需要1s，会错乱
结果：


hello,world, 1
hello,world, 0
hello,world, 4
hello,world, 2
hello,world, 3
hello,world, 5
hello,world, 6
hello,world, 7
hello,world, 8
hello,world, 9

real	0m2.076s
user	0m0.067s
sys	0m0.025s

#***************************************************8
#第四个
from multiprocessing import Process, Lock
import random,time


def f(l,i):
    l.acquire()
    try:
        time.sleep(random.randrange(0,2))#如果为sleep(1)，则此代码需要10秒跑完；
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()
    for num in range(10):
        p=Process(target=f, args=(lock,num))
        p.start()
结果：

hello world 0
hello world 1
hello world 2
hello world 3
hello world 4
hello world 5
hello world 6
hello world 7
hello world 8
hello world 9

real	0m3.088s
user	0m0.078s
sys	0m0.026s

如果为sleep(1)，则结果为：

hello world 0
hello world 1
hello world 2
hello world 3
hello world 4
hello world 5
hello world 6
hello world 7
hello world 8
hello world 9

real	0m10.090s
user	0m0.086s
sys	0m0.020s
#*********************************************************
#第五个
from multiprocessing import Process, Lock
import random,time


def f(l,i):
    time.sleep(random.randrange(0,4))#如果为sleep(1)，则此代码需要1秒跑完；
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()
    for num in range(10):
        p=Process(target=f, args=(lock,num))
        p.start()
结果：


hello world 0
hello world 2
hello world 3
hello world 6
hello world 7
hello world 5
hello world 9
hello world 1
hello world 4
hello world 8

real	0m3.087s
user	0m0.083s
sys	0m0.026s


如果为sleep(1)，则结果为:

hello world 0
hello world 1
hello world 2
hello world 3
hello world 4
hello world 5
hello world 6
hello world 7
hello world 8
hello world 9

real	0m1.080s
user	0m0.086s
sys	0m0.012s
#*****************************************************
"""
上述的第一个代码不会发生错乱，第二个不会发生错乱，第三个会，第四个不会；
第一个没锁也不会是因为只有打印语句，没有sleep语句，所以不会
第三个有sleep语句，且没有加锁，所以打印错乱
第二第四都加了锁，当然不会错乱
注意第四个和第五个的区别，sleep(1)在lock.acquire()之前和之后的巨大区别，所以如果计算
任务是在lock.acquire()之前完成的，在lock.acquire和lock.release()之间只有用时很少的写如文件任务
那么这个程序仍是并行的，不会因为锁存在而变慢，
"""

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
结果为：

work_with start
work_no_with start
work_with end
work_no_with end
end!!!

real	0m4.123s
user	0m0.057s
sys	0m0.024s

test.txt文件的内容为：

Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lockd acquired via with
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
Lock acquired directly
"""
此代码的运行时间为4s,说明‘锁’的作用范围是lock.acquire()和lock.release()之间，在lock.acquire()之前的
运算还是并行的,但是前提是没有加join()，否则，在lock>acquire()之前和lock.release之后的程还是变为
穿行的；
此代码验证了：虽然有锁lock，但是写文件的时间极端，可以在写之前进行其他运算，这时其他运算还是
并行的"""
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
    with open('/home/jack/文档/test.txt','a') as f:
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
    with open('/home/jack/文档/test.txt','a') as f:
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
    for i in range(10):
        #p1 = multiprocessing.Process(target = worker_with, args=(lock,i,))
        p = multiprocessing.Process(target = worker_without, args=(i,))
        #p1.start()
        p.start()
        p_list.append(p)
        #p_list.append(p1)#
        #p.join()

    for i in range(10):
        p_list[i].join()

    print("end!!!")

if __name__=='__main__':
    main()
#*******************************************************
