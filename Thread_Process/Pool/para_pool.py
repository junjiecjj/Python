
#***********************************************************************
#https://blog.csdn.net/topleeyap/article/details/78981848
'''
使用进程池Pool
'''
import os,time
from multiprocessing import Process,Pool

def worker():
    print("子进程执行中>>>pid={},ppid={}".format(os.getpid(),os.getppid()))
    time.sleep(2)
    print("子进程终止>>>pid={0}".format(os.getpid()))

def main():
    print("主进程执行中>>>pid{0}".format(os.getpid()))

    ps=Pool(10)

    for i in range(10):
        #ps.apply(worker,args=(i,))  #同步执行,每个进程依次进行，相当于串行
        ps.apply_async(worker,args=())  #异步执行

    ps.close() #关闭进程池，停止接受其他进程
    ps.join()  #阻塞进程

    print("主进程终止")

if __name__=='__main__':
    main()
结果：
主进程执行中>>>pid15340
子进程执行中>>>pid=15341,ppid=15340
子进程执行中>>>pid=15342,ppid=15340
子进程执行中>>>pid=15343,ppid=15340
子进程执行中>>>pid=15344,ppid=15340
子进程执行中>>>pid=15345,ppid=15340
子进程执行中>>>pid=15346,ppid=15340
子进程执行中>>>pid=15347,ppid=15340
子进程执行中>>>pid=15348,ppid=15340
子进程执行中>>>pid=15349,ppid=15340
子进程执行中>>>pid=15350,ppid=15340
子进程终止>>>pid=15342
子进程终止>>>pid=15348
子进程终止>>>pid=15341
子进程终止>>>pid=15345
子进程终止>>>pid=15343
子进程终止>>>pid=15347
子进程终止>>>pid=15349
子进程终止>>>pid=15350
子进程终止>>>pid=15346
子进程终止>>>pid=15344
主进程终止

real	0m2.100s
user	0m0.094s
sys	0m0.033s

如果是pool(3)，结果为：

主进程执行中>>>pid11175
子进程执行中>>>pid=11176,ppid=11175
子进程执行中>>>pid=11177,ppid=11175
子进程执行中>>>pid=11178,ppid=11175
子进程终止>>>pid=11176
子进程终止>>>pid=11177
子进程终止>>>pid=11178
子进程执行中>>>pid=11176,ppid=11175
子进程执行中>>>pid=11178,ppid=11175
子进程执行中>>>pid=11177,ppid=11175
子进程终止>>>pid=11176
子进程终止>>>pid=11178
子进程终止>>>pid=11177
子进程执行中>>>pid=11178,ppid=11175
子进程执行中>>>pid=11176,ppid=11175
子进程执行中>>>pid=11177,ppid=11175
子进程终止>>>pid=11177
子进程终止>>>pid=11178
子进程终止>>>pid=11176
子进程执行中>>>pid=11177,ppid=11175
子进程终止>>>pid=11177
主进程终止

real	0m8.100s
user	0m0.096s
sys	0m0.011s
#********************************************
import os,time
from multiprocessing import Process,Pool

def worker(i):
    print("子进程执行中>>>pid={},ppid={}".format(os.getpid(),os.getppid()))
    a=i
    b=i+1
    c=i/2
    d=i*2
    time.sleep(2)
    print("子进程终止>>>pid={0}".format(os.getpid()))
    with open('/home/jack/桌面/D_clust6.txt','a') as f:
        f.write("***************分割线*******************\na is %f; b is %f; c is %f; d is %f\n "%(a,b,c,d))
    return

def main():
    print("主进程执行中>>>pid{0}".format(os.getpid()))

    ps=Pool(4)

    for i in range(6):
        #ps.apply(worker,args=(i,))  #同步执行,每个进程依次进行，相当于串行
        ps.apply_async(worker,args=(i,))  #异步执行

    ps.close() #关闭进程池，停止接受其他进程
    ps.join()  #阻塞进程

    print("主进程终止")

if __name__=='__main__':
    main()

D_clust6.txt文件的内容如下:

***************分割线*******************
a is 1.000000; b is 2.000000; c is 0.500000; d is 2.000000
 ***************分割线*******************
a is 0.000000; b is 1.000000; c is 0.000000; d is 0.000000
 ***************分割线*******************
a is 2.000000; b is 3.000000; c is 1.000000; d is 4.000000
 ***************分割线*******************
a is 3.000000; b is 4.000000; c is 1.500000; d is 6.000000
 ***************分割线*******************
a is 5.000000; b is 6.000000; c is 2.500000; d is 10.000000
 ***************分割线*******************
a is 4.000000; b is 5.000000; c is 2.000000; d is 8.000000

#*********************************************

#https://www.cnblogs.com/linhaifeng/articles/7428874.html
#同步调用
from multiprocessing import Pool
import os,time
def work(n):
    print('%s run' %os.getpid())
    time.sleep(1)
    return n**2

if __name__ == '__main__':
    p=Pool(3) #进程池中从无到有创建三个进程,以后一直是这三个进程在执行任务
    res_l=[]
    for i in range(10):
        res=p.apply(work,args=(i,)) #同步调用，直到本次任务执行完毕拿到res，等待任务work执行的过程中可能有阻塞也可能没有阻塞，但不管该任务是否存在阻塞，同步调用都会在原地等着，只是等的过程中若是任务发生了阻塞就会被夺走cpu的执行权限
        res_l.append(res)
    print(res_l)
结果：
10264 run
10265 run
10266 run
10264 run
10265 run
10266 run
10264 run
10265 run
10266 run
10264 run
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

real	0m10.103s
user	0m0.080s
sys	0m0.031s

如果是pool(10)。结果为：

15482 run
15483 run
15484 run
15485 run
15486 run
15487 run
15488 run
15489 run
15490 run
15491 run
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

real	0m10.116s
user	0m0.091s
sys	0m0.051s

#***********************************************
#异步调用
from multiprocessing import Pool
import os,time
def work(n):
    print('%s run' %os.getpid())
    time.sleep(1)
    return n**2

if __name__ == '__main__':
    p=Pool(3) #进程池中从无到有创建三个进程,以后一直是这三个进程在执行任务
    res_l=[]
    for i in range(10):
        res=p.apply_async(work,args=(i,)) #同步运行,阻塞、直到本次任务执行完毕拿到res
        res_l.append(res)

    #异步apply_async用法：如果使用异步提交的任务，主进程需要使用jion，等待进程池内任务都处理完，然后可以用get收集结果，否则，主进程结束，进程池可能还没来得及执行，也就跟着一起结束了
    p.close()
    p.join()
    for res in res_l:
        print(res.get()) #使用get来获取apply_aync的结果,如果是apply,则没有get方法,因为apply是同步执行,立刻获取结果,也根本无需get
结果：

10179 run
10180 run
10181 run
10181 run
10180 run
10179 run
10181 run
10180 run
10179 run
10181 run
0
1
4
9
16
25
36
49
64
81

real	0m4.092s
user	0m0.079s
sys	0m0.025s

如果为pool(10)，结果为：

15645 run
15646 run
15647 run
15648 run
15649 run
15650 run
15651 run
15652 run
15653 run
15654 run
0
1
4
9
16
25
36
49
64
81

real	0m1.192s
user	0m0.085s
sys	0m0.039s
#*************************************
#一：使用进程池（异步调用,apply_async）
#coding: utf-8
from multiprocessing import Process,Pool
import time

def func(msg):
    print( "msg:", msg)
    time.sleep(1)
    return msg

if __name__ == "__main__":
    pool = Pool(processes = 3)
    res_l=[]
    for i in range(10):
        msg = "hello %d" %(i)
        res=pool.apply_async(func, (msg, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        res_l.append(res)
    print("==============================>") #没有后面的join，或get，则程序整体结束，进程池中的任务还没来得及全部执行完也都跟着主进程一起结束了

    pool.close() #关闭进程池，防止进一步操作。如果所有操作持续挂起，它们将在工作进程终止前完成
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    print(res_l) 
    #看到的是<multiprocessing.pool.ApplyResult object at 0x10357c4e0>对象组成的列表,
    #而非最终的结果,但这一步是在join后执行的,证明结果已经计算完毕,剩下的事情就是调用每个对象下的get方法去获取结果
    for i in res_l:
        print(i.get()) 
        #使用get来获取apply_aync的结果,如果是apply,则没有get方法,因为apply是同步执行,立刻获取结果,也根本无需get

结果：


==============================>
msg: hello 0
msg: hello 1
msg: hello 2
msg: hello 3
msg: hello 4
msg: hello 5
msg: hello 6
msg: hello 7
msg: hello 8
msg: hello 9

hello 0
hello 1
hello 2
hello 3
hello 4
hello 5
hello 6
hello 7
hello 8
hello 9

real	0m4.092s
user	0m0.079s
sys	0m0.026s

如果为pool(10)，结果为：

==============================>
msg: hello 0
msg: hello 1
msg: hello 2
msg: hello 3
msg: hello 4
msg: hello 5
msg: hello 6
msg: hello 7
msg: hello 8
msg: hello 9

hello 0
hello 1
hello 2
hello 3
hello 4
hello 5
hello 6
hello 7
hello 8
hello 9

real	0m1.185s
user	0m0.091s
sys	0m0.024s

#二：使用进程池（同步调用,apply）
#coding: utf-8
from multiprocessing import Process,Pool
import time

def func(msg):
    print( "msg:", msg)
    time.sleep(1)
    return msg

if __name__ == "__main__":
    pool = Pool(processes = 3)
    res_l=[]
    for i in range(10):
        msg = "hello %d" %(i)
        res=pool.apply(func, (msg, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        res_l.append(res) #同步执行，即执行完一个拿到结果，再去执行另外一个
    print("==============================>")
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    print(res_l) #看到的就是最终的结果组成的列表
    for i in res_l: #apply是同步的，所以直接得到结果，没有get()方法
        print(i)

结果：

msg: hello 0
msg: hello 1
msg: hello 2
msg: hello 3
msg: hello 4
msg: hello 5
msg: hello 6
msg: hello 7
msg: hello 8
msg: hello 9
==============================>
['hello 0', 'hello 1', 'hello 2', 'hello 3', 'hello 4', 'hello 5', 'hello 6', 'hello 7', 'hello 8', 'hello 9']
hello 0
hello 1
hello 2
hello 3
hello 4
hello 5
hello 6
hello 7
hello 8
hello 9

real	0m10.106s
user	0m0.097s
sys	0m0.019s

如果为pool(10)，结果为：

msg: hello 0
msg: hello 1
msg: hello 2
msg: hello 3
msg: hello 4
msg: hello 5
msg: hello 6
msg: hello 7
msg: hello 8
msg: hello 9
==============================>
['hello 0', 'hello 1', 'hello 2', 'hello 3', 'hello 4', 'hello 5', 'hello 6', 'hello 7', 'hello 8', 'hello 9']
hello 0
hello 1
hello 2
hello 3
hello 4
hello 5
hello 6
hello 7
hello 8
hello 9

real	0m10.117s
user	0m0.102s
sys	0m0.036s
#*************************************
#https://www.cnblogs.com/gregoryli/p/7892222.html


from multiprocessing import Pool
import os, time, random

def run_task(name):
    print('Task %s (pid = %s) is running...' % (name, os.getpid()))
    time.sleep(random.random() * 3) #sleep(2)
    print('Task %s end.' % name)

if __name__=='__main__':
    print('Current process %s.' % os.getpid())
    p = Pool(processes=3)
    for i in range(5):
        p.apply_async(run_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

结果：
Current process 15606.
Waiting for all subprocesses done...
Task 0 (pid = 15607) is running...
Task 1 (pid = 15608) is running...
Task 2 (pid = 15609) is running...
Task 1 end.
Task 3 (pid = 15608) is running...
Task 3 end.
Task 4 (pid = 15608) is running...
Task 2 end.
Task 0 end.
Task 4 end.
All subprocesses done.

real	0m2.871s
user	0m0.067s
sys	0m0.010s


当sleep(2)时，结果为：
Current process 15620.
Waiting for all subprocesses done...
Task 0 (pid = 15621) is running...
Task 1 (pid = 15622) is running...
Task 2 (pid = 15623) is running...
Task 0 end.
Task 2 end.
Task 1 end.
Task 3 (pid = 15621) is running...
Task 4 (pid = 15622) is running...
Task 3 end.
Task 4 end.
All subprocesses done.

real	0m4.090s
user	0m0.071s
sys	0m0.020s

如果为sleep(2),且Pool(process=5)，则结果为:

Current process 16806.
Waiting for all subprocesses done...
Task 0 (pid = 16807) is running...
Task 1 (pid = 16808) is running...
Task 2 (pid = 16809) is running...
Task 3 (pid = 16810) is running...
Task 4 (pid = 16811) is running...
Task 1 end.
Task 3 end.
Task 0 end.
Task 2 end.
Task 4 end.
All subprocesses done.

real	0m2.089s
user	0m0.082s
sys	0m0.023s
"""注意上述代码中，pool(3)，但是创建range(10)时的进程号，这时总是只有三个固定的进程号，而不是10个"""
#***************************************
from multiprocessing import Pool
import os, time, random

def run_task(name):
    print('Task %s (pid = %s) is running...' % (name, os.getpid()))
    time.sleep(1) #sleep(2)
    print('Task %s end.' % name)

if __name__=='__main__':
    print('Current process %s.' % os.getpid())
    p = Pool(processes=5)
    print("hello,jack")
    for i in range(5):
        print("hhh...")
        p.apply(run_task, args=(i,))
        print(i)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

结果为:

Current process 30447.
hello,jack
hhh...
Task 0 (pid = 30448) is running...
Task 0 end.
0
hhh...
Task 1 (pid = 30449) is running...
Task 1 end.
1
hhh...
Task 2 (pid = 30450) is running...
Task 2 end.
2
hhh...
Task 3 (pid = 30448) is running...
Task 3 end.
3
hhh...
Task 4 (pid = 30449) is running...
Task 4 end.
4
Waiting for all subprocesses done...
All subprocesses done.

real	0m5.076s
user	0m0.060s
sys	0m0.023s


如果为Pool(5)，则结果为：

Current process 30428.
hello,jack
hhh...
Task 0 (pid = 30429) is running...
Task 0 end.
0
hhh...
Task 1 (pid = 30430) is running...
Task 1 end.
1
hhh...
Task 2 (pid = 30431) is running...
Task 2 end.
2
hhh...
Task 3 (pid = 30432) is running...
Task 3 end.
3
hhh...
Task 4 (pid = 30433) is running...
Task 4 end.
4
Waiting for all subprocesses done...
All subprocesses done.

real	0m5.084s
user	0m0.077s
sys	0m0.024s
######################################################
#http://python.jobbole.com/82045/
#******************************************
#coding: utf-8  #使用进程池，非阻塞
import multiprocessing
import time
import os

def func(msg):
    print("msg:", msg,"子进程执行中>>>pid={},ppid={}".format(os.getpid(),os.getppid()))
    #print("子进程执行中>>>pid={},ppid={}".format(os.getpid(),os.getppid()))
    time.sleep(1)
    print("end")

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = 3)
    for i in range(6):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去

    print("Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~")
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("Sub-process(es) done.")
结果：


Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~
msg: hello 0 子进程执行中>>>pid=30699,ppid=30698
msg: hello 1 子进程执行中>>>pid=30700,ppid=30698
msg: hello 2 子进程执行中>>>pid=30701,ppid=30698
end
end
end
msg: hello 3 子进程执行中>>>pid=30699,ppid=30698
msg: hello 4 子进程执行中>>>pid=30700,ppid=30698
msg: hello 5 子进程执行中>>>pid=30701,ppid=30698
end
end
end
Sub-process(es) done.

real	0m2.071s
user	0m0.071s
sys	0m0.007s


如果为pool(6)，结果为：

Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~
msg: hello 0 子进程执行中>>>pid=30730,ppid=30729
msg: hello 1 子进程执行中>>>pid=30731,ppid=30729
msg: hello 2 子进程执行中>>>pid=30732,ppid=30729
msg: hello 3 子进程执行中>>>pid=30733,ppid=30729
msg: hello 4 子进程执行中>>>pid=30734,ppid=30729
msg: hello 5 子进程执行中>>>pid=30735,ppid=30729
end
end
end
end
end
end
Sub-process(es) done.

real	0m1.173s
user	0m0.057s
sys	0m0.041s
"""
执行说明：创建一个进程池pool，并设定进程的数量为3，range(6)会相继产生四个对象[0, 1, 2, 4,5,6]，四个对象被提交到pool中，
因pool指定进程数为3，所以0、1、2会直接送到进程中执行，当其中一个执行完事后才空出一个进程处理对象3，所以会出现输
出“msg: hello 3”出现在"end"后。因为为非阻塞，主函数会自己执行自个的，不搭理进程的执行，所以运行完for循环后直接输
出“mMsg: hark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~”，主程序在pool.join（）处等待各个进程的结束。"""
#*******************************
#coding: utf-8
#使用进程池，阻塞
import multiprocessing
import time

def func(msg):
    print("msg:", msg)
    time.sleep(1)
    print("end")

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = 3)
    for i in range(6):
        msg = "hello %d" %(i)
        pool.apply(func, (msg, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去

    print("Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~")
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("Sub-process(es) done.")
结果：
msg: hello 0
end
msg: hello 1
end
msg: hello 2
end
msg: hello 3
end
msg: hello 4
end
msg: hello 5
end
Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~
Sub-process(es) done.

real	0m6.098s
user	0m0.078s
sys	0m0.025s

如果为multiprocess.Pool(process=6)，结果为：

msg: hello 0
end
msg: hello 1
end
msg: hello 2
end
msg: hello 3
end
msg: hello 4
end
msg: hello 5
end
Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~
Sub-process(es) done.

real	0m6.101s
user	0m0.083s
sys	0m0.033s

"""从上面的例子可以看出，
当使用apply_async时非阻塞，是并行，且主进程中在close和join之前的语句都会立刻执行，不会等待子进程完成后再进行
但是close和join之后的语句会等所有的子进程结束后在进行；
当是有apply时，实际上程序变为了串行，不管有多少个子进程，每次只能执行一个，且主进程中在close和join之前和之后的语句都会
等所有的子进程结束后在进行；
"""
#********************
#使用进程池并关注结果
import multiprocessing
import time

def func(msg):
    print("msg:", msg)
    time.sleep(1)
    print("end")
    return "done" + msg

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    result = []
    for i in range(6):
        msg = "hello %d" %(i)
        result.append(pool.apply_async(func, (msg, )))
    pool.close()
    pool.join()
    for res in result:
        print(":::", res.get())
    print("Sub-process(es) done.")
结果：

msg: hello 0
msg: hello 1
msg: hello 2
msg: hello 3
end
end
end
end
msg: hello 4
msg: hello 5
end
end
::: donehello 0
::: donehello 1
::: donehello 2
::: donehello 3
::: donehello 4
::: donehello 5
Sub-process(es) done.

real	0m2.084s
user	0m0.070s
sys	0m0.026s

如果为multiprocess.Pool(process=6)，则结果为：

msg: hello 0
msg: hello 1
msg: hello 2
msg: hello 3
msg: hello 4
msg: hello 5
end
end
end
end
end
end
::: donehello 0
::: donehello 1
::: donehello 2
::: donehello 3
::: donehello 4
::: donehello 5
Sub-process(es) done.

real	0m1.183s
user	0m0.077s
sys	0m0.022s
#**********************************
#coding: utf-8
import multiprocessing
import os, time, random
#使用多个进程池

def Lee():
    print("\nRun task Lee-%s" %(os.getpid())) #os.getpid()获取当前的进程的ID
    start = time.time()
    time.sleep(random.random() * 10) #random.random()随机生成0-1之间的小数
    end = time.time()
    print('Task Lee, runs %0.2f seconds.' %(end - start))

def Marlon():
    print("\nRun task Marlon-%s" %(os.getpid()))
    start = time.time()
    time.sleep(random.random() * 40)
    end=time.time()
    print('Task Marlon runs %0.2f seconds.' %(end - start))

def Allen():
    print("\nRun task Allen-%s" %(os.getpid()))
    start = time.time()
    time.sleep(random.random() * 30)
    end = time.time()
    print('Task Allen runs %0.2f seconds.' %(end - start))

def Frank():
    print("\nRun task Frank-%s" %(os.getpid()))
    start = time.time()
    time.sleep(random.random() * 20)
    end = time.time()
    print('Task Frank runs %0.2f seconds.' %(end - start))

if __name__=='__main__':
    function_list=  [Lee, Marlon, Allen, Frank]
    print("parent process %s" %(os.getpid()))

    pool=multiprocessing.Pool(4)
    for func in function_list:
        pool.apply_async(func)     #Pool执行函数，apply执行函数,当有一个进程执行完毕后，会添加一个新的进程到pool中

    print('Waiting for all subprocesses done...')
    pool.close()
    pool.join()    #调用join之前，一定要先调用close() 函数，否则会出错, close()执行后不会有新的进程加入到pool,join函数等待素有子进程结束
    print('All subprocesses done.')
结果：
parent process 15590
Waiting for all subprocesses done...

Run task Lee-15591

Run task Marlon-15592

Run task Allen-15593

Run task Frank-15594
Task Lee, runs 2.35 seconds.
Task Allen runs 8.41 seconds.
Task Marlon runs 14.05 seconds.
Task Frank runs 19.35 seconds.
All subprocesses done.

real	0m19.442s
user	0m0.098s
sys	0m0.023s

#**********************************************

#***************************************************
import numpy as np
from multiprocessing import Process,Pool


class test(object):
    def __init__(self):
        self.a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        #self.b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])

    def chan(self,i):
        c = [1,2,3]
        self.a += i
        return self.a ,i**2,c
    def achan(self,i):
        self.a[i,:]+=i

def main():
    #a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    #b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
    hh = test()
    print("before change hh.a is :\n",hh.a,'\n')
    for i in range(4):
        hh.achan(i)
    print("first change a is:\n",hh.a,'\n')
    res_l = []
    arry_l = []
    pool = Pool(4)
    for i in range(4):
        res = pool.apply_async(hh.chan,(i,))
        res_l.append(res)
    pool.close()
    pool.join()

    for res in res_l:
        arry_l.append(res.get())
    for i in range(4):
        print("arry_l[%d] is :\n"%i)
        print(arry_l[i],'\n')
    print("second change hh.a is :\n",hh.a,'\n')
    for i in range(4):
        hh.achan(i)
    print("last change hh.a is:\n",hh.a,'\n')


if __name__=='__main__':
    main()

结果为:

before change hh.a is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]] 

first change a is:
 [[ 1  2  3  4]
 [ 6  7  8  9]
 [11 12 13 14]
 [16 17 18 19]] 

arry_l[0] is :

(array([[ 1,  2,  3,  4],
       [ 6,  7,  8,  9],
       [11, 12, 13, 14],
       [16, 17, 18, 19]]), 0, [1, 2, 3]) 

arry_l[1] is :

(array([[ 2,  3,  4,  5],
       [ 7,  8,  9, 10],
       [12, 13, 14, 15],
       [17, 18, 19, 20]]), 1, [1, 2, 3]) 

arry_l[2] is :

(array([[ 3,  4,  5,  6],
       [ 8,  9, 10, 11],
       [13, 14, 15, 16],
       [18, 19, 20, 21]]), 4, [1, 2, 3]) 

arry_l[3] is :

(array([[ 4,  5,  6,  7],
       [ 9, 10, 11, 12],
       [14, 15, 16, 17],
       [19, 20, 21, 22]]), 9, [1, 2, 3]) 

second change hh.a is :
 [[ 1  2  3  4]
 [ 6  7  8  9]
 [11 12 13 14]
 [16 17 18 19]] 

last change hh.a is:
 [[ 1  2  3  4]
 [ 7  8  9 10]
 [13 14 15 16]
 [19 20 21 22]] 


real	0m0.245s
user	0m0.127s
sys	0m0.034s

此例子非常重要，可以看出，在主进程中创建的数组，在子进程中改变后，程序回到主进程时这个数组不会改变，
这与process模块一致但是可以通过pool的返回值res以结果的形式传出来，且可以传多个值。
#*********************************
import numpy as np
from multiprocessing import Process,Pool



class test(object):
    def __init__(self):
        self.a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        #self.b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])

    def chan(self,i):
        self.a += i
        c=[1,2,3]
        return self.a ,i**2,c
    def achan(self,i):
        self.a[i,:]+=i


#def main():
    #a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    #b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
hh = test()
print("before change hh.a is :\n",hh.a,'\n')
for i in range(4):
    hh.achan(i)
print("first change a is:\n",hh.a,'\n')

res_l = []
arry_l = []
pool = Pool(4)
for i in range(4):
    res = pool.apply_async(hh.chan,(i,))
    res_l.append(res)
pool.close()
pool.join()

for res in res_l:
    arry_l.append(res.get())

for i in range(4):
    print("arry_l[%d] is :\n"%i)
    print(arry_l[i],'\n')
print("second change hh.a is :\n",hh.a,'\n')

for i,arry in enumerate(arry_l):
    print('the %d arry is:'%i,)
    print(arry,'\n')
    if i==0:
        train_mat = arry[0]
    else:
        train_mat = np.hstack([train_mat, arry[0]]) #np.hstack为水平拼接数组函数

print("trainmat:\n",train_mat,'\n')

for i in range(4):
    hh.achan(i)
print("last change hh.a is:\n",hh.a,'\n')

结果为:


before change hh.a is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]] 

first change a is:
 [[ 1  2  3  4]
 [ 6  7  8  9]
 [11 12 13 14]
 [16 17 18 19]] 

arry_l[0] is :
(array([[ 1,  2,  3,  4],
       [ 6,  7,  8,  9],
       [11, 12, 13, 14],
       [16, 17, 18, 19]]), 0, [1, 2, 3]) 

arry_l[1] is :
(array([[ 2,  3,  4,  5],
       [ 7,  8,  9, 10],
       [12, 13, 14, 15],
       [17, 18, 19, 20]]), 1, [1, 2, 3]) 

arry_l[2] is :
(array([[ 3,  4,  5,  6],
       [ 8,  9, 10, 11],
       [13, 14, 15, 16],
       [18, 19, 20, 21]]), 4, [1, 2, 3]) 

arry_l[3] is :
(array([[ 4,  5,  6,  7],
       [ 9, 10, 11, 12],
       [14, 15, 16, 17],
       [19, 20, 21, 22]]), 9, [1, 2, 3]) 

second change hh.a is :
 [[ 1  2  3  4]
 [ 6  7  8  9]
 [11 12 13 14]
 [16 17 18 19]] 

the 0 arry is:
(array([[ 1,  2,  3,  4],
       [ 6,  7,  8,  9],
       [11, 12, 13, 14],
       [16, 17, 18, 19]]), 0, [1, 2, 3]) 

the 1 arry is:
(array([[ 2,  3,  4,  5],
       [ 7,  8,  9, 10],
       [12, 13, 14, 15],
       [17, 18, 19, 20]]), 1, [1, 2, 3]) 

the 2 arry is:
(array([[ 3,  4,  5,  6],
       [ 8,  9, 10, 11],
       [13, 14, 15, 16],
       [18, 19, 20, 21]]), 4, [1, 2, 3]) 

the 3 arry is:
(array([[ 4,  5,  6,  7],
       [ 9, 10, 11, 12],
       [14, 15, 16, 17],
       [19, 20, 21, 22]]), 9, [1, 2, 3]) 

trainmat:
 [[ 1  2  3  4  2  3  4  5  3  4  5  6  4  5  6  7]
 [ 6  7  8  9  7  8  9 10  8  9 10 11  9 10 11 12]
 [11 12 13 14 12 13 14 15 13 14 15 16 14 15 16 17]
 [16 17 18 19 17 18 19 20 18 19 20 21 19 20 21 22]] 

last change hh.a is:
 [[ 1  2  3  4]
 [ 7  8  9 10]
 [13 14 15 16]
 [19 20 21 22]] 


real	0m0.222s
user	0m0.114s
sys	0m0.021s

###############################################
import numpy as np
from multiprocessing import Process,Pool


class test(object):
    def __init__(self):
        self.a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24],[25,26,27,28]])
        #self.b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])

    def chan(self,i):
        c = [0,0,0]
        c = [i,i,i]
        self.a += i
        return self.a ,i**2,c
    def achan(self,i):
        self.a[i,:]+=i

def main():
    #a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    #b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
    hh = test()
    print("before change hh.a is :\n",hh.a,'\n')
    for i in range(7):
        hh.achan(i)
    print("first change a is:\n",hh.a,'\n')
    res_l = []
    arry_l = []
    pool = Pool(3)
    for i in range(7):
        res = pool.apply_async(hh.chan,(i,))
        res_l.append(res)
    pool.close()
    pool.join()

    for res in res_l:
        arry_l.append(res.get())

    for i in range(7):
        print("arry_l[%d] is :\n"%i)
        print(arry_l[i],'\n')
    print("second change hh.a is :\n",hh.a,'\n')

    for i,arry in enumerate(arry_l):
        print('the %d arry is:'%i,)
        print(arry,'\n')
        if i==0:
            train_mat = arry[0]
        else:
            train_mat = np.hstack([train_mat, arry[0]]) #np.hstack为水平拼接数组函数
    print("trainmat:\n",train_mat,'\n')

    for i in range(7):
        hh.achan(i)
    print("last change hh.a is:\n",hh.a,'\n')


if __name__=='__main__':
    main()

结果为:
before change hh.a is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]
 [25 26 27 28]] 

first change a is:
 [[ 1  2  3  4]
 [ 6  7  8  9]
 [11 12 13 14]
 [16 17 18 19]
 [21 22 23 24]
 [26 27 28 29]
 [31 32 33 34]] 

arry_l[0] is :
(array([[ 1,  2,  3,  4],
       [ 6,  7,  8,  9],
       [11, 12, 13, 14],
       [16, 17, 18, 19],
       [21, 22, 23, 24],
       [26, 27, 28, 29],
       [31, 32, 33, 34]]), 0, [0, 0, 0]) 

arry_l[1] is :
(array([[ 2,  3,  4,  5],
       [ 7,  8,  9, 10],
       [12, 13, 14, 15],
       [17, 18, 19, 20],
       [22, 23, 24, 25],
       [27, 28, 29, 30],
       [32, 33, 34, 35]]), 1, [1, 1, 1]) 

arry_l[2] is :
(array([[ 3,  4,  5,  6],
       [ 8,  9, 10, 11],
       [13, 14, 15, 16],
       [18, 19, 20, 21],
       [23, 24, 25, 26],
       [28, 29, 30, 31],
       [33, 34, 35, 36]]), 4, [2, 2, 2]) 

arry_l[3] is :
(array([[ 4,  5,  6,  7],
       [ 9, 10, 11, 12],
       [14, 15, 16, 17],
       [19, 20, 21, 22],
       [24, 25, 26, 27],
       [29, 30, 31, 32],
       [34, 35, 36, 37]]), 9, [3, 3, 3]) 

arry_l[4] is :
(array([[ 5,  6,  7,  8],
       [10, 11, 12, 13],
       [15, 16, 17, 18],
       [20, 21, 22, 23],
       [25, 26, 27, 28],
       [30, 31, 32, 33],
       [35, 36, 37, 38]]), 16, [4, 4, 4]) 

arry_l[5] is :
(array([[ 6,  7,  8,  9],
       [11, 12, 13, 14],
       [16, 17, 18, 19],
       [21, 22, 23, 24],
       [26, 27, 28, 29],
       [31, 32, 33, 34],
       [36, 37, 38, 39]]), 25, [5, 5, 5]) 

arry_l[6] is :
(array([[ 7,  8,  9, 10],
       [12, 13, 14, 15],
       [17, 18, 19, 20],
       [22, 23, 24, 25],
       [27, 28, 29, 30],
       [32, 33, 34, 35],
       [37, 38, 39, 40]]), 36, [6, 6, 6]) 

second change hh.a is :
 [[ 1  2  3  4]
 [ 6  7  8  9]
 [11 12 13 14]
 [16 17 18 19]
 [21 22 23 24]
 [26 27 28 29]
 [31 32 33 34]] 

the 0 arry is:
(array([[ 1,  2,  3,  4],
       [ 6,  7,  8,  9],
       [11, 12, 13, 14],
       [16, 17, 18, 19],
       [21, 22, 23, 24],
       [26, 27, 28, 29],
       [31, 32, 33, 34]]), 0, [0, 0, 0]) 

the 1 arry is:
(array([[ 2,  3,  4,  5],
       [ 7,  8,  9, 10],
       [12, 13, 14, 15],
       [17, 18, 19, 20],
       [22, 23, 24, 25],
       [27, 28, 29, 30],
       [32, 33, 34, 35]]), 1, [1, 1, 1]) 

the 2 arry is:
(array([[ 3,  4,  5,  6],
       [ 8,  9, 10, 11],
       [13, 14, 15, 16],
       [18, 19, 20, 21],
       [23, 24, 25, 26],
       [28, 29, 30, 31],
       [33, 34, 35, 36]]), 4, [2, 2, 2]) 

the 3 arry is:
(array([[ 4,  5,  6,  7],
       [ 9, 10, 11, 12],
       [14, 15, 16, 17],
       [19, 20, 21, 22],
       [24, 25, 26, 27],
       [29, 30, 31, 32],
       [34, 35, 36, 37]]), 9, [3, 3, 3]) 

the 4 arry is:
(array([[ 5,  6,  7,  8],
       [10, 11, 12, 13],
       [15, 16, 17, 18],
       [20, 21, 22, 23],
       [25, 26, 27, 28],
       [30, 31, 32, 33],
       [35, 36, 37, 38]]), 16, [4, 4, 4]) 

the 5 arry is:
(array([[ 6,  7,  8,  9],
       [11, 12, 13, 14],
       [16, 17, 18, 19],
       [21, 22, 23, 24],
       [26, 27, 28, 29],
       [31, 32, 33, 34],
       [36, 37, 38, 39]]), 25, [5, 5, 5]) 

the 6 arry is:
(array([[ 7,  8,  9, 10],
       [12, 13, 14, 15],
       [17, 18, 19, 20],
       [22, 23, 24, 25],
       [27, 28, 29, 30],
       [32, 33, 34, 35],
       [37, 38, 39, 40]]), 36, [6, 6, 6]) 

trainmat:
 [[ 1  2  3  4  2  3  4  5  3  4  5  6  4  5  6  7  5  6  7  8  6  7  8  9
   7  8  9 10]
 [ 6  7  8  9  7  8  9 10  8  9 10 11  9 10 11 12 10 11 12 13 11 12 13 14
  12 13 14 15]
 [11 12 13 14 12 13 14 15 13 14 15 16 14 15 16 17 15 16 17 18 16 17 18 19
  17 18 19 20]
 [16 17 18 19 17 18 19 20 18 19 20 21 19 20 21 22 20 21 22 23 21 22 23 24
  22 23 24 25]
 [21 22 23 24 22 23 24 25 23 24 25 26 24 25 26 27 25 26 27 28 26 27 28 29
  27 28 29 30]
 [26 27 28 29 27 28 29 30 28 29 30 31 29 30 31 32 30 31 32 33 31 32 33 34
  32 33 34 35]
 [31 32 33 34 32 33 34 35 33 34 35 36 34 35 36 37 35 36 37 38 36 37 38 39
  37 38 39 40]] 

last change hh.a is:
 [[ 1  2  3  4]
 [ 7  8  9 10]
 [13 14 15 16]
 [19 20 21 22]
 [25 26 27 28]
 [31 32 33 34]
 [37 38 39 40]] 


real	0m0.242s
user	0m0.144s
sys	0m0.007s

从这个例子可以看出，当循环任务个数多余进程数时，也不会出现次序的错误，这是因为
    for i in range(7):
        res = pool.apply_async(hh.chan,(i,))
        res_l.append(res)
这几行代码是有次序的,res和i是一一对应的。
#************************************
from multiprocessing import Pool
import requests
import json
import os

def get_page(url):
    print('<进程%s> get %s' %(os.getpid(),url))
    respone=requests.get(url)
    if respone.status_code == 200:
        return {'url':url,'text':respone.text}

def pasrse_page(res):
    print('<进程%s> parse %s' %(os.getpid(),res['url']))
    parse_res='url:<%s> size:[%s]\n' %(res['url'],len(res['text']))
    with open('db.txt','a') as f:
        f.write(parse_res)


if __name__ == '__main__':
    urls=[
        'https://www.baidu.com',
        'https://www.python.org',
        'https://www.openstack.org',
        'https://help.github.com/',
        'http://www.sina.com.cn/'
    ]

    p=Pool(3)
    res_l=[]
    for url in urls:
        res=p.apply_async(get_page,args=(url,),callback=pasrse_page)
        res_l.append(res)

    p.close()
    p.join()
    print([res.get() for res in res_l]) #拿到的是get_page的结果,其实完全没必要拿该结果,该结果已经传给回调函数处理了

'''
打印结果:
<进程3388> get https://www.baidu.com
<进程3389> get https://www.python.org
<进程3390> get https://www.openstack.org
<进程3388> get https://help.github.com/
<进程3387> parse https://www.baidu.com
<进程3389> get http://www.sina.com.cn/
<进程3387> parse https://www.python.org
<进程3387> parse https://help.github.com/
<进程3387> parse http://www.sina.com.cn/
<进程3387> parse https://www.openstack.org
[{'url': 'https://www.baidu.com', 'text': '<!DOCTYPE html>\r\n...',...}]
'''
#*************************************
import numpy as np
from multiprocessing import Process,Pool


class test(object):
    def __init__(self):
        self.a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24],[25,26,27,28]])
        self.b = np.array([[67701,1.212,1],[67702,2.378,1],[67703,8.675,1],[67704,6.345,1],[67705,18.778,-1],[77098,7.659,-1],[89908,4.545,-1]])
        self.c = np.zeros(self.b.shape)
        #self.b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])

    def chan(self,i):
        c = [0,0,0]
        c = [i,i,i]
        self.a += i
        return self.a ,i**2,c
    def achan(self,i):
        self.a[i,:]+=i

def main():
    #a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    #b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
    hh = test()
    print("before change hh.a is :\n",hh.a,'\n')
    res_l = []
    arry_l = []
    pool = Pool(3)
    for i in range(7):
        res = pool.apply_async(hh.chan,(i,))
        res_l.append(res)
    pool.close()
    pool.join()

    for res in res_l:
        arry_l.append(res.get())

    for i in range(7):
        print("arry_l[%d] is :\n"%i)
        print(arry_l[i],'\n')
    print("first change hh.a is :\n",hh.a,'\n')

    all_test_shut = range(0,7)
    for i,arry in zip(all_test_shut,arry_l):
        print('the %d arry is:'%i,)
        print(arry,'\n')
        hh.c[i,0] = hh.b[i,0]
        hh.c[i,1] = arry[1]
        hh.c[i,2] = hh.b[i,2]
        if i==0:
            train_mat = arry[0]
        else:
            train_mat = np.hstack([train_mat, arry[0]]) #np.hstack为水平拼接数组函数
    print("trainmat:\n",train_mat,'\n')
    print("self.c is:",hh.c)


if __name__=='__main__':
    main()

结果为:

before change hh.a is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]
 [25 26 27 28]] 

arry_l[0] is :

(array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28]]), 0, [0, 0, 0]) 

arry_l[1] is :

(array([[ 2,  3,  4,  5],
       [ 6,  7,  8,  9],
       [10, 11, 12, 13],
       [14, 15, 16, 17],
       [18, 19, 20, 21],
       [22, 23, 24, 25],
       [26, 27, 28, 29]]), 1, [1, 1, 1]) 

arry_l[2] is :

(array([[ 3,  4,  5,  6],
       [ 7,  8,  9, 10],
       [11, 12, 13, 14],
       [15, 16, 17, 18],
       [19, 20, 21, 22],
       [23, 24, 25, 26],
       [27, 28, 29, 30]]), 4, [2, 2, 2]) 

arry_l[3] is :

(array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]]), 9, [3, 3, 3]) 

arry_l[4] is :

(array([[ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28],
       [29, 30, 31, 32]]), 16, [4, 4, 4]) 

arry_l[5] is :

(array([[ 6,  7,  8,  9],
       [10, 11, 12, 13],
       [14, 15, 16, 17],
       [18, 19, 20, 21],
       [22, 23, 24, 25],
       [26, 27, 28, 29],
       [30, 31, 32, 33]]), 25, [5, 5, 5]) 

arry_l[6] is :

(array([[ 7,  8,  9, 10],
       [11, 12, 13, 14],
       [15, 16, 17, 18],
       [19, 20, 21, 22],
       [23, 24, 25, 26],
       [27, 28, 29, 30],
       [31, 32, 33, 34]]), 36, [6, 6, 6]) 

first change hh.a is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]
 [25 26 27 28]] 

the 0 arry is:
(array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28]]), 0, [0, 0, 0]) 

the 1 arry is:
(array([[ 2,  3,  4,  5],
       [ 6,  7,  8,  9],
       [10, 11, 12, 13],
       [14, 15, 16, 17],
       [18, 19, 20, 21],
       [22, 23, 24, 25],
       [26, 27, 28, 29]]), 1, [1, 1, 1]) 

the 2 arry is:
(array([[ 3,  4,  5,  6],
       [ 7,  8,  9, 10],
       [11, 12, 13, 14],
       [15, 16, 17, 18],
       [19, 20, 21, 22],
       [23, 24, 25, 26],
       [27, 28, 29, 30]]), 4, [2, 2, 2]) 

the 3 arry is:
(array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]]), 9, [3, 3, 3]) 

the 4 arry is:
(array([[ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28],
       [29, 30, 31, 32]]), 16, [4, 4, 4]) 

the 5 arry is:
(array([[ 6,  7,  8,  9],
       [10, 11, 12, 13],
       [14, 15, 16, 17],
       [18, 19, 20, 21],
       [22, 23, 24, 25],
       [26, 27, 28, 29],
       [30, 31, 32, 33]]), 25, [5, 5, 5]) 

the 6 arry is:
(array([[ 7,  8,  9, 10],
       [11, 12, 13, 14],
       [15, 16, 17, 18],
       [19, 20, 21, 22],
       [23, 24, 25, 26],
       [27, 28, 29, 30],
       [31, 32, 33, 34]]), 36, [6, 6, 6]) 

trainmat:
 [[ 1  2  3  4  2  3  4  5  3  4  5  6  4  5  6  7  5  6  7  8  6  7  8  9
   7  8  9 10]
 [ 5  6  7  8  6  7  8  9  7  8  9 10  8  9 10 11  9 10 11 12 10 11 12 13
  11 12 13 14]
 [ 9 10 11 12 10 11 12 13 11 12 13 14 12 13 14 15 13 14 15 16 14 15 16 17
  15 16 17 18]
 [13 14 15 16 14 15 16 17 15 16 17 18 16 17 18 19 17 18 19 20 18 19 20 21
  19 20 21 22]
 [17 18 19 20 18 19 20 21 19 20 21 22 20 21 22 23 21 22 23 24 22 23 24 25
  23 24 25 26]
 [21 22 23 24 22 23 24 25 23 24 25 26 24 25 26 27 25 26 27 28 26 27 28 29
  27 28 29 30]
 [25 26 27 28 26 27 28 29 27 28 29 30 28 29 30 31 29 30 31 32 30 31 32 33
  31 32 33 34]] 

self.c is: [[ 6.7701e+04  0.0000e+00  1.0000e+00]
 [ 6.7702e+04  1.0000e+00  1.0000e+00]
 [ 6.7703e+04  4.0000e+00  1.0000e+00]
 [ 6.7704e+04  9.0000e+00  1.0000e+00]
 [ 6.7705e+04  1.6000e+01 -1.0000e+00]
 [ 7.7098e+04  2.5000e+01 -1.0000e+00]
 [ 8.9908e+04  3.6000e+01 -1.0000e+00]]

real	0m0.234s
user	0m0.125s
sys	0m0.019s
#********************************************************
import numpy as np
from multiprocessing import Process,Pool

arry = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24],[25,26,27,28]])

class test(object):
    def __init__(self):
        self.b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])

    def chan(self,i):
        global arry
        c = [0,0,0]
        c = [i,i,i]
        arry += i
        return arry ,i**2,c

def main():
    global arry
    #a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    #b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
    hh = test()
    print("before change Arry is :\n",arry,'\n')
    res_l = []
    arry_l = []
    pool = Pool(3)
    for i in range(7):
        res = pool.apply_async(hh.chan,(i,))
        res_l.append(res)
    pool.close()
    pool.join()

    for res in res_l:
        arry_l.append(res.get())

    for i in range(7):
        print("arry_l[%d] is :\n"%i)
        print(arry_l[i],'\n')
    print("first change hh.a is :\n",arry,'\n')

    all_test_shut = range(0,7)
    for i,arry in zip(all_test_shut,arry_l):
        if i==0:
            train_mat = arry[0]
        else:
            train_mat = np.hstack([train_mat, arry[0]]) #np.hstack为水平拼接数组函数
    print("trainmat:\n",train_mat,'\n')

if __name__=='__main__':
    main()

结果为:

before change Arry is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]
 [25 26 27 28]]

arry_l[0] is :

(array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28]]), 0, [0, 0, 0])

arry_l[1] is :

(array([[ 2,  3,  4,  5],
       [ 6,  7,  8,  9],
       [10, 11, 12, 13],
       [14, 15, 16, 17],
       [18, 19, 20, 21],
       [22, 23, 24, 25],
       [26, 27, 28, 29]]), 1, [1, 1, 1])

arry_l[2] is :

(array([[ 3,  4,  5,  6],
       [ 7,  8,  9, 10],
       [11, 12, 13, 14],
       [15, 16, 17, 18],
       [19, 20, 21, 22],
       [23, 24, 25, 26],
       [27, 28, 29, 30]]), 4, [2, 2, 2])

arry_l[3] is :

(array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]]), 9, [3, 3, 3])

arry_l[4] is :

(array([[ 6,  7,  8,  9],
       [10, 11, 12, 13],
       [14, 15, 16, 17],
       [18, 19, 20, 21],
       [22, 23, 24, 25],
       [26, 27, 28, 29],
       [30, 31, 32, 33]]), 16, [4, 4, 4])

arry_l[5] is :

(array([[ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28],
       [29, 30, 31, 32],
       [33, 34, 35, 36]]), 25, [5, 5, 5])

arry_l[6] is :

(array([[ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28],
       [29, 30, 31, 32],
       [33, 34, 35, 36]]), 36, [6, 6, 6])

first change hh.a is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]
 [25 26 27 28]]

trainmat:
 [[ 1  2  3  4  2  3  4  5  3  4  5  6  4  5  6  7  6  7  8  9  9 10 11 12
   9 10 11 12]
 [ 5  6  7  8  6  7  8  9  7  8  9 10  8  9 10 11 10 11 12 13 13 14 15 16
  13 14 15 16]
 [ 9 10 11 12 10 11 12 13 11 12 13 14 12 13 14 15 14 15 16 17 17 18 19 20
  17 18 19 20]
 [13 14 15 16 14 15 16 17 15 16 17 18 16 17 18 19 18 19 20 21 21 22 23 24
  21 22 23 24]
 [17 18 19 20 18 19 20 21 19 20 21 22 20 21 22 23 22 23 24 25 25 26 27 28
  25 26 27 28]
 [21 22 23 24 22 23 24 25 23 24 25 26 24 25 26 27 26 27 28 29 29 30 31 32
  29 30 31 32]
 [25 26 27 28 26 27 28 29 27 28 29 30 28 29 30 31 30 31 32 33 33 34 35 36
  33 34 35 36]]


real	0m0.249s
user	0m0.139s
sys	0m0.021s

如果为pool(7):

结果为：

before change Arry is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]
 [25 26 27 28]] 

arry_l[0] is :

(array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28]]), 0, [0, 0, 0]) 

arry_l[1] is :

(array([[ 2,  3,  4,  5],
       [ 6,  7,  8,  9],
       [10, 11, 12, 13],
       [14, 15, 16, 17],
       [18, 19, 20, 21],
       [22, 23, 24, 25],
       [26, 27, 28, 29]]), 1, [1, 1, 1]) 

arry_l[2] is :

(array([[ 3,  4,  5,  6],
       [ 7,  8,  9, 10],
       [11, 12, 13, 14],
       [15, 16, 17, 18],
       [19, 20, 21, 22],
       [23, 24, 25, 26],
       [27, 28, 29, 30]]), 4, [2, 2, 2]) 

arry_l[3] is :

(array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]]), 9, [3, 3, 3]) 

arry_l[4] is :

(array([[ 5,  6,  7,  8],
       [ 9, 10, 11, 12],
       [13, 14, 15, 16],
       [17, 18, 19, 20],
       [21, 22, 23, 24],
       [25, 26, 27, 28],
       [29, 30, 31, 32]]), 16, [4, 4, 4]) 

arry_l[5] is :

(array([[ 6,  7,  8,  9],
       [10, 11, 12, 13],
       [14, 15, 16, 17],
       [18, 19, 20, 21],
       [22, 23, 24, 25],
       [26, 27, 28, 29],
       [30, 31, 32, 33]]), 25, [5, 5, 5]) 

arry_l[6] is :

(array([[ 7,  8,  9, 10],
       [11, 12, 13, 14],
       [15, 16, 17, 18],
       [19, 20, 21, 22],
       [23, 24, 25, 26],
       [27, 28, 29, 30],
       [31, 32, 33, 34]]), 36, [6, 6, 6]) 

first change hh.a is :
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]
 [25 26 27 28]] 

trainmat:
 [[ 1  2  3  4  2  3  4  5  3  4  5  6  4  5  6  7  5  6  7  8  6  7  8  9
   7  8  9 10]
 [ 5  6  7  8  6  7  8  9  7  8  9 10  8  9 10 11  9 10 11 12 10 11 12 13
  11 12 13 14]
 [ 9 10 11 12 10 11 12 13 11 12 13 14 12 13 14 15 13 14 15 16 14 15 16 17
  15 16 17 18]
 [13 14 15 16 14 15 16 17 15 16 17 18 16 17 18 19 17 18 19 20 18 19 20 21
  19 20 21 22]
 [17 18 19 20 18 19 20 21 19 20 21 22 20 21 22 23 21 22 23 24 22 23 24 25
  23 24 25 26]
 [21 22 23 24 22 23 24 25 23 24 25 26 24 25 26 27 25 26 27 28 26 27 28 29
  27 28 29 30]
 [25 26 27 28 26 27 28 29 27 28 29 30 28 29 30 31 29 30 31 32 30 31 32 33
  31 32 33 34]] 


real	0m0.256s
user	0m0.129s
sys	0m0.055s
'''
从这个例子可以看出，当在进程池中使用全局变量时是一件非常危险的事情，虽然退回主进程后全局变量的值还是不变，但是在进程池中当创建的
进程数小于任务数时，非常危险，当进程数大于等于任务数时没事，
'''
##############################################
import numpy as np
from multiprocessing import Process,Pool,Lock
import os
import time

arry = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24],[25,26,27,28]])

class test(object):
    def __init__(self):
        self.b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])

    def chan(self,i):
        print("i is :%d" % i)
        #time.sleep(1)
        print("第{}个子进程执行中>>>pid={},ppid={}".format(i,os.getpid(),os.getppid()))

def main():
    #a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    #b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
    hh = test()
    pool = Pool(3)
    for i in range(7):
        pool.apply_async(hh.chan,(i,))
    #print("第一个进程池开始")
    pool.close()
    pool.join()
    print("第一个进程池结束\n")

    pool = Pool(2)
    for i in range(5):
        pool.apply_async(hh.chan,(i,))
    print("第二个进程池开始")
    pool.close()
    pool.join()
    print("第二个进程池结束\n")

if __name__=='__main__':
    main()

结果为:
i is :0
第0个子进程执行中>>>pid=6019,ppid=6018
i is :1
第1个子进程执行中>>>pid=6020,ppid=6018
i is :2
第2个子进程执行中>>>pid=6021,ppid=6018
i is :3
第3个子进程执行中>>>pid=6019,ppid=6018
i is :4
第4个子进程执行中>>>pid=6020,ppid=6018
i is :5
第5个子进程执行中>>>pid=6021,ppid=6018
i is :6
第6个子进程执行中>>>pid=6019,ppid=6018
第一个进程池结束

第二个进程池开始
i is :0
第0个子进程执行中>>>pid=6025,ppid=6018
i is :2
第2个子进程执行中>>>pid=6025,ppid=6018
i is :3
第3个子进程执行中>>>pid=6025,ppid=6018
i is :1
i is :4
第4个子进程执行中>>>pid=6025,ppid=6018
第1个子进程执行中>>>pid=6026,ppid=6018
第二个进程池结束


real	0m0.362s
user	0m0.152s
sys	0m0.029s
从这个例子可以看出，当前后相继创建了多个进程池时，只要每个进程池都调用了close和join方法，则前后的进程池相互不影响。
