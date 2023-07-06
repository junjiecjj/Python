#!/usr/bin/env python3
#-*-coding=utf-8-*-

###***************************************************************************
#以下代码主要在网址：https://blog.csdn.net/topleeyap/article/details/78981848中
'''
此方法是创建Process的实例，传入任务作为参数,此时可以不重写run()方法;
'''
import os,time
from multiprocessing import Process

def worker():
    print("子进程执行中>>> pid={0},ppid={1}".format(os.getpid(),os.getppid()))
    time.sleep(2)
    print("子进程终止>>>pid={0}".format(os.getpid()))

def main():
    print("主进程执行中>>>pid{0}".format(os.getpid()))

    ps=[]

    for i in range(5):
        p=Process(target=worker,name="worker"+str(i),args=())
        ps.append(p)

    for i in range(5):
        ps[i].start()

    for i in range(5):
        ps[i].join()

    print("主进程终止")

if __name__=='__main__':
    main()

结果：
主进程执行中>>>pid14497
子进程执行中>>> pid=14498,ppid=14497
子进程执行中>>> pid=14499,ppid=14497
子进程执行中>>> pid=14500,ppid=14497
子进程执行中>>> pid=14501,ppid=14497
子进程执行中>>> pid=14502,ppid=14497
子进程终止>>>pid=14499
子进程终止>>>pid=14498
子进程终止>>>pid=14500
子进程终止>>>pid=14501
子进程终止>>>pid=14502
主进程终止

real	0m2.074s
user	0m0.058s
sys	0m0.025s


##**************************************************************
'''
此方法是继承Process类，必须重写run()方法
'''
import os ,time
from multiprocessing import Process

class MyProcess(Process):
    def __init__(self):
        Process.__init__(self)

    def run(self):
        print("子进程执行中>>> pid={0},ppid={1}".format(os.getpid(),os.getppid()))
        time.sleep(2)
        print("子进程终止>>>pid={0}".format(os.getpid()))

def main():
    print("主进程开始>>>pid{}".format(os.getpid()))
    myp=MyProcess()
    myp.start() #start会自动调用run()
    myp.join()
    print("主进程终止")

if __name__=='__main__':
    main()

结果：
主进程开始>>>pid14581
子进程执行中>>> pid=14582,ppid=14581
子进程终止>>>pid=14582
主进程终止

real	0m2.069s
user	0m0.054s
sys	0m0.012s

##***********************************************
'''
使用进程池Pool
'''
import os,time
from multiprocessing import Pool

def worker(arg):
    print("子进程执行中>>>pid={},ppid={}".format(os.getpid(),os.getppid(),arg))
    time.sleep(2)
    print("子进程终止>>>pid={0}".format(os.getpid(),arg))

def main():
    print("主进程执行中>>>pid{0}".format(os.getpid()))

    ps=Pool(10)

    for i in range(10):
        #ps.apply(worker,args=(i,))  #同步执行,每个进程依次进行，相当于串行
        ps.apply_async(worker,args=(i,))  #异步执行

    ps.close() #关闭进程池，停止接受其他进程
    ps.join()  #阻塞进程

    print("主进程终止")

if __name__=='__main__':
    main()
结果：
主进程执行中>>>pid14688
子进程执行中>>>pid=14689,ppid=14688
子进程执行中>>>pid=14690,ppid=14688
子进程执行中>>>pid=14691,ppid=14688
子进程执行中>>>pid=14692,ppid=14688
子进程执行中>>>pid=14693,ppid=14688
子进程执行中>>>pid=14694,ppid=14688
子进程执行中>>>pid=14695,ppid=14688
子进程执行中>>>pid=14696,ppid=14688
子进程执行中>>>pid=14697,ppid=14688
子进程执行中>>>pid=14698,ppid=14688
子进程终止>>>pid=14689
子进程终止>>>pid=14696
子进程终止>>>pid=14690
子进程终止>>>pid=14698
子进程终止>>>pid=14697
子进程终止>>>pid=14695
子进程终止>>>pid=14693
子进程终止>>>pid=14692
子进程终止>>>pid=14691
子进程终止>>>pid=14694
主进程终止

real	0m2.115s
user	0m0.075s
sys	0m0.026s


#***************************************
#https://www.cnblogs.com/gregoryli/p/7892222.html

#**********************************************
from multiprocessing import Process
import time

def f(name):
    time.sleep(1)
    print('hello', name,time.ctime())

if __name__ == '__main__':
    p_list=[]
    for i in range(3):
        p = Process(target=f, args=('alvin',))
        p_list.append(p)
        p.start()
    for i in p_list:
        i.join()
    print('end')#一个主进程，三个子进程

结果：
hello alvin Mon Aug  6 00:06:26 2018
hello alvin Mon Aug  6 00:06:26 2018
hello alvin Mon Aug  6 00:06:26 2018
end

real	0m1.059s
user	0m0.049s
sys	0m0.014s


#*********************************************
from multiprocessing import Process
import time

class MyProcess(Process):
    def __init__(self):
        super(MyProcess, self).__init__()
        #self.name = name
    def run(self):
        time.sleep(1)
        print ('hello', self.name,time.ctime())

if __name__ == '__main__':
    p_list=[]
    for i in range(3):
        p = MyProcess()
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    print('end')
结果：
hello MyProcess-1 Mon Aug  6 00:07:46 2018
hello MyProcess-3 Mon Aug  6 00:07:46 2018
hello MyProcess-2 Mon Aug  6 00:07:46 2018
end

real	0m1.071s
user	0m0.056s
sys	0m0.019s


#***********************************************
from multiprocessing import Process
import os
import time

def info(title):
    print(title)
    print("module name:",__name__)
    print("parent process :",os.getppid())
    print("process id :%d\n" % (os.getpid()))

def f(name):
    info('\033[31;1mfunction f\033[0m')
    print('hello',name,'\n')

if __name__=='__main__':
    info('\033[32;1mmain process line\033[0m')
    time.sleep(2)
    p=Process(target=info,args=('bob',))
    p.start()
    p.join()

#***************************************************************
#https://www.cnblogs.com/linhaifeng/articles/7428874.html#_label4

#***********************************************************
#开进程的方法一:
import time
import random
from multiprocessing import Process
def piao(name):
    print('%s piaoing' %name)
    time.sleep(2)
    print('%s piao end' %name)



p1=Process(target=piao,args=('egon',)) #必须加,号
p2=Process(target=piao,args=('alex',))
p3=Process(target=piao,args=('wupeqi',))
p4=Process(target=piao,args=('yuanhao',))

p1.start()
p2.start()
p3.start()
p4.start()

p1.join()
p2.join()
p3.join()
p4.join()

print('主线程')

结果：
egon piaoing
alex piaoing
wupeqi piaoing
yuanhao piaoing
egon piao end
wupeqi piao end
alex piao end
yuanhao piao end
主线程

real	0m2.061s
user	0m0.047s
sys	0m0.020s

#*************************************************
#开进程的方法二:
import time
import random
from multiprocessing import Process


class Piao(Process):
    def __init__(self,name):
        super().__init__()
        self.name=name
    def run(self):
        print('%s piaoing' %self.name)

        time.sleep(2)
        print('%s piao end' %self.name)

p1=Piao('egon')
p2=Piao('alex')
p3=Piao('wupeiqi')
p4=Piao('yuanhao')

p1.start() #start会自动调用run
p2.start()
p3.start()
p4.start()

p1.join()
p2.join()
p3.join()
p4.join()
print('主线程')

结果：
egon piaoing
alex piaoing
wupeiqi piaoing
yuanhao piaoing
egon piao end
wupeiqi piao end
alex piao end
yuanhao piao end
主线程

real	0m2.061s
user	0m0.056s
sys	0m0.013s


#*****************************************
from multiprocessing import Process,Lock
import time
import numpy as np
import random

class CHAN(object):
    def __init__(self):
        self.name = 'chen'
        self.arry = np.array([[1,2,3],[4,5,6]])

    def f(self,lock,i):
        time.sleep(1)
        self.arry = self.arry+i
        lock.acquire()
        time.sleep(1)
        print("arry change is :\n",self.arry,'\n')
        lock.release()

if __name__ == '__main__':
    chan = CHAN()
    print("before change,arry is :\n",chan.arry,'\n')
    p_list=[]
    lock = Lock()
    for i in range(4):
        p = Process(target=chan.f, args=(lock,i,))
        p_list.append(p)
        p.start()
    for i in p_list:
        i.join()

    print("last arry is :\n",chan.arry,'\n')
    print('end')#一个主进程，三个子进程结果为：

结果为:

before change,arry is :
 [[1 2 3]
 [4 5 6]] 

arry change is :
 [[1 2 3]
 [4 5 6]] 

arry change is :
 [[2 3 4]
 [5 6 7]] 

arry change is :
 [[3 4 5]
 [6 7 8]] 

arry change is :
 [[4 5 6]
 [7 8 9]] 

last arry is :
 [[1 2 3]
 [4 5 6]] 

end

real	0m5.192s
user	0m0.171s
sys	0m0.028s

"""可以看到，arry是类的属性，在每个进程中都对arry进行了改变，但是当子进程都结束后，返回到主进程后
arry的值没变"""
#***********************************
from multiprocessing import Process
import multiprocessing
import time
import numpy as np
import random
import os

def test(i):
    name = multiprocessing.current_process().name
    start = time.ctime()
    print("process %s start at %s." % (name,start))
    Start = time.time()
    time.sleep(i)
    end = time.ctime()
    print("process %s end at %s." % (name,end))
    End = time.time()
    print("process %s time consum %s." % (name,End-Start))

def main():
    print("主进程执行中>>>pid{0}".format(os.getpid()))

    ps=[]

    for i in range(5):
        p=Process(target=test,name="worker"+str(i),args=(i,))
        ps.append(p)

    for i in range(5):
        ps[i].start()

    for i in range(5):
        ps[i].join()

    print("主进程终止")
结果为：

process worker0 start at Sat Aug 11 14:57:59 2018.
process worker0 end at Sat Aug 11 14:57:59 2018.
process worker0 time consum 3.790855407714844e-05.
process worker1 start at Sat Aug 11 14:57:59 2018.
process worker2 start at Sat Aug 11 14:57:59 2018.
process worker3 start at Sat Aug 11 14:57:59 2018.
process worker4 start at Sat Aug 11 14:57:59 2018.
process worker1 end at Sat Aug 11 14:58:00 2018.
process worker1 time consum 1.001197338104248.
process worker2 end at Sat Aug 11 14:58:01 2018.
process worker2 time consum 2.002206563949585.
process worker3 end at Sat Aug 11 14:58:02 2018.
process worker3 time consum 3.003229856491089.
process worker4 end at Sat Aug 11 14:58:03 2018.
process worker4 time consum 4.0042102336883545.
主进程终止

real	0m4.192s
user	0m0.163s
sys	0m0.038s
#**********************************************
