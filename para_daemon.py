#!/usr/bin/env python
#-*-coding=utf-8-*-


#http://python.jobbole.com/82045/
#**********************************************

import time,random
from multiprocessing import Process

def worker(interval):
    print("work start :{0}".format(time.ctime()))
    time.sleep(interval)
    print("work end :{0}".format(time.ctime()))

if __name__=="__main__":
    p = Process(target=worker,args=(2,))
    #p.daemon=True #不加这行，end!!语句总是最先执行；加了总是最后执行；
    p.start()
    p.join()
    #有没有这行影响很大，因为设置了守护进程，所以有这行会等所有子进程结束后再退出；没有就会直接退出

    print("end!!")

#*******************************************


#https://www.cnblogs.com/linhaifeng/articles/7428874.html

from multiprocessing import Process
import time
import random

class Piao(Process):
    def __init__(self,name):
        super().__init__()
        self.name=name
    def run(self):
        print('%s is piaoing' %self.name)
        time.sleep(random.randrange(1,3))
        print('%s is piao end' %self.name)


p=Piao('egon')
p.daemon=True #一定要在p.start()前设置,设置p为守护进程,禁止p创建子进程,并且父进程代码执行结束,
              #p即终止运行
p.start()
p.join()
print('主')

#******************************

#主进程代码运行完毕,守护进程就会结束
from multiprocessing import Process
from threading import Thread
import time
def foo():
    print(123)
    time.sleep(1)
    print("end123")

def bar():
    print(456)
    time.sleep(3)
    print("end456")


p1=Process(target=foo)
p2=Process(target=bar)

p1.daemon=True
p1.start()
p2.start()
#p1.join()
#p2.join()
print("main-------") #打印该行则主进程代码结束,则守护进程p1应该被终止,可能会有p1任务执行的打印信息
                     #123,因为主进程打印main----时,p1也执行了,但是随即被终止

