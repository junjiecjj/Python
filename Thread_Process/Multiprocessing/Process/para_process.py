#!/usr/bin/env python3
#-*-coding=utf-8-*-

"""


！！！！！一个重要提示：！！！！！
    在 Pool 和 Process中，如果是最开始(在主函数或者创建多进程之前)初始化一个类，如：
    class LDPC(object):
        def __init__(self):
            self.codedim  = 20         # 码的维数，编码前长度
            self.codelen  = 30         # 码的长度，编码后长度，码字长度
            self.codechk  = 10         # 校验位的个数
            # self.coderate = 0.0      # 码率
            # self.num_row  = 3
            # self.num_col  = 7
            # self.encH = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
            # self.encH     = np.array([[1,0,0,1,1,0,0],[0,1,0,0,1,1,1],[0,0,1,0,0,1,1]])
            self.encH     = np.random.randint(low = 0, high = 2, size = (self.codechk, self.codelen), dtype = np.int8 )
            self.a        = np.array([[0.1, 0.2],[0.3, 0.4],[0.5, 0.6],[0.7, 0.8]])
            ##
            print("LDPC初始化完成...")

        def chan(self,i):
            time.sleep(random.random())
            self.a += i
            time.sleep(random.random())
            c=[1,2,3]
            time.sleep(random.random())
            return self.a ,i**2, c

    coder = LDPC()

    然后在每个子进程中使用coder的成员变量a或者encH，虽然a和encH都是类coder的成员变量，且在函数外初始化LDPC的时候就已经确定了，且在每个进程中调用chan改变a的值，但是，接下来的这点特性非常重要：
    每个进程中使用在所有进程外初始化(全局初始化)的LDPC类的实例 coder，在进入子进程函数时，LDPC会依次拷贝一份，也就是说，每个进程中的LDPC.a和encH都是完全独立的，互不影响，每个子进程对 coder.a的操作都是基于最开始类初始化的那个值的基础上进行的。

    实际上每个进程中的共享类实例 coder 的np.array/list/num/dict 等初始化成员都是完全独立的，互不干扰，这也是合理的，不然就没必要搞进程的通信，同步了。且，我们创建多进程的目的往往是并行执行某些任务，在中间计算过程中需要通信和同步的时候使用特殊的方法，如：
    (一) 对于process：需要使用进程通信，如： Queue（用于进程通信，资源共享）/Value，Array（用于进程通信，资源共享）/Pipe（用于管道通信）/Manager（用于资源共享）
                                        或者：同步子进程模块：Condition（条件变量）/Event（事件）/Lock（互斥锁）/RLock（可重入的互斥锁(同一个进程可以多次获得它，同时不会造成阻塞)/Semaphore（信号量）等
    (二) 对于Pool只能使用 multiprocessing.Manager()中的锁或者共享数据类型。

    切记以上，这个特性会对新手程序设计带来很大的好处，总之，多进程程序在进入子进程之前对非特殊变量(常见数据类型:np、dict、list、int等等非并行程序中使用的数据类型)会进行单独拷贝，在子进程的计算中是完全独立的，子进程的通信和共享数据需要使用特殊的包和技术。


！！！！！一个重要提示：！！！！！
    (一)python多进程环境调用python自带的random在不同子进程中会生成不同的种子，
        而numpy.random不同子进程会fork相同的主进程中的种子, 默认每个进程会有相同的初始状态, 如果直接使用, 每个进程生成的随机序列会完全一致.

        如果不希望这种情况发生, 而是每个进程都是独立产生随机数, 有三种解决办法：
        (1) 需要在每个进程开始处对np.random初始化, 可以在每个进程用np.random.RandomState() 生成一个新的随机数引擎实例：
        import numpy as np
        import multiprocessing

        def gen_value(randomstate):
            values = []
            for i in range(10):
                values.append(randomstate.randint(100))   # 使用randomstart生成随机数
            print(values)

        procs = [multiprocessing.Process(target=gen_value, args=(np.random.RandomState(),)) for i in range(10)]

        for p in procs:
            p.start()
            p.join()


        (2)或者使用python原生的random模块替换np.random, 也会在每个进程初始化随机种子.
        (3)每个子进程使用不同的随机数种子初始化：np.random.seed(i)


        python自带的random在不同子进程中会生成不同的种子，而numpy.random不同子进程会fork相同的主进程中的种子。pytorch中的Dataloader类的__getitem__()会在不同子进程中发生不同的torch.seed()，并且种子与多进程的worker id有关（查看worker_init_fn参数说明）。但是三者互不影响，必须独立地处理。因此在写自己的数据准备代码时，如果使用了numpy中的随机化部件，一定要显示地在各个子进程中重新采样随机种子，或者使用python中的random发生随机种子。


"""


# https://www.cnblogs.com/linhaifeng/articles/7428874.html#_label5
#***************************************************************
#例一
#并发运行,效率高,但竞争同一打印终端,带来了打印错乱

from multiprocessing import Process
import os,time
import random

def fun1(name, i):
    print('测试%s多进程: %d' %(name, i))
    time.sleep(1)


if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    process_list = []
    for i in range(8):  #开启5个子进程执行fun1函数
        p = Process(target=fun1, args=('Python', i)) #实例化进程对象
        p.start()
        p.join() # 这里加了join()就变成串行了，且不会错乱，需要8s跑完；
        process_list.append(p)

    # for ps in process_list:
        # ps.join()

    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

#  /usr/bin/time -p ipython   test.py


# 结果：
# 加join()

# 测试Python多进程: 0
# 测试Python多进程: 1
# 测试Python多进程: 2
# 测试Python多进程: 3
# 测试Python多进程: 4
# 测试Python多进程: 5
# 测试Python多进程: 6
# 测试Python多进程: 7
# 结束测试
# 8.362524032592773

from multiprocessing import Process
import os,time
import random

def fun1(name, i):
    print('测试%s多进程: %d' %(name, i))
    time.sleep(1)


if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    process_list = []
    for i in range(8):  #开启5个子进程执行fun1函数
        p = Process(target=fun1, args=('Python', i)) #实例化进程对象
        p.start()
        # p.join() # 这里加了join()就变成串行了，且不会错乱，需要8s跑完；
        process_list.append(p)

    # for ps in process_list:
        # ps.join()

    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

# 测试Python多进程: 1测试Python多进程: 0
# 测试Python多进程: 3测试Python多进程: 2

# 测试Python多进程: 4测试Python多进程: 5

# 测试Python多进程: 6
# 结束测试
# 0.0588681697845459
# 测试Python多进程: 7

### 以上是并行，但是主进程不会等待子进程完成后再运行，而是不等待子进程，所以 "结束测试" 会比有些子进程提前打印
# 主线程等待p终止（强调：是主线程处于等的状态，而p是处于运行的状态）。timeout是可选的超时时间，需要强调的是，p.join只能join住start开启的进程，而不能join住run开启的进程



# 必须明确：p.join()是让谁等？
#很明显p.join()是让主线程等待p的结束，卡住的是主线程而绝非进程p，
# 所以上面的第一个程序：
        # p.start()
        # p.join() # 这里加了join()就变成串行了，且不会错乱，需要8s跑完；
 # p.join()紧跟着p.start()会导致程序变成串行，因为 join是告诉主进程等待当前子进程结束，而这时其他子进程还没启动，p.join()就紧跟着 start，当然当前子进程就结束了，程序变成串行。所以真正的 Process 并行程序如下：

#上述启动进程与join进程可以简写为
# p_l=[p1,p2,p3,p4]
#
# for p in p_l:
#     p.start()
#
# for p in p_l:
#     p.join()
#详细解析如下：
#进程只要start就会在开始运行了,所以p1-p4.start()时,系统中已经有四个并发的进程了
#而我们p1.join()是在等p1结束,没错p1只要不结束主线程就会一直卡在原地,这也是问题的关键
#join是让主线程等,而p1-p4仍然是并发执行的,p1.join的时候,其余p2,p3,p4仍然在运行,等#p1.join结束,可能p2,p3,p4早已经结束了,这样p2.join,p3.join.p4.join直接通过检测，无需等待
# 所以4个join花费的总时间仍然是耗费时间最长的那个进程运行的时间


## 正确使用 Process 且真正并行，主进程等待子进程结束后继续运行的程序如下两个：也就是join的位置非常关键；
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
        ps[i].join() #join应该这么用，千万别直接跟在start后面，这样会变成串行
    print("end")

# 结果：

# 1096 is running
# 1097 is running
# 1098 is running
# 1099 is running
# 1100 is running
# 1101 is running
# 1102 is running
# 1103 is running
# 1097 is done
# 1096 is done
# 1098 is done
# 1099 is done
# 1100 is done
# 1101 is done
# 1102 is done
# 1103 is done
# end

# real	0m1.078s
# user	0m0.082s
# sys	0m0.010s

from multiprocessing import Process
import os,time
import random


# function()   执行的程序

def fun1(name, i):
    print('测试%s多进程: %d \n' %(name, i))
    time.sleep(1)


if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    process_list = []
    for i in range(8):  #开启5个子进程执行fun1函数
        p = Process(target=fun1, args=('Python', i)) #实例化进程对象
        p.start()
        # p.join() # 这里加了join()就变成串行了，且不会错乱，需要8s跑完；
        process_list.append(p)

    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)


# 测试Python多进程: 0

# 测试Python多进程: 1

# 测试Python多进程: 2

# 测试Python多进程: 3

# 测试Python多进程: 4

# 测试Python多进程: 6
# 测试Python多进程: 5


# 测试Python多进程: 7

# 结束测试
# 1.0764784812927246


#*************************************************
## 多进程产生随机数的正确用法1

from multiprocessing import Process
import os,time
import random
import numpy as np

def work(r, i, ):
    # print(f'{os.getpid()} is running')
    np.random.seed()
    # a = RDM.binomial(n = 1, p = 0.5, size = 20)
    a = np.random.binomial(n = 1, p = 0.5, size = 10)
    print(f"round {r}, client {i}, {a}")
    # time.sleep(np.random.random())
    #注意，这里如果是sleep(1)则看不出来错乱；但是如果是sleep(random.randrange(0,2))则看得出错乱
    # print(f'{os.getpid()} is done')

if __name__ == '__main__':
    for r in range(3):
        ps=[]
        n = 4
        for i in range(n):
            # rdm = np.random.RandomState()
            p=Process(target=work, args = (r, i, ))
            p.start()
            ps.append(p)

        for i in range(n):
            ps[i].join() #join应该这么用，千万别直接跟在start后面，这样会变成串行
    print("end")
# round 0, client 0, [0 1 0 1 0 1 0 1 1 1]
# round 0, client 1, [0 0 0 0 0 0 0 0 0 0]
# round 0, client 2, [1 1 1 0 0 0 0 0 1 1]
# round 0, client 3, [1 0 1 1 0 1 1 1 0 1]
# round 1, client 0, [0 1 1 0 0 0 1 1 1 0]
# round 1, client 1, [1 1 1 0 0 0 1 1 1 1]
# round 1, client 2, [0 1 0 0 0 0 1 1 0 1]
# round 1, client 3, [1 1 0 0 0 0 1 0 1 0]
# round 2, client 0, [0 0 1 1 1 1 0 1 0 1]
# round 2, client 1, [0 0 1 0 0 1 1 1 1 1]
# round 2, client 2, [0 1 1 1 1 1 0 0 1 0]
# round 2, client 3, [0 0 1 1 0 0 1 0 0 0]
# end


#*************************************************
## 多进程产生随机数的正确用法2
def work(r, i, RDM):
    # print(f'{os.getpid()} is running')
    # np.random.seed()
    a = RDM.binomial(n = 1, p = 0.5, size = 20)
    a = np.random.binomial(n = 1, p = 0.5, size = 10)
    print(f"round {r}, client {i}, {a}")
    # time.sleep(1)
    #注意，这里如果是sleep(1)则看不出来错乱；但是如果是sleep(random.randrange(0,2))则看得出错乱
    # print(f'{os.getpid()} is done')

if __name__ == '__main__':
    for r in range(3):
        ps=[]
        n = 4
        for i in range(n):
            rdm = np.random.RandomState()
            p=Process(target=work, args = (r, i, rdm))
            p.start()
            ps.append(p)

        for i in range(n):
            ps[i].join() #join应该这么用，千万别直接跟在start后面，这样会变成串行
    print("end")

# round 0, client 0, [1 0 0 1 1 1 0 0 1 1 1 1 0 1 0 0 1 1 0 1]
# round 0, client 1, [1 0 1 1 1 1 1 1 1 1 0 1 0 0 0 0 1 0 1 0]
# round 0, client 2, [0 0 1 1 0 0 0 1 1 1 1 1 1 0 0 1 0 1 1 0]
# round 0, client 3, [1 0 1 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 1 1]
# round 1, client 0, [1 1 1 1 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0]
# round 1, client 1, [1 1 1 0 0 1 0 0 0 1 0 1 0 0 0 1 1 0 1 1]
# round 1, client 2, [1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 1 1 0 1 1]
# round 1, client 3, [1 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 1 1 0]
# round 2, client 0, [1 0 1 0 0 0 1 1 1 1 0 0 1 0 1 0 0 1 1 0]
# round 2, client 1, [0 1 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 1 1 0]
# round 2, client 2, [0 1 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1]
# round 2, client 3, [1 1 1 0 0 0 1 1 1 1 0 1 0 0 0 1 1 0 1 1]
# end


##  或者
#*************************************************
## 多进程产生随机数的正确用法3
def randomGene():
    return np.random.binomial(n = 1, p = 0.5, size = 10)


def work(r, i):
    # print(f'{os.getpid()} is running')
    np.random.seed(r+i)
    a = randomGene()
    # a = np.random.binomial(n = 1, p = 0.5, size = 10)
    print(f"round {r}, client {i}, {a}")
    # time.sleep(1)
    #注意，这里如果是sleep(1)则看不出来错乱；但是如果是sleep(random.randrange(0,2))则看得出错乱
    # print(f'{os.getpid()} is done')

if __name__ == '__main__':
    for r in range(3):
        ps=[]
        n = 4
        for i in range(n):
            p=Process(target=work, args = (r, i))
            p.start()
            ps.append(p)

        for i in range(n):
            ps[i].join() #join应该这么用，千万别直接跟在start后面，这样会变成串行
    print("end")
# round 0, client 0, [1 1 1 1 0 1 0 1 1 0]
# round 0, client 1, [0 1 0 0 0 0 0 0 0 1]
# round 0, client 2, [0 0 1 0 0 0 0 1 0 0]
# round 0, client 3, [1 1 0 1 1 1 0 0 0 0]
# round 1, client 0, [0 1 0 0 0 0 0 0 0 1]
# round 1, client 1, [0 0 1 0 0 0 0 1 0 0]
# round 1, client 2, [1 1 0 1 1 1 0 0 0 0]
# round 1, client 3, [1 1 1 1 1 0 1 0 0 0]
# round 2, client 0, [0 0 1 0 0 0 0 1 0 0]
# round 2, client 1, [1 1 0 1 1 1 0 0 0 0]
# round 2, client 2, [1 1 1 1 1 0 1 0 0 0]
# round 2, client 3, [0 1 0 1 0 1 1 1 0 0]
# end

np.random.seed(1)
for i in range(3):
    a = np.random.binomial(n = 1, p = 0.5, size = 10)
    print(f"{a}")




##  或者
#*************************************************
## 多进程产生随机数的正确用法 4
import multiprocessing as mp

def randomGene(param):
    p = param - np.floor(param)
    # p = torch.tensor([[0.38469, 0.55079, 0.16089, 0.40244],
    #         [0.39679, 0.15747, 0.64622, 0.64451],
    #         [0.46007, 0.04083, 0.31302, 0.80131]])
    # a = torch.bernoulli(p)
    # a = torch.bernoulli(torch.tensor(p)).numpy()
    G =  2**(8 - 1)
    p = (param * G + 1)/2
    p = np.clip(p, a_min = 0, a_max = 1, )
    f1 = np.frompyfunc(lambda x : int(np.random.binomial(1, x, 1)[0]), 1, 1)
    a = f1(p).astype(np.int8)
    return a


def work(r, i):
    # print(f'{os.getpid()} is running')
    np.random.seed(r+i)
    # torch.manual_seed(r+i)
    params = np.random.rand(10,)*0.001
    a = randomGene(params)
    # a = np.random.binomial(n = 1, p = 0.5, size = 10)
    print(f"round {r}, client {i}, {a}")
    # time.sleep(1)
    #注意，这里如果是sleep(1)则看不出来错乱；但是如果是sleep(random.randrange(0,2))则看得出错乱
    # print(f'{os.getpid()} is done')

if __name__ == '__main__':
    for r in range(3):
        ps=[]
        n = 4
        for i in range(n):
            p = mp.Process(target=work, args = (r, i))
            p.start()
            ps.append(p)

        for i in range(n):
            ps[i].join() #join应该这么用，千万别直接跟在start后面，这样会变成串行
    print("end")








###***************************************************************************
#以下代码主要在网址：https://blog.csdn.net/topleeyap/article/details/78981848 中
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

# 结果：
# 主进程执行中>>>pid14497
# 子进程执行中>>> pid=14498,ppid=14497
# 子进程执行中>>> pid=14499,ppid=14497
# 子进程执行中>>> pid=14500,ppid=14497
# 子进程执行中>>> pid=14501,ppid=14497
# 子进程执行中>>> pid=14502,ppid=14497
# 子进程终止>>>pid=14499
# 子进程终止>>>pid=14498
# 子进程终止>>>pid=14500
# 子进程终止>>>pid=14501
# 子进程终止>>>pid=14502
# 主进程终止

# real	0m2.074s
# user	0m0.058s
# sys	0m0.025s


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
# 结果：
# 主进程执行中>>>pid14688
# 子进程执行中>>>pid=14689,ppid=14688
# 子进程执行中>>>pid=14690,ppid=14688
# 子进程执行中>>>pid=14691,ppid=14688
# 子进程执行中>>>pid=14692,ppid=14688
# 子进程执行中>>>pid=14693,ppid=14688
# 子进程执行中>>>pid=14694,ppid=14688
# 子进程执行中>>>pid=14695,ppid=14688
# 子进程执行中>>>pid=14696,ppid=14688
# 子进程执行中>>>pid=14697,ppid=14688
# 子进程执行中>>>pid=14698,ppid=14688
# 子进程终止>>>pid=14689
# 子进程终止>>>pid=14696
# 子进程终止>>>pid=14690
# 子进程终止>>>pid=14698
# 子进程终止>>>pid=14697
# 子进程终止>>>pid=14695
# 子进程终止>>>pid=14693
# 子进程终止>>>pid=14692
# 子进程终止>>>pid=14691
# 子进程终止>>>pid=14694
# 主进程终止

# real	0m2.115s
# user	0m0.075s
# sys	0m0.026s


#***************************************
#https://www.cnblogs.com/gregoryli/p/7892222.html

#**********************************************
'''
此方法是创建Process的实例，传入任务作为参数,此时可以不重写run()方法;
'''
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

# 结果：
# hello alvin Mon Aug  6 00:06:26 2018
# hello alvin Mon Aug  6 00:06:26 2018
# hello alvin Mon Aug  6 00:06:26 2018
# end

# real	0m1.059s
# user	0m0.049s
# sys	0m0.014s


#*********************************************
'''
此方法是继承Process类，必须重写run()方法
'''
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

# 结果：
# hello MyProcess-1 Mon Aug  6 00:07:46 2018
# hello MyProcess-3 Mon Aug  6 00:07:46 2018
# hello MyProcess-2 Mon Aug  6 00:07:46 2018
# end

# real	0m1.071s
# user	0m0.056s
# sys	0m0.019s


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

## 从这个例子可以看出：多个进程并行，计算时间取决于最慢的那个。
#**********************************************
