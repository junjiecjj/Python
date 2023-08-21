
"""
(一) 关于Pool和Process中子进程使用全局初始化的类中的成员变量的说明：
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
    每个进程中使用在所有进程外初始化(全局初始化)的LDPC类的实例 coder，在进入子进程函数时，LDPC会依次拷贝一份，也就是说，每个进程中的LDPC.a和encH都是完全独立的，互不影响，每个子进程对 coder.a的操作都是基于最开始类初始化的那个值的基础上进行的，且子进程对 coder.a 和coder.encH的更改对主进程是看不到的，也就是说回到主进程后.a和.encH还是原来的值，这在下面的实验中可以看到。

    实际上每个进程中的共享类实例 coder 的np.array/list/num/dict 等初始化成员都是完全独立的，互不干扰，这也是合理的，不然就没必要搞进程的通信，同步了。且，我们创建多进程的目的往往是并行执行某些任务，在中间计算过程中需要通信和同步的时候使用特殊的方法，如：
    (一) 对于process：需要使用进程通信，如： Queue（用于进程通信，资源共享）/Value，Array（用于进程通信，资源共享）/Pipe（用于管道通信）/Manager（用于资源共享）
                                        或者：同步子进程模块：Condition（条件变量）/Event（事件）/Lock（互斥锁）/RLock（可重入的互斥锁(同一个进程可以多次获得它，同时不会造成阻塞)/Semaphore（信号量）等
    (二) 对于Pool只能使用 multiprocessing.Manager()中的锁或者共享数据类型。

    切记以上，这个特性会对新手程序设计带来很大的好处，总之，多进程程序在进入子进程之前对非特殊变量(常见数据类型:np、dict、list、int等等非并行程序中使用的数据类型)会进行单独拷贝，在子进程的计算中是完全独立的，子进程的通信和共享数据需要使用特殊的包和技术。

(二) 关于Pool 的一些说明
    (1)
        ps=Pool(10)
        Pool 使用get来获取 ps.apply_async的结果,
        如果是 ps.apply, 则没有get方法,因为apply是同步执行,立刻获取结果,也根本无需get
        apply：同步，一般不使用
        apply_async：异步
    (2)
        Pool.apply_async：异步返回的结果是和加入进程池的顺序一致的；
        快的子进程先执行完，慢的后执行完，但结果返回的顺序仍与进程添加的顺序一致。


(三) Pool和Process在通信，同步方面的区别：
    https://zhuanlan.zhihu.com/p/64702600
    https://zhuanlan.zhihu.com/p/166091204  重要
    进程是系统独立调度核分配系统资源（CPU、内存）的基本单位，进程之间是相互独立的，每启动一个新的进程相当于把数据进行了一次克隆，子进程里的数据修改无法影响到主进程中的数据，不同子进程之间的数据也不能共享，这是多进程在使用中与多线程最明显的区别。

    网上的说法：
        Python 多线程之间共享变量很简单，直接定义全局 global 变量即可。而多进程之间是相互独立的执行单元，这种方法就不可行了。
        不过 Python 标准库已经给我们提供了这样的能力，使用起来也很简单。但要分两种情况来看，一种是 Process 多进程，一种是 Pool 进程池的方式。
        (1) Process 多进程：使用 Process 定义的多进程之间共享变量可以直接使用 multiprocessing 下的 Value，Array，Queue 等，如果要共享 list，dict，可以使用强大的 Manager 模块。
        (2) Pool 进程池：进程池之间共享变量是不能使用上文方式的，因为进程池内进程关系并非父子进程，想要共享，必须使用 Manager 模块来定义。

    下面是我自己的总结。
    注意Pool和Process的区别：
    Pool强调的是 进程数量固定，任务数很多；相当于工人数量固定，工作很多，一个工人完成一个任务接着被分配另一个任务，直到完成所有的任务；
    Process强调的是一个工人一个任务，每个工人完成自己的任务就可以了。

    这点区别决定了Pool和Process在通信，同步方面很大的不同，从上面的特点可以看出，Pool其实是不太支持通信和同步的，它强调的是永无止境的工作，但是Process因为工作数和工人数都是固定的，强调的是每个进程完成自己的一份工作的同时进行{进程的通信和同步}，具体有以下不同:
        (1) Queue（用于进程通信，资源共享）/Value，Array（用于进程通信，资源共享）/Pipe（用于管道通信）/Manager（用于资源共享）
        (2) 同步子进程模块：Condition（条件变量）/Event（事件）/Lock（互斥锁）/RLock（可重入的互斥锁(同一个进程可以多次获得它，同时不会造成阻塞)/Semaphore（信号量）等
    的关系：

    上面的(1)和(2)和 Process 都是天生适配的，因为上述的通信和同步函数都是为了有限个进程(每个进程完成指定的一个任务)而设计的，但是Pool缺不能(也能用，程序不报错，但是结果不是预期的)使用上述的大部分功能，具体：
        (1) Pool 不可以与 queue 配合使用
            虽然程序不会报错，但是一旦数据多，结果就不是预期想要的。

        (2) 注意：Value和Array只适用于Process类，不适用于 Pool。

        (3) Manager模块常与Pool模块一起使用。
            ps =  multiprocessing.Pool()
            ps.apply_async 与 lock = multiprocessing.Lock不能混用，但是 ps.apply_async 可以与 multiprocessing.Manager().lock() 和 multiprocessing.Manager().Array或者multiprocessing.Manager().dict联用

            也就是说，ps.apply_async 如果想加锁或者共享变量, 则只能使用 multiprocessing.Manager()中的锁或者共享数据类型：
            (1) Lock 对象需要用 Manager.Lock() 生成 。lock = multiprocessing.Manager().lock()
            (2) 共享的数据类型如 int 、list等只能由 Manager 生成。
                multiprocessing.Manager().Array('i', range(10)) 或者
                multiprocessing.Manager().Value('d', 0.0) 或者
                multiprocessing.Manager().dict([(i, 0) for i in range(4)]) 或者
                multiprocessing.Manager().list() /  multiprocessing.Manager().list([1,2,3,4])

        (4) 同步子进程模块：Condition（条件变量）/Event（事件）/Lock（互斥锁）/RLock（可重入的互斥锁(同一个进程可以多次获得它，同时不会造成阻塞)/Semaphore（信号量）等 于 Pool一起使用没有实际的意义，
            再次解释：因为Pool是强调很多任务，但是进程数量有限，基本上是完成一个任务接着下一个任务，所以Pool的进程同步没有意义，但是Pool数据共享和获取每个进程对每个输入任务的输出是有意义的。


    综上，Pool一般不用来进程通信和同步，一般 Pool 使用两种情况
    (一) 使用 Manager中的锁或者  Manager中的Array、Value、dict、list进行进程间的数据共享和交换
    (二) 如果想获取Pool中每个进程对每个输入任务的返回结果，则采用 .get() 获取进程池中每个进程对每个输入的对应输出.

(四) 必须明确：p.join()是让谁等？不管是Process还是join
     很明显p.join()是让主线程等待p的结束，卡住的是主线程而绝非子进程p，join是让主线程等,而p1-p4仍然是并发执行的,p1.join的时候,其余p2,p3,p4仍然在运行,等#p1.join结束,可能p2,p3,p4早已经结束了,这样p2.join,p3.join.p4.join直接通过检测，无需等待，所以4个join花费的总时间仍然是耗费时间最长的那个进程运行的时间
      不要把join直接放在start后面
           p.start()
           p.join()
    而是分布启动后再让主进程等：
    for i in range(8):
        p=Process(target=work)
        p.start()
        ps.append(p)

    for i in range(8):
        ps[i].join()

"""



#***********************************************************************
# https://blog.csdn.net/topleeyap/article/details/78981848



'''
使用进程池Pool
'''
import os,time
from multiprocessing import Process, Pool

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
# 结果：
# 主进程执行中>>>pid15340
# 子进程执行中>>>pid=15341,ppid=15340
# 子进程执行中>>>pid=15342,ppid=15340
# 子进程执行中>>>pid=15343,ppid=15340
# 子进程执行中>>>pid=15344,ppid=15340
# 子进程执行中>>>pid=15345,ppid=15340
# 子进程执行中>>>pid=15346,ppid=15340
# 子进程执行中>>>pid=15347,ppid=15340
# 子进程执行中>>>pid=15348,ppid=15340
# 子进程执行中>>>pid=15349,ppid=15340
# 子进程执行中>>>pid=15350,ppid=15340
# 子进程终止>>>pid=15342
# 子进程终止>>>pid=15348
# 子进程终止>>>pid=15341
# 子进程终止>>>pid=15345
# 子进程终止>>>pid=15343
# 子进程终止>>>pid=15347
# 子进程终止>>>pid=15349
# 子进程终止>>>pid=15350
# 子进程终止>>>pid=15346
# 子进程终止>>>pid=15344
# 主进程终止

# real	0m2.100s
# user	0m0.094s
# sys	0m0.033s

def worker():
    print("子进程执行中>>>pid={},ppid={}".format(os.getpid(),os.getppid()))
    time.sleep(2)
    print("子进程终止>>>pid={0}".format(os.getpid()))

def main():
    print("主进程执行中>>>pid{0}".format(os.getpid()))
    ps=Pool(3)

    for i in range(10):
        #ps.apply(worker,args=(i,))  #同步执行,每个进程依次进行，相当于串行
        ps.apply_async(worker,args=())  #异步执行

    ps.close() #关闭进程池，停止接受其他进程
    ps.join()  #阻塞进程

    print("主进程终止")

if __name__=='__main__':
    main()

# 主进程执行中>>>pid2279929
# 子进程执行中>>>pid=2279949,ppid=2279929
# 子进程执行中>>>pid=2279950,ppid=2279929
# 子进程执行中>>>pid=2279951,ppid=2279929
# 子进程终止>>>pid=2279949
# 子进程终止>>>pid=2279950
# 子进程终止>>>pid=2279951
# 子进程执行中>>>pid=2279949,ppid=2279929
# 子进程执行中>>>pid=2279950,ppid=2279929
# 子进程执行中>>>pid=2279951,ppid=2279929
# 子进程终止>>>pid=2279949
# 子进程执行中>>>pid=2279949,ppid=2279929
# 子进程终止>>>pid=2279950
# 子进程终止>>>pid=2279951
# 子进程执行中>>>pid=2279950,ppid=2279929
# 子进程执行中>>>pid=2279951,ppid=2279929
# 子进程终止>>>pid=2279949
# 子进程执行中>>>pid=2279949,ppid=2279929
# 子进程终止>>>pid=2279950
# 子进程终止>>>pid=2279951
# 子进程终止>>>pid=2279949
# 主进程终止
# real 8.89
# user 0.44
# sys 0.04

## 从以上可以看出，pool(3)，十个任务，怎需要分10%3+1 = 4次  共需要8s左右。
#********************************************
import os,time
from multiprocessing import Process,Pool, Lock

def worker(i, lock):
    print("子进程执行中>>>pid={},ppid={}".format(os.getpid(),os.getppid()))
    a=i
    b=i+1
    c=i/2
    d=i*2
    time.sleep(2)
    print("子进程终止>>>pid={0}".format(os.getpid()))
    lock.acquire()
    with open('/home/jack/桌面/D_clust6.txt','a') as f:
        f.write("***************分割线*******************\na is %f; b is %f; c is %f; d is %f\n "%(a,b,c,d))
    lock.release()
    return

def main():
    lock = Lock()
    print("主进程执行中>>>pid{0}".format(os.getpid()))

    ps=Pool(4)

    for i in range(6):
        #ps.apply(worker,args=(i,))  #同步执行,每个进程依次进行，相当于串行
        ps.apply_async(worker,args=(i, lock))  #异步执行

    ps.close() #关闭进程池，停止接受其他进程
    ps.join()  #阻塞进程

    print("主进程终止")

if __name__=='__main__':
    main()


# 主进程执行中>>>pid3113315
# 主进程终止


#===================================================
import os,time
from multiprocessing import Process,Pool, Lock

def worker(i,  ):
    print("子进程执行中>>>pid={},ppid={}".format(os.getpid(),os.getppid()))
    a=i
    b=i+1
    c=i/2
    d=i*2
    time.sleep(2)
    print("子进程终止>>>pid={0}".format(os.getpid()))
    # lock.acquire()
    with open('/home/jack/桌面/D_clust6.txt','a') as f:
        f.write("***************分割线*******************\na is %f; b is %f; c is %f; d is %f\n "%(a,b,c,d))
    # lock.release()
    return

def main():
    # lock = Lock()
    print("主进程执行中>>>pid{0}".format(os.getpid()))

    ps=Pool(4)

    for i in range(6):
        #ps.apply(worker,args=(i,))  #同步执行,每个进程依次进行，相当于串行
        ps.apply_async(worker,args=(i,  ))  #异步执行

    ps.close() #关闭进程池，停止接受其他进程
    ps.join()  #阻塞进程

    print("主进程终止")

if __name__=='__main__':
    main()

# D_clust6.txt文件的内容如下:

# ***************分割线*******************
# a is 1.000000; b is 2.000000; c is 0.500000; d is 2.000000
#  ***************分割线*******************
# a is 0.000000; b is 1.000000; c is 0.000000; d is 0.000000
#  ***************分割线*******************
# a is 2.000000; b is 3.000000; c is 1.000000; d is 4.000000
#  ***************分割线*******************
# a is 3.000000; b is 4.000000; c is 1.500000; d is 6.000000
#  ***************分割线*******************
# a is 4.000000; b is 5.000000; c is 2.000000; d is 8.000000
#  ***************分割线*******************
# a is 5.000000; b is 6.000000; c is 2.500000; d is 10.000000


## 从上面2个例子可以看出：ps.apply_async与multiprocessing.lock不能混用

#===================================================
##  Pool 与 multiprocessing.Manager()中的锁和 Array/value/list/dict
#===================================================
# https://blog.csdn.net/qq_39694935/article/details/84552076
import os,time
import multiprocessing
from multiprocessing import Process, Pool, Manager

def worker(i, aim_dict, lock ):
    print("子进程执行中>>>pid={},ppid={} ".format(os.getpid(),os.getppid()))
    # time.sleep(i)
    for i in aim_dict.keys():
        # 获取锁
        lock.acquire()
        aim_dict[i] += 1
        # 完成字典操作，释放锁
        lock.release()
    print("子进程终止>>>pid={0}".format(os.getpid()))
    return

def main():
    # lock = Lock()
    print(f"主进程执行中 pid{os.getpid()}" )
    # 创建Manger对象用于管理进程间通信
    manager = multiprocessing.Manager()
    # 使用 Manager 生成锁
    lock = manager.Lock()

    pool = Pool(4)
    Aim_dict = manager.dict([(i, 0) for i in range(4)])
    print(Aim_dict)

    for i in range(4):
        #ps.apply(worker,args=(i,))  #同步执行,每个进程依次进行，相当于串行
        pool.apply_async(worker,args=(i, Aim_dict, lock ))  #异步执行
    pool.close() #关闭进程池，停止接受其他进程
    pool.join()  #阻塞进程

    print(Aim_dict)
    print(f"主进程终止 pid{os.getpid()}" )

if __name__=='__main__':
    main()

# 主进程执行中 pid4066976
# 子进程执行中>>>pid=4168834,ppid=4066976 子进程执行中>>>pid=4168837,ppid=4066976

# 子进程终止>>>pid=4168837子进程终止>>>pid=4168834

# 子进程执行中>>>pid=4168828,ppid=4066976 子进程执行中>>>pid=4168831,ppid=4066976

# 子进程终止>>>pid=4168831子进程终止>>>pid=4168828

# {0: 0, 1: 0, 2: 0, 3: 0}
# {0: 4, 1: 4, 2: 4, 3: 4}
# 主进程终止 pid4066976


#========================================================================
## https://zhuanlan.zhihu.com/p/166091204
import os,time
import multiprocessing
from multiprocessing import Process, Pool, Manager

def worker(i, aim_dict, aim_list, aim_list1, aim_val, aim_arr, lock):
    print("子进程执行中>>>pid={},ppid={} ".format(os.getpid(),os.getppid()))
    # time.sleep(i)
    aim_dict[i]  = i
    aim_list[i] += i
    aim_list1 = [1,2,3]  # 在共享 list 时，像这样写 func 是不起作用的。这样写相当于重新定义了一个局部变量，并没有作用到原来的 list 上，必须使用 append，extend, [i]  =xx 等方法。
    aim_arr[i] += i**2
    lock.acquire()
    aim_val.value += i  ## value must used,  use aim_val += 1 is error
    # 完成字典操作，释放锁
    lock.release()

    print("子进程终止>>>pid={0}".format(os.getpid()))
    return

def main():
    # lock = Lock()
    print(f"主进程执行中 pid{os.getpid()}" )
    # 创建Manger对象用于管理进程间通信
    manager = multiprocessing.Manager()
    # 使用 Manager 生成锁
    lock = manager.Lock()
    n = 10
    pool = Pool(n)
    Aim_dict = manager.dict([(i, 0) for i in range(n)])
    Aim_list = manager.list(range(n))
    Aim_list1 = manager.list()
    Aim_val = manager.Value('d', 0.0)
    Aim_arr = manager.Array('i', range(n))
    print(f"0: Aim_dict = {Aim_dict}")
    print(f"0: Aim_list = {Aim_list}")
    print(f"0: Aim_list1 = {Aim_list1}")
    print(f"0: Aim_val = {Aim_val}")
    print(f"0: Aim_arr = {Aim_arr}")

    for i in range(n):
        #ps.apply(worker,args=(i,))  #同步执行,每个进程依次进行，相当于串行
        pool.apply_async(worker,args=(i, Aim_dict,Aim_list,Aim_list1, Aim_val, Aim_arr, lock))  #异步执行
    pool.close() #关闭进程池，停止接受其他进程
    pool.join()  #阻塞进程

    print(f"1: Aim_dict = {Aim_dict}")
    print(f"1: Aim_list = {Aim_list}")
    print(f"1: Aim_list1 = {Aim_list1}")
    print(f"1: Aim_val = {Aim_val}")
    print(f"1: Aim_arr = {Aim_arr}")
    print(f"主进程终止 pid{os.getpid()}" )

if __name__=='__main__':
    main()


# 主进程执行中 pid4066976
# 子进程执行中>>>pid=12637,ppid=4066976 子进程执行中>>>pid=12658,ppid=4066976 子进程执行中>>>pid=12644,ppid=4066976
# 子进程执行中>>>pid=12651,ppid=4066976


# 子进程终止>>>pid=12637
# 子进程终止>>>pid=12658子进程终止>>>pid=12644

# 子进程终止>>>pid=12651
# 0: Aim_dict = {0: 0, 1: 0, 2: 0, 3: 0}
# 0: Aim_list = [0, 1, 2, 3]
# 0: Aim_val = Value('d', 0.0)
# 0: Aim_arr = array('i', [0, 1, 2, 3])
# 1: Aim_dict = {0: 0, 1: 1, 2: 2, 3: 3}
# 1: Aim_list = [0, 2, 4, 6]
# 1: Aim_val = Value('d', 6.0)
# 1: Aim_arr = array('i', [0, 2, 6, 12])
# 主进程终止 pid4066976


###  if  n = 10
# 主进程执行中 pid4066976
# 0: Aim_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
# 0: Aim_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 0: Aim_list1 = []
# 0: Aim_val = Value('d', 0.0)
# 0: Aim_arr = array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 1: Aim_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
# 1: Aim_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# 1: Aim_list1 = []
# 1: Aim_val = Value('d', 45.0)
# 1: Aim_arr = array('i', [0, 2, 6, 12, 20, 30, 42, 56, 72, 90])
# 主进程终止 pid4066976


## 从上面两个案例可以看出 multiprocessing.Pool()可以与 multiprocessing.Manager().lock()和 multiprocessing.Manager().Array或者multiprocessing.Manager().dict联用

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
        res = p.apply(work,args=(i,)) #同步调用，直到本次任务执行完毕拿到res，等待任务work执行的过程中可能有阻塞也可能没有阻塞，但不管该任务是否存在阻塞，同步调用都会在原地等着，只是等的过程中若是任务发生了阻塞就会被夺走cpu的执行权限
        res_l.append(res)  # 使用get来获取apply_aync的结果,如果是apply, 则没有get方法,因为apply是同步执行,立刻获取结果,也根本无需get
    print(res_l)
# 结果：
# 10264 run
# 10265 run
# 10266 run
# 10264 run
# 10265 run
# 10266 run
# 10264 run
# 10265 run
# 10266 run
# 10264 run
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# real	0m10.103s
# user	0m0.080s
# sys	0m0.031s

# 如果是pool(10)。结果为：

# 15482 run
# 15483 run
# 15484 run
# 15485 run
# 15486 run
# 15487 run
# 15488 run
# 15489 run
# 15490 run
# 15491 run
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# real	0m10.116s
# user	0m0.091s
# sys	0m0.051s

## 可以看出 apply 是阻塞的，也就是不是真的并行， 是串行的
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

# 2295657 run
# 2295658 run
# 2295659 run
# 2295657 run
# 2295658 run
# 2295659 run
# 2295658 run
# 2295657 run
# 2295659 run
# 2295658 run
# 0
# 1
# 4
# 9
# 16
# 25
# 36
# 49
# 64
# 81
# real 4.71
# user 0.42
# sys 0.04


from multiprocessing import Pool
import os,time
def work(n):
    print('%s run' %os.getpid())
    time.sleep(1)
    return n**2

if __name__ == '__main__':
    p=Pool(10) #进程池中从无到有创建三个进程,以后一直是这三个进程在执行任务
    res_l=[]
    for i in range(10):
        res=p.apply_async(work,args=(i,)) #同步运行,阻塞、直到本次任务执行完毕拿到res
        res_l.append(res)

    #异步apply_async用法：如果使用异步提交的任务，主进程需要使用jion，等待进程池内任务都处理完，然后可以用get收集结果，否则，主进程结束，进程池可能还没来得及执行，也就跟着一起结束了
    p.close()
    p.join()
    for res in res_l:
        print(res.get()) #使用get来获取apply_aync的结果,如果是apply,则没有get方法,因为apply是同步执行,立刻获取结果,也根本无需get

# 2296104 run
# 2296105 run
# 2296107 run
# 2296108 run
# 2296109 run
# 2296106 run
# 2296111 run
# 2296110 run
# 2296112 run
# 2296113 run
# 0
# 1
# 4
# 9
# 16
# 25
# 36
# 49
# 64
# 81
# real 1.72
# user 0.43
# sys 0.07


# 以上的过程可以简化为：

from multiprocessing import Pool
import os,time
def work(n):
    print('%s run' %os.getpid())
    time.sleep(1)
    return n**2

if __name__ == '__main__':
    p=Pool(10) #进程池中从无到有创建三个进程,以后一直是这三个进程在执行任务
    res_l=[]
    result = []
    for i in range(10):
        res=p.apply_async(work,args=(i,)) #同步运行,阻塞、直到本次任务执行完毕拿到res
        res_l.append(res)
    #异步apply_async用法：如果使用异步提交的任务，主进程需要使用jion，等待进程池内任务都处理完，然后可以用get收集结果，否则，主进程结束，进程池可能还没来得及执行，也就跟着一起结束了
    p.close()
    p.join()

    for res in res_l:
        result.append(res.get()) # 使用get来获取apply_aync的结果,如果是apply,则没有get方法,因为apply是同步执行,立刻获取结果,也根本无需get
    print(result)
# 2297559 run
# 2297558 run
# 2297561 run
# 2297560 run
# 2297562 run
# 2297563 run
# 2297564 run
# 2297565 run
# 2297566 run
# 2297567 run
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
# real 1.68
# user 0.40
# sys 0.10




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

# 结果：
# ==============================>
# msg: hello 0
# msg: hello 1
# msg: hello 2
# msg: hello 3
# msg: hello 4
# msg: hello 5
# msg: hello 6
# msg: hello 7
# msg: hello 8
# msg: hello 9

# hello 0
# hello 1
# hello 2
# hello 3
# hello 4
# hello 5
# hello 6
# hello 7
# hello 8
# hello 9

# real	0m4.092s
# user	0m0.079s
# sys	0m0.026s

# 如果为pool(10)，结果为：

# ==============================>
# msg: hello 0
# msg: hello 1
# msg: hello 2
# msg: hello 3
# msg: hello 4
# msg: hello 5
# msg: hello 6
# msg: hello 7
# msg: hello 8
# msg: hello 9

# hello 0
# hello 1
# hello 2
# hello 3
# hello 4
# hello 5
# hello 6
# hello 7
# hello 8
# hello 9

# real	0m1.185s
# user	0m0.091s
# sys	0m0.024s

## apply_async是真并行

#=============================================================

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
    for i in res_l: # apply是同步的，所以直接得到结果，没有get()方法
        print(i)

# 结果：

# msg: hello 0
# msg: hello 1
# msg: hello 2
# msg: hello 3
# msg: hello 4
# msg: hello 5
# msg: hello 6
# msg: hello 7
# msg: hello 8
# msg: hello 9
# ==============================>
# ['hello 0', 'hello 1', 'hello 2', 'hello 3', 'hello 4', 'hello 5', 'hello 6', 'hello 7', 'hello 8', 'hello 9']
# hello 0
# hello 1
# hello 2
# hello 3
# hello 4
# hello 5
# hello 6
# hello 7
# hello 8
# hello 9

# real	0m10.106s
# user	0m0.097s
# sys	0m0.019s

# 如果为pool(10)，结果为：

# msg: hello 0
# msg: hello 1
# msg: hello 2
# msg: hello 3
# msg: hello 4
# msg: hello 5
# msg: hello 6
# msg: hello 7
# msg: hello 8
# msg: hello 9
# ==============================>
# ['hello 0', 'hello 1', 'hello 2', 'hello 3', 'hello 4', 'hello 5', 'hello 6', 'hello 7', 'hello 8', 'hello 9']
# hello 0
# hello 1
# hello 2
# hello 3
# hello 4
# hello 5
# hello 6
# hello 7
# hello 8
# hello 9

# real	0m10.117s
# user	0m0.102s
# sys	0m0.036s

## 可以看出 apply是 伪并行的，是串行的
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

# 结果：
# Current process 15606.
# Waiting for all subprocesses done...
# Task 0 (pid = 15607) is running...
# Task 1 (pid = 15608) is running...
# Task 2 (pid = 15609) is running...
# Task 1 end.
# Task 3 (pid = 15608) is running...
# Task 3 end.
# Task 4 (pid = 15608) is running...
# Task 2 end.
# Task 0 end.
# Task 4 end.
# All subprocesses done.

# real	0m2.871s
# user	0m0.067s
# sys	0m0.010s


def run_task(name):
    print('Task %s (pid = %s) is running...' % (name, os.getpid()))
    # time.sleep(random.random() * 3) #
    time.sleep(2)
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

# Current process 2300895.
# Waiting for all subprocesses done...
# Task 0 (pid = 2300900) is running...
# Task 2 (pid = 2300902) is running...
# Task 1 (pid = 2300901) is running...
# Task 0 end.
# Task 2 end.
# Task 1 end.
# Task 3 (pid = 2300900) is running...
# Task 4 (pid = 2300902) is running...
# Task 4 end.
# Task 3 end.
# All subprocesses done.
# real 4.75
# user 0.49
# sys 0.06


def run_task(name):
    print('Task %s (pid = %s) is running...' % (name, os.getpid()))
    # time.sleep(random.random() * 3) #
    time.sleep(2)
    print('Task %s end.' % name)

if __name__=='__main__':
    print('Current process %s.' % os.getpid())
    p = Pool(processes=5)
    for i in range(5):
        p.apply_async(run_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

# Current process 2301423.
# Waiting for all subprocesses done...
# Task 0 (pid = 2301431) is running...
# Task 2 (pid = 2301434) is running...
# Task 1 (pid = 2301432) is running...
# Task 3 (pid = 2301433) is running...
# Task 4 (pid = 2301435) is running...
# Task 1 end.
# Task 2 end.
# Task 3 end.
# Task 0 end.
# Task 4 end.
# All subprocesses done.
# real 2.87
# user 0.42
# sys 0.05
"""

注意上述代码中，pool(3)，但是创建range(5)时的进程号，这时总是只有三个固定的进程号，而不是5个,也就是说，进程的个数始终等于3，这三个进程轮流执行，
当执行完后，系统会分配下一个for循环的任务给空闲的进程；

"""

## apply_async是真并行
#***************************************
from multiprocessing import Pool
import os, time, random

def run_task(name):
    print('Task %s (pid = %s) is running...' % (name, os.getpid()))
    time.sleep(1) #sleep(2)
    print('Task %s end.' % name)

if __name__=='__main__':
    time_start = time.time()  # 记录开始时间
    print('Current process %s.' % os.getpid())
    p = Pool(processes=3)
    print("hello,jack")
    for i in range(10):
        print("hhh...")
        p.apply(run_task, args=(i,))
        print(i)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

# Current process 488153.
# Task 0 (pid = 555790) is running...
# hello,jack
# hhh...
# Task 0 end.
# Task 1 (pid = 555791) is running...
# 0
# hhh...
# Task 1 end.
# Task 2 (pid = 555792) is running...
# 1
# hhh...
# Task 2 end.
# Task 3 (pid = 555790) is running...
# 2
# hhh...
# Task 3 end.
# Task 4 (pid = 555791) is running...
# 3
# hhh...
# Task 4 end.
# Task 5 (pid = 555792) is running...
# 4
# hhh...
# Task 5 end.
# Task 6 (pid = 555790) is running...
# 5
# hhh...
# Task 6 end.
# Task 7 (pid = 555791) is running...
# 6
# hhh...
# Task 7 end.
# Task 8 (pid = 555792) is running...
# 7
# hhh...
# Task 8 end.
# Task 9 (pid = 555790) is running...
# 8
# hhh...
# Task 9 end.
# 9
# Waiting for all subprocesses done...
# 结束测试
# 10.230293273925781


def run_task(name):
    print('Task %s (pid = %s) is running...' % (name, os.getpid()))
    time.sleep(1) #sleep(2)
    print('Task %s end.' % name)

if __name__=='__main__':
    time_start = time.time()  # 记录开始时间
    print('Current process %s.' % os.getpid())
    p = Pool(processes=5)
    print("hello,jack")
    for i in range(10):
        print("hhh...")
        p.apply(run_task, args=(i,))
        print(i)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('结束测试')
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

# Current process 488153.
# Task 0 (pid = 556401) is running...
# hello,jack
# hhh...
# Task 0 end.
# Task 1 (pid = 556402) is running...
# 0
# hhh...
# Task 1 end.
# Task 2 (pid = 556403) is running...
# 1
# hhh...
# Task 2 end.
# Task 3 (pid = 556404) is running...
# 2
# hhh...
# Task 3 end.
# Task 4 (pid = 556405) is running...
# 3
# hhh...
# Task 4 end.
# Task 5 (pid = 556401) is running...
# 4
# hhh...
# Task 5 end.
# Task 6 (pid = 556402) is running...
# 5
# hhh...
# Task 6 end.
# Task 7 (pid = 556403) is running...
# 6
# hhh...
# Task 7 end.
# Task 8 (pid = 556404) is running...
# 7
# hhh...
# Task 8 end.
# Task 9 (pid = 556405) is running...
# 8
# hhh...
# Task 9 end.
# 9
# Waiting for all subprocesses done...
# 结束测试
# 10.205894947052002


## 注意看上述两个程序的子进程号
## apply是伪并行，


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
    print(os.getpid() + " end")

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = 3)
    for i in range(6):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去

    print("Mark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~")
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("Sub-process(es) done.")


# 结果：

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
执行说明：创建一个进程池pool，并设定进程的数量为3，range(6)会相继产生四个对象[0, 1, 2,3, 4,5 ]，四个对象被提交到pool中，
因pool指定进程数为3，所以0、1、2会直接送到进程中执行，当其中一个执行完事后才空出一个进程处理对象3，所以会出现输
出“msg: hello 3”出现在"end"后。因为为非阻塞，主函数会自己执行自个的，不搭理进程的执行，所以运行完for循环后直接输
出“mMsg: hark~ Mark~ Mark~~~~~~~~~~~~~~~~~~~~~~”，
主程序在pool.join（）处等待各个进程的结束。也就是说，pool.join()处主进程会等所有的子进程完成计算任务后才会继续执行join后的程序。
将 mMsg: hark~ Mark~ Mark~ 放在join后，则mMsg: hark~ Mark~ Mark~一定会在完成所有子进程打印后再打印
"""



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

## 可以看出，异步进程池的执行时间取决于最长的那个子进程

#********************************************************************************
#                       使用进程池并关注结果
#********************************************************************************

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

#**********************************************

#***************************************************
import numpy as np
from multiprocessing import Process,Pool
import time, random

class test(object):
    def __init__(self):
        self.a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        self.c = 0
        #self.b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])

    def chan(self,i):
        time.sleep(random.random())
        c = [1,2,3]
        self.a += i
        time.sleep(random.random())
        self.c = i
        time.sleep(random.random())
        print(f"{i}:{self.c}")
        return self.a ,i**2, c
    def achan(self,i):
        self.a[i,:]+=i

def main():
    #a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    #b = np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
    hh = test()
    print("before change hh.a is :\n",hh.a,'\n')
    # for i in range(4):
        # hh.achan(i)
    # print("first change a is:\n",hh.a,'\n')
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

# 结果为:

# before change hh.a is :
#  [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]
#  [13 14 15 16]]

# first change a is:
#  [[ 1  2  3  4]
#  [ 6  7  8  9]
#  [11 12 13 14]
#  [16 17 18 19]]

# arry_l[0] is :

# (array([[ 1,  2,  3,  4],
#        [ 6,  7,  8,  9],
#        [11, 12, 13, 14],
#        [16, 17, 18, 19]]), 0, [1, 2, 3])

# arry_l[1] is :

# (array([[ 2,  3,  4,  5],
#        [ 7,  8,  9, 10],
#        [12, 13, 14, 15],
#        [17, 18, 19, 20]]), 1, [1, 2, 3])

# arry_l[2] is :

# (array([[ 3,  4,  5,  6],
#        [ 8,  9, 10, 11],
#        [13, 14, 15, 16],
#        [18, 19, 20, 21]]), 4, [1, 2, 3])

# arry_l[3] is :

# (array([[ 4,  5,  6,  7],
#        [ 9, 10, 11, 12],
#        [14, 15, 16, 17],
#        [19, 20, 21, 22]]), 9, [1, 2, 3])

# second change hh.a is :
#  [[ 1  2  3  4]
#  [ 6  7  8  9]
#  [11 12 13 14]
#  [16 17 18 19]]

# last change hh.a is:
#  [[ 1  2  3  4]
#  [ 7  8  9 10]
#  [13 14 15 16]
#  [19 20 21 22]]


# real	0m0.245s
# user	0m0.127s
# sys	0m0.034s

# 此例子非常重要，可以看出，在主进程中创建的np.array数组，在子进程中改变后，程序回到主进程时这个数组不会改变，不仅如此，进入每个进程里时，每个进程的self.a是最开始的类的一个拷贝，互不影响，每个进程对self.a的操作都是基于最开始类初始化的那个值的基础上进行的。完全独立的，互不影响
# 这与process模块一致但是可以通过pool的返回值res以结果的形式传出来，且可以传多个值。

# 但是,在Pool之间共享数组可以通过
#                            manager = multiprocessing.Manager()
#                            Aim_dict = manager.dict([(i, 0) for i in range(4)])
#                            Aim_arr =  manager.Array('i', range(10))
#                            实现。
#    但是一般因为进程池强调的是任务很多，工人数量固定，因此还是采用.get()获取每个输入对应的输出为好.

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

看1977行的arry_l[4]和arry_l[3]就知道了，当多个子进程都需要改变一个全局变量时，这时候就会出现各个子进程读取的是同一个全局变量而不是预期的其他子进程更改后的全局变量，
所以多进程程序中不要更改全局变量，除非加锁。但是是没法保证子进程更改全局变量的顺序的。

但是类的成员(通过非Multiprocess.Manager()方式初始化的，如np.array, num, dict, list)在进入子进程的时候都已经被拷贝了一份，各个子进程的这些数据都是完全独立的，互不影响；

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
