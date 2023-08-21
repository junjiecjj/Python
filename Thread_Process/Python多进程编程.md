

本文档主要是记录Python多进程编程，包括：

+ 多进程的创建
+ 多进程的工作方式
+ 多进程的通信

因为python中GIL的缘故，多线程是“伪线程”，并不是真正的多线程，因此不做介绍。本文档写于2023-08-18中山大学博士期间，因为需要用到多进程加速，所以记录相关技术，但是对多进程的通信没有需求，因此主要是介绍多进程的创建、进程内的运算、以及计算结果需要按顺序返回。



注意：因为这里主要介绍 Python的自带的多进程包 multiprocessing，实际上，python还有第三方库， mpi4py：

+ Python 提供了很多MPI模块写并行程序。其中 `mpi4py` 是一个又有意思的库。它在MPI-1/2顶层构建，提供了面向对象的接口，紧跟C++绑定的 MPI-2。MPI的C语言用户可以无需学习新的接口就可以上手这个库。所以，它成为了Python中最广泛使用的MPI库。
+ 此模块包含的主要应用有：
  + 点对点通讯
  + 集体通讯
  + 拓扑

+ mpi4py 完全复刻 C/C++中的  open  MPI 编程, 相当不错。

**multiprocessing Vs. mpi4py, 实际上，**

`multiprocessing` 是Python标准库中的模块，实现了共享内存机制，也就是说，可以让运行在不同处理器核心的进程能读取共享内存。

`mpi4py` 库实现了消息传递的编程范例（设计模式）。简单来说，就是进程之间不靠任何共享信息来进行通讯（也叫做shared nothing），所有的交流都通过传递信息代替。

这方面与使用共享内存通讯，通过加锁或类似机制实现互斥的技术行成对比。在信息传递的代码中，进程通过 `send()` 和 `receive` 进行交流。

+ 如果是本地的多进程，multiprocessing 一般够用，既可以进程通信又可以进行多进程的计算，我用它主要利用它的多进程计算；

+ mpi4py 一般用在分布式内存的多进程编程，多个主机间进行通信，强调的是多进程通信。

+ 而且，multiprocessing程序启动是正常启动，但是 mpi4py的启动如下，得专用启动：

  `mpiexec -n 9 python pointToPointCommunication.py`



教程：

https://python-parallel-programmning-cookbook.readthedocs.io/zh_CN/latest/chapter3/01_Introduction.html

网页：

https://zhuanlan.zhihu.com/p/64702600

https://zhuanlan.zhihu.com/p/340657122

https://www.jianshu.com/p/91d8657f72ba

https://blog.csdn.net/Victor2code/article/details/109005171

https://www.cnblogs.com/linhaifeng/articles/7428874.html#_label5

https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd

https://zhuanlan.zhihu.com/p/454590612

https://zhuanlan.zhihu.com/p/166091204      重要

https://blog.csdn.net/AlfaCuton/article/details/118496696

https://zhuanlan.zhihu.com/p/46368084



#  Python的多进程包 multiprocessing

Python的threading包主要运用多线程的开发，但由于GIL的存在，Python中的多线程其实并不是真正的多线程，如果想要充分地使用多核CPU的资源，大部分情况需要使用多进程。在Python 2.6版本的时候引入了multiprocessing包，它完整的复制了一套threading所提供的接口方便迁移。唯一的不同就是它使用了多进程而不是多线程。每个进程有自己的独立的GIL，因此也不会出现进程之间的GIL争抢。

借助这个multiprocessing，你可以轻松完成从单进程到并发执行的转换。multiprocessing支持子进程、通信和共享数据、执行不同形式的同步，提供了Process、Queue、Pipe、Lock等组件。

## Multiprocessing产生的背景

除了应对Python的GIL以外，产生multiprocessing的另外一个原因时Windows操作系统与Linux/Unix系统的不一致。

Unix/Linux操作系统提供了一个fork()系统调用，它非常特殊。普通的函数，调用一次，返回一次，但是fork()调用一次，返回两次，因为操作系统自动把当前进程（父进程）复制了一份（子进程），然后，分别在父进程和子进程内返回。子进程永远返回0，而父进程返回子进程的ID。

这样做的理由是，一个父进程可以fork出很多子进程，所以，父进程要记下每个子进程的ID，而子进程只需要调用getpid()就可以拿到父进程的ID。

Python的os模块封装了常见的系统调用，其中就包括fork，可以在Python程序中轻松创建子进程：

有了fork调用，一个进程在接到新任务时就可以复制出一个子进程来处理新任务，常见的Apache服务器就是由父进程监听端口，每当有新的http请求时，就fork出子进程来处理新的http请求。

由于Windows没有fork调用，上面的代码在Windows上无法运行。由于Python是跨平台的，自然也应该提供一个跨平台的多进程支持。multiprocessing模块就是跨平台版本的多进程模块。multiprocessing模块封装了fork()调用，使我们不需要关注fork()的细节。由于Windows没有fork调用，因此，multiprocessing需要“模拟”出fork的效果。

##  multiprocessing 常用组件及功能

创建管理进程模块：

- Process（用于创建进程）
- Pool（用于创建管理进程池）
- Queue（用于进程通信，资源共享）
- Pipe（用于管道通信）
- **Value，Array（用于进程通信，资源共享）**
- **Manager（用于资源共享）**

同步子进程模块：

- Condition（条件变量）
- Event（事件）
- Lock（互斥锁）
- RLock（可重入的互斥锁(同一个进程可以多次获得它，同时不会造成阻塞)
- Semaphore（信号量）

multiprocessing相比于 Linux  中的进程相关内容会少很多，但是基本够用。接下来就一起来学习下每个组件及功能的具体使用方法。



# Process（用于创建进程）

一次只能创建一个

multiprocessing模块提供了一个Process类来代表一个进程对象。

在multiprocessing中，每一个进程都用一个Process类来表示。

构造方法：Process([group [, target [, name [, args [, kwargs]]]]])

- group：分组，实际上不使用，值始终为None
- target：表示调用对象，即子进程要执行的任务，你可以传入方法名
- name：为子进程设定名称
- args：要传给target函数的位置参数，以元组方式进行传入。
- kwargs：要传给target函数的字典参数，以字典方式进行传入。

实例方法：

- start()：启动进程，并调用该子进程中的p.run()
- run()：进程启动时运行的方法，正是它去调用target指定的函数，我们自定义类的类中一定要实现该方法
- terminate()：强制终止进程p，不会进行任何清理操作，如果p创建了子进程，该子进程就成了僵尸进程，使用该方法需要特别小心这种情况。如果p还保存了一个锁那么也将不会被释放，进而导致死锁
- is_alive()：返回进程是否在运行。如果p仍然运行，返回True
- join([timeout])：进程同步，主进程等待子进程完成后再执行后面的代码。线程等待p终止（强调：是主线程处于等的状态，而p是处于运行的状态）。timeout是可选的超时时间（超过这个时间，父线程不再等待子线程，继续往下执行），需要强调的是，p.join只能join住start开启的进程，而不能join住run开启的进程

属性介绍：

- daemon：默认值为False，如果设为True，代表p为后台运行的守护进程；当p的父进程终止时，p也随之终止，并且设定为True后，p不能创建自己的新进程；必须在p.start()之前设置
- name：进程的名称
- pid：进程的pid
- exitcode：进程在运行时为None、如果为–N，表示被信号N结束(了解即可)
- authkey：进程的身份验证键,默认是由os.urandom()随机生成的32字符的字符串。这个键的用途是为涉及网络连接的底层进程间通信提供安全性，这类连接只有在具有相同的身份验证键时才能成功（了解即可）









#  进程池 Pool（用于创建管理进程池）

https://zhuanlan.zhihu.com/p/46368084

很多时候系统都需要创建多个进程以提高CPU的利用率，当数量较少时，可以手动生成一个个Process实例。当进程数量很多时，或许可以利用循环，但是这需要程序员手动管理系统中并发进程的数量，有时会很麻烦。这时进程池Pool就可以发挥其功效了。可以通过传递参数限制并发进程的数量，默认值为CPU的核数。

Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果进程池还没有满，就会创建一个新的进程来执行请求。如果池满，请求就会告知先等待，直到池中有进程结束，才会创建新的进程来执行这些请求。



下面介绍一下multiprocessing 模块下的Pool类的几个方法：

1.apply_async

函数原型：apply_async(func[, args=()[, kwds={}[, callback=None]]])

**其作用是向进程池提交需要执行的函数及参数， 各个进程采用非阻塞（异步）的调用方式，即每个子进程只管运行自己的，不管其它进程是否已经完成。这是默认方式。**

2.map()

函数原型：map(func, iterable[, chunksize=None])

Pool类中的map方法，与内置的map函数用法行为基本一致，它会使进程阻塞直到结果返回。 注意：虽然第二个参数是一个迭代器，但在实际使用中，必须在整个队列都就绪后，程序才会运行子进程。

3.map_async()

函数原型：map_async(func, iterable[, chunksize[, callback]])
与map用法一致，但是它是非阻塞的。其有关事项见apply_async。

4.close()

关闭进程池（pool），使其不在接受新的任务。

\5. terminate()

结束工作进程，不在处理未处理的任务。

6.join()

主进程阻塞等待子进程的退出， join方法要在close或terminate之后使用。

-----

------



进程池内部维护一个进程序列，当使用时，则去进程池中获取一个进程，如果进程池序列中没有可供使用的进进程，那么程序就会等待，直到进程池中有可用进程为止。就是固定有几个进程可以使用。

进程池中有两个方法：

+ apply：同步，一般不使用
+ apply_async：异步

![图片](https://mmbiz.qpic.cn/mmbiz_png/fhujzoQe7TrMS1mPJIK4ia993XvTmTMY9P1U35Clr8lUV1dCibv1GEZcD2NomziccxkEdduGCx81GWy1EKcUSgvLg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

Pool类用于需要执行的目标很多，而手动限制进程数量又太繁琐时，如果目标少且不用控制进程数量则可以用Process类。Pool可以提供指定数量的进程，供用户调用，当有新的请求提交到Pool中时，如果池还没有满，那么就会创建一个新的进程用来执行该请求；但如果池中的进程数已经达到规定最大值，那么该请求就会等待，直到池中有进程结束，就重用进程池中的进程。

构造方法：Pool([processes[, initializer[, initargs[, maxtasksperchild[, context]]]]])

- processes ：要创建的进程数，如果省略，将默认使用cpu_count()返回的数量。
- initializer：每个工作进程启动时要执行的可调用对象，默认为None。如果initializer是None，那么每一个工作进程在开始的时候会调用initializer(*initargs)。
- initargs：是要传给initializer的参数组。
- maxtasksperchild：工作进程退出之前可以完成的任务数，完成后用一个新的工作进程来替代原进程，来让闲置的资源被释放。maxtasksperchild默认是None，意味着只要Pool存在工作进程就会一直存活。
- context: 用在制定工作进程启动时的上下文，一般使用Pool() 或者一个context对象的Pool()方法来创建一个池，两种方法都适当的设置了context。

实例方法：

- apply(func[, args[, kwargs]])：在一个池工作进程中执行func(args,*kwargs),然后返回结果。需要强调的是：此操作并不会在所有池工作进程中并执行func函数。如果要通过不同参数并发地执行func函数，必须从不同线程调用p.apply()函数或者使用p.apply_async()。它是阻塞的。apply很少使用
- apply_async(func[, arg[, kwds={}[, callback=None]]])：在一个池工作进程中执行func(args,*kwargs),然后返回结果。此方法的结果是AsyncResult类的实例，callback是可调用对象，接收输入参数。当func的结果变为可用时，将理解传递给callback。callback禁止执行任何阻塞操作，否则将接收其他异步操作中的结果。它是非阻塞。
- map(func, iterable[, chunksize=None])：Pool类中的map方法，与内置的map函数用法行为基本一致，它会使进程阻塞直到返回结果。注意，虽然第二个参数是一个迭代器，但在实际使用中，必须在整个队列都就绪后，程序才会运行子进程。
- map_async(func, iterable[, chunksize=None])：map_async与map的关系同apply与apply_async
- imap()：imap 与 map的区别是，map是当所有的进程都已经执行完了，并将结果返回了，imap()则是立即返回一个iterable可迭代对象。
- imap_unordered()：不保证返回的结果顺序与进程添加的顺序一致。
- close()：关闭进程池，防止进一步操作。如果所有操作持续挂起，它们将在工作进程终止前完成。
- join()：等待所有工作进程退出。此方法只能在close()或teminate()之后调用，让其不再接受新的Process。
- terminate()：结束工作进程，不再处理未处理的任务。

方法apply_async()和map_async()的返回值是AsyncResul的实例obj。实例具有以下方法：

- get()：返回结果，如果有必要则等待结果到达。timeout是可选的。如果在指定时间内还没有到达，将引发异常。如果远程操作中引发了异常，它将在调用此方法时再次被引发。
- ready()：如果调用完成，返回True
- successful()：如果调用完成且没有引发异常，返回True，如果在结果就绪之前调用此方法，引发异常
- wait([timeout])：等待结果变为可用。
- terminate()：立即终止所有工作进程，同时不执行任何清理或结束任何挂起工作。如果p被垃圾回收，将自动调用此函数







#  Python多线程的通信 

并行应用常常需要在进程之间交换数据。Multiprocessing库有两个Communication Channel可以交换对象：队列(queue)和管道（pipe）。

但是这两个都是适合两个进程间通信的，不适合多个进程通信。

  **Python多线程的通信**

https://zhuanlan.zhihu.com/p/166091204      重要

https://zhuanlan.zhihu.com/p/64702600   



进程是系统独立调度核分配系统资源（CPU、内存）的基本单位，进程之间是相互独立的，每启动一个新的进程相当于把数据进行了一次克隆，子进程里的数据修改无法影响到主进程中的数据，不同子进程之间的数据也不能共享，这是多进程在使用中与多线程最明显的区别。但是难道Python多进程中间难道就是孤立的吗？当然不是，python也提供了多种方法实现了多进程中间的通信和数据共享（可以修改一份数据）



##  队列 Queue

Queue在多线程中也说到过，在生成者消费者模式中使用，是线程安全的，是生产者和消费者中间的数据管道，那在python多进程中，它其实就是进程之间的数据管道，实现进程通信。

在使用多进程的过程中，最好不要使用共享资源。普通的全局变量是不能被子进程所共享的，只有通过Multiprocessing组件构造的数据结构可以被共享。

**Queue**是用来创建进程间资源共享的队列的类，使用Queue可以达到多进程间数据传递的功能（缺点：只适用Process类，不能在Pool进程池中使用）。

构造方法：Queue([maxsize])

- maxsize是队列中允许最大项数，省略则无大小限制。

实例方法：

- put()：用以插入数据到队列。put方法还有两个可选参数：blocked和timeout。如果blocked为True（默认值），并且timeout为正值，该方法会阻塞timeout指定的时间，直到该队列有剩余的空间。如果超时，会抛出Queue.Full异常。如果blocked为False，但该Queue已满，会立即抛出Queue.Full异常。
- get()：可以从队列读取并且删除一个元素。get方法有两个可选参数：blocked和timeout。如果blocked为True（默认值），并且timeout为正值，那么在等待时间内没有取到任何元素，会抛出Queue.Empty异常。如果blocked为False，有两种情况存在，如果Queue有一个值可用，则立即返回该值，否则，如果队列为空，则立即抛出Queue.Empty异常。若不希望在empty的时候抛出异常，令blocked为True或者参数全部置空即可。
- get_nowait()：同q.get(False)
- put_nowait()：同q.put(False)
- empty()：调用此方法时q为空则返回True，该结果不可靠，比如在返回True的过程中，如果队列中又加入了项目。
- full()：调用此方法时q已满则返回True，该结果不可靠，比如在返回True的过程中，如果队列中的项目被取走。
- qsize()：返回队列中目前项目的正确数量，结果也不可靠，理由同q.empty()和q.full()一样







##  管道 Pipe

管道Pipe和Queue的作用大致差不多，也是实现进程间的通信。



多进程还有一种数据传递方式叫管道原理和 Queue相同。Pipe可以在进程之间创建一条管道，并返回元组（conn1,conn2）,其中conn1，conn2表示管道两端的连接对象，强调一点：必须在产生Process对象之前产生管道。

构造方法：Pipe([duplex])

- dumplex:默认管道是全双工的，如果将duplex射成False，conn1只能用于接收，conn2只能用于发送。

实例方法：

- send(obj)：通过连接发送对象。obj是与序列化兼容的任意对象
- recv()：接收conn2.send(obj)发送的对象。如果没有消息可接收，recv方法会一直阻塞。如果连接的另外一端已经关闭，那么recv方法会抛出EOFError。
- close():关闭连接。如果conn1被垃圾回收，将自动调用此方法
- fileno():返回连接使用的整数文件描述符
- poll([timeout]):如果连接上的数据可用，返回True。timeout指定等待的最长时限。如果省略此参数，方法将立即返回结果。如果将timeout射成None，操作将无限期地等待数据到达。
- recv_bytes([maxlength]):接收c.send_bytes()方法发送的一条完整的字节消息。maxlength指定要接收的最大字节数。如果进入的消息，超过了这个最大值，将引发IOError异常，并且在连接上无法进行进一步读取。如果连接的另外一端已经关闭，再也不存在任何数据，将引发EOFError异常。
- send_bytes(buffer [, offset [, size]])：通过连接发送字节数据缓冲区，buffer是支持缓冲区接口的任意对象，offset是缓冲区中的字节偏移量，而size是要发送字节数。结果数据以单条消息的形式发出，然后调用c.recv_bytes()函数进行接收
- recv_bytes_into(buffer [, offset]):接收一条完整的字节消息，并把它保存在buffer对象中，该对象支持可写入的缓冲区接口（即bytearray对象或类似的对象）。offset指定缓冲区中放置消息处的字节位移。返回值是收到的字节数。如果消息长度大于可用的缓冲区空间，将引发BufferTooShort异常。







## 共享内存 Manager

Queue和Pipe只是实现了数据交互，并没实现数据共享，即一个进程去更改另一个进程的数据,那么要用到Managers.

Manager()返回的manager对象控制了一个server进程，此进程包含的python对象可以被其他的进程通过proxies来访问。从而达到多进程间数据通信且安全。Manager模块常与Pool模块一起使用。

Manager支持的类型有list,dict,Namespace,Lock,RLock,Semaphore,BoundedSemaphore,Condition,Event,Queue,Value和Array。

管理器是独立运行的子进程，其中存在真实的对象，并以服务器的形式运行，其他进程通过使用代理访问共享对象，这些代理作为客户端运行。Manager()是BaseManager的子类，返回一个启动的SyncManager()实例，可用于创建共享对象并返回访问这些共享对象的代理。







## Value可以保存数值，Array可以保存数组

multiprocessing 中Value和Array的实现原理都是在共享内存中创建ctypes()对象来达到共享数据的目的，两者实现方法大同小异，只是选用不同的ctypes数据类型而已。

**Value**

构造方法：Value((typecode_or_type, args[, lock])

- typecode_or_type：定义ctypes()对象的类型，可以传Type code或 C Type，具体对照表见下文。
- args：传递给typecode_or_type构造函数的参数
- lock：默认为True，创建一个互斥锁来限制对Value对象的访问，如果传入一个锁，如Lock或RLock的实例，将用于同步。如果传入False，Value的实例就不会被锁保护，它将不是进程安全的。

typecode_or_type支持的类型：

```c++
| Type code | C Type             | Python Type       | Minimum size in bytes |

| --------- | ------------------ | ----------------- | --------------------- |

| `'b'`     | signed char        | int               | 1                     |

| `'B'`     | unsigned char      | int               | 1                     |

| `'u'`     | Py_UNICODE         | Unicode character | 2                     |

| `'h'`     | signed short       | int               | 2                     |

| `'H'`     | unsigned short     | int               | 2                     |

| `'i'`     | signed int         | int               | 2                     |

| `'I'`     | unsigned int       | int               | 2                     |

| `'l'`     | signed long        | int               | 4                     |

| `'L'`     | unsigned long      | int               | 4                     |

| `'q'`     | signed long long   | int               | 8                     |

| `'Q'`     | unsigned long long | int               | 8                     |

| `'f'`     | float              | float             | 4                     |

| `'d'`     | double             | float             | 8                     |
```

参考地址：https://docs.python.org/3/library/array.html

**Array**

构造方法：Array(typecode_or_type, size_or_initializer, **kwds[, lock])

- typecode_or_type：同上
- size_or_initializer：如果它是一个整数，那么它确定数组的长度，并且数组将被初始化为零。否则，size_or_initializer是用于初始化数组的序列，其长度决定数组的长度。
- kwds：传递给typecode_or_type构造函数的参数
- lock：同上





# 同步子进程模块

## Lock（互斥锁）

Lock锁的作用是当多个进程需要访问共享资源的时候，避免访问的冲突。加锁保证了多个进程修改同一块数据时，同一时间只能有一个修改，即串行的修改，牺牲了速度但保证了数据安全。Lock包含两种状态——锁定和非锁定，以及两个基本的方法。

构造方法：Lock()

实例方法：

- acquire([timeout]): 使线程进入同步阻塞状态，尝试获得锁定。
- release(): 释放锁。使用前线程必须已获得锁定，否则将抛出异常。

## RLock（可重入的互斥锁(同一个进程可以多次获得它，同时不会造成阻塞)

RLock（可重入锁）是一个可以被同一个线程请求多次的同步指令。RLock使用了“拥有的线程”和“递归等级”的概念，处于锁定状态时，RLock被某个线程拥有。拥有RLock的线程可以再次调用acquire()，释放锁时需要调用release()相同次数。可以认为RLock包含一个锁定池和一个初始值为0的计数器，每次成功调用 acquire()/release()，计数器将+1/-1，为0时锁处于未锁定状态。

构造方法：RLock()

实例方法：

- acquire([timeout])：同Lock
- release(): 同Lock





## Semaphore（信号量）

信号量是一个更高级的锁机制。信号量内部有一个计数器而不像锁对象内部有锁标识，而且只有当占用信号量的线程数超过信号量时线程才阻塞。这允许了多个线程可以同时访问相同的代码区。比如厕所有3个坑，那最多只允许3个人上厕所，后面的人只能等里面有人出来了才能再进去，如果指定信号量为3，那么来一个人获得一把锁，计数加1，当计数等于3时，后面的人均需要等待。一旦释放，就有人可以获得一把锁。

构造方法：Semaphore([value])

- value：设定信号量，默认值为1

实例方法：

- acquire([timeout])：同Lock
- release(): 同Lock



##  Condition（条件变量）

可以把Condition理解为一把高级的锁，它提供了比Lock, RLock更高级的功能，允许我们能够控制复杂的线程同步问题。Condition在内部维护一个锁对象（默认是RLock），可以在创建Condigtion对象的时候把琐对象作为参数传入。

Condition也提供了acquire, release方法，其含义与锁的acquire, release方法一致，其实它只是简单的调用内部锁对象的对应的方法而已。Condition还提供了其他的一些方法。

构造方法：Condition([lock/rlock])

- 可以传递一个Lock/RLock实例给构造方法，否则它将自己生成一个RLock实例。

实例方法：

- acquire([timeout])：首先进行acquire，然后判断一些条件。如果条件不满足则wait
- release()：释放 Lock
- wait([timeout]): 调用这个方法将使线程进入Condition的等待池等待通知，并释放锁。使用前线程必须已获得锁定，否则将抛出异常。处于wait状态的线程接到通知后会重新判断条件。
- notify(): 调用这个方法将从等待池挑选一个线程并通知，收到通知的线程将自动调用acquire()尝试获得锁定（进入锁定池）；其他线程仍然在等待池中。调用这个方法不会释放锁定。使用前线程必须已获得锁定，否则将抛出异常。
- notifyAll(): 调用这个方法将通知等待池中所有的线程，这些线程都将进入锁定池尝试获得锁定。调用这个方法不会释放锁定。使用前线程必须已获得锁定，否则将抛出异常。

## Event（事件）

Event内部包含了一个标志位，初始的时候为false。可以使用set()来将其设置为true；或者使用clear()将其从新设置为false；可以使用is_set()来检查标志位的状态；另一个最重要的函数就是wait(timeout=None)，用来阻塞当前线程，直到event的内部标志位被设置为true或者timeout超时。如果内部标志位为true则wait()函数理解返回。































































































































































