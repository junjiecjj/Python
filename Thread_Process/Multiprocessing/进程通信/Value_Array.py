#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:29:41 2022

@author: jack

https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd

https://docs.python.org/zh-cn/3/library/multiprocessing.html

进程间共享状态
如上所述，在进行并发编程时，通常最好尽量避免使用共享状态。使用多个进程时尤其如此。

但是，如果你真的需要使用一些共享数据，那么 multiprocessing 提供了两种方法。

(一) 共享内存
    可以使用 Value 或 Array 将数据存储在共享内存映射中。 注意：Value和Array只适用于Process类。
    Value，Array（用于进程通信，资源共享）
    multiprocessing 中Value和Array的实现原理都是在共享内存中创建ctypes()对象来达到共享数据的目的，两者实现方法大同小异，只是选用不同的ctypes数据类型而已。

    (1) Value
    构造方法：Value((typecode_or_type, args[, lock])
    typecode_or_type：定义ctypes()对象的类型，可以传Type code或 C Type，具体对照表见下文。
    args：传递给typecode_or_type构造函数的参数
    lock：默认为True，创建一个互斥锁来限制对Value对象的访问，如果传入一个锁，如Lock或RLock的实例，将用于同步。如果传入False，Value的实例就不会被锁保护，它将不是进程安全的。
    typecode_or_type支持的类型：

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

    (2) Array

    构造方法：Array(typecode_or_type, size_or_initializer, **kwds[, lock])

    typecode_or_type：同上
    size_or_initializer：如果它是一个整数，那么它确定数组的长度，并且数组将被初始化为零。否则，size_or_initializer是用于初始化数组的序列，其长度决定数组的长度。
    kwds：传递给typecode_or_type构造函数的参数
    lock：同上

使用 Process 定义的多进程之间共享变量可以直接使用 multiprocessing 下的 Value，Array，Queue 等，如果要共享 list，dict，可以使用强大的 Manager 模块。也就是说multiprocessing下的共享数据一般只有Value和Array，queue和pipe是特殊的单独的技巧，不属于数据类型，而Manager下有：Value，Array， list, dict

(二) 服务进程
    # https://www.jianshu.com/p/cf617911e0f0

    Manager()返回的manager对象控制了一个server进程，此进程包含的python对象可以被其他的进程通过proxies来访问。从而达到多进程间数据通信且安全。Manager模块常与Pool模块一起使用。
    由 Manager() 返回的管理器对象控制一个服务进程，该进程保存Python对象并允许其他进程使用代理操作它们。
    Manager() 返回的管理器支持类型： list 、 dict 、 Namespace 、 Lock 、 RLock 、 Semaphore 、 BoundedSemaphore 、 Condition 、 Event 、 Barrier 、 Queue 、            Value 和 Array 。
    管理器是独立运行的子进程，其中存在真实的对象，并以服务器的形式运行，其他进程通过使用代理访问共享对象，这些代理作为客户端运行。Manager()是BaseManager的子类，返回一个启动的SyncManager()实例，可用于创建共享对象并返回访问这些共享对象的代理。

    manager = Manager()
    return_list = manager.list()
    pool = Pool(processes=len(BOHAO))
    for host, port in BOHAO.items():
        pool.apply_async(getIpFromAdsl, args=(host, port, return_list))
    pool.close()
    pool.join()
    return_list = list(return_list)




"""

##===================================================================================


import multiprocessing

def f(n, a):
    n.value = 3.14
    a[0] = 5

if __name__ == '__main__':
    num = multiprocessing.Value('d', 0.0)
    arr = multiprocessing.Array('i', range(10))
    p = multiprocessing.Process(target=f, args=(num, arr))
    p.start()
    p.join()
    print(num.value)
    print(arr[:])

# 创建 num 和 arr 时使用的 'd' 和 'i' 参数是 array 模块使用的类型的 typecode ： 'd' 表示双精度浮点数， 'i' 表示有符号整数。这些共享对象将是进程和线程安全的。
# 为了更灵活地使用共享内存，可以使用 multiprocessing.sharedctypes 模块，该模块支持创建从共享内存分配的任意ctypes对象。

# def Value(typecode_or_type: Any, *args: Any, lock: bool = ...) -> sharedctypes._Value: ...
# typecode_to_type = {
#     'c': ctypes.c_char,     'u': ctypes.c_wchar,
#     'b': ctypes.c_byte,     'B': ctypes.c_ubyte,
#     'h': ctypes.c_short,    'H': ctypes.c_ushort,
#     'i': ctypes.c_int,      'I': ctypes.c_uint,
#     'l': ctypes.c_long,     'L': ctypes.c_ulong,
#     'q': ctypes.c_longlong, 'Q': ctypes.c_ulonglong,
#     'f': ctypes.c_float,    'd': ctypes.c_double
#     }
# 3.14
# [5, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#=================================================
## 2
import multiprocessing, time, random
import numpy as np

def f(i, numlock, numulock, idxarr, ):
    time.sleep(random.uniform(0,0.1))
    numlock.value += i
    numulock.value += i
    idxarr[i] = i**2



if __name__ == '__main__':
    n = 100
    process_list = []
    num_lock = multiprocessing.Value('d', 0.0, lock=True)
    num_ulock = multiprocessing.Value('d', 0.0, lock=False)
    arrIdx = multiprocessing.Array('i', range(n))
    # arr_lock = multiprocessing.Array('f', np.zeros((n, )), lock=True)
    # arr_ulock = multiprocessing.Array('f', np.zeros((n, )), lock=False)

    for i in range(100):
        p = multiprocessing.Process(target=f, args=(i, num_lock, num_ulock, arrIdx,  ))
        p.start()
        process_list.append(p)

    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    print(f"num_lock = {num_lock.value}")
    print(f"num_ulock = {num_ulock.value}")
    print(arrIdx[:])



# num_lock = 4892.0
# num_ulock = 4950.0
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936, 2025, 2116, 2209, 2304, 2401, 2500, 2601, 2704, 2809, 2916, 3025, 3136, 3249, 3364, 3481, 3600, 3721, 3844, 3969, 4096, 4225, 4356, 4489, 4624, 4761, 4900, 5041, 5184, 5329, 5476, 5625, 5776, 5929, 6084, 6241, 6400, 6561, 6724, 6889, 7056, 7225, 7396, 7569, 7744, 7921, 8100, 8281, 8464, 8649, 8836, 9025, 9216, 9409, 9604, 9801]

## 很奇怪的是，numlock明明是lock的，但是还是会出错，所以最好别在多进程中改变某个全局变量的值，实在要变，则加锁

#=================================================
def f(i, numlock, numulock, idxarr, ):
    time.sleep(random.uniform(0,0.1))
    numlock.value += i
    numulock.value += i
    idxarr[i] = i**2



if __name__ == '__main__':
    n = 6
    process_list = []
    num_lock = multiprocessing.Value('d', 0.0, lock=True)
    num_ulock = multiprocessing.Value('d', 0.0, lock=False)
    arrIdx = multiprocessing.Array('i', [1,2,3,4,5,6])
    # arr_lock = multiprocessing.Array('f', np.zeros((n, )), lock=True)
    # arr_ulock = multiprocessing.Array('f', np.zeros((n, )), lock=False)

    for i in range(n):
        p = multiprocessing.Process(target=f, args=(i, num_lock, num_ulock, arrIdx,  ))
        p.start()
        process_list.append(p)

    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    print(f"num_lock = {num_lock.value}")
    print(f"num_ulock = {num_ulock.value}")
    print(arrIdx[:])

#=================================================

def f(i, numlock, numulock, idxarr, arr_lock, arr_ulock ):
    time.sleep(random.uniform(0,0.1))
    numlock.value += i
    numulock.value += i
    idxarr[i] = i**2
    arr_lock[i] += i
    # arr_ulock[:] += i


if __name__ == '__main__':
    n = 100
    process_list = []
    num_lock = multiprocessing.Value('d', 0.0, lock=True)
    num_ulock = multiprocessing.Value('d', 0.0, lock=False)
    arrIdx = multiprocessing.Array('i', range(n))
    arr_lock = multiprocessing.Array('f', range(n), lock=True)
    arr_ulock = multiprocessing.Array('f', np.zeros((n, )), lock=False)

    for i in range(n):
        p = multiprocessing.Process(target=f, args=(i, num_lock, num_ulock, arrIdx,arr_lock, arr_ulock ))
        p.start()
        process_list.append(p)

    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行

    print(f"num_lock = {num_lock.value}")
    print(f"num_ulock = {num_ulock.value}")
    print(f"arrIdx[:] = {arrIdx[:]}\n")
    print(f"arr_lock = {arr_lock[:]}\n")
    print(f"arr_ulock = {arr_ulock[:] }")

# num_lock = 4950.0
# num_ulock = 4950.0
# arrIdx[:] = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1089, 1156, 1225, 1296, 1369, 1444, 1521, 1600, 1681, 1764, 1849, 1936, 2025, 2116, 2209, 2304, 2401, 2500, 2601, 2704, 2809, 2916, 3025, 3136, 3249, 3364, 3481, 3600, 3721, 3844, 3969, 4096, 4225, 4356, 4489, 4624, 4761, 4900, 5041, 5184, 5329, 5476, 5625, 5776, 5929, 6084, 6241, 6400, 6561, 6724, 6889, 7056, 7225, 7396, 7569, 7744, 7921, 8100, 8281, 8464, 8649, 8836, 9025, 9216, 9409, 9604, 9801]

# arr_lock = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 122.0, 124.0, 126.0, 128.0, 130.0, 132.0, 134.0, 136.0, 138.0, 140.0, 142.0, 144.0, 146.0, 148.0, 150.0, 152.0, 154.0, 156.0, 158.0, 160.0, 162.0, 164.0, 166.0, 168.0, 170.0, 172.0, 174.0, 176.0, 178.0, 180.0, 182.0, 184.0, 186.0, 188.0, 190.0, 192.0, 194.0, 196.0, 198.0]

# arr_ulock = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


##===================================================================================


###  https://zhuanlan.zhihu.com/p/340657122
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double

class Point(Structure):
    _fields_ = [('x', c_double), ('y', c_double)]

def modify(n, x, s, A):
    n.value **= 2
    x.value **= 2
    s.value = s.value.upper()
    for a in A:
        a.x **= 2
        a.y **= 2

if __name__ == '__main__':
    lock = Lock()

    n = Value('i', 7)
    x = Value(c_double, 1.0/3.0, lock=False)
    s = Array('c', b'hello world', lock=lock)
    A = Array(Point, [(1.875,-6.25), (-5.75,2.0), (2.375,9.5)], lock=lock)

    p = Process(target=modify, args=(n, x, s, A))
    p.start()
    p.join()

    print(n.value)
    print(x.value)
    print(s.value)
    print([(a.x, a.y) for a in A])

# 49
# 0.1111111111111111
# b'HELLO WORLD'
# [(3.515625, 39.0625), (33.0625, 4.0), (5.640625, 90.25)]


##===================================================================================



from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double

class Point(Structure):
    _fields_ = [('x', c_double), ('y', c_double)]

def modify(i, n, x, s, A):
    n.value += i
    x.value += i
    s.value = s.value.upper()
    for a in A:
        a.x  += 2
        a.y += 3

if __name__ == '__main__':
    process_list = []
    lock = Lock()
    N = 10
    n = Value('i', 7)
    x = Value(c_double, 1.0/3.0, lock=False)
    s = Array('c', b'hello world', lock=lock)
    A = Array(Point, [(1.875,-6.25), (-5.75,2.0), (2.375,9.5)], lock=lock)


    for i in range(N):
        p = multiprocessing.Process(target=modify, args=(i, n, x, s, A))
        p.start()
        process_list.append(p)

    for ps in process_list:
        ps.join()  #join应该这么用，千万别直接跟在start后面，这样会变成串行


    print(n.value)
    print(x.value)
    print(s.value)
    print([(a.x, a.y) for a in A])

# 52
# 45.33333333333333
# b'HELLO WORLD'
# [(21.875, 23.75), (14.25, 32.0), (22.375, 39.5)]




















