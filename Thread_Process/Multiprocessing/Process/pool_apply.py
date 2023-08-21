#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:27:12 2022
https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd
@author: jack
"""



## 异步进程池（非阻塞）

from multiprocessing import Pool

def test(i):
    print(f"{i}")
    return

if __name__ == "__main__":

    pool = Pool(8)

    for i in range(1000):
        '''
            实际测试发现，for循环内部执行步骤：
            （1）遍历100个可迭代对象，往进程池放一个子进程
            （2）执行这个子进程，等子进程执行完毕，再往进程池放一个子进程，再执行。（同时只执行一个子进程）
            for循环执行完毕，再执行print函数。
        '''
        pool.apply(test, args=(i,))  ## 维持执行的进程总数为8，当一个进程执行完后启动一个新进程.

    print("test")
    pool.close()
    pool.join()


# import multiprocessing

# def function_square(data):
#     result = data*data
#     return result

# if __name__ == '__main__':
#     inputs = list(range(100))
#     pool = multiprocessing.Pool(processes=4)
#     pool_outputs = pool.map(function_square, inputs)
#     pool.close()
#     pool.join()
#     print ('Pool    :', pool_outputs)
