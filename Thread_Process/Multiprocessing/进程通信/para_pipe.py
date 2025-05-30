#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:06:44 2022

@author: jack
"""
import multiprocessing

def create_items(pipe):
    output_pipe, _ = pipe
    for item in range(10):
        output_pipe.send(item)
    output_pipe.close()

def multiply_items(pipe_1, pipe_2):
    close, input_pipe = pipe_1
    close.close()
    output_pipe, _ = pipe_2
    try:
        while True:
            item = input_pipe.recv()
            output_pipe.send(item * item)
    except EOFError:
        output_pipe.close()

if __name__== '__main__':
    # 第一个进程管道发出数字
    pipe_1 = multiprocessing.Pipe(True)
    process_pipe_1 = multiprocessing.Process(target=create_items, args=(pipe_1,))
    process_pipe_1.start()
    # 第二个进程管道接收数字并计算
    pipe_2 = multiprocessing.Pipe(True)
    process_pipe_2 = multiprocessing.Process(target=multiply_items, args=(pipe_1, pipe_2,))
    process_pipe_2.start()
    pipe_1[0].close()
    pipe_2[0].close()
    try:
        while True:
            print(pipe_2[1].recv())
    except EOFError:
        print("End")






##==========================================================================

from multiprocessing import Process, Pipe

import time

# 子进程执行方法

def f(Subconn):

    time.sleep(1)

    Subconn.send("吃了吗")

    print("来自父亲的问候:", Subconn.recv())

    Subconn.close()

if __name__ == "__main__":

    parent_conn, child_conn = Pipe()  # 创建管道两端

    p = Process(target=f, args=(child_conn,))  # 创建子进程

    p.start()

    print("来自儿子的问候:", parent_conn.recv())

    parent_conn.send("嗯")
