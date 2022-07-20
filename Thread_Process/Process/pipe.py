#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:30:36 2022

@author: jack
"""

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