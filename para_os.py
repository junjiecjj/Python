#!/usr/bin/env python3
#-*-coding=utf-8-*-
"""
pid=os.fork()
    1.只用在Unix系统中有效，Windows系统中无效
    2.fork函数调用一次，返回两次：在父进程中返回值为子进程id，在子进程中返回值为0
"""
import os

print("Process (%s) get start..."% os.getpid())
pid = os.fork()
if pid==0:
    print("执行子进程，子进程pid=%s;父进程pid=%s"% (os.getpid(),os.getppid()))
    print("pid1=%s"% pid)
else:
    print("执行父进程，子进程pid=%s;父进程pid=%s"% (pid,os.getpid()))
    print('pid2=%s'% pid)
