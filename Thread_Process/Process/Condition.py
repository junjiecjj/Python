#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:32:20 2022
https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd
@author: jack
"""

import multiprocessing

import time

def stage_1(cond):

    """perform first stage of work,

    then notify stage_2 to continue

    """

    name = multiprocessing.current_process().name

    print('Starting', name)

    with cond:

        print('{} done and ready for stage 2'.format(name))

        cond.notify_all()

def stage_2(cond):

    """wait for the condition telling us stage_1 is done"""

    name = multiprocessing.current_process().name

    print('Starting', name)

    with cond:

        cond.wait()

        print('{} running'.format(name))

if __name__ == '__main__':

    condition = multiprocessing.Condition()

    s1 = multiprocessing.Process(name='s1',

                                 target=stage_1,

                                 args=(condition,))

    s2_clients = [

        multiprocessing.Process(

            name='stage_2[{}]'.format(i),

            target=stage_2,

            args=(condition,),

        )

        for i in range(1, 3)

    ]

    for c in s2_clients:

        c.start()

        time.sleep(1)

    s1.start()

    s1.join()

    for c in s2_clients:

        c.join()