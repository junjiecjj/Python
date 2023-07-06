#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:32:39 2022
https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844248&idx=1&sn=4b452678e385a29eb86ef450f2d2f6e4&chksm=87eaa0d1b09d29c70e6105aafc3f8805597e69c8104d8cf13342a49fab91b6eada19ec1a0eb2&mpshare=1&scene=1&srcid=1223Fw3CaLXesQMsTOrnXvZG&sharer_sharetime=1647653001990&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ARd7mWwCOVGBFtoIucc13cw%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd
@author: jack
"""

import multiprocessing

import time

def wait_for_event(e):

    """Wait for the event to be set before doing anything"""

    print('wait_for_event: starting')

    e.wait()

    print('wait_for_event: e.is_set()->', e.is_set())

def wait_for_event_timeout(e, t):

    """Wait t seconds and then timeout"""

    print('wait_for_event_timeout: starting')

    e.wait(t)

    print('wait_for_event_timeout: e.is_set()->', e.is_set())

if __name__ == '__main__':

    e = multiprocessing.Event()

    w1 = multiprocessing.Process(

        name='block',

        target=wait_for_event,

        args=(e,),

    )

    w1.start()

    w2 = multiprocessing.Process(

        name='nonblock',

        target=wait_for_event_timeout,

        args=(e, 2),

    )

    w2.start()

    print('main: waiting before calling Event.set()')

    time.sleep(3)

    e.set()

    print('main: event is set')