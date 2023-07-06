#!/usr/bin/env python
#-*-coding=utf-8-*-

import pp

def loop1(name):
    print("Hello,",name)
    x=0
    while True:
        x+=1

job_server=pp.Server()
name_list=['chen','wang','zhang','jack','cahcau','liang','qing','xia','wo','aini']
print("pp 可以工作的核心数:%d" % job_server.get_ncpus())
#fun_list=[loop1,loop2,loop3,loop4]
for i in range(job_server.get_ncpus()):
    job_server.submit(loop1,(name_list[i],))


"""这是测试利用多线程pp模块能否达到真正的多进行并行程序的测试代码，结果表明可以。且用了相同的测试
函数，只是传入不同的参数。此代码创建了四个进程，每个进程都是一个死循环，此电脑有4个核，最后对系统
监测表明:四个核都用了，且每个核的利用率为100%，说明是真正的多进程编程"""


