#!/usr/bin/env python
#-*-coding=utf-8-*-


from multiprocessing import Pool

def loop(name):
    print('Hello,',name)
    x=0
    while True:
        x+=1

def main():
    name_list=['chen','wang','zhang','jack']
    ps = Pool(4)
    for i in range(4):
        ps.apply_async(loop,(name_list[i],))
    ps.close()
    ps.join()# 这一行必须有，否则进程直接结束
    print("ending!!!!!!!")

if __name__=='__main__':
    main()

"""这是测试利用进程池Pool模块能否达到真正的多进行并行程序的测试代码，结果表明可以。且用了相同的测试
函数，只是传入不同的参数。此代码创建了四个进程，每个进程都是一个死循环，此电脑有4个核，最后对系统
监测表明:四个核都用了，且每个核的利用率为100%，说明是真正的多进程编程"""

