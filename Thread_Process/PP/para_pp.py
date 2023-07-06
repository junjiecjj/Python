#!/usr/bin/env python
#-*-coding=utf-8-*-

import pp,time,math

def isprime(n):
    if not isinstance(n,int):
        raise TypeError("argument passed to is_prime is not of 'int' type")
    if n<2:
        return False
    if n==2:
        return True
    max=int(math.ceil(math.sqrt(n)))
    i=2
    while i<=max:
        if n%i==0:
            return False
        i+=1
    return True

def sum_prime(n):
    return sum([x for x in range(2,n) if isprime(x)])
'''
#串行测试代码
print("{beg}串行程序{beg}".format(beg='-'*16))

startTime=time.time()

inputs=(100000,100100,100200,0,100900,101000,102000,1000000000)
results =[(input,sum_prime(input)) for input in inputs]

for input,result in results :
    print("sum of primes below %s is %s"%(input,result))

print("用时:%.3fs"%(time.time()-startTime))
'''
#并行代码

print("{beg}并行程序{beg}".format(beg='-'*16))
startTime=time.time()

job_server=pp.Server()
print("pp 可以工作的核心线程数",job_server.get_ncpus(),"worker")

#inputs =(100000,100100,100200,100300,100400,100500,100600,100700)
inputs =(100000000000000,1001000000000000,100200000000000,1003000050000000,100000000000000,1001000000000000,100200000000000,1003000050000000,100000000000000,1001000000000000,100200000000000,1003000050000000,100000000000000,1001000000000000,100200000000000,1003000050000000,100000000000000,1001000000000000,100200000000000,1003000050000000,100000000000000,1001000000000000,100200000000000,1003000050000000,100000000000000,1001000000000000,100200000000000,1003000050000000,100000000000000,1001000000000000,100200000000000,1003000050000000,)

jobs=[(input,job_server.submit(sum_prime,(input,),(isprime,),("math",))) for
      input in inputs]

for input,job in jobs:
    print("sum of primes below %s is %s"%(input,job()))

print("用时:%.3fs"%(time.time()-startTime))
job_server.print_stats()

