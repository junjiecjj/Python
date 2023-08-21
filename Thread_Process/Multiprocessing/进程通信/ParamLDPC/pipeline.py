

import numpy  as np
import os,time, random
# from multiprocessing import Process, Pool
import multiprocessing

from para_LDPC import LDPC
from sourcesink import SourceSink



coder = LDPC()
print(f"最开始的 encH 为:\n {coder.encH} \n")
print(f"最开始的 a 为:\n {coder.a} \n")
# for i in range(4):
    # coder.achan(i)
# print(f"改变一次后的 a 为:\n {coder.a} \n")




def pipe_code(i , param_W = '', snr = 2.0 , quantBits = 8, com_round = 0, dic = '', lock = None):
    random.seed(i)
    np.random.seed(i)
    ## 信源、统计初始化
    source = SourceSink()
    # source.InitLog(promargs = topargs, codeargs = coderargs)

    name = multiprocessing.current_process().name
    # uu = source.GenerateBitStr(coder.codedim)
    uu = np.random.randint(low = 0, high = 2, size = (coder.codedim, ), dtype = np.int8 )
    cc = coder.encoder(uu)
    uu_hat = np.zeros((coder.codedim, )) + i

    param_recover = {}
    for key in param_W:
        param_recover[key] = param_W[key] + i

    # print(f"{name}: {snr:.2f}, {quantBits} bits, cc = {cc}, uu = {uu} \n")

    dic[name] = param_recover

    # coder.a[i,:] += i
    a, b, c = coder.chan(i)
    if lock != None:
        print(f"i = {i}, {name}: {coder.a}, {b}, {c}")

    # while 1:
        # 1^1

    return













































































