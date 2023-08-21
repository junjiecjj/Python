

import numpy  as np
import os,time
from multiprocessing import Process, Pool, Lock
import multiprocessing

from pipeline import pipe_code







clients = ["client0", "client1", "client2", "client3",]  # "client4", "client5", "client6", "client7", "client8", "client9"]



process_list  = []
server = multiprocessing.Manager()
dict_res = server.dict()
lock = Lock()

for i, client in enumerate(clients):
    # print(client)
    param_W = {}
    param_W["a"] = np.arange(6).reshape(2, 3) * 0                                                   ##  i, param_W,  2.0 ,  8,  1, dict_res
    param_W["b"] = np.arange(2) * 0
    p = Process(name = client, target = pipe_code, args = (i, param_W,  2.0 ,  8,  1, dict_res, lock) )  ## i = i, param_W = param_W, snr = 2.0 , quantBits = 8, com_round = 1, dic = dict_res
    p.start()
    process_list.append(p)

for pro in process_list:
    pro.join() #join应该这么用，千万别直接跟在start后面，这样会变成串行

print(dict_res)


from pipeline import coder

print(f"最后的 a 为:\n {coder.a} \n")


# for i in range(4):
#     hh.achan(i)
# print("last change hh.a is:\n",hh.a,'\n')









































































