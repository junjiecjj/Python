#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:27:26 2023

@author: jack


http://mdelrosario.com/2019/06/27/ldpc-example.html


"""

import numpy as np
import time
from pyldpc import make_ldpc, encode, decode, get_message


n = 15
d_v = 4
d_c = 5
snr = 10
H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
k = G.shape[1]
v = np.random.randint(2, size=k)
y = encode(G, v, snr)
d = decode(H, y, snr, maxiter=100, log=True)
x = get_message(G, d)
assert abs(x - v).sum() == 0



# encode/decode messages for different SNR vals
mess_num = int(1e3)
tic_incr = mess_num/4
v = np.random.randint(2, size=(mess_num,k))
min_snr=0
max_snr=10
snrs = np.arange(min_snr,max_snr,0.5)
errs = np.array(())
times = np.array(())
for snr in snrs:
    print
    err_num = 0
    time_tot = 0
    current = time.time()
    for i in range(mess_num):
        v_i = v[i,:]
        y = encode(G, v_i, snr)
        d = decode(H, y, snr, maxiter = 100)
        x = get_message(G, d)
        if abs(x-v_i).sum() != 0 :
            err_num = err_num + 1
        if (i+1) % tic_incr == 0:
            time_tot = timer_update(i, current, time_tot, tic_incr)
    err = float(err_num)/mess_num
    print('SNR: {:04.3f}:\n -> BER: {:03.2f}\n -> Total Time: {:03.2f}s'.format(snr,err,time_tot))
    errs=np.append(errs,err)
    times=np.append(times, time_tot)
