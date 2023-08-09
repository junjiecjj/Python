#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:57:44 2022

https://zhuanlan.zhihu.com/p/435395340

@author: jack
"""

import os
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator
fontpath = "/usr/share/fonts/truetype/windows/"
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font2 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
font2 = FontProperties(fname=fontpath2+"Caskaydia Cove Light Nerd Font Complete Mono.otf", size=20)
font2 = FontProperties(fname=fontpath2+"Caskaydia Cove SemiLight Nerd Font Complete.otf", size=20)
font2 = FontProperties(fname=fontpath2+"Caskaydia Cove Regular Nerd Font Complete Mono.otf", size=20)


P = np.array([[1,1,0,1,1,0,1,1,0], 
              [0,1,1,0,1,1,0,1,1], 
              [1,0,1,1,0,1,1,0,1]])

H = np.hstack([P,np.eye(3,dtype=int)])



def QPSK(cc):
    for i in range(len(cc)):
        # print(f"int(cc[{i}] = {int(cc[i])}")
        if int(cc[i]) == 1:
            cc[i] = -1.0
        elif int(cc[i]) == 0:
            cc[i] = 1.0
    return cc

def ldpc_decode_data(h,delta):
    length = len(h[0])
    uu = np.random.randint(low = 0, high= 2, size=length, dtype='int')
    encode_data = np.zeros(length + len(h),dtype='float')
    encode_data[0:len(uu)] = uu

    for i in range(len(h)):
        tmp = 0
        for j in range(length):
            tmp ^= (h[i][j]&uu[j])
        encode_data[len(uu) + i] = tmp

    encode_data = QPSK(encode_data)
    #encode_data -= 0.5
    #encode_data *= -2

    #numpy.random. normal ( loc=0.0 , scale=1.0 , size=None )loc 均值，scale 标准差，size大小
    yy = np.random.normal(loc=0.0 ,scale=delta, size=len(encode_data))
    yy = encode_data + yy
    
    return uu, yy




def ldpc_decode(h, yy, iter_num = 5, alpha = 0.75):

    LPn = np.zeros(h.shape, dtype = 'float')
    
    #  从变量节点向校验节点发送的置信度
    Lqmn = np.zeros(h.shape, dtype = 'float')
    
    #  从校验节点向变量节点发送的置信度
    Lrmn = np.zeros(h.shape, dtype = 'float')
    
    # 
    LQn = np.zeros(len(Lqmn[0]), dtype = 'float')

    check_data = yy[len(Lqmn[0]):]

    #print(yy)
    #print(check_data)

    for row in range(len(Lqmn)):
        for col in range(len(Lqmn[0])):
            if h[row][col] == 1:
                LPn[row][col] = yy[col]
                Lqmn[row][col] = yy[col]

    for iter in range(iter_num):
        for row in range(len(Lqmn)):
            for col in range(len(Lqmn[0])):
                if h[row][col] == 1:
                    sign = 1.0
                    if check_data[row] < 0:
                        sign = -1.0
                    min_data = abs(check_data[row])

                    for col_idx in range(len(Lqmn[0])):
                        if h[row][col_idx] == 1 and col_idx != col:
                            if Lqmn[row][col_idx] < 0:
                                sign *= -1

                            if abs(Lqmn[row][col_idx]) < min_data:
                                min_data = abs(Lqmn[row][col_idx])

                    Lrmn[row][col] = min_data * alpha * sign
                else:
                    Lrmn[row][col] = 0.0

        #print ("Lrmn:",Lrmn)

        for row in range(len(Lrmn)):
            for col in range(len(Lrmn[0])):
                if h[row][col] == 1:
                    sum_tmp = 0
                    for row_idx in range(len(Lrmn)):
                        if row_idx != row:
                            sum_tmp += Lrmn[row_idx][col]

                    Lqmn[row][col] = LPn[row][col] + sum_tmp

        #print ("Lqmn:",Lqmn)

    for col in range(len(LQn)):
        sum_tmp = 0.0
        for row in range(len(Lqmn)):
            sum_tmp += Lrmn[row][col]

        LQn[col] = sum_tmp

    #print ("LQnIn:",LQn)
    #print ("decode_data:",decode_data)
    LQn = LQn + yy[0:len(LQn)]
    #print ("LQnOut:",LQn)

    bits = bit_judge(LQn)

    return bits



"""
输入：信道编码后以及过信道后接受到的信号,yy。
输出：生成与yy等长的全0的码字，bits，如果yy[i]<0,则bits[i] = 1
"""
def bit_judge(data_in):
    bits = np.zeros(len(data_in),dtype = 'int')
    for i in range(len(bits)):
        if data_in[i] < 0 :
            bits[i] = 1
    return bits


"""
输入：两个等长的比特串。
输出：两个比特串的汉明距离，即不同位的长度。
"""
def err_bit_count(bits0,bits1):
    number = 0
    for i in range(len(bits0)):
        if bits0[i] != bits1[i]:
            number += 1
    return number


def test_once(h,delta = 0.1):
    global epoch

    # 由生成矩阵生成信息位和码字(信道编码后)
    uu, yy = ldpc_decode_data(h, delta)

    #v 软判决译码, 得到译码码字。
    uu_soft_hat = ldpc_decode(h, yy)

    # 硬判决译码, 得到译码码字。
    uu_hard_hat = bit_judge(yy[0:len(h[0])])


    if epoch == 1000:
        print(f"epoch = {epoch}, uu.shape = {uu.shape}, yy.shape = {yy.shape}, uu_soft_hat.shape = {uu_soft_hat.shape},\n")
        # uu.shape = (9,), yy.shape = (12,), uu_hat.shape = (9,)
        print(f"\tuu = {uu}\n\tuu_soft_hat = {uu_soft_hat}\n\tuu_hard_hat = {uu_hard_hat}\n\tyy = {yy}\n")



    total_number = len(uu)
    err_bits_num_soft = err_bit_count(uu, uu_soft_hat)
    err_bits_num_hard = err_bit_count(uu, uu_hard_hat)
    if epoch == 1000:
        print(f"\t\t uu_hard_hat.shape = {uu_hard_hat.shape}, total_number = {total_number}, err_bits_num_soft = {err_bits_num_soft}, err_bits_num_hard = {err_bits_num_hard}\n")

    epoch +=1

    return total_number, err_bits_num_soft, err_bits_num_hard



#if __name__ == '__main__':

# 信噪比起止和步进
snr_start = -1.0
snr_end = 15
snr_step = 1
SNR = []


#
loop_num = 1000
snr_num = int((snr_end - snr_start)/snr_step)
err_soft_log = np.zeros(snr_num,dtype = 'float')
err_hard_log = np.zeros(snr_num,dtype = 'float')

for i in range(snr_num):
    total_acc = 0.0
    err_soft_acc = 0.0
    err_hard_acc = 0.0
    snr = snr_start + snr_step * i
    SNR.append(snr)
    delta = (10**(-snr/10)) ** 0.5
    epoch = 1
    for _ in range(loop_num):
        total_number, err_bits_num_soft, err_bits_num_hard =test_once(P, delta = delta)
        total_acc += total_number
        err_soft_acc += err_bits_num_soft
        err_hard_acc += err_bits_num_hard

    err_soft_log[i] = err_soft_acc/total_acc
    err_hard_log[i] = err_hard_acc/total_acc

print (err_soft_log)
print (err_hard_log)

fig, axs = plt.subplots(1, 1, figsize=(8, 6))

x = np.arange(snr_start, snr_end, snr_step)
axs.plot(x, err_soft_log, label = r"$\mathrm{log(err_{soft})}$")
axs.plot(x, err_hard_log, label = r"$\mathrm{log(err_{hard})}$")
axs.set_yscale('log')

font3 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font3,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs.tick_params(direction='in', axis='both', labelsize=16,top=True,right=True, width=3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('SNR', fontproperties=font)
axs.set_ylabel("log(BER)", fontproperties=font)
axs.set_title("BER", fontproperties=font)


plt.show()
